from pathlib import Path
import re
import os
import joblib
import fitz
import cv2
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MODEL_PATH = ROOT / "model.joblib"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"}
MAX_BYTES = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
UPLOAD_DIR.mkdir(exist_ok=True)

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_document(path, extension):
    text, page_count, image_for_qr = "", 1, None
    if extension == "pdf":
        doc = fitz.open(path)
        page_count = len(doc)
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
        text = "\n".join(chunks)
        if doc.page_count:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_for_qr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                image_for_qr = cv2.cvtColor(image_for_qr, cv2.COLOR_RGBA2BGR)
            else:
                image_for_qr = cv2.cvtColor(image_for_qr, cv2.COLOR_RGB2BGR)
        # OCR only when the PDF has almost no selectable text
        if len(text.strip()) < 25 and image_for_qr is not None:
            try:
                text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(image_for_qr, cv2.COLOR_BGR2RGB)))
            except Exception:
                pass
    else:
        image_for_qr = cv2.imread(str(path))
        if image_for_qr is None:
            raise ValueError("The image could not be read. Try a PNG or JPG file.")
        try:
            text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(image_for_qr, cv2.COLOR_BGR2RGB)))
        except Exception:
            text = ""
    return text, page_count, image_for_qr

def get_features(filename, text, page_count, image):
    lower = text.lower()
    doc_type = "other"
    name = filename.lower()
    for key, words in {
        "certificate": ["certificate", "certified", "degree", "diploma"],
        "offer_letter": ["offer letter", "employment offer", "joining date", "designation"],
        "attendance": ["attendance", "present", "absent", "roll no", "register number"],
        "internship": ["internship", "intern", "training completion"]
    }.items():
        if any(w in lower for w in words):
            doc_type = key
            break
    qr_text = ""
    qr_found = False
    if image is not None:
        try:
            detector = cv2.QRCodeDetector()
            decoded, points, _ = detector.detectAndDecode(image)
            qr_found = points is not None
            qr_text = decoded or ""
        except Exception:
            pass
    has_number = bool(re.search(r"\b(?:cert(?:ificate)?\s*(?:no|number|#)?|ref(?:erence)?\s*(?:no|number)?|id)\s*[:#\-]?\s*[A-Z0-9][A-Z0-9\-/]{3,}\b", text, re.I))
    has_issuer = any(w in lower for w in ["issued by", "registrar", "principal", "human resources", "hr department", "university", "institute", "college", "company"])
    has_date = bool(re.search(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|20\d{2})\b", text, re.I))
    # A signature is only estimated from text cues; visual signature validation is not performed.
    has_signature = any(w in lower for w in ["signature", "signed by", "authorized signatory", "digitally signed", "e-signed"])
    # Heuristic metadata consistency: useful as a weak signal, not proof.
    metadata_consistent = int(len(text.strip()) >= 80 and page_count >= 1 and (has_date or has_number))
    return {
        "document_type": doc_type,
        "has_qr": int(qr_found),
        "qr_decodable": int(bool(qr_text.strip())),
        "has_signature": int(has_signature),
        "has_document_number": int(has_number),
        "has_issuer": int(has_issuer),
        "has_date": int(has_date),
        "text_length": len(text.strip()),
        "page_count": page_count,
        "metadata_consistent": metadata_consistent,
    }, qr_text

def assess(path, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    text, page_count, image = extract_document(path, ext)
    features, qr_text = get_features(filename, text, page_count, image)
    if not MODEL_PATH.exists():
        raise RuntimeError("Model is not trained yet. Run: python train_model.py")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    import pandas as pd
    row = pd.DataFrame([features], columns=bundle["features"])
    prediction = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    confidence = round(float(probabilities[classes.index(prediction)]) * 100, 1)
    checks = [
        {"name": "QR code detected", "status": "pass" if features["has_qr"] else "review",
         "detail": "A QR code was detected." if features["has_qr"] else "No QR code was detected; some legitimate documents do not use QR codes."},
        {"name": "QR content readable", "status": "pass" if features["qr_decodable"] else ("review" if features["has_qr"] else "neutral"),
         "detail": "QR content could be decoded." if features["qr_decodable"] else "No readable QR content was found."},
        {"name": "Document / certificate number", "status": "pass" if features["has_document_number"] else "review",
         "detail": "A number-like identifier was found in extracted text." if features["has_document_number"] else "No recognizable document number was found."},
        {"name": "Issuer information", "status": "pass" if features["has_issuer"] else "review",
         "detail": "Issuer-related wording was found." if features["has_issuer"] else "Issuer information was not clearly detected."},
        {"name": "Date", "status": "pass" if features["has_date"] else "review",
         "detail": "A date-like value was found." if features["has_date"] else "A date was not clearly detected."},
        {"name": "Signature wording", "status": "pass" if features["has_signature"] else "review",
         "detail": "Signature-related wording was found; the signature itself is not authenticated." if features["has_signature"] else "No signature-related wording was detected; visual signature verification is not performed."},
    ]
    return {
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "document_type": features["document_type"].replace("_", " ").title(),
        "text_length": features["text_length"],
        "page_count": page_count,
        "qr_text": qr_text[:500],
        "checks": checks,
        "extracted_preview": re.sub(r"\s+", " ", text).strip()[:700],
        "note": "This is a machine-learning screening result based on limited document features and a demonstration dataset. It is not proof of authenticity. Verify with the issuing organization before making decisions."
    }

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/analyze")
def analyze():
    if "document" not in request.files:
        return jsonify({"error": "Choose a document first."}), 400
    file = request.files["document"]
    if not file.filename:
        return jsonify({"error": "Choose a document first."}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIF."}), 400
    safe_name = secure_filename(file.filename)
    path = UPLOAD_DIR / safe_name
    # Avoid overwriting another upload with the same name
    stem, suffix = path.stem, path.suffix
    counter = 1
    while path.exists():
        path = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    try:
        file.save(path)
        result = assess(path, safe_name)
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Document analysis failed")
        return jsonify({"error": str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Could not analyze this file. Check that it is a valid PDF or image."}), 500
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File is too large. Maximum upload size is 12 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True)
