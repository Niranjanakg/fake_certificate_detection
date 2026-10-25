# VeriDoc — Fake Certificate Detection (student ML prototype)

## Important project limitation
`data/document_features.csv` is a **synthetic teaching dataset**, generated for demonstrating a machine-learning workflow. It is not a collection of verified genuine/fake documents, so test accuracy and predictions are not evidence of real-world detection quality. Replace it with a properly labelled, lawfully obtained dataset before presenting the model as validated.

## Features
- Premium responsive dashboard UI.
- Flask backend and upload endpoint.
- PDF text extraction using PyMuPDF; image OCR through Tesseract when available.
- QR code detection/decoding using OpenCV.
- Feature-based Random Forest model trained from CSV.
- Displays model screening result, document signals, and extracted text preview.
- Uploaded files are deleted after processing by the demo server.

## Setup in VS Code (Windows)
1. Install Python 3.10+ and VS Code.
2. Extract this ZIP and open the `fake_certificate_detection` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Train the model:
   ```powershell
   python train_model.py
   ```
   This reads `data/document_features.csv` and creates `model.joblib`.
7. Start the website:
   ```powershell
   python app.py
   ```
8. Open `http://127.0.0.1:5000` in your browser.

## OCR note
For scanned PDFs and images, install the Tesseract OCR application separately and make sure `tesseract.exe` is on PATH. Without it, text extraction from images may be empty. PDFs with selectable text can work without Tesseract.

## How the ML dataset works
Required CSV columns:
- `document_type` (certificate, offer_letter, attendance, internship, other)
- `has_qr`, `qr_decodable`, `has_signature`, `has_document_number`, `has_issuer`, `has_date`
- `text_length`, `page_count`, `metadata_consistent`
- `label` (`genuine` or `suspicious`)

Use labels established from trusted issuer checks or documented review. Do not label a file fake just because it lacks a QR code, signature, or a particular field. Keep separate training and test sets and report precision, recall, F1-score, and false-positive rate.

## Known limitations
- A QR code being present or readable does not prove its contents are authentic. This prototype decodes QR data but does not verify it against an issuer.
- Signature cues are based on extracted text. It does not visually authenticate a handwritten signature or validate cryptographic signatures.
- The demo model uses coarse features and synthetic training data. Its results are for project demonstration only.
- Avoid uploading confidential documents to an exposed/public server. This is a local student prototype, not a production service.
