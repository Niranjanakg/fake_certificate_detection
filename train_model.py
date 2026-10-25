"""Train the demo classifier using data/document_features.csv."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "document_features.csv"
MODEL = ROOT / "model.joblib"

FEATURES = [
    "document_type", "has_qr", "qr_decodable", "has_signature",
    "has_document_number", "has_issuer", "has_date",
    "text_length", "page_count", "metadata_consistent"
]
TARGET = "label"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    X, y = df[FEATURES], df[TARGET].astype(str).str.lower()
    if not set(y.unique()).issubset({"genuine", "suspicious"}):
        raise ValueError("label must contain only 'genuine' or 'suspicious'")
    categorical = ["document_type"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42
        ))
    ])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Holdout accuracy (demo data only): {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump({"model": model, "features": FEATURES}, MODEL)
    print(f"Saved trained model to: {MODEL}")

if __name__ == "__main__":
    main()
