# Optional CLI trainer (Do Not Execute Here)
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

RANDOM_STATE = 42

def main(input_csv: str, out_dir: str = "models"):
    df = pd.read_csv(input_csv)
    assert {"text","label"}.issubset(df.columns), "CSV must have text,label"
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"ham":0,"spam":1}).fillna(df["label"]).astype(int)
    X = df["text"].astype(str).values
    y = df["label"].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), strip_accents="unicode", sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE, C=1.0, penalty="l2"))
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_valid)[:,1]
    pr = float(average_precision_score(y_valid, y_proba))
    roc = float(roc_auc_score(y_valid, y_proba))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "best_model.joblib")
    with open(out / "metadata.json", "w") as f:
        json.dump({"selected_model":"tfidf+logreg", "pr_auc": pr, "roc_auc": roc}, f, indent=2)
    print("Saved model with PR-AUC", pr, "ROC-AUC", roc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with text,label")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args.input_csv, args.out_dir)
