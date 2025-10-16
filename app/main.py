from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib, os, math
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")

app = FastAPI(title="Spam Detector", version="1.1.0")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

_model = None
def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def predict_with_confidence(pipeline, text: str):
    """
    Returns: dict(label_int, label_str, confidence, calibrated, explanation)
    - confidence in [0,1]; 'calibrated' marks if it's true probability (predict_proba)
      or an approximation (from decision_function).
    - explanation: list of {token, weight, contribution} when available.
    """
    # Prediction
    label = int(pipeline.predict([text])[0])
    label_str = "Spam" if label == 1 else "Not Spam"

    # Confidence
    calibrated = False
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        # True probabilities available
        calibrated = True
        confidence = float(pipeline.predict_proba([text])[0, 1])
    elif hasattr(pipeline, "decision_function"):
        # Approximate probability from margin
        margin = np.atleast_1d(pipeline.decision_function([text]))[0]
        confidence = float(_sigmoid(margin))
        calibrated = False
    else:
        # Fallback: distance via predict on [text, text]? Not reliable. Use 0.5 neutral.
        confidence = 0.5
        calibrated = False

    # Explanation (feature contributions) — best effort
    explanation = []
    vec = None
    clf = None
    try:
        # Try common step names
        steps = getattr(pipeline, "named_steps", {})
        # Find a vectorizer that exposes feature names
        for name in ["tfidf", "vectorizer"]:
            if name in steps and hasattr(steps[name], "get_feature_names_out"):
                vec = steps[name]
                break
        # Last step is typically classifier
        if steps:
            clf = list(steps.values())[-1]
        else:
            clf = pipeline  # in case it's not a Pipeline
    except Exception:
        pass

    # Only explain when we can get feature names AND some weights
    if vec is not None and hasattr(vec, "get_feature_names_out"):
        try:
            feature_names = vec.get_feature_names_out()
            X = vec.transform([text])  # sparse
            # Get class weights depending on estimator type
            weights = None
            # LogisticRegression / LinearSVC (via base estimator)
            if hasattr(clf, "coef_"):
                weights = clf.coef_[0]
            else:
                # CalibratedClassifierCV with LinearSVC
                # Try to dig out base estimator
                base = getattr(clf, "base_estimator", None)
                if base is not None and hasattr(base, "coef_"):
                    weights = base.coef_[0]
                # MultinomialNB: use log odds between classes
                if weights is None and hasattr(clf, "feature_log_prob_"):
                    log_odds = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
                    weights = log_odds

            if weights is not None:
                # contribution = value * weight for each feature present
                # Convert sparse row to (indices, values)
                row = X.tocoo()
                contribs = []
                for i, v in zip(row.col, row.data):
                    w = weights[i]
                    contribs.append((feature_names[i], float(w), float(v * w)))
                # Sort by absolute contribution and take top few
                contribs.sort(key=lambda t: abs(t[2]), reverse=True)
                for token, w, c in contribs[:10]:
                    explanation.append({"token": token, "weight": w, "contribution": c})
        except Exception:
            # If anything fails, explanation remains empty
            explanation = []

    return {
        "prediction": label,
        "label": label_str,
        "confidence": confidence,
        "calibrated": calibrated,
        "explanation": explanation
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Spam Detector"})

@app.post("/predict", response_class=JSONResponse)
async def predict(payload: dict):
    text = payload.get("text", "")
    model = get_model()
    result = predict_with_confidence(model, text)
    return result
