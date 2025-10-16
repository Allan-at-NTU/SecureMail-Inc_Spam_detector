from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib, os, json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "metadata.json")

app = FastAPI(title="SecureMail Spam Detector", version="2.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

_model = None
def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "SecureMail • Spam Detector"})

@app.post("/predict", response_class=JSONResponse)
async def predict(payload: dict):
    text = (payload or {}).get("text", "")
    model = get_model()
    pred = int(model.predict([text])[0])
    label = "Spam" if pred == 1 else "Not Spam"
    return {"prediction": pred, "label": label}

@app.get("/metrics", response_class=JSONResponse)
async def metrics():
    """
    Used by the Performance section. We return whatever exists in metadata.json.
    Optional keys: accuracy, precision, recall, f1, pr_auc, roc_auc, confusion_matrix.
    """
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
    else:
        meta = {}
    # Provide safe defaults if some metrics aren't present
    meta.setdefault("selected_model", "N/A")
    meta.setdefault("pr_auc", None)
    meta.setdefault("roc_auc", None)
    return meta
