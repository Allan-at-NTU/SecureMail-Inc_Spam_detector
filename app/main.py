from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")

app = FastAPI(title="Spam Detector", version="1.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

# Lazy load
_model = None
def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Spam Detector"})

@app.post("/predict", response_class=JSONResponse)
async def predict(payload: dict):
    text = payload.get("text", "")
    model = get_model()
    pred = int(model.predict([text])[0])
    result = "Spam" if pred == 1 else "Not Spam"
    return {"prediction": pred, "label": result}
