from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.detector import NLPDetector
from src.transformer_detector import TransformerDetector

app = FastAPI(title="Fake News / Spam Detector")
templates = Jinja2Templates(directory="templates")

sk_detector = NLPDetector(models_dir="models")
tr_detector = TransformerDetector()


class PredictRequest(BaseModel):
    task: str = Field(pattern="^(spam|fake_news)$")
    text: str = Field(min_length=3)
    use_transformer: bool = False


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "models_dir": str(Path('models').resolve())}


@app.post("/predict")
def predict(payload: PredictRequest):
    if payload.use_transformer:
        try:
            return tr_detector.predict(payload.task, payload.text)
        except Exception:
            return sk_detector.predict(payload.task, payload.text)

    return sk_detector.predict(payload.task, payload.text)
