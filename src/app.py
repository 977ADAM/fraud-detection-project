from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import logging

try:
    from .schema import PredictRequest
    from .inference import FraudModel
    from .model_repository import ModelRepository
    from .logging_config import setup_logging
except ImportError:
    from schema import PredictRequest
    from inference import FraudModel
    from model_repository import ModelRepository
    from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        repo = ModelRepository()
        model = FraudModel(repository=repo)
        app.state.model = model
    except Exception:
        logger.exception("Ошибка инициализации модели")
        app.state.model = None
    yield

app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model_not_loaded"
        )
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    model = getattr(app.state, "model", None)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = model.predict(req.customer.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "prediction": result.prediction,
        "label": result.label,
        "probability": result.probability,
    }
