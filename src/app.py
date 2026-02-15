from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import joblib

from config import config
from schema import FEATURE_SCHEMA

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = joblib.load(str())
    app.state.pipe = pipe
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
    pipe = getattr(app.state, "pipe", None)
    return {
        "status": "ok" if pipe is not None else "pipeline_not_loaded",
    }

@app.post("/predict")
def predict(req):
    pipe = getattr(app.state, "pipe", None)
    return ()
