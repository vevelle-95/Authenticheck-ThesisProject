"""Local HTTP inference for the trained six-aspect CLIP-CA-CG baseline."""

from contextlib import asynccontextmanager
import logging
import os
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from datasets.authenticheck_data import MODEL_VERSION
from inference import BaselinePredictor
from runtime import resolve_path

SCHEMA_VERSION = "1.0"
logger = logging.getLogger(__name__)
prediction_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = None
    app.state.model_error = None
    try:
        checkpoint = os.getenv("CLIP_CA_CG_CHECKPOINT", "").strip() or "outputs/checkpoints/best.pt"
        predictor = BaselinePredictor(resolve_path(checkpoint))
        predictor.validate_cache()
        app.state.predictor = predictor
    except Exception as error:
        app.state.model_error = str(error)
        logger.warning("Baseline setup failed: %s", error)
    yield
    app.state.predictor = None


app = FastAPI(title="CLIP-CA-CG baseline", version=MODEL_VERSION, lifespan=lifespan)
app.state.predictor = None
app.state.model_error = "Models have not been checked."
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-p]{32}|http://(?:127\.0\.0\.1|localhost)(?::\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    imageUrls: list[HttpUrl] = Field(default_factory=list, max_length=5)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Review text must be nonblank")
        return cleaned


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schemaVersion: Literal["1.0"] = SCHEMA_VERSION
    reviews: list[Review] = Field(min_length=1, max_length=20)

    @field_validator("reviews")
    @classmethod
    def unique_ids(cls, reviews):
        if len({review.id for review in reviews}) != len(reviews):
            raise ValueError("Review IDs must be unique")
        return reviews


class AspectPrediction(BaseModel):
    category: Literal["aesthetics", "product_quality", "accuracy_of_description", "design", "value", "seller_service"]
    text: str
    sentiment: Literal["negative", "neutral", "positive"]


class ReviewPrediction(BaseModel):
    id: str
    aspects: list[AspectPrediction]


class PredictionResponse(BaseModel):
    schemaVersion: Literal["1.0"] = SCHEMA_VERSION
    modelVersion: str
    reviews: list[ReviewPrediction]


@app.get("/health")
def health():
    return {"status": "ok", "schemaVersion": SCHEMA_VERSION, "modelVersion": MODEL_VERSION}


@app.get("/ready")
def ready():
    return {
        "ready": app.state.predictor is not None,
        "schemaVersion": SCHEMA_VERSION,
        "modelVersion": MODEL_VERSION,
        "error": app.state.model_error,
    }


@app.post("/predict-aspects", response_model=PredictionResponse)
def predict_aspects(request: PredictionRequest):
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail=app.state.model_error or "Baseline model is not ready.")
    frame = pd.DataFrame([
        {"review_id": review.id, "review_text": review.text,
         "image_urls": [str(url) for url in review.imageUrls]}
        for review in request.reviews
    ])
    try:
        with prediction_lock:
            output, _, _ = predictor.predict(frame, prepare_cache=True)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Baseline prediction failed")
        raise HTTPException(status_code=500, detail="Baseline prediction failed. Check the server logs.") from error
    return {"schemaVersion": SCHEMA_VERSION, "modelVersion": MODEL_VERSION, **output}
