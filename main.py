import os
import logging
import time
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from model_contract import MODEL_VERSION as DEFAULT_MODEL_VERSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log incoming request
        logger.info(f"→ {request.method} {request.url.path}")
        
        # Log request body for POST requests
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    logger.info(f"  Request body size: {len(body)} bytes")
            except Exception as e:
                logger.error(f"  Error reading request body: {e}")
        
        # Track execution time
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response status and timing
            logger.info(f"← {request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time:.2f}s")
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"✗ {request.method} {request.url.path} | Error: {str(e)} | Time: {process_time:.2f}s")
            raise

SCHEMA_VERSION = "1.0"
MODEL_VERSION = os.getenv("AUTHENTICHECK_MODEL_VERSION", DEFAULT_MODEL_VERSION)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    rating: int = Field(ge=1, le=5)
    hasImage: bool = False
    imageUrls: list[HttpUrl] = Field(default_factory=list, max_length=5)

    @field_validator("text")
    @classmethod
    def clean_review_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Review text cannot be blank.")
        return cleaned


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["1.0"] = SCHEMA_VERSION
    platform: Literal["shopee", "lazada"]
    url: HttpUrl
    productTitle: str = Field(default="", max_length=500)
    productDescription: str = Field(default="", max_length=2000)
    productCategory: str = Field(default="", max_length=100, description="Optional legacy metadata; ignored by the model")
    marketplaceRating: float | None = Field(default=None, ge=1, le=5)
    extraction: dict[str, Any] = Field(default_factory=dict)
    reviews: list[ReviewRequest] = Field(min_length=1, max_length=20)


@lru_cache(maxsize=1)
def _pipeline():
    from model_contract import get_inference_bundle
    from stage2.online_inference import OnlineInference

    bundle = get_inference_bundle()
    return OnlineInference(
        roberta_model=bundle / "dost_roberta",
        xgb_path=bundle / "xgboost_meta_classifier.json",
        absa_dir=bundle / "absa_model",
    )


class LazyPipeline:
    """Delay model imports so health endpoints remain available during setup."""

    def __getattr__(self, name):
        return getattr(_pipeline(), name)


pipeline = LazyPipeline()


app = FastAPI(title="AuthentiCheck Backend", version=MODEL_VERSION)
app.state.model_ready = False
app.state.model_error = "Models have not been checked."

# Add request logging middleware (must be added before CORS middleware)
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-p]{32}|http://(?:127\.0\.0\.1|localhost)(?::\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.on_event("startup")
def load_models():
    try:
        if os.getenv("AUTHENTICHECK_PRELOAD_MODELS", "1") == "1":
            pipeline.preload()
        else:
            pipeline.check_artifacts()
        app.state.model_ready = True
        app.state.model_error = None
    except Exception as error:  # Keep health endpoints available for setup diagnostics.
        app.state.model_ready = False
        app.state.model_error = str(error)


@app.get("/health")
def health():
    return {"status": "ok", "schemaVersion": SCHEMA_VERSION, "modelVersion": MODEL_VERSION}


@app.get("/ready")
def ready():
    try:
        pipeline.check_artifacts()
        artifacts_present = True
        artifact_error = None
    except Exception as error:
        artifacts_present = False
        artifact_error = str(error)
    return {
        "ready": bool(app.state.model_ready and artifacts_present),
        "schemaVersion": SCHEMA_VERSION,
        "modelVersion": MODEL_VERSION,
        "error": app.state.model_error or artifact_error,
    }


@app.get("/analyze")
def analyze_usage():
    return {
        "status": "ready" if app.state.model_ready else "not_ready",
        "message": "This endpoint analyzes reviews through an HTTP POST request. The browser extension sends that request automatically.",
        "method": "POST",
        "schemaVersion": SCHEMA_VERSION,
        "modelVersion": MODEL_VERSION,
        "documentation": "/docs",
        "readiness": "/ready",
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail=app.state.model_error or "Model pipeline is not ready.")

    import pandas as pd

    records = [
        {
            "review_id": review.id,
            "text": review.text,
            "star_rating": review.rating,
            "image_urls": [str(url) for url in review.imageUrls],
            "product_description": request.productDescription,
            "product_title": request.productTitle,
            "product_id": str(request.url),
        }
        for review in request.reviews
    ]
    try:
        result = pipeline.run(pd.DataFrame(records))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error
    return {"schemaVersion": SCHEMA_VERSION, "modelVersion": MODEL_VERSION, **result}
