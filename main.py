import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent / "stage2"))
import absa_model
from online_inference import OnlineInference, DEFAULT_XGB_PATH

app = FastAPI(title="AuthentiCheck Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Review(BaseModel):
    review_id: str | None = None
    review_text: str
    image_url: str = ""
    star_rating: float


class AnalyzeRequest(BaseModel):
    product_description: str = ""
    reviews: list[Review]


pipeline = OnlineInference(
    roberta_model=absa_model.STAGE1_MODEL_DIR,
    xgb_path=DEFAULT_XGB_PATH,
    absa_dir=absa_model.DEFAULT_ABSA_MODEL_DIR,
)


@app.on_event("startup")
def load_models():
    pipeline.check_artifacts()


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    df = pd.DataFrame([r.model_dump() for r in request.reviews])
    df = df.rename(columns={"review_text": "text"})
    result = pipeline.run(df)
    return result