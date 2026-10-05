"""Comparison API; engine failures are reported separately."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.authenticheck_adapter import run_authenticheck
from backend.services.lu_et_al_adapter import BaselineExecutionError, BaselineUnavailable, run_baseline
from backend.services.taxonomy import CATEGORIES

router = APIRouter(prefix="/api/compare", tags=["comparison"])


class ComparisonReview(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=10000)
    star_rating: float | None = Field(default=None, ge=1, le=5)
    image_url: str = ""
    image_urls: list[str] = Field(default_factory=list, max_length=5)


class ComparisonRequest(BaseModel):
    product_title: str = Field(default="", max_length=100)
    product_description: str = Field(default="", max_length=2000)
    reviews: list[ComparisonReview] = Field(min_length=1, max_length=20)


def _metrics(auth, baseline):
    if not auth or not baseline:
        return None
    rows = []
    for ours, theirs in zip(auth, baseline):
        for category in CATEGORIES:
            left = sorted({a["sentiment"] for a in ours["aspects"] if a["category"] == category})
            right = sorted({a["sentiment"] for a in theirs["aspects"] if a["category"] == category})
            if left or right:
                rows.append({
                    "review_id": ours["id"], "category": category,
                    "authenticheck": left, "baseline": right,
                    "status": "agree" if left == right else "diverge",
                })
    return {
        "compared_category_rows": len(rows),
        "agreements": sum(row["status"] == "agree" for row in rows),
        "divergences": sum(row["status"] == "diverge" for row in rows),
        "rows": rows,
        "note": "Descriptive agreement only; baseline aspect sentiment may be projected from holistic sentiment.",
    }


@router.post("/lu-et-al")
def compare_lu_et_al(request: ComparisonRequest):
    reviews = [review.model_dump() for review in request.reviews]
    if len({review["id"] for review in reviews}) != len(reviews):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Review IDs must be unique")

    engines = {}
    try:
        ours = run_authenticheck(reviews, request.product_title, request.product_description)
        engines["authenticheck"] = {"status": "ok", "reviews": ours}
    except Exception as error:
        ours = None
        engines["authenticheck"] = {"status": "unavailable", "message": str(error)}

    try:
        baseline = run_baseline(reviews)
        engines["lu_et_al"] = {"status": "ok", "reviews": baseline}
    except BaselineUnavailable as error:
        baseline = None
        engines["lu_et_al"] = {"status": "unavailable", "message": str(error)}
    except BaselineExecutionError as error:
        baseline = None
        engines["lu_et_al"] = {"status": "error", "message": str(error)}

    return {
        "taxonomy": list(CATEGORIES), "reviews": reviews,
        "engines": engines, "metrics": _metrics(ours, baseline),
    }
