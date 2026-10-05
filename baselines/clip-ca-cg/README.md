# CLIP-CA-CG baseline: setup, prediction, and frontend connection

This folder contains AuthentiCheck's adapted multimodal comparison model.
It reads review text and buyer photos, then predicts aspect categories and
positive, neutral, or negative sentiment. It does not classify review quality.

The six supported categories are: `aesthetics`, `product_quality`,
`accuracy_of_description`, `design`, `value`, and `seller_service`.

AuthentiCheck's own model keeps its ten-category taxonomy. Use these six shared
categories when comparing the two models.

## 1. Set up after cloning

Install Python 3.11. From the AuthentiCheck project root, run:

```powershell
cd baselines\clip-ca-cg
py -3.11 -m venv .venv
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

For a machine without a supported NVIDIA GPU, add `-CpuOnly`
to the setup command. The remaining commands use this folder's interpreter;
activation is optional.

**GitHub provides the source code. The environment, trained weights, downloaded
models, cached features, and generated reports are ignored by Git.**

Choose one of the following ways to obtain a trained model.

### Use the team's trained model

Ask the person who trained it for an archive containing these directories,
and extract them into this baseline folder:

```text
outputs/
  checkpoints/
    best.pt
    tokenizer/
    training_metadata.json
    training_history.json
  clip_cache/
    encoder.json
    ...cached feature files...
  image_cache/
    ...cached review photos...
```

Keep the tokenizer and CLIP cache together with their matching checkpoint.
`encoder.json` pins the CLIP revision used during training.
The pretrained `outputs/model_cache/` can be shared too; otherwise
CLIP weights are downloaded when uncached reviews need feature extraction.

You can predict new reviews immediately after restoring these files. Moving
this folder does not require another training run.

### Train a model yourself

The shared files are in AuthentiCheck's main `data/` folder:

```text
Authenticheck-ThesisProject/data/test_reviews.csv
Authenticheck-ThesisProject/data/splits.json
```

The config resolves them as `../../data/test_reviews.csv` and
`../../data/splits.json`. The old baseline-local data folder is
not needed. Development split files can stay ignored while the dataset is
changing. After setting up the baseline environment, generate the initial
local splits from the **AuthentiCheck project root**:

```powershell
.\baselines\clip-ca-cg\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits.json
```

This reuses AuthentiCheck's splitter; it does not train a model. An existing
matching file is reused. For a paired experiment, both models must read the
same saved split file and the same dataset snapshot.

When the CSV changes, an old split file is rejected because its dataset
fingerprint no longer matches. Preserve the old experiment and generate a
new file, for example:

```powershell
.\baselines\clip-ca-cg\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits_v2.json
```

Set this baseline's `configs/config.yaml` entry `data.splits` to
`../../data/splits_v2.json`, and pass `--splits data/splits_v2.json` to
AuthentiCheck's own training workflow for that experiment. Train and evaluate
against that dataset version. Archive its CSV, splits, and model metadata
when you freeze an experiment; committing development split files is optional.

From this baseline folder:

```powershell
.\.venv\Scripts\python.exe main.py --validate-only
.\.venv\Scripts\python.exe cache_clip_features.py
.\.venv\Scripts\python.exe main.py
```

The first command checks the data and partitions. The second downloads available
buyer photos and computes frozen CLIP features once. The last trains for the
configured five epochs, evaluates validation data after each epoch, and saves
the best checkpoint.

Training and validation use ground-truth Authentic reviews. Labels come from
`aspect_annotations`; other categories are ignored in this baseline.
No product-category column is required. Test products are held out.

Each fresh training run replaces the files in `outputs/checkpoints/`.
Copy that directory first if you want to retain an earlier run.

## 2. Predict new reviews

For one review, with no image:

```powershell
.\.venv\Scripts\python.exe predict.py --text "Sira agad ang material" --prepare-cache --output outputs/single_review.json
```

To include a buyer photo, add `--image-url "https://..."`.
Repeat that option for multiple photos; the model uses up to five.

For a batch, create a CSV such as `new_reviews.csv`:

```csv
review_id,review_text,review_image_urls
new_001,"Matibay ang material at sulit sa presyo.",
new_002,"Maganda ang kulay.",
```

Then run:

```powershell
.\.venv\Scripts\python.exe predict.py --csv new_reviews.csv --partition all --prepare-cache --output outputs/new_predictions.json
```

New CSVs require unique `review_id` values, nonblank
`review_text`, and a `review_image_urls` column.
Image cells can be blank, pipe-separated URLs, or JSON lists. Prediction requires
no annotations, quality labels, product category, or split file when using
`--partition all`.

`--prepare-cache` computes missing CLIP features before loading the
trained model. Missing or unusable photos produce zero CLIP image vectors and a
CLIP score of 0; the model then uses its text path.

## 3. Read the output

Single-review prediction saves the aspect list directly. This is an illustrative
format, not a guaranteed prediction:

```json
[
  {
    "category": "product_quality",
    "text": "Sira agad ang material",
    "sentiment": "negative"
  }
]
```

CSV prediction wraps those lists with review IDs:

```json
{
  "reviews": [
    {
      "id": "new_001",
      "aspects": [
        {
          "category": "product_quality",
          "text": "Matibay ang material at sulit sa presyo.",
          "sentiment": "positive"
        }
      ]
    }
  ]
}
```

Each detected category receives one sentiment. `text` is the full
review context; the model does not extract evidence snippets. An empty
`aspects` list means no category reached the saved detection threshold.

| Location | Contents |
| --- | --- |
| `outputs/checkpoints/best.pt` | Trained model selected using validation results |
| `outputs/checkpoints/tokenizer/` | Tokenizer required for prediction |
| `outputs/checkpoints/training_history.json` | Per-epoch loss and validation metrics |
| `outputs/clip_cache/` | Frozen CLIP features and encoder revision |
| `outputs/image_cache/` | Downloaded buyer photos |
| `outputs/model_cache/` | Downloaded pretrained model assets |
| `outputs/*.json` | Prediction and evaluation reports |

## 4. Evaluate after training

Once training choices are fixed, run:

```powershell
.\.venv\Scripts\python.exe cache_clip_features.py --partition test
.\.venv\Scripts\python.exe evaluate.py
```

Results are saved to `outputs/test_metrics.json`. They measure aspect
detection and sentiment classification given the annotated categories, on
ground-truth Authentic test reviews. Sentiment accuracy alone does not measure
whether the model found the correct aspects.

The evaluator refuses to overwrite an existing report. Use
`evaluate.py --output outputs/test_metrics_run2.json` for another filename.
The default `predict.py` command predicts every held-out test review.

For comparison with AuthentiCheck, score the same test IDs and the same six
categories, with matching annotation rules. Keep quality-filter evaluation
separate. The current fixture contains synthetic records and draft annotations;
its scores demonstrate the workflow rather than final thesis performance.

## 5. Connect to the frontend

**Current state:** this baseline has Python/CSV prediction scripts, but no HTTP
API or frontend comparison connection. The following is a copyable integration
example; `api.py` is not included yet.

The browser sends reviews to Python services:

```text
Frontend sends the same review IDs, text, and buyer-photo URLs
  -> AuthentiCheck API, port 8000: quality labels and aspect sentiments
  -> Baseline API, port 8001: aspect sentiments
Frontend joins responses by review ID and shows the six shared categories
```

### A. Create the baseline API wrapper

Create `api.py` inside this baseline folder with:

<details>
<summary>Show the complete Python API wrapper</summary>

```python
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, field_validator

from datasets.authenticheck_data import MODEL_VERSION
from inference import predict_frame
from runtime import resolve_path

app = FastAPI(title="CLIP-CA-CG baseline")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-p]{32}|http://(?:127\.0\.0\.1|localhost)(?::\d+)?)$",
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Accept"],
)
prediction_lock = Lock()


class Review(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    imageUrls: list[HttpUrl] = Field(default_factory=list, max_length=5)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value):
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Review text must be nonblank")
        return cleaned


class PredictionRequest(BaseModel):
    schemaVersion: Literal["1.0"] = "1.0"
    reviews: list[Review] = Field(min_length=1, max_length=20)

    @field_validator("reviews")
    @classmethod
    def unique_ids(cls, reviews):
        if len({review.id for review in reviews}) != len(reviews):
            raise ValueError("Review IDs must be unique")
        return reviews


@app.post("/predict-aspects")
def predict_aspects(request: PredictionRequest):
    frame = pd.DataFrame([
        {
            "review_id": review.id,
            "review_text": review.text,
            "image_urls": [str(url) for url in review.imageUrls],
        }
        for review in request.reviews
    ])
    try:
        with prediction_lock:
            output, _, _ = predict_frame(
                frame,
                resolve_path("outputs/checkpoints/best.pt"),
                prepare_cache=True,
            )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "schemaVersion": "1.0",
        "modelVersion": MODEL_VERSION,
        **output,
    }
```

</details>

The wrapper accepts the existing frontend's review fields. Extra fields such as
ratings, platform, and product descriptions are ignored by this baseline.
It returns `reviews[].aspects` containing the canonical category/text/sentiment
objects. It does not invent quality labels or an authentic-review percentage.

Install the HTTP dependencies and start the wrapper from this baseline folder:

```powershell
.\.venv\Scripts\python.exe -m pip install "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "pydantic>=2.7,<3"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

Use `http://127.0.0.1:8001/docs` to test `POST /predict-aspects`
after creating the wrapper. The request needs only:

```json
{
  "schemaVersion": "1.0",
  "reviews": [
    {
      "id": "review_001",
      "text": "Matibay ang material.",
      "imageUrls": []
    }
  ]
}
```

The first request for uncached text/images may download CLIP weights and photos.
This simple wrapper loads the aspect model for each request and serializes
prediction calls. It is a local integration starting point; it does not train
the model or calculate accuracy on new, unlabeled reviews.

### B. Add the frontend comparison call

The existing AuthentiCheck service remains on
`http://127.0.0.1:8000/analyze`. Start it using the project-root
environment and its own configured trained bundle:

```powershell
# Run from the AuthentiCheck project root in another terminal.
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Its `/ready` endpoint must report `ready: true`.
The baseline setup does not install AuthentiCheck's own model artifacts.

For a web frontend, call the new baseline endpoint from the comparison handler:

<details>
<summary>Show the web frontend request example</summary>

```javascript
async function getBaselinePredictions(reviews) {
  const response = await fetch("http://127.0.0.1:8001/predict-aspects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      schemaVersion: "1.0",
      reviews: reviews.map(({ id, text, imageUrls }) => ({
        id,
        text,
        imageUrls: imageUrls ?? []
      }))
    })
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(typeof result.detail === "string"
      ? result.detail
      : `Baseline request failed (${response.status})`);
  }
  if (result.schemaVersion !== "1.0" || !Array.isArray(result.reviews)) {
    throw new Error("Invalid baseline response");
  }
  return new Map(result.reviews.map(review => [String(review.id), review.aspects]));
}
```

</details>

Send the same reviews to AuthentiCheck's existing request handler. Join its
`reviews[].aspectSentiment` with the baseline map by `id`,
filter AuthentiCheck's results to the six shared categories, and normalize its
sentiment names to lowercase. Render review text using `textContent`.
For reviews AuthentiCheck rejects at Stage 1, show that it skipped ABSA; an empty
result there does not mean the models analyzed the same review.

| Frontend file | Integration work |
| --- | --- |
| [extension/background.js](../../extension/background.js) | Add a separate baseline prediction message/HTTP call using the baseline endpoint. The existing AuthentiCheck call stays on port 8000. |
| [extension/content.js](../../extension/content.js) | Send the same analyzable reviews through the new message, join responses by ID, and render a comparison view. |
| [prototype/app.js](../../prototype/app.js) | Replace the illustrative demo with real API calls and comparison rendering if this is the frontend being used. |

The current extension's `normalizeApiResult()` requires quality counts,
review labels, and summary fields. The baseline response needs its own comparison
renderer; it cannot be substituted into that normalizer. The existing extension
manifest already permits localhost API requests.

Serve a web prototype over localhost so the wrapper's CORS rules allow it.
From the project root, an example development server is:

```powershell
.\baselines\clip-ca-cg\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory prototype
```

Open `http://127.0.0.1:5500`. Creating the wrapper and adding the
frontend calls/rendering are separate implementation steps; editing this README
does not create those files or connect the services.

## Source and model scope

The architecture uses RoBERTa with a Bi-GRU, frozen ResNet image regions, cached
frozen CLIP embeddings, cross-attention, gating, and six-aspect sentiment heads.
Frozen encoders, missing-image handling, and aspect-level outputs are adaptations
for this project. This is an adapted baseline, not an official or exact
reproduction of the paper's reported experiments.

Starting repository:
[charlesczar/CLIP-CA-CG-model](https://github.com/charlesczar/CLIP-CA-CG-model),
commit `fa5cf5dc36cf278c57c2eb2a25bb217c5ca92e56`.
Architecture reference:
[Lu et al. (2024)](https://thesai.org/Downloads/Volume15No2/Paper_90-Cross_Modal_Sentiment_Analysis_Based_on_CLIP_Image.pdf).
