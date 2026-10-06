# AuthentiCheck project guide

This is the guide for the dataset, AuthentiCheck model, CLIP-CA-CG baseline,
APIs, browser extension, and frontend comparison. Commands use Windows
PowerShell. **Project root** means the `Authenticheck-ThesisProject` folder.

The dataset is still being collected and annotated. Both models have local
practice checkpoints and working APIs. The frontend currently connects to
AuthentiCheck; the CLIP-CA-CG comparison call and display remain to be connected.
Practice results include draft/synthetic data and are development checks.

- [1. What runs](#1-what-runs)
- [2. Set up the environments](#2-set-up-the-environments)
- [3. Dataset and annotation rules](#3-dataset-and-annotation-rules)
- [4. Train, validate, and test](#4-train-validate-and-test)
- [5. Saved files and new-review prediction](#5-saved-files-and-new-review-prediction)
- [6. Run the APIs and connect the frontend](#6-run-the-apis-and-connect-the-frontend)
- [7. Browser extraction and interface behavior](#7-browser-extraction-and-interface-behavior)
- [8. File map, tests, and troubleshooting](#8-file-map-tests-and-troubleshooting)
- [9. Thesis scope and baseline sources](#9-thesis-scope-and-baseline-sources)

## 1. What runs

| Component | Inputs | Output |
| --- | --- | --- |
| AuthentiCheck quality filter | Review text, buyer photos, and stars | `authentic`, `deceptive`, `liv`, or `irrelevant` |
| AuthentiCheck ABSA | Text of reviews predicted Authentic | Ten aspect categories and their sentiments |
| CLIP-CA-CG baseline | Review text and optional buyer photos | Six aspect categories and their sentiments |
| Extension | Buyer reviews already loaded on Shopee/Lazada | Review evidence, coverage, and local/API results |
| Prototype | Illustrative demo data | Interface demonstration |

ABSA means **aspect-based sentiment analysis**. AuthentiCheck first applies this
quality filter:

```text
Review text -> DOST RoBERTa -> four quality probabilities
Review text + usable buyer photos -> multilingual CLIP similarity
Stars -> (stars - 1) / 4
Those six values -> XGBoost -> final quality label
Authentic reviews -> independent DOST ABSA -> aspects and sentiments
```

Both AuthentiCheck encoders start from `dost-asti/RoBERTa-tl-cased`. ABSA is
trained independently; it does not reuse the quality classifier's fine-tuned
weights. XGBoost belongs to the logical quality stage even though its script
lives in `stage2/`. CLIP is pretrained and is not fine-tuned by this pipeline.
The maximum review/photo cosine similarity is used; no usable photos means `0`.
Product titles, descriptions, and category metadata are not learned inputs.

CLIP-CA-CG uses RoBERTa with a Bi-GRU, frozen ResNet image regions, cached frozen
CLIP embeddings, cross-attention, gating, and aspect/sentiment heads. It has no
quality classifier. Missing photos use zero CLIP image features and a masked
visual path. Both models return one sentiment per detected category, rather
than separate opinion spans.

## 2. Set up the environments

Use Python 3.11. Git contains the source, but `.venv`, trained weights, downloaded
encoders, feature caches, and generated reports are ignored. On another machine,
set up the environments and train the models before serving predictions.

**AuthentiCheck environment: run from the project root.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the environment already exists, install dependencies with its interpreter.
Activation is optional; `.\.venv\Scripts\Activate.ps1` enables the shorter
`python` command. Examples here use explicit interpreter paths.

**Baseline environment: open another terminal at the project root.**

```powershell
cd baselines\clip-ca-cg
py -3.11 -m venv .venv
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

For a machine without a supported NVIDIA GPU, add `-CpuOnly` to the setup
command. The baseline has its own environment for its CUDA/Torchvision and
Transformers dependencies. If it was set up before the API was added, run
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` in this folder.

Compatible trained files can also be restored from an existing experiment.
Keep each bundle's tokenizer, configuration, and metadata with its weights;
see [saved files](#5-saved-files-and-new-review-prediction). Moving a compatible
bundle to another folder does not require retraining.

## 3. Dataset and annotation rules

Keep **one row per review** in `data/test_reviews.csv`. Its filename does not
make every row a test example; the saved split manifest determines partitions.

| Column | Use |
| --- | --- |
| `review_id` | Unique, permanent local review identifier |
| `product_id` | Groups reviews from the same actual product listing |
| `review_text` | Buyer-authored text |
| `review_image_urls` | JSON list or pipe-separated buyer-photo URLs; blank if none |
| `star_rating` | Integer from 1 to 5 |
| `ground_truth` | Human quality label: `authentic`, `deceptive`, `liv`, or `irrelevant` |
| `aspect_annotations` | JSON list of category, evidence text, and sentiment |
| `product_title`, `product_description`, `text_label` | Context/source metadata, not learned inputs or targets |
| `aspects`, `aspect_text`, `sentiment` | May remain in the CSV, but training reads `aspect_annotations` |

`product_category` is not required. Annotators use product context to apply the
sensory definition; the model does not automatically suppress sensory predictions
for non-beauty products. Unknown or missing photos produce a CLIP score of `0`.

For "Matibay ang casing pero mahal.", an annotation cell can contain:

```json
[
  {"category": "product_quality", "text": "Matibay ang casing", "sentiment": "positive"},
  {"category": "value", "text": "mahal", "sentiment": "negative"}
]
```

Use `[]` when annotation is complete and no aspects apply. Evidence may
paraphrase the review; exact character matching is not required. An absent
aspect is not a neutral sentiment. Multiple sentiments for the same category
are allowed: preserve positive fragrance and negative texture, for example.
Repeated evidence with the same category/polarity is grouped once. Training
uses an equal-weight distribution over distinct polarities, while inference
returns one sentiment per detected category.

The CSV includes assistant-filled draft annotations and synthetic practice
reviews. Their local records are `reports/generated/csv_annotation_fill.json`
and `reports/generated/synthetic_reviews_added.json`; synthetic IDs start with
`synthetic_review_` and `synthetic_prod_`. Review drafts before thesis evaluation.
The pipeline does not generate synthetic data itself.

### Fixed taxonomy

The table preserves AuthentiCheck's category order. Code and saved label mappings
use the identifiers defined in [model_contract.py](../model_contract.py).

| Category | Definition |
| --- | --- |
| `product_quality` | Overall build, material quality, durability, or craftsmanship. |
| `functionality` | Whether the product has the expected features or capabilities. |
| `performance` | How well functions work: speed, responsiveness, lag, accuracy, sound, battery performance, results, effectiveness, or reliability. |
| `design` | Ergonomics, physical form, layout, fit, comfort, feel in the hand, non-beauty texture, or ease of use. |
| `aesthetics` | Purely visual appeal: color, appearance, style, or look. |
| `sensory_experience` | Beauty, skincare, and personal care only: smell, fragrance, texture, consistency, or feel on the skin; for example, "mabango", "walang amoy", "creamy", or "foamy". |
| `value` | Whether the product is worth its price: sulit, affordable, expensive, or overpriced. |
| `packaging` | Box condition, wrapping, protection, presentation, or secure packing. |
| `seller_service` | Concrete seller responsiveness, communication, helpfulness, problem resolution, support, or assistance; general thanks/praise alone does not qualify. |
| `accuracy_of_description` | Reviewer-stated agreement or mismatch with advertised size, color, variation, specifications, or features. |

Bluetooth availability is functionality; Bluetooth lag is performance. Comfort
is design; appearance is aesthetics. A listing comparison is
accuracy_of_description, while a standalone color preference is aesthetics.
Delivery speed/courier behavior has no category here and should not automatically
be assigned to seller service or packaging. Mentioning a property alone does not
establish an evaluative opinion. Aspect annotation and review-quality labeling
are separate decisions; being outside this taxonomy does not determine quality.

The baseline supports these six shared categories, in this order:
`aesthetics`, `product_quality`, `accuracy_of_description`, `design`, `value`,
`seller_service`. Its loader ignores the other four categories without merging
them into these six. The thesis methods should describe the agreed taxonomy and
the current annotation/target policy.

## 4. Train, validate, and test

Training teaches the models. Validation selects settings/checkpoints. Final
testing measures the selected models on products they have not trained on.
Use the same dataset snapshot and saved split manifest for both models.

### Prepare shared product splits

Run from the project root:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --validate-only
.\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits.json
```

These commands validate the CSV and create/reuse `data/splits.json`; they do not
train weights. The baseline environment can run the same preparation script
from the root using `baselines\clip-ca-cg\.venv\Scripts\python.exe`.

The approximate split is 70% train, 15% validation, and 15% test, with each product
entirely in one partition. Training also has five product-disjoint OOF folds.
The splitter requires all four quality classes in each partition, at least five
training products per class, and all fitting classes in every OOF run. It fails
if suitable groups cannot be formed; it does not silently use overlapping
product partitions. AuthentiCheck ABSA also requires all three sentiments among
Authentic training reviews.

OOF means **out-of-fold**: each training review receives quality probabilities
from a RoBERTa model fitted on the other four folds. Those probabilities train
XGBoost. Final RoBERTa is a separate fit on the full training partition and is
used for validation/new reviews. The fold models are not passed into final
RoBERTa, and normal prediction does not load them.

### Train AuthentiCheck

Run from the project root:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --splits data/splits.json --epochs 3 --batch-size 2
```

The pipeline generates OOF probabilities, trains final RoBERTa, extracts six
features, fits XGBoost, then trains independent ABSA. Test products are excluded.
Fold training uses its fixed epoch setting; held-fold reviews do not select its
checkpoint. Final RoBERTa and XGBoost are selected using validation macro-F1;
XGBoost class weights come from training data. ABSA uses human-labeled Authentic
reviews and selects its checkpoint using detection and sentiment validation
macro-F1. Its validation-selected detection threshold controls which aspects
appear, not which sentiment wins.

For a separate experiment, choose separate destinations:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --splits data/experiment_01/splits.json --work-dir data/experiment_01/features --output models/experiment_01 --epochs 3 --batch-size 2
```

### Train CLIP-CA-CG

Run from `baselines/clip-ca-cg` after preparing the shared splits:

```powershell
.\.venv\Scripts\python.exe main.py --validate-only
.\.venv\Scripts\python.exe cache_clip_features.py
.\.venv\Scripts\python.exe main.py
```

[configs/config.yaml](../baselines/clip-ca-cg/configs/config.yaml) points to
`../../data/test_reviews.csv` and `../../data/splits.json`. It defaults to five
epochs, batch size 2, maximum text length 128, up to five photos, and automatic
device selection. Caching prepares training/validation CLIP features and photos.
Training/validation use ground-truth Authentic reviews and their six-category
annotations. A fresh run replaces its configured checkpoint files; use separate
output directories in a copied config to retain experiments.

### When the dataset changes

Manifests contain a dataset fingerprint. Editing CSV content invalidates the old
manifest. Create a new version from the project root:

```powershell
.\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits_v2.json
```

Use `--splits data/splits_v2.json` for AuthentiCheck. Set baseline `data.splits`
to `../../data/splits_v2.json` and train against the same snapshot. Preserve the
old CSV/splits/models if retaining that experiment. Development splits can stay
ignored while annotation changes; archive the frozen experiment for final work.

### Evaluate selected models

For the default AuthentiCheck bundle, run from the project root:

```powershell
.\.venv\Scripts\python.exe stage2/evaluate_models.py --data data/test_reviews.csv --splits data/splits.json --bundle models/own_model_v2 --output results/own_model_test.json
```

For an experiment or the existing practice bundle, pass its matching `--bundle`,
`--splits`, and a new report filename. Evaluation checks training/validation
membership and available dataset fingerprints. It reports quality metrics,
aspect detection, sentiment given human target categories, and product sentiment
errors with/without filtering. Product sentiment averages category outputs
using positive `+1`, neutral `0`, negative `-1`; it is separate from average
stars. Undefined product aggregates are excluded from paired error analysis.

For the baseline, run from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe cache_clip_features.py --partition test
.\.venv\Scripts\python.exe evaluate.py
```

This saves `outputs/test_metrics.json` for ground-truth Authentic test reviews.
Existing final reports are not overwritten; use a new `--output` for a separate
experiment. Freeze training choices before final testing. A new filename does
not make a test set used for tuning an unbiased final evaluation set.

For ABSA comparison, score the same human-labeled Authentic test IDs and six
shared categories, supplying the same human target categories for sentiment
metrics. Compare detection separately and assess the quality filter separately.
Sentiment accuracy alone does not establish correct aspect detection. There is
no general paired-comparison CLI or completed formal statistical protocol yet.

## 5. Saved files and new-review prediction

| Location, relative to the project root | Contents |
| --- | --- |
| `data/splits.json` | Shared review/product partitions and OOF folds |
| `data/training_v2/oof_probabilities.csv` | OOF probabilities for training reviews |
| `data/training_v2/6d_features.csv` | Training/validation features for XGBoost |
| `models/own_model_v2/dost_roberta/` | Final quality encoder, tokenizer, configuration, and metadata |
| `models/own_model_v2/xgboost_meta_classifier.json` | Quality filter, with adjacent `.metadata.json` |
| `models/own_model_v2/absa_model/` | `model.pt`, `absa_config.json`, `encoder/config.json`, tokenizer, and metadata |
| `models/own_model_v2/oof/fold_0/` through `fold_4/` | Training-feature generation models |
| `models/own_model_v2/manifest.json` | Bundle versions, environment, taxonomy, and experiment settings |
| `models/practice_run/` | Current local AuthentiCheck practice bundle, with the same structure |
| `results/*.json` | AuthentiCheck prediction/evaluation reports |
| `reports/generated/*.json` | Draft annotation/synthetic generation records |
| `baselines/clip-ca-cg/outputs/checkpoints/` | `best.pt`, tokenizer, training metadata, and epoch history |
| `baselines/clip-ca-cg/outputs/clip_cache/` | Content-keyed features and `encoder.json` with pinned CLIP revision |
| `baselines/clip-ca-cg/outputs/image_cache/` | Cached buyer photos |
| `baselines/clip-ca-cg/outputs/model_cache/` | Downloaded pretrained model assets |
| `baselines/clip-ca-cg/outputs/*.json` | Baseline predictions and evaluation reports |

Final RoBERTa saves epoch checkpoints under its `checkpoints/` directory, limits
retention, and exports the selected model in `dost_roberta/`. ABSA saves the best
validation model to `absa_model/model.pt`; its history is in training metadata.
The baseline saves its best model to `outputs/checkpoints/best.pt` and records
epoch results in `training_history.json`.

If restoring an existing baseline, keep `outputs/checkpoints/`,
`outputs/clip_cache/`, and `outputs/image_cache/` together. The model cache can
also be copied; otherwise uncached reviews may download CLIP assets. The saved
CLIP revision must match training. An empty `outputs` directory is created as
needed during training; cloning source alone does not provide trained files.

The tracked `models/manifest.json` and [scripts/setup_models.py](../scripts/setup_models.py)
describe an older download bundle. That installer does not create the current
ten-category model. Compatibility is determined from saved versions/configuration,
not the directory name.

### Predict a new CSV with AuthentiCheck

New data needs review text and integer stars. Include IDs for matching/grouping;
photos are optional. Human labels and annotations are not needed. For the current
practice bundle, run from the root:

```powershell
.\.venv\Scripts\python.exe stage2/online_inference.py --data data/new_reviews.csv --roberta-model models/practice_run/dost_roberta --xgb-path models/practice_run/xgboost_meta_classifier.json --absa-dir models/practice_run/absa_model --output results/new_review_predictions.json
```

The API bundle environment variable does not set these CLI arguments. To use
the default trained bundle, omit the three model-path options. ABSA-only diagnosis
is available through `stage2/predict_absa.py`; it bypasses the full quality filter.

### Predict with the baseline CLI

Run from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe predict.py --text "Sira agad ang material" --prepare-cache --output outputs/single_review.json
.\.venv\Scripts\python.exe predict.py --csv new_reviews.csv --partition all --prepare-cache --output outputs/new_predictions.json
```

A new baseline CSV needs `review_id`, `review_text`, and `review_image_urls`.
Photo cells can be blank. With `--partition all`, it needs no split file. For
single-review photos, add `--image-url "https://..."`, repeating it as needed.
Single-review output is an aspect list; CSV output wraps lists under
`reviews[].aspects` with IDs. `--prepare-cache` computes missing features before
prediction. No detected categories gives `aspects: []`; aspect text is full
review context, not an extracted evidence snippet.

## 6. Run the APIs and connect the frontend

### Start both services

**Terminal A: project root, using the current practice bundle.**

```powershell
$env:AUTHENTICHECK_MODEL_BUNDLE = "models/practice_run"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

`AUTHENTICHECK_MODEL_BUNDLE` selects all three AuthentiCheck models for the API.
Relative paths resolve from the project root. Without it, the default is
`models/own_model_v2`. Restart the API after changing the setting. It affects
API inference; pipeline training still uses its default or explicit `--output`.

**Terminal B: open at the project root, then enter the baseline folder.**

```powershell
cd baselines\clip-ca-cg
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

The baseline loads `outputs/checkpoints/best.pt` once and serializes predictions.
`CLIP_CA_CG_CHECKPOINT` can select another checkpoint before startup; relative
paths resolve from the baseline folder. The tokenizer path is recorded in the
checkpoint configuration. Keep matching CLIP metadata available at startup.

| Service | Health | Readiness | Interactive API docs | Prediction |
| --- | --- | --- | --- | --- |
| AuthentiCheck | `http://127.0.0.1:8000/health` | `http://127.0.0.1:8000/ready` | `http://127.0.0.1:8000/docs` | `POST /analyze` |
| CLIP-CA-CG | `http://127.0.0.1:8001/health` | `http://127.0.0.1:8001/ready` | `http://127.0.0.1:8001/docs` | `POST /predict-aspects` |

`/health` confirms the server is running; `/ready` must report `ready: true`
before model use. Missing/incompatible artifacts produce readiness errors and
HTTP 503 for predictions. API requests accept 1 to 20 reviews, text up to 2,000
characters, and up to five HTTP/HTTPS photo URLs per review. Use unique IDs.
Requests with invalid fields return 422; unexpected inference failures return
500. New uncached photos/features may take longer. Both APIs permit localhost
web frontends and Chrome/Edge extension origins.

### Request and response fields

AuthentiCheck needs platform/product context and review stars. A valid example:

```json
{
  "schemaVersion": "1.0",
  "platform": "shopee",
  "url": "https://shopee.ph/product/demo",
  "productTitle": "Product name",
  "productDescription": "Visible description",
  "reviews": [
    {"id": "review_001", "text": "Matibay ang material.", "rating": 5, "hasImage": false, "imageUrls": []}
  ]
}
```

Only send supported review fields to AuthentiCheck; keep extraction eligibility
and missing-field diagnostics in the frontend. Its quality model requires an
integer rating from 1 to 5. The baseline needs only this subset:

```json
{
  "schemaVersion": "1.0",
  "reviews": [
    {"id": "review_001", "text": "Matibay ang material.", "imageUrls": []}
  ]
}
```

The baseline ignores additional ratings/platform/product fields, so the existing
AuthentiCheck payload can also be sent to it. It uses no ground-truth labels.
Baseline response structure, with illustrative predictions:

```json
{
  "schemaVersion": "1.0",
  "modelVersion": "adapted-clip-ca-cg-aspects-v1",
  "reviews": [
    {
      "id": "review_001",
      "aspects": [
        {"category": "product_quality", "text": "Matibay ang material.", "sentiment": "positive"}
      ]
    }
  ]
}
```

| Meaning | AuthentiCheck | Baseline |
| --- | --- | --- |
| Review identifier | `reviews[].id` | `reviews[].id` |
| Review text | `reviews[].text` | Request text; also complete normalized text in each aspect |
| Review quality | `reviews[].label`, probabilities, confidence | Not provided |
| Aspect predictions | `reviews[].aspectSentiment` | `reviews[].aspects` |
| Category | Aspect item's `category` | Aspect item's `category` |
| Sentiment | `Positive`, `Neutral`, `Negative` | `positive`, `neutral`, `negative` |
| Quality/UI summaries | `reviewCount`, `counts`, `authenticShare`, `verifiedRating`, `sentimentCounts`, `aspects` | Not provided |

AuthentiCheck's `authenticShare` is the percentage classified Authentic, not
statistical model confidence. `verifiedRating` averages stars of reviews
classified Authentic. Predicted sentiments and product aggregates are separate
from these stars. The baseline preserves every ID even if its aspect list is empty.

### Frontend developer handoff

The baseline HTTP service is implemented. The remaining frontend work is:

1. Send the same IDs, text, and buyer-photo URLs to both endpoints.
2. Join responses by ID and show results on the same review row.
3. Compare the six shared categories and normalize sentiment names to lowercase.
4. For reviews AuthentiCheck rejects, show **ABSA skipped by quality filter**.
   Distinguish this from an analyzed review with no detected aspects.
5. Keep AuthentiCheck's quality summaries with AuthentiCheck. Show loading,
   empty, unavailable-input, and API-error states; render review text with `textContent`.

| File | Connection work |
| --- | --- |
| [extension/background.js](../extension/background.js) | Add a separate baseline message/HTTP call; current AuthentiCheck call uses port 8000. |
| [extension/content.js](../extension/content.js) | Send matched reviews, join IDs, and render the comparison. |
| [extension/popup.js](../extension/popup.js) | Extend settings if exposing a baseline endpoint or comparison mode. |
| [prototype/app.js](../prototype/app.js) | Connect real requests/rendering if using this standalone demo as the frontend. |

The existing extension's `normalizeApiResult()` requires AuthentiCheck quality
counts, labels, and summaries. Baseline responses need their own comparison
handler. The extension manifest already allows localhost requests. For a web
frontend, a baseline request can use:

```javascript
async function getBaselinePredictions(reviews) {
  const response = await fetch("http://127.0.0.1:8001/predict-aspects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      schemaVersion: "1.0",
      reviews: reviews.map(({ id, text, imageUrls }) => ({ id, text, imageUrls: imageUrls ?? [] }))
    })
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Baseline failed (${response.status})`);
  if (result.schemaVersion !== "1.0" || !Array.isArray(result.reviews)) throw new Error("Invalid baseline response");
  return new Map(result.reviews.map(review => [review.id, review.aspects]));
}
```

This is a handoff example; the frontend files do not call it yet. Validate that
returned IDs/categories match the request and handle timeouts/errors in the UI.
The existing AuthentiCheck extension call has a timeout of at least 60 seconds.

## 7. Browser extraction and interface behavior

Load the `extension/` folder through Chrome/Edge's **Load unpacked** option in
developer mode. Open a Shopee Philippines or Lazada Philippines product page,
scroll until buyer reviews are visible, and use the floating badge. Reload the
extension after source changes. To use trained results, enable **Use model API**
in its popup and set `http://127.0.0.1:8000/analyze`.

The default local estimate uses Tagalog/Taglish heuristics for interface testing.
It is not the trained model. API failures are shown explicitly; the user can
deliberately select the labeled local preview. The standalone prototype also
uses illustrative data until real calls are connected. Serve it from the root:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory prototype
```

Open `http://127.0.0.1:5500` so localhost CORS rules apply.

### Buyer-review boundary

Adapters extract one record per buyer review already loaded in the page:

```json
{
  "id": "shopee-1",
  "text": "Buyer-authored review text only",
  "rating": 5,
  "imageUrls": ["https://example.com/buyer-photo.jpg"],
  "hasImage": true,
  "analysisEligible": true,
  "missingFields": []
}
```

Text, stars, and up to five photos must come from the same review-card root.
Missing text is `""`; an undetected rating is `null`; `hasImage` follows the URL
list. `analysisEligible` currently indicates available written text. Reviews
without text remain visible as **Not analyzed** and are excluded from model
requests. Missing ratings still need to be handled for AuthentiCheck's quality
model; do not invent them.

Remove seller/shop replies before reading buyer text/media. Exclude reviewer
usernames/avatars, seller names/logos, timestamps, variation labels, buttons,
like counts, and report controls. Never read account, cart, payment, delivery
address, or checkout data. Deduplicate nested card matches and duplicate
nonempty text; rating-only/photo-only cards remain separate evidence records.
Process the first 20 loaded reviews without automated scrolling, pagination,
or private marketplace API calls.

Diagnostics include candidate/accepted cards, analyzable reviews, duplicates,
reviews without text, detected stars, buyer photos, and removed seller responses.
Use them to inspect extraction before interpreting predictions. Marketplace DOM
changes belong in the appropriate [extractor](../extension/extractors/);
shared buyer-only rules live in `buyer-reviews.js`.

Keep loading, empty, model-error, recovery, and result states clear. Evidence
search, quality filters, input coverage, and JSON export refer to the actual
analyzed reviews. The interface supports keyboard use and responsive layouts.
Buyer photo bytes are processed in memory by AuthentiCheck; the baseline
caches photos/features locally. Review content is sent to the local service
when the user enables API mode.

## 8. File map, tests, and troubleshooting

| File/directory | Purpose |
| --- | --- |
| [pipeline.py](../pipeline.py) | Runs AuthentiCheck preparation, OOF fitting, training, and validation. |
| [model_contract.py](../model_contract.py) | Shared labels, taxonomy, feature order, accepted formats, and API bundle selection. |
| [model_data.py](../model_data.py) | CSV validation, fingerprints, and product split logic. |
| [model_metrics.py](../model_metrics.py) | Accuracy, precision, recall, F1, and confusion matrices. |
| [stage1/prepare_splits.py](../stage1/prepare_splits.py) | CLI for the shared split logic; either environment can invoke it from the root. |
| [stage1/generate_oof_features.py](../stage1/generate_oof_features.py) | Five fold fits and held-fold quality probabilities. |
| [stage1/train_roberta.py](../stage1/train_roberta.py) | Final quality text encoder. |
| [stage1/features.py](../stage1/features.py) | Shared offline/online six-feature extraction. |
| [stage1/extract_6d_features.py](../stage1/extract_6d_features.py) | Writes training/validation feature CSV. |
| [stage2/train_xgboost.py](../stage2/train_xgboost.py) | Fits the final quality filter. |
| [stage2/absa_model.py](../stage2/absa_model.py) and [fine_tune_absa.py](../stage2/fine_tune_absa.py) | Category detection, conditioned sentiment, and ABSA training. |
| [stage2/online_inference.py](../stage2/online_inference.py) | Full prediction and aggregation. |
| [stage2/predict_absa.py](../stage2/predict_absa.py) | ABSA-only diagnosis. |
| [stage2/evaluate_models.py](../stage2/evaluate_models.py) | Held-out AuthentiCheck evaluation. |
| [main.py](../main.py) | AuthentiCheck HTTP API. |
| [baseline main.py](../baselines/clip-ca-cg/main.py) | Baseline validation/training entry point. |
| [baseline datasets/](../baselines/clip-ca-cg/datasets/) | CSV mapping, shared manifests, cached CLIP, and photos. |
| [baseline models/](../baselines/clip-ca-cg/models/) and [training/](../baselines/clip-ca-cg/training/) | Baseline architecture, losses, fitting, and metrics. |
| [baseline inference.py](../baselines/clip-ca-cg/inference.py) and [predict.py](../baselines/clip-ca-cg/predict.py) | Reusable trained predictor and CLI. |
| [baseline evaluate.py](../baselines/clip-ca-cg/evaluate.py) | Held-out baseline evaluation. |
| [baseline api.py](../baselines/clip-ca-cg/api.py) | HTTP predictions and readiness checks. |
| `extension/`, `prototype/` | Browser interface/extractors and standalone demonstration. |

Keep `tests/`; it checks software behavior and is separate from the thesis's
test partition. Tests do not create the team's trained bundle. The own-model
integration test uses a temporary tiny random encoder and mocked image features;
baseline tests also use miniature offline fixtures. Their scores are not thesis
accuracy measurements.

From the root, run the relevant own-model/API checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_own_model*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_api_contract.py -v
```

From `baselines/clip-ca-cg`, run baseline/API checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Extension checks, when Node is installed, run from the root:

```powershell
node tests/check_extension_contract.mjs
node tests/test_extension_runtime.mjs
```

| Problem | Check |
| --- | --- |
| `/health` works but `/ready` is false | Read its `error`; confirm matching weights/tokenizers/metadata and the selected folder, then restart. |
| Dataset fingerprint mismatch | Generate a new split version and use it for both newly trained models. |
| Five product folds cannot be formed | Check product/class coverage; a row-level fallback is not used. |
| CLIP revision mismatch or cached image bytes missing | Restore the matching cache; rebuild features with `--force` if photos/features are inconsistent. |
| HTTP 422 | Check schema version, supported fields, nonblank text, review count, and photo URLs; baseline IDs must be unique. |
| Inference fails with missing/invalid stars | AuthentiCheck requires integer ratings 1 to 5; inspect extraction instead of filling invented ratings. |
| No baseline comparison appears | The service exists, but the frontend comparison message/call/rendering still needs implementation. |
| No models/outputs after cloning | Train locally or restore a compatible bundle; ignored files are not supplied by Git. |

## 9. Thesis scope and baseline sources

Current code supports the fixed taxonomy, shared product splits, OOF features,
separate final testing, missing-image handling, both prediction APIs, and local
practice runs. Final real-dataset training/evaluation, annotation agreement,
formal statistical tests, empirical Tagalog/Taglish CLIP alignment, and frontend
comparison integration remain work to complete. Use human-labeled unseen products
for final accuracy claims, and record dataset versions, seeds, environments,
thresholds, training settings, and model versions for reproducibility.

The baseline is an adaptation for this project's categories, frozen encoders,
missing photos, and aspect-level outputs. Its reference does not establish an
exact reproduction of the paper's reported experiments.

Starting repository: [charlesczar/CLIP-CA-CG-model](https://github.com/charlesczar/CLIP-CA-CG-model),
commit `fa5cf5dc36cf278c57c2eb2a25bb217c5ca92e56`.
Architecture reference: [Lu et al. (2024)](https://thesai.org/Downloads/Volume15No2/Paper_90-Cross_Modal_Sentiment_Analysis_Based_on_CLIP_Image.pdf).
