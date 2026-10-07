# AuthentiCheck

AuthentiCheck filters reviews by quality, then predicts aspect sentiments for
reviews classified as Authentic.

## Start here

The team is currently collecting and annotating the dataset. Use the single
[project guide](docs/PROJECT_GUIDE.md) for both models, dataset annotations,
training, saved outputs, APIs, frontend comparison, and browser extraction.
This README is the project overview and quick start.

Both APIs are implemented and saved local practice bundles exist. AuthentiCheck's
older review-only quality models need retraining for the current title +
description + review input. The extension connects to AuthentiCheck; baseline
comparison rendering is still pending frontend integration. Environments,
weights, caches, and generated reports are ignored by Git. Another machine must
set up and train or restore compatible models using the guide.

A dependency-free Manifest V3 browser extension plus a standalone frontend prototype. The extension injects a floating review-analysis panel into Shopee Philippines and Lazada Philippines product pages. It follows the AuthentiCheck thesis pipeline: review-quality classification first, then aspect sentiment using only reviews classified as authentic.

## Install the extension in Chrome or Edge

1. Open `chrome://extensions` in Chrome or `edge://extensions` in Edge.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select the `extension` folder in this project.
5. Open a product page on `shopee.ph` or `lazada.com.ph` and scroll until reviews are visible.
6. Tap the floating AuthentiCheck badge. Use the extension popup to rescan or configure the analyzer.

When extension source files change, select **Reload** on the extension card.

## Analysis modes

### Local estimate (works immediately)

The default mode classifies currently visible review text using transparent Tagalog/Taglish-aware heuristics. It checks detail level, logistics-only content, basic rating/sentiment consistency, and whether buyer media is present. This mode is for interface testing and must not be presented as the trained thesis model's output.

### Model API

Enable **Use model API** in the popup and provide a local endpoint such as `http://127.0.0.1:8000/analyze`. The extension posts:

```json
{
  "schemaVersion": "1.0",
  "platform": "shopee",
  "url": "https://shopee.ph/...",
  "productTitle": "Product name",
  "productDescription": "Visible product description",
  "marketplaceRating": 4.8,
  "reviews": [
    {
      "id": "shopee-1",
      "text": "...",
      "rating": 5,
      "hasImage": true,
      "imageUrls": ["https://example-cdn.test/buyer-photo.jpg"]
    }
  ]
}
```

The versioned API returns `authenticShare`, `verifiedRating`, `counts`, `sentimentCounts`, `aspects`, and classified `reviews`. Incomplete or incompatible model responses are rejected; they are never silently mixed with local heuristic values. Version 0.3 permits local API hosts only (`127.0.0.1` or `localhost`).

The quality model and M-CLIP use product title + description + buyer review text,
with the same combination in training, evaluation, and live prediction. XGBoost
also uses the buyer's stars. ABSA uses the buyer review alone, and the interface
displays that original review text.

When API mode is enabled, connection and response errors are shown explicitly. The extension does not silently replace a failed model response with heuristic output; the user may deliberately select **Use local preview**, which remains labeled as a preliminary estimate.

## Mock-defense readiness

- Complete loading, empty, model-error, recovery, and result states.
- Review evidence search and classification filters.
- Input-coverage counts for reviews, ratings, and buyer media.
- JSON export containing the exact analysis input and displayed result.
- Responsive and keyboard-friendly panel behavior.
- Stable local API boundary with a 60-second timeout and strict schema validation.

The standalone prototype and local estimate are appropriate for demonstrating the proposed workflow and interface during the mock defense. They are not empirical model results and should not be used as thesis performance evidence. Replace the local preview with the trained API for the tool defense.

## Privacy and scope

- Reads only public review content already loaded in the current product page.
- Extracts only buyer-authored review text, the star rating belonging to that review, and buyer-review pictures from the same review card.
- Explicitly removes seller or shop responses before review text and pictures are selected.
- Uses separate Shopee and Lazada adapters so platform-specific page changes can be maintained independently.
- Injects the floating interface only after detecting a supported product page.
- Does not read account credentials, messages, cart data, addresses, or checkout information.
- Does not automate scrolling or send content anywhere unless the user enables the local model API.
- Shopee and Lazada can change their page markup; the extraction selectors are deliberately broad, and the popup includes a manual rescan.

## User-facing terminology

- **Authentic review share** is the percentage of analyzed reviews classified as authentic. It is not the classifier's statistical confidence score.
- **Authenticity-adjusted rating** is the average star rating after excluding reviews classified as deceptive, irrelevant, or low-information.
- **Local estimate** means the bundled heuristic analyzer produced the result.
- **Model API** means a connected AuthentiCheck inference service produced the result.

## Standalone visual prototype

Open `prototype/index.html` directly, or serve the repository with a static file server. This mock product page remains useful for design review without installing the extension.

## Prototype interactions

- Start a staged mock analysis with progress updates and rotating review-literacy tips.
- Replay the presentation sequence with **Run demo again**.
- Open and close the floating AuthentiCheck panel.
- Explore Overview, Insights, and Reviews tabs.
- Select any review-quality category to jump to its evidence view.
- Switch the mock host marketplace by clicking the Shopmall/LazMall logo.
- Try color and protection selectors, the methodology info button, filters, and feedback.

All product and analysis values in the standalone prototype are illustrative placeholders.

## Repository layout

```text
extension/   Browser extension source that should be committed
prototype/   Standalone interface demonstration that should be committed
data/        Centralized input data and generated 6D features
models/      Centralized trained models
main.py      Versioned FastAPI boundary used by the extension
stage1/      RoBERTa training and 6D feature extraction scripts
stage2/      XGBoost, fixed ten-category ABSA, inference, and held-out evaluation
requirements.txt  Shared Python dependencies
output/      Generated ZIP packages; ignored by Git
```

Full datasets, scraped buyer images, trained weights, secrets, and generated experiment output are intentionally excluded by `.gitignore`. The tracked `models/manifest.json` describes the older download bundle; new training saves a separate manifest under `models/own_model_v2/`.

## Run the integrated local API

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline.py --data data/test_reviews.csv --validate-only
python pipeline.py --data data/test_reviews.csv
uvicorn main:app --host 127.0.0.1 --port 8000
```

`pipeline.py` validates the finalized dataset, persists product-disjoint partitions, generates five-fold out-of-fold RoBERTa probabilities, trains final RoBERTa and XGBoost, and independently trains fixed ten-category ABSA. It selects models and thresholds using validation data. Both models default to the shared `data/splits_real_v2.json` manifest. Training and the backend default to `models/own_model_v2/`. See the [training and evaluation guide](docs/PROJECT_GUIDE.md#4-train-validate-and-test) for the CSV workflow and separate held-out test command.

AuthentiCheck's paths and training defaults are in [model_contract.py](model_contract.py).
Edit `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, or `MAX_LENGTH` there; training CLI
options can override the exposed settings for an individual run. The baseline
keeps its separate `baselines/clip-ca-cg/configs/config.yaml`.

To use an existing compatible bundle, set `AUTHENTICHECK_MODEL_BUNDLE` before
starting the API. To rebuild for the combined input and select a separate bundle,
run these commands from `Authenticheck-ThesisProject`:

```powershell
.\.venv\Scripts\Activate.ps1
python pipeline.py --data data/test_reviews.csv --splits data/splits_real_v2.json --work-dir data/context_run/features --output models/context_run --epochs 3
$env:AUTHENTICHECK_MODEL_BUNDLE = "models/context_run"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Relative bundle paths are resolved from the project root. The selected folder
must contain `dost_roberta/`, `xgboost_meta_classifier.json` with its metadata,
and `absa_model/`. The API checks their compatibility before accepting analysis
requests. This setting affects API inference; training still saves to
`models/own_model_v2/` unless you pass `--output` to `pipeline.py`.

The saved `models/practice_run/` quality models use the previous review-only
input contract and are rejected by current inference. Rebuild OOF probabilities,
RoBERTa, six-feature CSVs, and XGBoost; changing a version tag is insufficient.
The full pipeline command above also trains independent ABSA.

Restart the API after changing the setting. It applies to the current PowerShell
session; without it, the API uses the default bundle. Check
`http://127.0.0.1:8000/health` for the server and `/ready` for model readiness.

The existing `scripts/setup_models.py` installs/verifies the older download manifest and is not a substitute for preparing a compatible training bundle.

The backend processes review photos in memory and discards their bytes after M-CLIP feature extraction; it does not create a permanent review-image store.

`data/test_reviews.csv` is used for dataset-format testing and includes review IDs. Assistant-added draft annotations are recorded in `reports/generated/csv_annotation_fill.json`; review those drafts before thesis evaluation. Missing buyer images produce a CLIP score of 0.

For baseline API startup and the frontend developer handoff, see
[run the APIs and connect the frontend](docs/PROJECT_GUIDE.md#6-run-the-apis-and-connect-the-frontend).
The [browser extraction rules](docs/PROJECT_GUIDE.md#7-browser-extraction-and-interface-behavior)
are in the same guide.
