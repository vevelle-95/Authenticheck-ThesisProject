# AuthentiCheck browser extension

## Model comparison in the extension

Run `python -m uvicorn main:app --reload` from the repository root, enable the
model API in the extension popup, and open a supported product page. The
floating analysis panel has a **Compare** tab. Select one visible buyer review
and press **Compare this review**. The extension sends its text, rating, up to
five buyer image URLs, and product description to `POST /api/compare/lu-et-al`, then
shows both engines and category-level discrepancies inside the panel. The API
also accepts up to 20 reviews per call for research scripts.

The Lu et al. checkout remains outside this repository. Set `LU_ET_AL_REPO` to
its absolute directory, `LU_ET_AL_PYTHON` to the Python executable in its own
environment, and `LU_ET_AL_ENTRYPOINT` to `module:function` in that checkout.
The callable receives a list of review objects and returns either a list of
results or `{"reviews": [...]}`. Each result needs the same `id` and a holistic
`sentiment` of `positive`, `negative`, or `neutral`.

Alternatively, set `LU_ET_AL_COMMAND` to a baseline inference command. It runs
from the external checkout, receives one JSON object on stdin, and writes one
JSON object on stdout (logs go to stderr). The command setting takes precedence
over the Python entrypoint. Request example:

```json
{"product_description":"Wireless headphones","reviews":[{"id":"r1","text":"Great sound but late delivery","star_rating":3,"image_url":"https://example.org/photo.jpg","image_urls":["https://example.org/photo.jpg"]}]}
```

Expected response, in the same order and with the same IDs:

```json
{"reviews":[{"id":"r1","sentiment":"negative"}]}
```

The runner may optionally include `segments`, for example
`{"segments":[{"text":"Great sound","sentiment":"positive"}]}`. Segment text
must occur verbatim in the review. If segment sentiment is omitted, the adapter
inherits the holistic label and marks it `holistic_projection`. The taxonomy
rules classify matching clauses into the six thesis categories. They are an
explicit evaluation projection, not native Lu et al. aspect predictions.

No Lu et al. implementation or weights are included here. Until its checkout and
entrypoint or runner command are configured, the response marks the baseline unavailable. AuthentiCheck
likewise needs its trained model artifacts to produce results; missing artifacts
are shown as an independent unavailable status.

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

The request also includes `productDescription` and an `imageUrls` array for each review when buyer media is visible. This keeps the frontend contract ready for product-description similarity and M-CLIP visual-grounding features without claiming that those models are already running.

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
stage2/      XGBoost meta-classifier and open-ended ABSA scripts
requirements.txt  Shared Python dependencies
output/      Generated ZIP packages; ignored by Git
```

Full datasets, scraped buyer images, trained weights, secrets, and generated experiment output are intentionally excluded by `.gitignore`. The tracked `models/manifest.json` documents the required local model artifacts.

## Run the integrated local API

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline.py
uvicorn main:app --host 127.0.0.1 --port 8000
```

`pipeline.py` trains Stage 1 DOST-RoBERTa, extracts the six-dimensional features, trains XGBoost, and trains the authentic-only ABSA model using `data/test_reviews.csv`. Check `http://127.0.0.1:8000/health` for the server and `/ready` for model readiness. `python scripts/setup_models.py --verify-only` lists any artifact that the training pipeline did not produce.

If a teammate distributes already-trained artifacts instead, `python scripts/setup_models.py --archive C:\path\to\authenticheck-models.zip` installs and verifies that optional bundle without committing its weights to Git.

The backend processes review photos in memory and discards their bytes after M-CLIP feature extraction; it does not create a permanent review-image store.

The included 50 synthetic reviews are suitable for integration and demonstration testing. They are not sufficient for reporting the final comparative performance or statistical significance of the thesis models.

The exact browser extraction boundary is documented in `docs/EXTRACTION_CONTRACT.md`.
