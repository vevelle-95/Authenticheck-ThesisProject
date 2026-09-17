# AuthentiCheck browser extension

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
  "platform": "Shopee",
  "url": "https://shopee.ph/...",
  "productTitle": "Product name",
  "marketplaceRating": 4.8,
  "reviews": [
    { "id": "shopee-1", "text": "...", "rating": 5, "hasImage": true }
  ]
}
```

The API may return `authenticShare`, `verifiedRating`, `counts`, `sentimentCounts`, `aspects`, and classified `reviews`. For backward compatibility, `confidence` is also accepted as an alias for `authenticShare`. Missing fields fall back to the local estimate. Version 0.1 permits local API hosts only (`127.0.0.1` or `localhost`).

## Privacy and scope

- Reads only public review content already loaded in the current product page.
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
output/      Generated ZIP packages; ignored by Git
```

Future backend, machine-learning, tests, and documentation source should be committed under `backend/`, `ml/`, `tests/`, and `docs/`. Full datasets, scraped buyer images, trained weights, secrets, and generated experiment output are intentionally excluded by `.gitignore`.
