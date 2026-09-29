# Browser review extraction contract

AuthentiCheck's browser extension extracts one record per buyer review that is already loaded in the active Shopee or Lazada product page.

## Included fields

```json
{
  "id": "shopee-1",
  "text": "Buyer-authored review text only",
  "rating": 5,
  "imageUrls": ["https://example.com/buyer-photo.jpg"],
  "hasImage": true
}
```

- `text` is the buyer-authored review body.
- `rating` is the star rating attached to that same review, or `null` when it cannot be detected reliably.
- `imageUrls` contains up to five buyer-review photo URLs from that same review card.
- `hasImage` is derived from `imageUrls.length > 0`.

## Explicitly excluded

- Seller or shop replies and responses
- Reviewer usernames and profile images
- Seller names, seller avatars, and shop logos
- Review dates and timestamps
- Variation labels and interaction controls
- Like counts and report controls
- Account, cart, delivery-address, payment, and checkout data

## Boundary rules

1. A marketplace adapter selects complete review-card roots.
2. Nested selector matches are deduplicated so one buyer review creates one record.
3. Seller-response containers are removed before review text or media is selected.
4. Review text, rating, and photos are read from the same review-card root.
5. Duplicate review text is discarded.
6. Only the first 20 currently loaded buyer reviews are processed per analysis request.
7. No scrolling, pagination, or private marketplace API is automated.

## Extraction diagnostics

Every scan records:

- Candidate review-card count
- Accepted buyer-review count
- Duplicate count
- Star ratings detected
- Reviews with buyer photos
- Seller-response containers excluded

These diagnostics appear in the extension panel so missing or contaminated extraction can be detected before model results are interpreted.

## Maintenance

Shopee and Lazada may change their DOM structure without notice. Update only the relevant adapter in `extension/extractors/`; shared buyer-only rules live in `buyer-reviews.js`. Then test that text, rating, and images still originate from the same buyer review and that seller replies remain excluded.
