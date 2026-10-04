(() => {
  const registry = globalThis.AuthentiCheckExtractors;
  const common = registry.common;
  const REVIEW_SELECTORS = [
    ".mod-reviews .item",
    ".review-item",
    "[class*='review-item'][data-review-id]",
    "[data-spm*='review'] [class*='review-item']"
  ];

  const EXTRACTION_CONFIG = {
    contentSelectors: [
      ".item-content",
      ".review-content-sl",
      "[class*='review-content']",
      "[class*='review-comment']"
    ],
    ratingSelectors: [
      ".starCtn",
      "[class*='review-star']",
      "[aria-label*='star' i]"
    ],
    imageSelectors: [
      ".review-image img",
      ".item-content img",
      "[class*='review-media'] img",
      "[class*='review-image'] img",
      "[class*='review-image'] [style*='background-image']"
    ],
    sellerResponseSelectors: [
      ".seller-reply",
      ".seller-response",
      "[class*='seller-reply']",
      "[class*='seller-response']"
    ]
  };

  registry.register({
    id: "lazada",
    name: "Lazada",
    matches: location => /(^|\.)lazada\.com\.ph$/i.test(location.hostname),
    isProductPage() {
      return /\/products\/.+-i\d+/i.test(location.pathname)
        || common.schemaProductPresent()
        || Boolean(document.querySelector(".pdp-product-title, .pdp-mod-product-badge-title, .mod-reviews"));
    },
    getProductTitle() {
      return common.titleFrom([
        "meta[property='og:title']",
        ".pdp-mod-product-badge-title",
        ".pdp-product-title",
        "h1"
      ]);
    },
    getProductDescription() {
      return common.descriptionFrom([
        ".pdp-product-desc",
        ".html-content",
        "[class*='product-desc']"
      ]);
    },
    getMarketplaceRating() {
      return common.ratingFromPage([
        ".score-average",
        ".pdp-review-summary__link",
        "[class*='rating'] [class*='score']"
      ]);
    },
    extractReviews() {
      const knownNodes = [...document.querySelectorAll(REVIEW_SELECTORS.join(","))];
      return common.extractReviews(common.discoverReviewNodes(knownNodes), "lazada", EXTRACTION_CONFIG);
    }
  });
})();
