(() => {
  const registry = globalThis.AuthentiCheckExtractors;
  const common = registry.common;
  const REVIEW_SELECTORS = [
    ".mod-reviews .item",
    ".review-item",
    "[class*='review-item']",
    "[class*='review-content']",
    "[data-spm*='review'] .item-content"
  ];

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
    getMarketplaceRating() {
      return common.ratingFromPage([
        ".score-average",
        ".pdp-review-summary__link",
        "[class*='rating'] [class*='score']"
      ]);
    },
    extractReviews() {
      return common.extractReviews([...document.querySelectorAll(REVIEW_SELECTORS.join(","))], "lazada");
    }
  });
})();
