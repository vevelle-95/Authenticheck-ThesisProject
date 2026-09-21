(() => {
  const registry = globalThis.AuthentiCheckExtractors;
  const common = registry.common;
  const REVIEW_SELECTORS = [
    ".shopee-product-rating",
    ".shopee-product-rating__main",
    "[class*='product-rating'][class*='main']",
    "[class*='product-rating'] [class*='comment']"
  ];

  registry.register({
    id: "shopee",
    name: "Shopee",
    matches: location => /(^|\.)shopee\.ph$/i.test(location.hostname),
    isProductPage() {
      return /(?:-i\.|\/product\/|\/product\.)\d+/i.test(location.pathname)
        || common.schemaProductPresent()
        || Boolean(document.querySelector(".shopee-product-rating, [class*='product-detail']"));
    },
    getProductTitle() {
      return common.titleFrom([
        "meta[property='og:title']",
        "h1",
        "[class*='product-title']",
        "[class*='product-name']"
      ]);
    },
    getProductDescription() {
      return common.descriptionFrom([
        ".shopee-product-detail__description",
        "[class*='product-detail'][class*='description']",
        "[class*='product-description']"
      ]);
    },
    getMarketplaceRating() {
      return common.ratingFromPage([
        ".product-rating-overview__briefing",
        "[class*='rating-overview'] [class*='score']",
        "[class*='product-rating']"
      ]);
    },
    extractReviews() {
      return common.extractReviews([...document.querySelectorAll(REVIEW_SELECTORS.join(","))], "shopee");
    }
  });
})();
