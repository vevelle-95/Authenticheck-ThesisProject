(() => {
  const registry = globalThis.AuthentiCheckExtractors;
  const common = registry.common;
  const REVIEW_SELECTORS = [
    ".shopee-product-rating",
    "[class*='product-rating'][data-review-id]",
    "[data-testid*='review-item']"
  ];

  const EXTRACTION_CONFIG = {
    contentSelectors: [
      ".shopee-product-rating__content",
      ".shopee-product-rating__comment",
      "[class*='rating__content']",
      "[class*='review-comment']"
    ],
    ratingSelectors: [
      ".shopee-product-rating__rating",
      "[class*='rating__rating']",
      "[aria-label*='star' i]"
    ],
    imageSelectors: [
      ".shopee-product-rating__image-list-wrapper img",
      ".shopee-product-rating__image-list-wrapper [style*='background-image']",
      "[class*='rating-media'] img",
      "[class*='review-image'] img",
      "[class*='review-image'] [style*='background-image']"
    ],
    sellerResponseSelectors: [
      ".shopee-product-rating__report-menu-button + div",
      "[class*='seller-comment']",
      "[class*='shop-reply']"
    ]
  };

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
      return common.extractReviews([...document.querySelectorAll(REVIEW_SELECTORS.join(","))], "shopee", EXTRACTION_CONFIG);
    }
  });
})();
