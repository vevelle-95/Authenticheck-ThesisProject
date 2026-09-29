(() => {
  const registry = globalThis.AuthentiCheckExtractors ||= { adapters: [] };
  const MAX_REVIEWS = 20;
  const MAX_IMAGES_PER_REVIEW = 5;
  const SELLER_REPLY_PATTERN = /^(?:seller|shop|merchant)\s*(?:reply|response)|^(?:reply|response)\s+from\s+(?:seller|shop)|^(?:tugon|sagot)\s+(?:ng|mula sa)\s+(?:seller|shop|tindahan)/i;
  const DEFAULT_EXCLUSIONS = [
    "[class*='seller-reply']",
    "[class*='seller-response']",
    "[class*='shop-reply']",
    "[class*='merchant-reply']",
    "[data-testid*='seller-reply']",
    "[data-testid*='seller-response']"
  ];

  function cleanText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function metaContent(selector) {
    return document.querySelector(selector)?.getAttribute("content")?.trim() || "";
  }

  function schemaProductPresent() {
    return Boolean(
      document.querySelector("[itemtype*='schema.org/Product'], [itemtype*='schema.org/product']")
      || [...document.querySelectorAll("script[type='application/ld+json']")]
        .some(node => /["']@type["']\s*:\s*["']Product["']/i.test(node.textContent || ""))
      || /product/i.test(metaContent("meta[property='og:type']"))
    );
  }

  function titleFrom(selectors) {
    for (const selector of selectors) {
      const value = selector.startsWith("meta")
        ? metaContent(selector)
        : document.querySelector(selector)?.textContent?.trim();
      if (value) return cleanText(value).slice(0, 500);
    }
    return cleanText(document.title.split(/[|–—]/)[0]).slice(0, 500);
  }

  function descriptionFrom(selectors = []) {
    const structured = metaContent("meta[property='og:description']")
      || metaContent("meta[name='description']")
      || document.querySelector("[itemprop='description']")?.textContent?.trim();
    if (structured) return cleanText(structured).slice(0, 2000);
    for (const selector of selectors) {
      const value = document.querySelector(selector)?.textContent?.trim();
      if (value) return cleanText(value).slice(0, 2000);
    }
    return "";
  }

  function isInsideExcluded(element, root, selectors) {
    return selectors.some(selector => {
      const excluded = element.closest?.(selector);
      return Boolean(excluded && root.contains(excluded));
    });
  }

  function parseRatingFromNode(node, ratingSelectors = []) {
    const candidates = [node, ...ratingSelectors.flatMap(selector => [...node.querySelectorAll(selector)])];
    for (const candidate of candidates) {
      const aria = candidate.getAttribute?.("aria-label")
        || candidate.querySelector?.("[aria-label*='star' i]")?.getAttribute("aria-label")
        || "";
      const dataRating = candidate.getAttribute?.("data-rating") || candidate.getAttribute?.("data-star-rating") || "";
      const match = `${aria} ${dataRating} ${candidate.textContent || ""}`
        .match(/(?:^|\s)([1-5](?:\.\d)?)\s*(?:out of 5|stars?|\/\s*5)(?:\s|$)/i);
      if (match) return Number(match[1]);
    }

    const ratingRoot = ratingSelectors.map(selector => node.querySelector(selector)).find(Boolean) || node;
    const filled = ratingRoot.querySelectorAll?.(
      "[class*='star'][class*='full'], [class*='star'][class*='filled'], [class*='star'][class*='active'], .icon-rating-solid"
    )?.length || 0;
    if (filled > 0 && filled <= 5) return filled;

    const widthMatch = [...ratingRoot.querySelectorAll("[style*='width']")]
      .map(element => element.getAttribute("style") || "")
      .join(" ")
      .match(/width\s*:\s*(\d{1,3})%/i);
    if (widthMatch) return Math.max(1, Math.min(5, Math.round(Number(widthMatch[1]) / 20)));
    return null;
  }

  function ratingFromPage(selectors) {
    const structured = metaContent("meta[itemprop='ratingValue']")
      || document.querySelector("[itemprop='ratingValue']")?.textContent?.trim()
      || metaContent("meta[property='product:rating:value']");
    if (structured && Number(structured) >= 1 && Number(structured) <= 5) return Number(structured);
    for (const selector of selectors) {
      for (const node of [...document.querySelectorAll(selector)].slice(0, 30)) {
        const match = cleanText(node.textContent).match(/^([1-5](?:\.[0-9])?)$/);
        if (match) return Number(match[1]);
      }
    }
    return null;
  }

  function findSellerResponseElements(root, selectors) {
    const found = new Set();
    selectors.forEach(selector => {
      if (root.matches?.(selector)) found.add(root);
      root.querySelectorAll(selector).forEach(element => found.add(element));
    });
    root.querySelectorAll("div, section, article, p, span").forEach(element => {
      const ownText = cleanText(element.childNodes.length === 1 ? element.textContent : "");
      if (ownText && SELLER_REPLY_PATTERN.test(ownText)) found.add(element.parentElement || element);
    });
    return [...found];
  }

  function extractReviewText(root, contentSelectors, exclusionSelectors) {
    const textFromElement = element => {
      const clone = element.cloneNode(true);
      findSellerResponseElements(clone, exclusionSelectors).forEach(excluded => excluded.remove());
      [
        ...exclusionSelectors,
        "script", "style", "button", "svg", "img", "video",
        "[class*='username']", "[class*='author']", "[class*='date']", "[class*='time']",
        "[class*='variation']", "[class*='rating']", "[class*='like']", "[class*='action']"
      ].forEach(selector => clone.querySelectorAll(selector).forEach(child => child.remove()));
      return cleanText(clone.innerText || clone.textContent || "");
    };

    const candidates = [];
    contentSelectors.forEach(selector => {
      const elements = [...root.querySelectorAll(selector)];
      if (root.matches?.(selector)) elements.unshift(root);
      elements.forEach(element => {
        if (isInsideExcluded(element, root, exclusionSelectors)) return;
        const text = textFromElement(element);
        if (text.length >= 1 && text.length <= 2000 && !SELLER_REPLY_PATTERN.test(text)) candidates.push(text);
      });
    });
    if (!candidates.length) {
      const text = textFromElement(root);
      if (text.length >= 1) candidates.push(text);
    }
    return candidates
      .filter(text => !/^\d{1,2}[\s/:.-]/.test(text))
      .sort((a, b) => b.length - a.length)[0]
      ?.slice(0, 2000) || "";
  }

  function normalizeImageUrl(value) {
    if (!value) return null;
    const cleaned = String(value).trim().replace(/^['"]|['"]$/g, "");
    try {
      const url = new URL(cleaned, location.href);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function extractImageUrls(root, imageSelectors, exclusionSelectors) {
    const selected = imageSelectors.flatMap(selector => [...root.querySelectorAll(selector)]);
    const elements = selected.length ? selected : [...root.querySelectorAll("img, [style*='background-image'], video[poster]")];
    const urls = [];
    elements.forEach(element => {
      if (isInsideExcluded(element, root, exclusionSelectors)) return;
      const descriptor = `${element.className || ""} ${element.getAttribute?.("alt") || ""}`.toLowerCase();
      if (/avatar|profile|seller|shop|logo|icon|star|emoji/.test(descriptor)) return;
      const styleUrl = (element.getAttribute?.("style") || "").match(/background-image\s*:\s*url\(([^)]+)\)/i)?.[1];
      const raw = element.currentSrc
        || element.getAttribute?.("src")
        || element.getAttribute?.("data-src")
        || element.getAttribute?.("data-ks-lazyload")
        || element.getAttribute?.("poster")
        || styleUrl;
      const url = normalizeImageUrl(raw);
      if (url && !urls.includes(url)) urls.push(url);
    });
    return urls.slice(0, MAX_IMAGES_PER_REVIEW);
  }

  function topLevelUniqueNodes(nodes) {
    const unique = [...new Set(nodes)].filter(Boolean);
    return unique.filter(node => !unique.some(other => other !== node && other.contains(node)));
  }

  function extractReviews(nodes, platform, config = {}) {
    const exclusionSelectors = [...DEFAULT_EXCLUSIONS, ...(config.sellerResponseSelectors || [])];
    const contentSelectors = config.contentSelectors || ["[class*='comment']", "[class*='content']", "[class*='review-text']", "p"];
    const imageSelectors = config.imageSelectors || [];
    const ratingSelectors = config.ratingSelectors || [];
    const reviewNodes = topLevelUniqueNodes(nodes);
    const reviews = [];
    const seen = new Set();
    let duplicateReviews = 0;
    let sellerResponsesExcluded = 0;

    reviewNodes.forEach((node, index) => {
      sellerResponsesExcluded += findSellerResponseElements(node, exclusionSelectors).length;
      const text = extractReviewText(node, contentSelectors, exclusionSelectors);
      const key = text.toLocaleLowerCase();
      if (!text) return;
      if (seen.has(key)) {
        duplicateReviews += 1;
        return;
      }
      seen.add(key);
      const imageUrls = extractImageUrls(node, imageSelectors, exclusionSelectors);
      const sourceId = node.getAttribute("data-review-id") || node.id || `${index + 1}`;
      reviews.push({
        id: `${platform.toLowerCase()}-${sourceId}`,
        text,
        rating: parseRatingFromNode(node, ratingSelectors),
        hasImage: imageUrls.length > 0,
        imageUrls
      });
    });

    const accepted = reviews.slice(0, MAX_REVIEWS);
    return {
      reviews: accepted,
      stats: {
        candidateNodes: reviewNodes.length,
        acceptedReviews: accepted.length,
        duplicateReviews,
        sellerResponsesExcluded,
        withImages: accepted.filter(review => review.hasImage).length,
        withRatings: accepted.filter(review => Number.isFinite(review.rating)).length
      }
    };
  }

  registry.common = { metaContent, schemaProductPresent, titleFrom, descriptionFrom, ratingFromPage, extractReviews };
  registry.register = adapter => registry.adapters.push(adapter);
  registry.getActive = () => registry.adapters.find(adapter => adapter.matches(location));
})();
