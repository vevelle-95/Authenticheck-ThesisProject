(() => {
  const registry = globalThis.AuthentiCheckExtractors ||= { adapters: [] };

  function metaContent(selector) {
    return document.querySelector(selector)?.getAttribute("content")?.trim() || "";
  }

  function schemaProductPresent() {
    return Boolean(
      document.querySelector("[itemtype*='schema.org/Product'], [itemtype*='schema.org/product']")
      || [...document.querySelectorAll("script[type='application/ld+json']")].some(node => /["']@type["']\s*:\s*["']Product["']/i.test(node.textContent || ""))
      || /product/i.test(metaContent("meta[property='og:type']"))
    );
  }

  function titleFrom(selectors) {
    for (const selector of selectors) {
      const value = selector.startsWith("meta")
        ? metaContent(selector)
        : document.querySelector(selector)?.textContent?.trim();
      if (value) return value;
    }
    return document.title.split(/[|–—]/)[0].trim();
  }

  function descriptionFrom(selectors = []) {
    const structured = metaContent("meta[property='og:description']")
      || metaContent("meta[name='description']")
      || document.querySelector("[itemprop='description']")?.textContent?.trim();
    if (structured) return structured.replace(/\s+/g, " ").slice(0, 2000);
    for (const selector of selectors) {
      const value = document.querySelector(selector)?.textContent?.trim();
      if (value) return value.replace(/\s+/g, " ").slice(0, 2000);
    }
    return "";
  }

  function parseRatingFromNode(node) {
    const aria = node.querySelector("[aria-label*='star' i]")?.getAttribute("aria-label") || "";
    const text = `${aria} ${node.textContent || ""}`;
    const match = text.match(/([1-5](?:\.\d)?)\s*(?:out of 5|stars?|\/\s*5)/i);
    if (match) return Number(match[1]);

    const filled = ratingRoot.querySelectorAll?.("[class*='star'][class*='full'], [class*='star'][class*='filled'], [class*='star'][class*='active'], .icon-rating-solid")?.length || 0;
    if (filled > 0 && filled <= 5) return filled;

    const widthStyle = [...ratingRoot.querySelectorAll("[style*='width']")]
      .map(element => element.getAttribute("style") || "")
      .join(" ")
      .match(/width\s*:\s*(\d{1,3})%/i);
    if (widthStyle) return Math.max(1, Math.min(5, Math.round(Number(widthStyle[1]) / 20)));
    return null;
  }

  function ratingFromPage(selectors) {
    const structured = metaContent("meta[itemprop='ratingValue']")
      || document.querySelector("[itemprop='ratingValue']")?.textContent?.trim()
      || metaContent("meta[property='product:rating:value']");
    if (structured && Number(structured) <= 5) return Number(structured);

    for (const selector of selectors) {
      for (const node of [...document.querySelectorAll(selector)].slice(0, 30)) {
        const match = (node.textContent || "").trim().match(/^([1-5](?:\.[0-9])?)$/);
        if (match) return Number(match[1]);
      }
    }
    return null;
  }

  function findSellerResponseElements(root, selectors) {
    const found = new Set();
    selectors.forEach(selector => root.querySelectorAll(selector).forEach(element => found.add(element)));
    root.querySelectorAll("div, section, article, p, span").forEach(element => {
      const ownText = cleanText(element.childNodes.length === 1 ? element.textContent : "");
      if (ownText && SELLER_REPLY_PATTERN.test(ownText)) {
        found.add(element.parentElement || element);
      }
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
      root.querySelectorAll(selector).forEach(element => {
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
    const cleaned = value.trim().replace(/^['"]|['"]$/g, "");
    try {
      const url = new URL(cleaned, location.href);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function extractImageUrls(root, imageSelectors, exclusionSelectors) {
    const selected = imageSelectors.flatMap(selector => [...root.querySelectorAll(selector)]);
    const elements = selected.length
      ? selected
      : [...root.querySelectorAll("img, [style*='background-image'], video[poster]")];
    const urls = [];

    elements.forEach(element => {
      if (isInsideExcluded(element, root, exclusionSelectors)) return;
      const descriptor = `${element.className || ""} ${element.getAttribute?.("alt") || ""}`.toLowerCase();
      if (/avatar|profile|seller|shop|logo|icon|star/.test(descriptor)) return;

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
    return urls.slice(0, 5);
  }

  function topLevelUniqueNodes(nodes) {
    return [...new Set(nodes)].filter(node => !nodes.some(other => other !== node && other.contains(node)));
  }

  function extractReviews(nodes, platform, config = {}) {
    const exclusionSelectors = [...DEFAULT_EXCLUSIONS, ...(config.sellerResponseSelectors || [])];
    const contentSelectors = config.contentSelectors || ["[class*='comment']", "[class*='content']", "[class*='review-text']", "p"];
    const imageSelectors = config.imageSelectors || [];
    const ratingSelectors = config.ratingSelectors || [];
    const reviewNodes = topLevelUniqueNodes(nodes);
    const seen = new Set();
    return nodes.map((node, index) => {
      const preferred = node.querySelector("[class*='comment'], [class*='content'], [class*='review-text'], p");
      const candidates = [preferred, ...node.querySelectorAll("p, span, div")]
        .filter(Boolean)
        .map(element => (element.innerText || "").trim())
        .filter(text => text.length >= 5 && text.length <= 800 && !/^\d{1,2}[\s/:.-]/.test(text));
      const text = candidates.sort((a, b) => b.length - a.length)[0] || (node.innerText || "").trim();
      const clean = text.replace(/\s+/g, " ").slice(0, 700);
      if (!clean || clean.length < 5 || seen.has(clean)) return null;
      seen.add(clean);
      const imageUrls = [...node.querySelectorAll("img")]
        .map(img => img.currentSrc || img.src || img.getAttribute("data-src") || "")
        .filter(url => /^https?:/i.test(url) && !/avatar|profile|icon|emoji/i.test(url))
        .filter((url, imageIndex, urls) => urls.indexOf(url) === imageIndex)
        .slice(0, 5);
      return {
        id: `${platform.toLowerCase()}-${index + 1}`,
        text: clean,
        rating: parseRatingFromNode(node),
        hasImage: imageUrls.length > 0,
        imageUrls
      };
    }).filter(Boolean).slice(0, 100);

    return {
      reviews,
      stats: {
        candidateNodes: reviewNodes.length,
        acceptedReviews: reviews.length,
        duplicateReviews,
        sellerResponsesExcluded,
        withImages: reviews.filter(review => review.hasImage).length,
        withRatings: reviews.filter(review => Number.isFinite(review.rating)).length
      }
    };
  }

  registry.common = { metaContent, schemaProductPresent, titleFrom, descriptionFrom, ratingFromPage, extractReviews };
  registry.register = adapter => registry.adapters.push(adapter);
  registry.getActive = () => registry.adapters.find(adapter => adapter.matches(location));
})();
