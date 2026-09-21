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

    const filled = node.querySelectorAll("[class*='star'][class*='full'], [class*='star'][class*='filled'], [class*='star'][class*='active']").length;
    if (filled > 0 && filled <= 5) return filled;

    const widthStyle = [...node.querySelectorAll("[style*='width']")]
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

  function extractReviews(nodes, platform) {
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
  }

  registry.common = { metaContent, schemaProductPresent, titleFrom, descriptionFrom, ratingFromPage, extractReviews };
  registry.register = adapter => registry.adapters.push(adapter);
  registry.getActive = () => registry.adapters.find(adapter => adapter.matches(location));
})();
