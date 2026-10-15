(() => {
  if (window.__authentiCheckInjected) return;
  window.__authentiCheckInjected = true;

  const DEFAULTS = { enabled: true, autoAnalyze: true, useApi: false };
  const POSITIVE = ["good","great","best","love","excellent","perfect","comfortable","clear","quality","worth","nice","sulit","maganda","maayos","malinaw","matibay","ganda","okay","ayos","recommend"];
  const NEGATIVE = ["bad","poor","broken","fake","damaged","disappointed","waste","uncomfortable","mahina","pangit","sira","mali","sayang","hindi gumagana","defective","kulang","madumi"];
  const GENERIC = ["good product","nice product","okay","ok","good quality","recommended","highly recommended","perfect product","maganda","maayos","sulit"];
  const DELIVERY = ["delivery","shipping","courier","rider","parcel","dumating","mabilis dumating","seller responsive","packaging"];
  const PRODUCT_SPECIFIC = ["quality","matibay","durable","material","build","battery","sound","audio","camera","gumagana","price","presyo","mahal","affordable","color","kulay","design","size","laki"];
  const ASPECTS = {
    Quality: ["quality","matibay","durable","material","build","ganda","finish","sira"],
    Performance: ["performance","works","gumagana","fast","mabilis","battery","sound","audio","camera"],
    Value: ["price","presyo","sulit","worth","mahal","cheap","affordable"],
    Delivery: DELIVERY,
    Appearance: ["color","kulay","design","look","size","laki","maliit"]
  };
  const LOADING_TIPS = [
    "Hindi lahat ng reviews ay dapat paniwalaan.",
    "Mas kapaki-pakinabang ang reviews na may tiyak na detalye.",
    "Tinitingnan din kung tugma ang rating at review text.",
    "Buyer photos can provide useful product context.",
    "Delivery comments may not describe the product itself."
  ];

  let settings = { ...DEFAULTS };
  let currentUrl = location.href;
  let refreshTimer;
  let loadingTipTimer;
  let loadingTipIndex = -1;
  let analysisRun = 0;
  let host;
  let root;
  let lastPayload;
  let displayedPayload;
  let lastResult;
  let lastExtractionSignature = "";
  let extensionContextInvalidated = false;
  let adapter;
  let activeReviewFilter = "all";
  let activeReviewQuery = "";
  let reviewLimit = 20;
  let lastAnalyzedAt = null;
  let reviewsChanged = false;
  let freshnessTimer;
  let activeReviewSort = "original";
  let activePage = "overview";
  let scanInProgress = false;
  let scanChangeMessage = "";
  const pageScrollPositions = new Map();
  const expandedReviews = new Set();
  const expandedReasons = new Set();

  const shield = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2 8 3v6c0 5.1-3.4 9.6-8 11-4.6-1.4-8-5.9-8-11V5l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></svg>`;
  const closeIcon = `<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>`;
  const infoIcon = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/></svg>`;

  async function start() {
    settings = await chrome.storage.sync.get(DEFAULTS);
    observePage();
    await syncPageContext();
  }

  async function syncPageContext() {
    adapter = globalThis.AuthentiCheckExtractors?.getActive();
    const isProductPage = Boolean(adapter?.isProductPage());
    if (!isProductPage) {
      destroyShell();
      return;
    }
    createShell();
    const previousProduct = displayedPayload || lastPayload;
    if (previousProduct && previousProduct.url !== location.href) {
      activeReviewFilter = "all";
      activeReviewQuery = "";
      reviewLimit = 20;
      activeReviewSort = "original";
      lastResult = null;
      displayedPayload = null;
      expandedReviews.clear();
      expandedReasons.clear();
      pageScrollPositions.clear();
      activePage = "overview";
      showPage("overview");
      root.querySelector(".ac-scroll").scrollTop = 0;
    }
    applyEnabledState();
    await analyzePage();
  }

  function createShell() {
    if (host?.isConnected) return;
    host = document.createElement("div");
    host.id = "authenticheck-extension-root";
    host.style.all = "initial";
    root = host.attachShadow({ mode: "open" });
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = chrome.runtime.getURL("content.css");
    root.append(stylesheet);

    const wrapper = document.createElement("div");
    wrapper.innerHTML = `
      <div class="ac-dim"></div>
      <button class="ac-trigger" aria-label="Open AuthentiCheck analysis" aria-expanded="false" aria-controls="ac-analysis-panel">
        <span class="ac-trigger-logo">${shield}</span>
        <span class="ac-trigger-copy"><strong>Checking visible reviews…</strong><small>AuthentiCheck · Tap to view</small></span>
        <span class="ac-trigger-score">—</span>
      </button>
      <aside class="ac-panel" id="ac-analysis-panel" role="dialog" aria-modal="true" tabindex="-1" data-state="loading" data-view="overview" inert aria-hidden="true" aria-label="AuthentiCheck review analysis">
        <header class="ac-head">
          <div class="ac-brand"><span class="ac-brand-mark">${shield}</span><div><strong>AuthentiCheck</strong><small>Review intelligence</small></div></div>
          <div class="ac-head-actions"><button class="ac-icon-btn ac-info" aria-label="About this analysis" aria-expanded="false" aria-controls="ac-about">${infoIcon}</button><button class="ac-icon-btn ac-close" aria-label="Close AuthentiCheck">${closeIcon}</button></div>
        </header>
        <div class="ac-scroll">
          <section class="ac-about" id="ac-about" hidden aria-label="About this analysis">
            <h2>Understand your results</h2>
            <p>AuthentiCheck classifies review quality, then summarizes product sentiment from reviews classified as authentic. Results cover reviews loaded on this page.</p>
            <p>Authentic review share is the proportion of analyzed reviews classified as authentic, not a model confidence score. This scan covers loaded reviews only and does not represent all product reviews.</p>
            <p>Model evaluation is ongoing. Predictions are provisional. Local estimates use heuristics; published test scores are available in Model performance.</p>
          </section>
          <section class="ac-state ac-loading" role="status" aria-live="polite"><span class="ac-spinner" aria-hidden="true"></span><h2>Analyzing reviews…</h2><p>Preparing review text, ratings, and available buyer media.</p><div class="ac-loading-tip"><strong>Review tip</strong><span></span></div></section>
          <section class="ac-state ac-empty"><span class="ac-empty-icon">${shield}</span><h2>Reviews aren't loaded yet</h2><p>Scroll to the product reviews so the marketplace loads them, then run the scan again.</p><button class="ac-rescan">Scan visible reviews</button></section>
          <section class="ac-state ac-error" role="alert"><span class="ac-empty-icon ac-error-icon">!</span><h2>Analysis unavailable</h2><p class="ac-error-message">The local inference service could not be reached.</p><div class="ac-error-actions"><button class="ac-retry">Try again</button><button class="ac-local-preview">Use local preview</button></div><details class="ac-error-details"><summary>Technical details</summary><p></p></details><small>Local preview uses heuristics. It does not replace trained model results.</small></section>
          <div class="ac-results hidden">
            <section class="ac-hero">
              <div class="ac-product-context"><span class="ac-eyebrow">Review analysis</span><h1 class="ac-product-title">Current product</h1></div>
              <div class="ac-status"><i></i><span class="ac-status-text">Analysis complete</span><span class="ac-mode">Local estimate</span></div>
              <div class="ac-score-row"><div class="ac-ring"><div><strong>0</strong><span>%</span></div></div><div class="ac-score-copy"><span class="ac-eyebrow">Authentic review share</span><h2>Analyzing…</h2><details class="ac-summary-details"><summary>About this result</summary><p></p></details></div></div>
              <p class="ac-development-notice" role="note">Loaded-review sample only · Predictions are provisional, not an overall product verdict.</p>
            </section>
          </div>
            <nav class="ac-tabs" role="tablist" aria-label="Analysis views"><button class="ac-tab active" id="ac-tab-overview" role="tab" aria-selected="true" aria-controls="ac-page-overview" data-tab="overview">Overview</button><button class="ac-tab" id="ac-tab-insights" role="tab" aria-selected="false" aria-controls="ac-page-insights" tabindex="-1" data-tab="insights">Insights</button><button class="ac-tab" id="ac-tab-reviews" role="tab" aria-selected="false" aria-controls="ac-page-reviews" tabindex="-1" data-tab="reviews">Reviews</button><button class="ac-tab" id="ac-tab-performance" role="tab" aria-selected="false" aria-controls="ac-page-performance" tabindex="-1" data-tab="performance">Model performance</button><button class="ac-tab" id="ac-tab-testing" role="tab" aria-selected="false" aria-controls="ac-page-testing" tabindex="-1" data-tab="testing">Test results</button></nav>
            <section class="ac-page active" id="ac-page-overview" role="tabpanel" aria-labelledby="ac-tab-overview" tabindex="0" data-page="overview"></section>
            <section class="ac-page" id="ac-page-insights" role="tabpanel" aria-labelledby="ac-tab-insights" tabindex="0" data-page="insights"></section>
            <section class="ac-page" id="ac-page-reviews" role="tabpanel" aria-labelledby="ac-tab-reviews" tabindex="0" data-page="reviews"></section>
            <section class="ac-page" id="ac-page-performance" role="tabpanel" aria-labelledby="ac-tab-performance" tabindex="0" data-page="performance">
              <div class="ac-title"><div><h3>Model performance</h3><p class="ac-performance-score">Test Macro-F1 · Not evaluated</p></div></div>
              <div class="ac-performance-body"></div>
            </section>
            <section class="ac-page" id="ac-page-testing" role="tabpanel" aria-labelledby="ac-tab-testing" tabindex="0" data-page="testing">
              <div class="ac-title"><div><h3>Test results</h3><p>Held-out CSV evaluation against annotated labels</p></div></div>
              <div class="ac-testing-body"></div>
            </section>
        </div>
        <div class="ac-scan-status" role="status" aria-live="polite"><span class="ac-freshness">Ready to scan loaded reviews</span><button class="ac-scan-new" hidden>Scan updated reviews</button></div>
        <footer class="ac-footer"><span><i></i><b>Visible reviews only</b></span><div><button class="ac-export" disabled>Export JSON</button><button class="ac-rescan">Rescan reviews</button></div></footer>
      </aside>`;
    root.append(wrapper);
    document.documentElement.append(host);
    renderPerformance();
    renderTestResults();
    freshnessTimer = setInterval(updateFreshness, 30000);

    root.querySelector(".ac-trigger").addEventListener("click", openPanel);
    root.querySelector(".ac-close").addEventListener("click", closePanel);
    root.querySelector(".ac-dim").addEventListener("click", closePanel);
    root.querySelectorAll(".ac-rescan").forEach(button => button.addEventListener("click", analyzePage));
    root.querySelector(".ac-retry").addEventListener("click", () => extensionContextInvalidated ? location.reload() : analyzePage());
    root.querySelector(".ac-scan-new").addEventListener("click", analyzePage);
    root.querySelector(".ac-local-preview").addEventListener("click", useLocalPreview);
    root.querySelector(".ac-export").addEventListener("click", exportAnalysis);
    root.querySelector(".ac-info").addEventListener("click", () => {
      const about = root.querySelector(".ac-about");
      about.hidden = !about.hidden;
      root.querySelector(".ac-info").setAttribute("aria-expanded", String(!about.hidden));
      if (!about.hidden) about.scrollIntoView({ block: "nearest" });
    });
    root.querySelectorAll(".ac-tab").forEach(button => button.addEventListener("click", () => showPage(button.dataset.tab)));
    root.querySelector(".ac-tabs").addEventListener("keydown", event => {
      const tabs = [...root.querySelectorAll(".ac-tab")];
      const index = tabs.indexOf(event.target);
      if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      showPage(tabs[next].dataset.tab);
      tabs[next].focus();
    });
    root.addEventListener("keydown", handlePanelKeydown);
  }

  function destroyShell() {
    analysisRun += 1;
    stopLoadingTips();
    clearInterval(freshnessTimer);
    if (host?.isConnected) host.remove();
    host = null;
    root = null;
    lastPayload = null;
    lastResult = null;
    displayedPayload = null;
    lastExtractionSignature = "";
    lastAnalyzedAt = null;
    reviewsChanged = false;
    activeReviewFilter = "all";
    activeReviewQuery = "";
    reviewLimit = 20;
    activeReviewSort = "original";
    activePage = "overview";
    scanInProgress = false;
    scanChangeMessage = "";
    pageScrollPositions.clear();
    expandedReviews.clear();
    expandedReasons.clear();
  }

  function openPanel() {
    const wasOpen = root.querySelector(".ac-panel").classList.contains("open");
    root.querySelector(".ac-panel").inert = false;
    root.querySelector(".ac-panel").classList.add("open");
    root.querySelector(".ac-panel").setAttribute("aria-hidden", "false");
    root.querySelector(".ac-trigger").classList.add("hidden");
    root.querySelector(".ac-dim").classList.add("open");
    root.querySelector(".ac-trigger").setAttribute("aria-expanded", "true");
    root.querySelector(".ac-trigger").inert = true;
    if (!wasOpen) root.querySelector(".ac-close").focus({ preventScroll: true });
  }

  function handlePanelKeydown(event) {
    const panel = root?.querySelector(".ac-panel");
    if (!panel?.classList.contains("open")) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closePanel();
      return;
    }
    if (event.key !== "Tab") return;
    const controls = [...panel.querySelectorAll("button, input, select, textarea, a[href], summary, [tabindex]")]
      .filter(element => element.tabIndex >= 0 && !element.disabled
        && !element.closest("[hidden], [inert], [aria-hidden='true']") && element.getClientRects().length > 0);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (!first) { event.preventDefault(); panel.focus(); return; }
    const current = root.activeElement;
    if (!controls.includes(current) || event.shiftKey && current === first || !event.shiftKey && current === last) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    }
  }

  function closePanel() {
    if (!root) return;
    const wasOpen = root.querySelector(".ac-panel").classList.contains("open");
    root.querySelector(".ac-panel").inert = true;
    root.querySelector(".ac-panel").classList.remove("open");
    root.querySelector(".ac-panel").setAttribute("aria-hidden", "true");
    root.querySelector(".ac-trigger").classList.remove("hidden");
    root.querySelector(".ac-dim").classList.remove("open");
    root.querySelector(".ac-trigger").inert = false;
    root.querySelector(".ac-trigger").setAttribute("aria-expanded", "false");
    if (wasOpen) root.querySelector(".ac-trigger").focus({ preventScroll: true });
  }

  function showPage(name) {
    const scroll = root.querySelector(".ac-scroll");
    if (name !== activePage) pageScrollPositions.set(activePage, scroll.scrollTop || 0);
    root.querySelector(".ac-panel").setAttribute("data-view", name);
    root.querySelectorAll(".ac-tab").forEach(tab => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    root.querySelectorAll(".ac-page").forEach(page => page.classList.toggle("active", page.dataset.page === name));
    if (name !== activePage) scroll.scrollTop = pageScrollPositions.get(name) || 0;
    activePage = name;
  }

  function openReviews(filter = "all") {
    activeReviewFilter = filter;
    activeReviewQuery = "";
    reviewLimit = 20;
    renderReviews(lastResult);
    showPage("reviews");
    const page = root.querySelector('[data-page="reviews"]');
    page.scrollIntoView({ block: "start" });
    root.querySelector(".ac-review-filter").focus({ preventScroll: true });
  }

  function updateFreshness() {
    if (!root || lastAnalyzedAt == null || scanInProgress) return;
    const status = root.querySelector(".ac-freshness");
    const minutes = lastAnalyzedAt == null ? null : Math.max(0, Math.floor((Date.now() - lastAnalyzedAt) / 60000));
    status.textContent = reviewsChanged ? "Loaded reviews changed · Results may be out of date"
      : minutes == null ? "No completed scan yet" : minutes === 0 ? "Last scanned just now"
      : `Last scanned ${minutes} minute${minutes === 1 ? "" : "s"} ago`;
    if (!reviewsChanged && scanChangeMessage) status.textContent += ` · ${scanChangeMessage}`;
    root.querySelector(".ac-scan-new").hidden = !reviewsChanged;
    root.querySelector(".ac-scan-status").classList.toggle("stale", reviewsChanged);
  }

  function applyEnabledState() {
    if (!host) return;
    host.style.display = settings.enabled ? "block" : "none";
  }

  function wordHits(text, terms) {
    return terms.reduce((sum, term) => {
      const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return sum + (new RegExp(`(?:^|[^a-zà-ž0-9])${escaped}(?:$|[^a-zà-ž0-9])`, "i").test(text) ? 1 : 0);
    }, 0);
  }

  function classifyReview(review) {
    const text = String(review.text || "").toLowerCase();
    if (!review.analysisEligible || !text) {
      const missing = review.missingFields?.length ? review.missingFields.filter(field => field !== "image").join(", ") : "text or integer star rating";
      return {
        ...review,
        label: "unavailable",
        sentiment: null,
        signals: [`Not sent to the text model; missing ${missing}`],
        positiveHits: 0,
        negativeHits: 0
      };
    }
    const tokens = text.match(/[a-zà-ž0-9]+/gi) || [];
    const pos = wordHits(text, POSITIVE);
    const neg = wordHits(text, NEGATIVE);
    const genericOnly = GENERIC.some(term => text === term || text === `${term}!`) || (tokens.length <= 6 && GENERIC.some(term => text.includes(term)));
    const deliveryHits = wordHits(text, DELIVERY);
    const productAspectHits = wordHits(text, PRODUCT_SPECIFIC);
    const highRatingNegative = review.rating >= 4 && neg > pos;
    const lowRatingPositive = review.rating && review.rating <= 2 && pos > neg;

    let label = "authentic";
    const signals = [];
    if (tokens.length <= 4 || text.length < 24 || genericOnly) {
      label = "liv";
      signals.push("Limited product detail");
    } else if (deliveryHits > 0 && productAspectHits === 0 && tokens.length < 18) {
      label = "irrelevant";
      signals.push("Mostly logistics-related");
    } else if (highRatingNegative || lowRatingPositive) {
      label = "deceptive";
      signals.push("Text and rating conflict");
    } else {
      signals.push("Text and rating appear consistent");
      if (review.hasImage) signals.push("Buyer media detected");
      if (tokens.length > 15) signals.push("Specific experience details");
    }

    const sentiment = pos === neg ? "neutral" : pos > neg ? "positive" : "negative";
    return { ...review, label, sentiment, signals, positiveHits: pos, negativeHits: neg };
  }

  function analyzeLocally(payload) {
    const reviews = payload.reviews.map(classifyReview);
    const counts = { authentic: 0, liv: 0, irrelevant: 0, deceptive: 0 };
    const analyzed = reviews.filter(review => review.label !== "unavailable");
    analyzed.forEach(review => counts[review.label]++);
    const authentic = reviews.filter(review => review.label === "authentic");
    const sentimentCounts = { positive: 0, neutral: 0, negative: 0 };
    authentic.forEach(review => sentimentCounts[review.sentiment]++);
    const confidence = analyzed.length ? Math.round((counts.authentic / analyzed.length) * 100) : 0;
    const ratedAuthentic = authentic.filter(review => review.rating);
    const verifiedRating = ratedAuthentic.length
      ? ratedAuthentic.reduce((sum, review) => sum + review.rating, 0) / ratedAuthentic.length
      : null;
    const aspects = Object.entries(ASPECTS).map(([name, terms]) => {
      const matching = authentic.filter(review => wordHits(review.text.toLowerCase(), terms));
      const positive = matching.filter(review => review.sentiment === "positive").length;
      return { name, mentions: matching.length, positivePercent: matching.length ? Math.round(positive / matching.length * 100) : 0 };
    }).filter(aspect => aspect.mentions > 0).sort((a, b) => b.mentions - a.mentions).slice(0, 5);
    return { mode: analyzed.length ? "local" : "coverage", reviews, counts, sentimentCounts, authenticShare: confidence, verifiedRating, aspects };
  }

  function extractionSignature(reviews = []) {
    return reviews.map(review => [
      review.id,
      review.username || "",
      String(review.text || "").trim(),
      Number.isFinite(review.rating) ? review.rating : "",
      ...(review.imageUrls || [])
    ].join("\u001f")).join("\u001e");
  }

  async function sendRuntimeMessage(message) {
    try {
      return await chrome.runtime.sendMessage(message);
    } catch (error) {
      const invalidated = /extension context invalidated/i.test(String(error?.message || error));
      if (invalidated) extensionContextInvalidated = true;
      return {
        ok: false,
        contextInvalidated: invalidated,
        error: invalidated
          ? "AuthentiCheck was reloaded or updated. Refresh this Shopee/Lazada tab, then run the scan again."
          : "The extension could not contact its background service. Try rescanning the page."
      };
    }
  }

  async function analyzePage() {
    if (!root || !settings.enabled || !adapter?.isProductPage()) return;
    if (extensionContextInvalidated) {
      renderError("AuthentiCheck was reloaded or updated. Refresh this Shopee/Lazada tab, then run the scan again.");
      return;
    }
    const runId = ++analysisRun;
    lastPayload = null;
    try {
      reviewsChanged = false;
      renderLoading();
      const minimumLoadingTime = new Promise(resolve => setTimeout(resolve, 1200));
      const extraction = adapter.extractReviews();
      const reviews = extraction.reviews;
      const analyzableReviews = reviews.filter(review => review.analysisEligible && review.text.trim());
      const payload = {
        schemaVersion: "1.0",
        platform: adapter.id,
        url: location.href,
        productTitle: adapter.getProductTitle(),
        productDescription: adapter.getProductDescription?.() || "",
        marketplaceRating: adapter.getMarketplaceRating(),
        extraction: extraction.stats,
        reviews
      };
      lastPayload = payload;
      lastExtractionSignature = extractionSignature(reviews);

      if (!reviews.length) {
        await minimumLoadingTime;
        if (runId !== analysisRun) return;
        renderEmpty(extraction.stats);
        return;
      }

      let result = null;
      if (settings.useApi && analyzableReviews.length) {
        const apiPayload = {
          ...payload,
          reviews: analyzableReviews.map(({ id, text, rating, hasImage, imageUrls }) => ({
            id,
            text,
            rating,
            hasImage,
            imageUrls
          }))
        };
        const response = await sendRuntimeMessage({ type: "AUTHENTICHECK_ANALYZE", payload: apiPayload });
        if (runId !== analysisRun) return;
        if (response?.ok && response.result) {
          try {
            result = normalizeApiResult(response.result, payload);
          } catch (error) {
            await minimumLoadingTime;
            if (runId !== analysisRun) return;
            renderError(error.message || "The model API returned an incompatible response.");
            return;
          }
        } else {
          await minimumLoadingTime;
          if (runId !== analysisRun) return;
          renderError(response?.error || "The model API did not respond. Check that the local service is running.");
          return;
        }
      }
      if (!result) result = analyzeLocally(payload);
      await minimumLoadingTime;
      if (runId !== analysisRun) return;
      renderResult(result, payload);
    } catch (error) {
      if (runId !== analysisRun || !root) return;
      renderError(`Page scan failed: ${error.message || "The loaded reviews could not be read."}`);
    }
  }

  function normalizeApiResult(api, payload) {
    if (api?.schemaVersion !== "1.0") throw new Error("The model API response uses an unsupported schema version.");
    const numberOr = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => {
      if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
        throw new Error("The model API response contains an invalid numeric field.");
      }
      return value;
    };
    const labelMap = { authentic:"authentic", deceptive:"deceptive", irrelevant:"irrelevant", liv:"liv", "low-value":"liv", low_value:"liv", lowValue:"liv" };
    if (!Array.isArray(api.reviews) || !api.counts || !api.sentimentCounts || !Array.isArray(api.aspects)) {
      throw new Error("The model API response is incomplete.");
    }
    const responseById = new Map(api.reviews.map(review => [String(review?.id || ""), review]));
    const eligibleIds = new Set(payload.reviews.filter(review => review.analysisEligible && review.text.trim()).map(review => String(review.id)));
    if (responseById.size !== api.reviews.length || responseById.size !== eligibleIds.size
      || [...responseById.keys()].some(id => !eligibleIds.has(id))) {
      throw new Error("The model API returned duplicate or unexpected review IDs.");
    }
    const apiReviews = payload.reviews.map(source => {
      if (!source.analysisEligible || !source.text.trim()) {
        const missing = source.missingFields?.length ? source.missingFields.filter(field => field !== "image").join(", ") : "text or integer star rating";
        return { ...source, label: "unavailable", sentiment: null, signals: [`Not sent to the model; missing ${missing}`] };
      }
      const review = responseById.get(String(source.id));
      if (!review) throw new Error("The model API did not return every analyzable review.");
      const label = labelMap[String(review.label || "").toLowerCase()];
      if (!label) throw new Error("The model API returned an unknown review classification.");
      return {
        ...source,
        ...review,
        username: source.username || "",
        text: String(review.text ?? source.text ?? ""),
        rating: review.starRating ?? source.rating ?? null,
        label,
        signals: Array.isArray(review.signals) ? review.signals.map(String).slice(0, 8) : []
      };
    });
    const apiCounts = api.counts;
    const expectedCounts = { authentic: 0, liv: 0, irrelevant: 0, deceptive: 0 };
    apiReviews.filter(review => review.label !== "unavailable").forEach(review => expectedCounts[review.label]++);
    for (const [label, expected] of Object.entries(expectedCounts)) {
      if (numberOr(apiCounts[label]) !== expected) throw new Error("The model API counts do not match its review classifications.");
    }
    const expectedShare = eligibleIds.size ? expectedCounts.authentic / eligibleIds.size * 100 : 0;
    if (Math.abs(numberOr(api.authenticShare, 0, 100) - expectedShare) > 0.51) {
      throw new Error("The model API authentic share does not match its review classifications.");
    }
    for (const key of ["positive", "neutral", "negative"]) {
      if (!Number.isInteger(numberOr(api.sentimentCounts[key]))) throw new Error("The model API returned an invalid sentiment count.");
    }
    return {
      mode: "api",
      modelVersion: String(api.modelVersion || "unknown"),
      authenticShare: numberOr(api.authenticShare, 0, 100),
      verifiedRating: api.verifiedRating == null ? null : numberOr(api.verifiedRating, 0, 5),
      counts: {
        authentic: numberOr(apiCounts.authentic),
        liv: numberOr(apiCounts.liv),
        irrelevant: numberOr(apiCounts.irrelevant),
        deceptive: numberOr(apiCounts.deceptive)
      },
      sentimentCounts: {
        positive: numberOr(api.sentimentCounts.positive),
        neutral: numberOr(api.sentimentCounts.neutral),
        negative: numberOr(api.sentimentCounts.negative)
      },
      reviews: apiReviews,
      aspects: api.aspects.slice(0, 12).map(aspect => ({
        name: String(aspect?.name || "Unnamed aspect"),
        mentions: numberOr(aspect?.mentions),
        positivePercent: numberOr(aspect?.positivePercent, 0, 100)
      }))
    };
  }

  function renderEmpty(extraction = {}) {
    hideStates();
    scanInProgress = false;
    lastResult = null;
    displayedPayload = null;
    lastAnalyzedAt = null;
    root.querySelector(".ac-panel").setAttribute("data-has-results", "false");
    root.querySelector(".ac-results").setAttribute("aria-busy", "false");
    root.querySelector(".ac-panel").setAttribute("data-state", "empty");
    root.querySelector(".ac-freshness").textContent = "No reviews available to analyze";
    setTriggerLoading(false);
    root.querySelector(".ac-empty").classList.add("show");
    root.querySelector(".ac-results").classList.add("hidden");
    const cardsFound = Number(extraction.candidateNodes || 0);
    const withoutText = Number(extraction.withoutWrittenText || 0);
    const title = root.querySelector(".ac-empty h2");
    const message = root.querySelector(".ac-empty p");
    if (cardsFound > 0 && withoutText > 0) {
      title.textContent = "No written review text found";
      message.textContent = `${cardsFound} review card${cardsFound === 1 ? " was" : "s were"} detected, but ${withoutText} contained only ratings or media. Text is required for review classification and sentiment analysis.`;
      root.querySelector(".ac-trigger-copy strong").textContent = "Reviews found without written text";
      root.querySelector(".ac-trigger-copy small").textContent = `${cardsFound} rating/media-only review${cardsFound === 1 ? "" : "s"}`;
    } else {
      title.textContent = "Reviews aren't loaded yet";
      message.textContent = "Scroll to the product reviews so the marketplace loads them, then run the scan again.";
      root.querySelector(".ac-trigger-copy strong").textContent = "Open product reviews to analyze";
      root.querySelector(".ac-trigger-copy small").textContent = "Scroll to reviews, then tap";
    }
    root.querySelector(".ac-trigger-score").textContent = "—";
  }

  function hideStates() {
    stopLoadingTips();
    root.querySelectorAll(".ac-state").forEach(state => state.classList.remove("show"));
  }

  function renderLoading() {
    hideStates();
    scanInProgress = true;
    const updating = Boolean(lastResult && displayedPayload?.url === location.href);
    root.querySelector(".ac-panel").setAttribute("data-has-results", String(updating));
    root.querySelector(".ac-panel").setAttribute("data-state", updating ? "updating" : "loading");
    root.querySelector(".ac-export").disabled = true;
    root.querySelector(".ac-results").setAttribute("aria-busy", "true");
    root.querySelector(".ac-freshness").textContent = updating ? "Updating… Showing your previous scan until results are ready." : "Scanning loaded reviews…";
    root.querySelector(".ac-scan-new").hidden = true;
    root.querySelector(".ac-scan-status").classList.remove("stale");
    root.querySelector(".ac-results").classList.toggle("hidden", !updating);
    if (!updating) root.querySelector(".ac-loading").classList.add("show");
    root.querySelector(".ac-trigger-copy strong").textContent = settings.useApi ? "Waiting for model analysis…" : "Checking visible reviews…";
    setTriggerLoading(true);
    startLoadingTips();
  }

  function errorPresentation(message) {
    if (/Page scan failed:/i.test(message)) return {
      title: "Could not read the loaded reviews", message: "Refresh the product page and try scanning again. The marketplace review layout may have changed.", retry: "Try scan again"
    };
    if (/reloaded or updated|context invalidated/i.test(message)) return {
      title: "Refresh this product page", message: "AuthentiCheck was updated. Refresh this Shopee/Lazada tab, then run the scan again.", retry: "Refresh page"
    };
    if (/contract mismatch|input-compatible|Missing artifacts|retrain/i.test(message)) return {
      title: "Model update required", message: "The saved models do not match this version of AuthentiCheck. Rebuild a compatible model bundle and restart the local API, then try again.", retry: "Check model again"
    };
    if (/timed out/i.test(message)) return {
      title: "The model took too long", message: "The local model did not finish in time. Check that the service is running and try again.", retry: "Try again"
    };
    if (/schema|incomplete|invalid|unexpected|do not match|does not match|incompatible|numeric/i.test(message)) return {
      title: "Model response unavailable", message: "The model returned a response this extension could not use. Check the API version and try again.", retry: "Try again"
    };
    return { title: "Cannot reach the model", message: "Check that your local model API is running and that its address is saved in extension settings. Then try again, or choose a local preview.", retry: "Try again" };
  }

  function useLocalPreview() {
    if (lastPayload?.url !== location.href || !lastPayload?.reviews.length) return;
    analysisRun += 1;
    renderResult(analyzeLocally(lastPayload), lastPayload);
  }

  function renderError(message) {
    hideStates();
    scanInProgress = false;
    // Failed scans must not leave previous results looking current.
    root.querySelector(".ac-panel").setAttribute("data-has-results", "false");
    lastAnalyzedAt = null;
    const problem = errorPresentation(String(message));
    root.querySelector(".ac-panel").setAttribute("data-state", "error");
    root.querySelector(".ac-results").setAttribute("aria-busy", "false");
    root.querySelector(".ac-export").disabled = true;
    root.querySelector(".ac-local-preview").disabled = lastPayload?.url !== location.href || !lastPayload?.reviews?.length;
    root.querySelector(".ac-scan-new").hidden = true;
    root.querySelector(".ac-scan-status").classList.remove("stale");
    root.querySelector(".ac-freshness").textContent = "Scan incomplete · No current model result";
    setTriggerLoading(false);
    root.querySelector(".ac-results").classList.add("hidden");
    root.querySelector(".ac-error").classList.add("show");
    root.querySelector(".ac-error h2").textContent = problem.title;
    root.querySelector(".ac-error-message").textContent = problem.message;
    root.querySelector(".ac-error-details p").textContent = message;
    root.querySelector(".ac-error-details").open = false;
    root.querySelector(".ac-retry").textContent = problem.retry;
    root.querySelector(".ac-trigger-copy strong").textContent = problem.title;
    root.querySelector(".ac-trigger-copy small").textContent = "Open for recovery options";
    root.querySelector(".ac-trigger-score").textContent = "!";
    openPanel();
  }

  function setTriggerLoading(isLoading) {
    const score = root.querySelector(".ac-trigger-score");
    score.classList.toggle("loading", isLoading);
    if (isLoading) score.textContent = "";
  }

  function startLoadingTips() {
    stopLoadingTips();
    if (loadingTipIndex < 0) {
      try {
        const stored = globalThis.sessionStorage?.getItem("authenticheck:last-review-tip");
        const previous = stored == null ? -1 : Number(stored);
        if (Number.isInteger(previous) && previous >= 0 && previous < LOADING_TIPS.length) loadingTipIndex = previous;
      } catch { /* Tip rotation still works when page storage is unavailable. */ }
    }
    const tip = root.querySelector(".ac-loading-tip span");
    const showNextTip = () => {
      loadingTipIndex = (loadingTipIndex + 1) % LOADING_TIPS.length;
      tip.textContent = LOADING_TIPS[loadingTipIndex];
      try {
        globalThis.sessionStorage?.setItem("authenticheck:last-review-tip", String(loadingTipIndex));
      } catch { /* Do not interrupt analysis if storage is blocked. */ }
    };
    showNextTip();
    loadingTipTimer = setInterval(showNextTip, 7500);
  }

  function stopLoadingTips() {
    clearInterval(loadingTipTimer);
    loadingTipTimer = null;
  }

  function percent(value, total) { return total ? Math.round(value / total * 100) : 0; }
  function escapeHtml(value = "") { const div = document.createElement("div"); div.textContent = String(value); return div.innerHTML; }
  function sentimentPercent(result, key) { const total = Object.values(result.sentimentCounts).reduce((a,b) => a+b, 0); return percent(result.sentimentCounts[key] || 0, total); }
  function verdictTitle(label) { return ({ authentic:"Authentic", liv:"Low-value", irrelevant:"Irrelevant", deceptive:"Deceptive", unavailable:"Not analyzed" })[label] || label; }

  function renderResult(result, payload) {
    const scroll = root.querySelector(".ac-scroll");
    const previousScroll = scroll.scrollTop || 0;
    const focused = root.activeElement;
    const focusedSelector = [".ac-review-search", ".ac-review-filter", ".ac-review-sort"].find(selector => focused?.matches(selector));
    const selection = focusedSelector === ".ac-review-search" ? [focused.selectionStart, focused.selectionEnd] : null;
    scanChangeMessage = compareScans(lastResult, displayedPayload, result, payload);
    scanInProgress = false;
    const total = Object.values(result.counts).reduce((a,b) => a+b, 0);
    const confidence = Math.max(0, Math.min(100, Math.round(result.authenticShare || 0)));
    const status = total === 0 ? "No eligible reviews" : confidence >= 75 ? "Mostly authentic reviews" : confidence >= 50 ? "Mixed review quality" : "Few authentic reviews";

    lastResult = result;
    displayedPayload = payload;
    lastAnalyzedAt = Date.now();
    reviewsChanged = adapter ? extractionSignature(adapter.extractReviews().reviews) !== lastExtractionSignature : false;
    hideStates();
    root.querySelector(".ac-panel").setAttribute("data-state", "ready");
    root.querySelector(".ac-panel").setAttribute("data-has-results", "true");
    root.querySelector(".ac-results").setAttribute("aria-busy", "false");
    updateFreshness();
    setTriggerLoading(false);
    root.querySelector(".ac-results").classList.remove("hidden");
    root.querySelector(".ac-export").disabled = false;
    root.querySelector(".ac-product-title").textContent = payload.productTitle || "Current product";
    root.querySelector(".ac-status-text").textContent = `${total} of ${payload.reviews.length} loaded review${payload.reviews.length === 1 ? "" : "s"} analyzed`;
    const developmentModel = result.mode === "api" && String(result.modelVersion || "").startsWith("development-");
    root.querySelector(".ac-mode").textContent = developmentModel ? "Development model" : result.mode === "api" ? "Model API" : result.mode === "coverage" ? "Input coverage" : "Local estimate";
    root.querySelector(".ac-ring").style.setProperty("--score", confidence);
    root.querySelector(".ac-ring strong").textContent = total ? confidence : "—";
    root.querySelector(".ac-score-copy h2").textContent = status;
    root.querySelector(".ac-score-copy p").textContent = developmentModel
      ? "Compact test model trained on draft/synthetic reviews. Uses text and stars; photos are omitted. Software testing only."
      : result.mode === "api"
      ? "Model results combine review text, ratings, and available buyer media."
      : result.mode === "coverage"
        ? "Visible buyer reviews are listed, but written text and an integer star rating are required for classification and ABSA sentiment."
        : "A local estimate from review text, ratings, and buyer media. These results use heuristics.";
    root.querySelector(".ac-trigger-score").textContent = total ? confidence : "—";
    root.querySelector(".ac-trigger-copy strong").textContent = total ? `${confidence}% classified authentic` : `${payload.reviews.length} review${payload.reviews.length === 1 ? "" : "s"} detected`;
    root.querySelector(".ac-trigger-copy small").textContent = `${total} of ${payload.reviews.length} loaded review${payload.reviews.length === 1 ? "" : "s"} analyzed`;

    renderOverview(result, total, payload);
    renderInsights(result);
    renderReviews(result);
    if (focusedSelector) {
      const next = root.querySelector(focusedSelector);
      next.focus({ preventScroll: true });
      if (selection) next.setSelectionRange(...selection);
    }
    scroll.scrollTop = previousScroll;
  }

  function compareScans(previous, previousPayload, result, payload) {
    if (!previous || previousPayload?.url !== payload.url || previous.mode !== result.mode || previous.modelVersion !== result.modelVersion) return "";
    const priorIds = new Set(previous.reviews.filter(review => review.label !== "unavailable").map(review => String(review.id)));
    const additional = result.reviews.filter(review => review.label !== "unavailable" && !priorIds.has(String(review.id))).length;
    return additional ? `${additional} additional review${additional === 1 ? "" : "s"} analyzed` : "Results refreshed · No additional reviews analyzed";
  }

  function renderOverview(result, total, payload) {
    const extraction = payload.extraction || {};
    const p = key => percent(result.counts[key] || 0, total);
    const positive = sentimentPercent(result, "positive");
    const neutral = sentimentPercent(result, "neutral");
    const negative = sentimentPercent(result, "negative");
    const skipped = result.reviews.filter(review => review.label === "unavailable");
    const missingText = skipped.filter(review => !String(review.text || "").trim()).length;
    const missingRating = skipped.filter(review => !Number.isInteger(review.rating) || review.rating < 1 || review.rating > 5).length;
    root.querySelector('[data-page="overview"]').innerHTML = `
      <div class="ac-title"><div><h3>Review quality</h3><p>How eligible visible reviews were classified</p></div><span class="ac-lang">Tagalog + Taglish</span></div>
      <div class="ac-card"><div class="ac-quality-total"><div><strong>${result.counts.authentic || 0}</strong><span>Authentic reviews</span></div><b>${p("authentic")}%</b></div>
        <div class="ac-stack"><i class="ac-q-auth" style="width:${p("authentic")}%"></i><i class="ac-q-liv" style="width:${p("liv")}%"></i><i class="ac-q-irrel" style="width:${p("irrelevant")}%"></i><i class="ac-q-decep" style="width:${p("deceptive")}%"></i></div>
        <div class="ac-quality-grid">
          ${qualityRow("ac-q-auth","Authentic",result.counts.authentic,p("authentic"))}
          ${qualityRow("ac-q-liv","Low-value",result.counts.liv,p("liv"))}
          ${qualityRow("ac-q-irrel","Irrelevant",result.counts.irrelevant,p("irrelevant"))}
          ${qualityRow("ac-q-decep","Deceptive",result.counts.deceptive,p("deceptive"))}
        </div>
      </div>
      <div class="ac-title" style="margin-top:20px"><div><h3>Scan coverage</h3><p>Only reviews loaded on this page are included</p></div></div>
      <div class="ac-card ac-coverage-card">
        <div class="ac-coverage-summary"><div><strong>${payload.reviews.length}</strong><span>Loaded</span></div><div><strong>${total}</strong><span>Analyzed</span></div><button class="ac-show-skipped" ${skipped.length ? "" : "disabled"}><strong>${skipped.length}</strong><span>Skipped${skipped.length ? " · View" : ""}</span></button></div>
        <details class="ac-coverage-details"><summary>What was included?</summary>
          <div class="ac-scope-grid"><div class="ac-scope"><strong>${payload.reviews.filter(review => review.text?.trim()).length}</strong><span>With written text</span></div><div class="ac-scope"><strong>${payload.reviews.filter(review => Number.isInteger(review.rating) && review.rating >= 1 && review.rating <= 5).length}</strong><span>With valid ratings</span></div><div class="ac-scope"><strong>${payload.reviews.filter(review => review.imageUrls?.length).length}</strong><span>With photos</span></div><div class="ac-scope excluded"><strong>${extraction.sellerResponsesExcluded || 0}</strong><span>Seller replies excluded</span></div></div>
          <p>${skipped.length ? `${missingText} missing written text · ${missingRating} missing a valid star rating. A review may be missing both.` : "All loaded reviews met the text and star-rating requirements."} Buyer photos are optional.</p>
        </details>
      </div>
      <div class="ac-title" style="margin-top:16px"><div><h3>Authentic sentiment</h3><p>Based on reviews classified as authentic</p></div></div>
      ${Object.values(result.sentimentCounts).some(count => count > 0) ? `<div class="ac-card ac-sentiment"><div class="ac-donut" style="--positive:${positive};--neutral:${neutral}"><div><strong>${positive}%</strong><span>positive</span></div></div><div class="ac-legend"><div><i class="positive"></i><strong>Positive</strong><b>${positive}%</b></div><div><i class="neutral"></i><strong>Neutral</strong><b>${neutral}%</b></div><div><i class="negative"></i><strong>Negative</strong><b>${negative}%</b></div></div></div>` : `<div class="ac-no-match">No sentiment results are available from authentic reviews in this scan.</div>`}
      <div class="ac-note">${shield}<p><strong>${result.mode === "api" ? "Model-connected analysis." : result.mode === "coverage" ? "Input coverage only." : "Preliminary local estimate."}</strong> Only content already visible in your browser is read. No account, seller-response, or checkout information is collected.</p></div>`;
    root.querySelectorAll(".ac-quality").forEach(button => button.addEventListener("click", () => openReviews(button.dataset.filter)));
    root.querySelector(".ac-show-skipped").addEventListener("click", () => openReviews("unavailable"));
  }

  function qualityRow(colorClass, title, count = 0, pct = 0) {
    const filter = { "ac-q-auth": "authentic", "ac-q-liv": "liv", "ac-q-irrel": "irrelevant", "ac-q-decep": "deceptive" }[colorClass];
    return `<button class="ac-quality" data-filter="${filter}" aria-label="View ${count} ${title.toLowerCase()} reviews"><i class="${colorClass}" aria-hidden="true"></i><span><strong>${title}</strong><small>${count} review${count === 1 ? "" : "s"} · View</small></span><b>${pct}%</b></button>`;
  }

  function renderInsights(result) {
    const aspects = result.aspects;
    root.querySelector('[data-page="insights"]').innerHTML = `
      <div class="ac-title"><div><h3>Product aspects</h3><p>What authentic visible reviews mention</p></div></div>
      <div class="ac-aspects">${aspects.length ? aspects.map((aspect,index) => {
        const evidence = aspectEvidence(result, aspect);
        return `<div class="ac-aspect"><span class="ac-aspect-icon" aria-hidden="true">${["✦","◎","₱","◴","◇"][index] || "•"}</span><div><div class="ac-aspect-head"><strong>${escapeHtml(aspect.name.replaceAll("_", " "))}</strong><b>${aspect.positivePercent}% positive</b></div><span class="ac-bar"><i style="width:${aspect.positivePercent}%"></i></span><p>${aspect.mentions} authentic mention${aspect.mentions === 1 ? "" : "s"} found in the loaded reviews.</p>${evidence.length ? `<details class="ac-aspect-evidence"><summary>Supporting reviews (${evidence.length}${evidence.length === 2 ? " shown" : ""})</summary>${evidence.map(review => `<blockquote>${escapeHtml(String(review.text).slice(0, 240))}${String(review.text).length > 240 ? "…" : ""}</blockquote>`).join("")}</details>` : `<p class="ac-evidence-unavailable">Per-review evidence was not supplied for this aspect.</p>`}</div></div>`;
      }).join("") : `<div class="ac-no-match">No aspect mentions were found in the reviews classified as authentic. Load more reviews and rescan for a broader sample.</div>`}</div>
      <button class="ac-view-authentic">View authentic reviews</button>
      <div class="ac-takeaway"><span>✦</span><div><strong>Scope note</strong><p>Aspect coverage grows as the marketplace loads more reviews. Scroll further and rescan for a broader sample.</p></div></div>`;
    root.querySelector(".ac-view-authentic").addEventListener("click", () => openReviews("authentic"));
  }

  function aspectEvidence(result, aspect) {
    return result.reviews.filter(review => review.label === "authentic" && (result.mode === "api"
      ? Array.isArray(review.aspectSentiment) && review.aspectSentiment.some(item => (item.category ?? item.aspect) === aspect.name)
      : ASPECTS[aspect.name] && wordHits(String(review.text || "").toLowerCase(), ASPECTS[aspect.name]) > 0)).slice(0, 2);
  }

  function renderReviews(result) {
    const filters = [["all", "All reviews"], ["authentic", "Authentic"], ["liv", "Low-value"], ["irrelevant", "Irrelevant"], ["deceptive", "Deceptive"], ["unavailable", "Skipped"]];
    root.querySelector('[data-page="reviews"]').innerHTML = `
      <div class="ac-title"><div><h3>Review evidence</h3><p>Every detected buyer review and its analysis status</p></div></div>
      <div class="ac-review-tools"><input class="ac-review-search" type="search" placeholder="Search review text" aria-label="Search visible review text"><select class="ac-review-filter" aria-label="Filter review classification">${filters.map(([value, title]) => `<option value="${value}">${title} (${value === "all" ? result.reviews.length : result.reviews.filter(review => review.label === value).length})</option>`).join("")}</select></div>
      <label class="ac-sort-wrap"><span>Sort by</span><select class="ac-review-sort"><option value="original">Original order</option><option value="lowest">Lowest rating</option><option value="highest">Highest rating</option><option value="longest">Longest review</option></select></label>
      <div class="ac-review-toolbar"><p class="ac-review-count" role="status" aria-live="polite"></p><button class="ac-clear-filters" hidden>Clear filters</button></div><div class="ac-review-list"></div><button class="ac-load-more" hidden>Load more reviews</button>`;
    const search = root.querySelector(".ac-review-search");
    const filter = root.querySelector(".ac-review-filter");
    const sort = root.querySelector(".ac-review-sort");
    search.value = activeReviewQuery;
    filter.value = activeReviewFilter;
    sort.value = activeReviewSort;
    const clear = root.querySelector(".ac-clear-filters");
    const more = root.querySelector(".ac-load-more");
    const update = () => {
      const query = search.value.trim().toLowerCase();
      const selected = filter.value;
      activeReviewFilter = selected;
      activeReviewQuery = search.value;
      activeReviewSort = sort.value;
      const matching = sortReviews(result.reviews.filter(review => (selected === "all" || review.label === selected) && (!query || String(review.text || "").toLowerCase().includes(query))), sort.value);
      const visible = matching.slice(0, reviewLimit);
      root.querySelector(".ac-review-count").textContent = `${visible.length} of ${matching.length} matching reviews · ${result.reviews.length} loaded`;
      root.querySelector(".ac-review-list").innerHTML = visible.length ? visible.map(review => reviewCard(review, result.reviews.indexOf(review), query)).join("") : `<div class="ac-no-match">No reviews match your search and classification. Clear filters to see all loaded reviews.</div>`;
      bindReviewExpansions(result, query);
      clear.hidden = !query && selected === "all";
      more.hidden = visible.length >= matching.length;
      more.textContent = `Load ${Math.min(20, matching.length - visible.length)} more reviews`;
    };
    const resetPage = () => { reviewLimit = 20; update(); };
    search.addEventListener("input", resetPage);
    filter.addEventListener("change", resetPage);
    sort.addEventListener("change", resetPage);
    clear.addEventListener("click", () => { search.value = ""; filter.value = "all"; resetPage(); search.focus(); });
    more.addEventListener("click", () => {
      const previousCount = Math.min(reviewLimit, root.querySelectorAll(".ac-review").length);
      reviewLimit += 20;
      update();
      const added = root.querySelectorAll(".ac-review")[previousCount];
      if (added) { added.tabIndex = -1; added.focus({ preventScroll: true }); }
    });
    update();
  }

  function sortReviews(reviews, order) {
    const validRating = review => Number.isInteger(review.rating) && review.rating >= 1 && review.rating <= 5;
    return reviews.map((review, index) => ({ review, index })).sort((a, b) => {
      if (order === "longest") return String(b.review.text || "").trim().length - String(a.review.text || "").trim().length || a.index - b.index;
      if (order === "lowest" || order === "highest") {
        if (validRating(a.review) !== validRating(b.review)) return validRating(a.review) ? -1 : 1;
        if (validRating(a.review)) return (order === "lowest" ? 1 : -1) * (a.review.rating - b.review.rating) || a.index - b.index;
      }
      return a.index - b.index;
    }).map(item => item.review);
  }

  function reviewExcerpt(text, query = "") {
    const limit = Math.max(320, query.length + 80);
    if (text.length <= limit) return text;
    const match = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
    const start = match < 0 ? 0 : Math.max(0, match - 80);
    const end = Math.min(text.length, start + limit);
    return `${start ? "…" : ""}${text.slice(start, end)}${end < text.length ? "…" : ""}`;
  }

  function bindReviewExpansions(result, query) {
    root.querySelectorAll(".ac-review-expand").forEach(button => button.addEventListener("click", () => {
      const review = result.reviews[Number(button.dataset.reviewIndex)];
      const id = String(review.id);
      const expanded = !expandedReviews.has(id);
      if (expanded) expandedReviews.add(id); else expandedReviews.delete(id);
      button.setAttribute("aria-expanded", String(expanded));
      button.textContent = expanded ? "Show less" : "Read full review";
      button.closest(".ac-review").querySelector(".ac-review-text").innerHTML = highlightText(expanded ? String(review.text).trim() : reviewExcerpt(String(review.text).trim(), query), query);
    }));
    root.querySelectorAll(".ac-signals-details").forEach(details => details.addEventListener("toggle", () => {
      if (!details.isConnected) return;
      const review = result.reviews[Number(details.dataset.reviewIndex)];
      if (details.open) expandedReasons.add(String(review.id)); else expandedReasons.delete(String(review.id));
    }));
  }

  function highlightText(text, query = "") {
    if (!query) return escapeHtml(text);
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return String(text).split(new RegExp(`(${escaped})`, "gi")).map((part, index) => index % 2 ? `<mark>${escapeHtml(part)}</mark>` : escapeHtml(part)).join("");
  }

  function reviewCard(review, index, query = "") {
    const username = String(review.username || "").trim() || "Reviewer name unavailable";
    const initial = review.username ? (username.match(/[\p{L}\p{N}]/u)?.[0] || "•").toUpperCase() : "•";
    const media = review.imageUrls?.length ? `<span>▣ ${review.imageUrls.length} media</span>` : "";
    const text = String(review.text || "").trim();
    const expanded = expandedReviews.has(String(review.id));
    const excerpt = reviewExcerpt(text, query);
    const stars = review.rating ? `${"★".repeat(Math.max(0,Math.min(5,review.rating)))} <small>${review.rating}/5</small>` : `<small>No star rating</small>`;
    const stage2Status = review.label === "authentic"
      ? `<span>✓ Included in sentiment analysis</span>`
      : ["deceptive", "liv", "irrelevant"].includes(review.label)
        ? `<span class="warn">Not included in sentiment analysis</span>`
        : "";
    const reasons = review.signals || [];
    const fallback = { authentic: "Classified as authentic based on the available review inputs.", liv: "The review offers limited product detail.", irrelevant: "The review was classified as unrelated to the product.", deceptive: "The analysis flagged potentially misleading review signals.", unavailable: "Written text and a valid star rating are required for analysis." };
    return `<article class="ac-review"><header><div class="ac-review-user"><span class="ac-avatar">${escapeHtml(initial)}</span><div><strong>${escapeHtml(username)}</strong><span class="ac-stars">${stars}</span></div></div><span class="ac-verdict ${escapeHtml(review.label)}">${escapeHtml(verdictTitle(review.label))}</span></header><p class="ac-review-text ${text ? "" : "missing"}" id="ac-review-text-${index}">${text ? highlightText(expanded ? text : excerpt, query) : "No written buyer comment was provided."}</p>${excerpt !== text ? `<button class="ac-review-expand" data-review-index="${index}" aria-expanded="${expanded}" aria-controls="ac-review-text-${index}">${expanded ? "Show less" : "Read full review"}</button>` : ""}<p class="ac-review-reason">${escapeHtml(reasons[0] || fallback[review.label] || "Review classification available.")}</p><details class="ac-signals-details" data-review-index="${index}" ${expandedReasons.has(String(review.id)) ? "open" : ""}><summary>Why this classification?</summary><div class="ac-signals">${reasons.slice(1).map(signal => `<span class="${review.label === "authentic" ? "" : "warn"}">${escapeHtml(signal)}</span>`).join("")}${stage2Status}${media}${!reasons.length ? `<span>No additional classification details were supplied.</span>` : ""}</div></details></article>`;
  }

  function metricScore(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? `${(value * 100).toFixed(2)}%` : "Not evaluated";
  }

  function renderTestResults() {
    const model = (globalThis.AuthentiCheckPerformance?.models || []).find(model => model.name === "AuthentiCheck");
    const quality = model?.tasks?.find(task => task.name === "Review quality · RoBERTa + XGBoost");
    const metrics = quality?.metrics;
    const evaluation = model?.testEvaluation;
    const samples = evaluation?.partition === "test" && Array.isArray(evaluation.samples) ? evaluation.samples : [];
    const tasks = (model?.tasks || []).filter(task => task.name.startsWith("Review quality") || task.name === "RoBERTa · text target");
    root.querySelector(".ac-testing-body").innerHTML = `
      <div class="ac-note">${shield}<p><strong>Ground truth comes from the labeled CSV.</strong> Predictions are compared with annotations on the held-out test split. Live Shopee and Lazada reviews have no ground-truth labels available to this extension, so a live scan cannot measure accuracy.</p></div>
      <div class="ac-test-source"><span class="ac-lang">Test split only</span><p>${model ? `${escapeHtml(model.reviewCount ?? "Unknown")} test reviews · ${escapeHtml(model.productCount ?? "Unknown")} products` : "No held-out test report has been published yet."}</p></div>
      <div class="ac-metric-grid">${[["Accuracy", "accuracy"], ["Macro precision", "macro_precision"], ["Macro recall", "macro_recall"], ["Macro F1", "macro_f1"]].map(([title, key]) => `<div><span>${title}</span><strong>${metricScore(metrics?.[key])}</strong></div>`).join("")}</div>
      <details class="ac-metric-definitions"><summary>What do these scores mean?</summary><dl><dt>Accuracy</dt><dd>The fraction of test reviews whose predicted quality label matches ground_truth.</dd><dt>Macro precision</dt><dd>Precision calculated for each of the four quality classes, then averaged equally.</dd><dt>Macro recall</dt><dd>Recall calculated for each quality class, then averaged equally.</dd><dt>Macro F1</dt><dd>F1 calculated for each quality class, then averaged equally. Undefined class scores use zero, matching model_metrics.py.</dd></dl></details>
      ${model ? `<section class="ac-eval-model"><h3>Evaluation source</h3><dl><dt>Evaluated model bundle</dt><dd>${escapeHtml(model.version)}</dd><dt>CSV dataset</dt><dd>${escapeHtml(model.dataset)}</dd><dt>Split manifest</dt><dd>${escapeHtml(model.split)}</dd><dt>Evaluation date</dt><dd>${escapeHtml(model.evaluatedAt)}</dd><dt>Primary target</dt><dd>ground_truth · final review-quality classification</dd></dl></section>` : `<p class="ac-eval-note">Scores will appear after a compatible trained bundle is evaluated on the test split and its report is published. Missing scores are not zero. Scanning a product page does not create or change these scores.</p>`}
      ${tasks.length ? `<h3>Quality evaluation by target</h3><p class="ac-eval-note">The final classifier uses ground_truth. RoBERTa's separate text-target evaluation uses text_label when included. These targets answer different questions.</p><div class="ac-table-scroll" tabindex="0" role="region" aria-label="Test metrics by target"><table class="ac-metrics-table"><caption>Test split · model predictions versus CSV labels</caption><thead><tr><th scope="col">Model / target</th><th scope="col">Accuracy</th><th scope="col">Macro precision</th><th scope="col">Macro recall</th><th scope="col">Macro F1</th></tr></thead><tbody>${tasks.map(task => `<tr><th scope="row">${escapeHtml(task.name)}<small>Target: ${escapeHtml(task.targetColumn || "ground_truth")}</small></th>${["accuracy", "macro_precision", "macro_recall", "macro_f1"].map(key => `<td>${metricScore(task.metrics?.[key])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : ""}
      ${quality ? renderMetricDetails(quality) : ""}
      <h3>Ground truth vs. prediction</h3>
      ${samples.length ? `<p class="ac-eval-note">Showing ${samples.length} example${samples.length === 1 ? "" : "s"} from ${escapeHtml(evaluation.comparisonCount)} recorded test predictions, in report order. Scores above use the full test split. Review text and usernames are not included.</p><div class="ac-table-scroll" tabindex="0" role="region" aria-label="Ground truth versus prediction examples"><table class="ac-metrics-table"><caption>CSV review IDs · final classifier</caption><thead><tr><th scope="col">Review ID</th><th scope="col">Ground truth</th><th scope="col">Prediction</th><th scope="col">Match</th></tr></thead><tbody>${samples.map(row => `<tr><th scope="row">${escapeHtml(row.id)}</th><td>${escapeHtml(verdictTitle(row.groundTruth))}</td><td>${escapeHtml(verdictTitle(row.prediction))}</td><td>${row.groundTruth === row.prediction ? "Correct" : "Incorrect"}</td></tr>`).join("")}</tbody></table></div>` : `<p class="ac-eval-note">No per-review test comparisons have been published. Publish a report from evaluate_models.py to show review IDs, annotated labels, and model predictions here.</p>`}
      <p class="ac-eval-note">These results describe the named evaluated bundle. They do not certify the model connected to your current live scan.</p>`;
  }

  function renderPerformance() {
    const models = globalThis.AuthentiCheckPerformance?.models || [];
    const own = models.find(model => model.name === "AuthentiCheck");
    const quality = own?.tasks?.find(task => task.name === "Review quality · RoBERTa + XGBoost");
    root.querySelector(".ac-performance-score").textContent = `Test Macro-F1 · ${metricScore(quality?.metrics?.macro_f1)}`;
    const headings = "<tr><th scope='col'>Model / task</th><th scope='col'>Macro-F1</th><th scope='col'>Precision (macro)</th><th scope='col'>Recall (macro)</th><th scope='col'>Accuracy</th></tr>";
    const rows = models.flatMap(model => (model.tasks || []).map(task => `<tr><th scope="row">${escapeHtml(model.name)}<br>${escapeHtml(task.name)}<small>${escapeHtml(task.scope)}</small></th>${["macro_f1", "macro_precision", "macro_recall", "accuracy"].map(key => `<td>${task.metrics?.[key] == null && key === "accuracy" && task.name === "Aspect detection" ? "Not reported" : metricScore(task.metrics?.[key])}</td>`).join("")}</tr>`));
    if (!own) rows.unshift(`<tr><th scope="row">AuthentiCheck<small>Quality classification · ten-aspect ABSA</small></th><td colspan="4">Not evaluated</td></tr>`);
    if (!models.some(model => model.name === "CLIP-CA-CG")) rows.push(`<tr><th scope="row">CLIP-CA-CG<small>Six-aspect ABSA · no quality classifier</small></th><td colspan="4">Not evaluated</td></tr>`);
    root.querySelector(".ac-performance-body").innerHTML = `
      <p class="ac-eval-note">Held-out test performance on labeled reviews. These scores stay fixed across product scans. Live review confidence and authentic share are separate measures.</p>
      <div class="ac-metric-grid"><div><span>Quality Macro-F1</span><strong>${metricScore(quality?.metrics?.macro_f1)}</strong></div><div><span>Macro-precision</span><strong>${metricScore(quality?.metrics?.macro_precision)}</strong></div><div><span>Macro-recall</span><strong>${metricScore(quality?.metrics?.macro_recall)}</strong></div><div><span>Accuracy</span><strong>${metricScore(quality?.metrics?.accuracy)}</strong></div></div>
      <h3>Model comparison</h3>
      <p class="ac-eval-note">AuthentiCheck uses ten aspects; CLIP-CA-CG uses six. Their overall ABSA scores have different scopes. Direct comparison requires evaluation on shared aspects, labels, and held-out reviews. No winner is inferred here.</p>
      <div class="ac-table-scroll" tabindex="0" role="region" aria-label="Model comparison table"><table class="ac-metrics-table"><caption>Held-out metrics by model and task</caption><thead>${headings}</thead><tbody>${rows.join("")}</tbody></table></div>
      ${models.length ? models.map(model => `<section class="ac-eval-model"><h3>${escapeHtml(model.name)} · evaluation record</h3><dl><dt>Evaluated bundle / version</dt><dd>${escapeHtml(model.version)}</dd><dt>Dataset</dt><dd>${escapeHtml(model.dataset)}</dd><dt>Split</dt><dd>${escapeHtml(model.split)}</dd><dt>Evaluated</dt><dd>${escapeHtml(model.evaluatedAt)}</dd><dt>Test coverage</dt><dd>${escapeHtml(model.reviewCount ?? "Unknown")} reviews · ${escapeHtml(model.productCount ?? "Unknown")} products</dd></dl>${(model.tasks || []).map(task => renderMetricDetails(task)).join("")}</section>`).join("") : `<p class="ac-eval-note">No held-out reports have been published. Scores and confusion matrices will appear after evaluation. Missing scores are not zero.</p>`}
      <p class="ac-eval-note">These are stored evaluation records for the named bundles. They do not certify the model connected to this browser session. Local preview has no measured test performance.</p>`;
  }

  function renderMetricDetails(task) {
    const metrics = task.metrics;
    if (!metrics?.per_class?.length) return "";
    const names = metrics.labels;
    return `<details class="ac-metric-details"><summary>${escapeHtml(task.name)} · class results &amp; confusion matrix</summary>
      <div class="ac-table-scroll" tabindex="0" role="region" aria-label="Per-class metrics"><table class="ac-metrics-table"><caption>Per-class precision, recall, F1, and support</caption><thead><tr><th scope="col">Class</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Support</th></tr></thead><tbody>${metrics.per_class.map(row => `<tr><th scope="row">${escapeHtml(row.name)}</th><td>${metricScore(row.precision)}</td><td>${metricScore(row.recall)}</td><td>${metricScore(row.f1)}</td><td>${escapeHtml(row.support)}</td></tr>`).join("")}</tbody></table></div>
      <div class="ac-table-scroll" tabindex="0" role="region" aria-label="Confusion matrix"><table class="ac-metrics-table"><caption>Confusion matrix · rows: actual, columns: predicted</caption><thead><tr><th scope="col">Actual / predicted</th>${names.map(name => `<th scope="col">${escapeHtml(name)}</th>`).join("")}</tr></thead><tbody>${metrics.confusion_matrix.map((row, i) => `<tr><th scope="row">${escapeHtml(names[i])}</th>${row.map(value => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;
  }

  function exportAnalysis() {
    if (scanInProgress || !displayedPayload || !lastResult) return;
    const exportData = { schemaVersion: "1.0", exportedAt: new Date().toISOString(), input: displayedPayload, analysis: lastResult, heldOutEvaluation: globalThis.AuthentiCheckPerformance || { schemaVersion: 1, models: [] } };
    const url = URL.createObjectURL(new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `authenticheck-${adapter?.id || "analysis"}-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function observePage() {
    const observer = new MutationObserver(() => {
      if (extensionContextInvalidated) return;
      if (location.href !== currentUrl) {
        currentUrl = location.href;
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(syncPageContext, 1800);
      } else if (root && adapter?.isProductPage()) {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(() => {
          if (!root || !adapter?.isProductPage()) return;
          try {
            const reviews = adapter.extractReviews().reviews;
            reviewsChanged = extractionSignature(reviews) !== lastExtractionSignature;
            if (reviewsChanged && settings.autoAnalyze) analyzePage();
            else updateFreshness();
          } catch (error) {
            lastPayload = null;
            renderError(`Page scan failed: ${error.message || "The loaded reviews could not be read."}`);
          }
        }, 2200);
      }
    });
    observer.observe(document.body, { childList:true, subtree:true });
  }

  function getPageStatus() {
    const analysisState = !settings.enabled ? "disabled" : root?.querySelector(".ac-panel").getAttribute("data-state") || "idle";
    const visibleResult = ["ready", "updating"].includes(analysisState) && displayedPayload?.url === location.href ? lastResult : null;
    const payload = lastPayload?.url === location.href ? lastPayload : displayedPayload?.url === location.href ? displayedPayload : null;
    return {
      productPage: Boolean(adapter?.isProductPage()), platform: adapter?.name || null,
      reviewCount: payload?.reviews?.length || 0,
      candidateCount: payload?.extraction?.candidateNodes || 0,
      analyzableCount: payload?.extraction?.analyzableReviews || 0,
      analysisState, resultMode: visibleResult?.mode || null, modelVersion: visibleResult?.modelVersion || null
    };
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type === "AUTHENTICHECK_RESCAN") {
      analyzePage();
      sendResponse({ requested: true });
    }
    if (message?.type === "AUTHENTICHECK_STATUS") {
      adapter = globalThis.AuthentiCheckExtractors?.getActive();
      sendResponse(getPageStatus());
    }
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    Object.entries(changes).forEach(([key, value]) => { settings[key] = value.newValue; });
    applyEnabledState();
    if (settings.enabled) analyzePage();
  });

  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closePanel();
  });

  start();
})();
