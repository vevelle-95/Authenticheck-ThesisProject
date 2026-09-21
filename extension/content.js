(() => {
  if (window.__authentiCheckInjected) return;
  window.__authentiCheckInjected = true;

  const DEFAULTS = { enabled: true, autoAnalyze: true, useApi: false };
  const POSITIVE = ["good","great","best","love","excellent","perfect","comfortable","clear","quality","worth","nice","sulit","maganda","maayos","malinaw","matibay","ganda","okay","ayos","recommend"];
  const NEGATIVE = ["bad","poor","broken","fake","damaged","disappointed","waste","uncomfortable","mahina","pangit","sira","mali","sayang","hindi gumagana","defective","kulang","madumi"];
  const GENERIC = ["good product","nice product","okay","ok","good quality","recommended","highly recommended","perfect product","maganda","maayos","sulit"];
  const DELIVERY = ["delivery","shipping","courier","rider","parcel","dumating","mabilis dumating","seller responsive","packaging"];
  const ASPECTS = {
    Quality: ["quality","matibay","durable","material","build","ganda","finish","sira"],
    Performance: ["performance","works","gumagana","fast","mabilis","battery","sound","audio","camera"],
    Value: ["price","presyo","sulit","worth","mahal","cheap","affordable"],
    Delivery: DELIVERY,
    Appearance: ["color","kulay","design","look","size","laki","maliit"]
  };

  let settings = { ...DEFAULTS };
  let currentUrl = location.href;
  let refreshTimer;
  let analysisRun = 0;
  let host;
  let root;
  let lastPayload;
  let lastResult;
  let adapter;

  const shield = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2 8 3v6c0 5.1-3.4 9.6-8 11-4.6-1.4-8-5.9-8-11V5l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></svg>`;
  const closeIcon = `<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>`;
  const infoIcon = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/></svg>`;
  const arrowIcon = `<svg viewBox="0 0 24 24"><path d="M5 12h14M15 8l4 4-4 4"/></svg>`;

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
      <button class="ac-trigger" aria-label="Open AuthentiCheck analysis">
        <span class="ac-trigger-logo">${shield}</span>
        <span class="ac-trigger-copy"><strong>Checking visible reviews…</strong><small>AuthentiCheck · Tap to view</small></span>
        <span class="ac-trigger-score">—</span>
      </button>
      <aside class="ac-panel" aria-hidden="true" aria-label="AuthentiCheck review analysis">
        <header class="ac-head">
          <div class="ac-brand"><span class="ac-brand-mark">${shield}</span><div><strong>AuthentiCheck</strong><small>Review intelligence</small></div></div>
          <div class="ac-head-actions"><button class="ac-icon-btn ac-info" aria-label="About this analysis">${infoIcon}</button><button class="ac-icon-btn ac-close" aria-label="Close AuthentiCheck">${closeIcon}</button></div>
        </header>
        <div class="ac-scroll">
          <section class="ac-state ac-loading"><span class="ac-spinner"></span><h2>Analyzing visible reviews</h2><p>Preparing review text, ratings, and available buyer media.</p></section>
          <section class="ac-state ac-empty"><span class="ac-empty-icon">${shield}</span><h2>Reviews aren't loaded yet</h2><p>Scroll to the product reviews so the marketplace loads them, then run the scan again.</p><button class="ac-rescan">Scan visible reviews</button></section>
          <section class="ac-state ac-error"><span class="ac-empty-icon ac-error-icon">!</span><h2>Model connection failed</h2><p class="ac-error-message">The local inference service could not be reached.</p><div class="ac-error-actions"><button class="ac-retry">Try model again</button><button class="ac-local-preview">Use local preview</button></div><small>Local preview is heuristic-only and is never presented as model output.</small></section>
          <div class="ac-results hidden">
            <section class="ac-hero">
              <div class="ac-status"><i></i><span class="ac-status-text">Analysis complete</span><span class="ac-mode">Local estimate</span></div>
              <div class="ac-score-row"><div class="ac-ring"><div><strong>0</strong><span>%</span></div></div><div class="ac-score-copy"><span class="ac-eyebrow">Authentic review share</span><h2>Analyzing…</h2><p>Checking review detail, rating consistency, and visible buyer media.</p></div></div>
              <div class="ac-rating"><div><span>Marketplace rating</span><strong class="ac-market-rating">—</strong></div>${arrowIcon}<div><span>Authenticity-adjusted rating</span><strong class="ac-verified-rating">—</strong></div><span class="ac-delta">—</span></div>
            </section>
            <nav class="ac-tabs"><button class="ac-tab active" data-tab="overview">Overview</button><button class="ac-tab" data-tab="insights">Insights</button><button class="ac-tab" data-tab="reviews">Reviews</button></nav>
            <section class="ac-page active" data-page="overview"></section>
            <section class="ac-page" data-page="insights"></section>
            <section class="ac-page" data-page="reviews"></section>
          </div>
        </div>
        <footer class="ac-footer"><span><i></i><b>Analyzes visible public reviews only</b></span><div><button class="ac-export" disabled>Export JSON</button><button class="ac-rescan">Rescan</button></div></footer>
      </aside>`;
    root.append(wrapper);
    document.documentElement.append(host);

    root.querySelector(".ac-trigger").addEventListener("click", openPanel);
    root.querySelector(".ac-close").addEventListener("click", closePanel);
    root.querySelector(".ac-dim").addEventListener("click", closePanel);
    root.querySelectorAll(".ac-rescan").forEach(button => button.addEventListener("click", analyzePage));
    root.querySelector(".ac-retry").addEventListener("click", analyzePage);
    root.querySelector(".ac-local-preview").addEventListener("click", () => {
      if (lastPayload) renderResult(analyzeLocally(lastPayload), lastPayload);
    });
    root.querySelector(".ac-export").addEventListener("click", exportAnalysis);
    root.querySelector(".ac-info").addEventListener("click", () => {
      root.querySelector(".ac-mode").textContent = settings.useApi ? "Model API" : "Local estimate · not a model verdict";
    });
    root.querySelectorAll(".ac-tab").forEach(button => button.addEventListener("click", () => showPage(button.dataset.tab)));
  }

  function destroyShell() {
    analysisRun += 1;
    if (host?.isConnected) host.remove();
    host = null;
    root = null;
    lastPayload = null;
    lastResult = null;
  }

  function openPanel() {
    root.querySelector(".ac-panel").classList.add("open");
    root.querySelector(".ac-panel").setAttribute("aria-hidden", "false");
    root.querySelector(".ac-trigger").classList.add("hidden");
    root.querySelector(".ac-dim").classList.add("open");
  }

  function closePanel() {
    root.querySelector(".ac-panel").classList.remove("open");
    root.querySelector(".ac-panel").setAttribute("aria-hidden", "true");
    root.querySelector(".ac-trigger").classList.remove("hidden");
    root.querySelector(".ac-dim").classList.remove("open");
  }

  function showPage(name) {
    root.querySelectorAll(".ac-tab").forEach(tab => tab.classList.toggle("active", tab.dataset.tab === name));
    root.querySelectorAll(".ac-page").forEach(page => page.classList.toggle("active", page.dataset.page === name));
  }

  function applyEnabledState() {
    host.style.display = settings.enabled ? "block" : "none";
  }

  function wordHits(text, terms) {
    return terms.reduce((sum, term) => sum + (text.includes(term) ? 1 : 0), 0);
  }

  function classifyReview(review) {
    const text = review.text.toLowerCase();
    const tokens = text.match(/[a-zà-ž0-9]+/gi) || [];
    const pos = wordHits(text, POSITIVE);
    const neg = wordHits(text, NEGATIVE);
    const genericOnly = GENERIC.some(term => text === term || text === `${term}!`) || (tokens.length <= 6 && GENERIC.some(term => text.includes(term)));
    const deliveryHits = wordHits(text, DELIVERY);
    const productAspectHits = Object.entries(ASPECTS).filter(([name]) => name !== "Delivery").reduce((sum, [,terms]) => sum + wordHits(text, terms), 0);
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
    reviews.forEach(review => counts[review.label]++);
    const authentic = reviews.filter(review => review.label === "authentic");
    const sentimentCounts = { positive: 0, neutral: 0, negative: 0 };
    authentic.forEach(review => sentimentCounts[review.sentiment]++);
    const confidence = reviews.length ? Math.round((counts.authentic / reviews.length) * 100) : 0;
    const ratedAuthentic = authentic.filter(review => review.rating);
    const verifiedRating = ratedAuthentic.length
      ? ratedAuthentic.reduce((sum, review) => sum + review.rating, 0) / ratedAuthentic.length
      : payload.marketplaceRating;
    const aspects = Object.entries(ASPECTS).map(([name, terms]) => {
      const matching = authentic.filter(review => wordHits(review.text.toLowerCase(), terms));
      const positive = matching.filter(review => review.sentiment === "positive").length;
      return { name, mentions: matching.length, positivePercent: matching.length ? Math.round(positive / matching.length * 100) : 0 };
    }).filter(aspect => aspect.mentions > 0).sort((a, b) => b.mentions - a.mentions).slice(0, 5);
    return { mode: "local", reviews, counts, sentimentCounts, authenticShare: confidence, verifiedRating, aspects };
  }

  async function analyzePage() {
    if (!root || !settings.enabled || !adapter?.isProductPage()) return;
    const runId = ++analysisRun;
    renderLoading();
    const reviews = adapter.extractReviews();
    const payload = {
      platform: adapter.name,
      url: location.href,
      productTitle: adapter.getProductTitle(),
      productDescription: adapter.getProductDescription?.() || "",
      marketplaceRating: adapter.getMarketplaceRating(),
      reviews
    };
    lastPayload = payload;

    if (!reviews.length) {
      renderEmpty();
      return;
    }

    let result = null;
    if (settings.useApi) {
      const response = await chrome.runtime.sendMessage({ type: "AUTHENTICHECK_ANALYZE", payload }).catch(() => null);
      if (runId !== analysisRun) return;
      if (response?.ok && response.result) result = normalizeApiResult(response.result, payload);
      else {
        renderError(response?.error || "The model API did not respond. Check that the local service is running.");
        return;
      }
    }
    if (!result) result = analyzeLocally(payload);
    renderResult(result, payload);
  }

  function normalizeApiResult(api, payload) {
    const local = analyzeLocally(payload);
    const numberOr = (value, fallback, min = 0, max = Number.MAX_SAFE_INTEGER) => {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? Math.max(min, Math.min(max, numeric)) : fallback;
    };
    const labelMap = { authentic:"authentic", deceptive:"deceptive", irrelevant:"irrelevant", liv:"liv", "low-value":"liv", low_value:"liv", lowValue:"liv" };
    const apiReviews = Array.isArray(api.reviews) ? api.reviews.map((review, index) => {
      const fallback = local.reviews.find(item => item.id === review?.id) || local.reviews[index] || {};
      return {
        ...fallback,
        ...review,
        text: String(review?.text ?? fallback.text ?? ""),
        label: labelMap[review?.label] || fallback.label || "liv",
        signals: Array.isArray(review?.signals) ? review.signals.map(String).slice(0, 8) : (fallback.signals || [])
      };
    }) : local.reviews;
    const apiCounts = api.counts || {};
    return {
      ...local,
      mode: "api",
      authenticShare: numberOr(api.authenticShare ?? api.confidence, local.authenticShare, 0, 100),
      verifiedRating: numberOr(api.verifiedRating, local.verifiedRating, 0, 5),
      counts: {
        authentic: numberOr(apiCounts.authentic, local.counts.authentic),
        liv: numberOr(apiCounts.liv ?? apiCounts.lowValue ?? apiCounts["low-value"], local.counts.liv),
        irrelevant: numberOr(apiCounts.irrelevant, local.counts.irrelevant),
        deceptive: numberOr(apiCounts.deceptive, local.counts.deceptive)
      },
      sentimentCounts: {
        positive: numberOr(api.sentimentCounts?.positive, local.sentimentCounts.positive),
        neutral: numberOr(api.sentimentCounts?.neutral, local.sentimentCounts.neutral),
        negative: numberOr(api.sentimentCounts?.negative, local.sentimentCounts.negative)
      },
      reviews: apiReviews,
      aspects: Array.isArray(api.aspects) ? api.aspects.slice(0, 12).map(aspect => ({
        name: String(aspect?.name || "Unnamed aspect"),
        mentions: numberOr(aspect?.mentions, 0),
        positivePercent: numberOr(aspect?.positivePercent, 0, 0, 100)
      })) : local.aspects
    };
  }

  function renderEmpty() {
    hideStates();
    root.querySelector(".ac-empty").classList.add("show");
    root.querySelector(".ac-results").classList.add("hidden");
    root.querySelector(".ac-trigger-copy strong").textContent = "Open product reviews to analyze";
    root.querySelector(".ac-trigger-copy small").textContent = "Scroll to reviews, then tap";
    root.querySelector(".ac-trigger-score").textContent = "—";
  }

  function hideStates() {
    root.querySelectorAll(".ac-state").forEach(state => state.classList.remove("show"));
  }

  function renderLoading() {
    hideStates();
    root.querySelector(".ac-results").classList.add("hidden");
    root.querySelector(".ac-loading").classList.add("show");
    root.querySelector(".ac-trigger-copy strong").textContent = settings.useApi ? "Waiting for model analysis…" : "Checking visible reviews…";
    root.querySelector(".ac-trigger-score").textContent = "…";
  }

  function renderError(message) {
    hideStates();
    root.querySelector(".ac-results").classList.add("hidden");
    root.querySelector(".ac-error").classList.add("show");
    root.querySelector(".ac-error-message").textContent = message;
    root.querySelector(".ac-trigger-copy strong").textContent = "Model connection failed";
    root.querySelector(".ac-trigger-copy small").textContent = "Open for recovery options";
    root.querySelector(".ac-trigger-score").textContent = "!";
    openPanel();
  }

  function percent(value, total) { return total ? Math.round(value / total * 100) : 0; }
  function escapeHtml(value = "") { const div = document.createElement("div"); div.textContent = String(value); return div.innerHTML; }
  function sentimentPercent(result, key) { const total = Object.values(result.sentimentCounts).reduce((a,b) => a+b, 0); return percent(result.sentimentCounts[key] || 0, total); }
  function verdictTitle(label) { return ({ authentic:"Authentic", liv:"Low-value", irrelevant:"Irrelevant", deceptive:"Potential mismatch" })[label] || label; }

  function renderResult(result, payload) {
    const total = Object.values(result.counts).reduce((a,b) => a+b, 0);
    const confidence = Math.max(0, Math.min(100, Math.round(result.authenticShare || 0)));
    const marketplace = Number(payload.marketplaceRating);
    const verified = Number(result.verifiedRating);
    const delta = Number.isFinite(marketplace) && Number.isFinite(verified) ? verified - marketplace : null;
    const status = confidence >= 75 ? "Highly trustworthy" : confidence >= 50 ? "Mixed review quality" : "Use extra caution";

    lastResult = result;
    hideStates();
    root.querySelector(".ac-results").classList.remove("hidden");
    root.querySelector(".ac-export").disabled = false;
    root.querySelector(".ac-status-text").textContent = `${total} visible review${total === 1 ? "" : "s"} analyzed`;
    root.querySelector(".ac-mode").textContent = result.mode === "api" ? "Model API" : "Local estimate";
    root.querySelector(".ac-ring").style.setProperty("--score", confidence);
    root.querySelector(".ac-ring strong").textContent = confidence;
    root.querySelector(".ac-score-copy h2").textContent = status;
    root.querySelector(".ac-score-copy p").textContent = result.mode === "api"
      ? "Model results combine review text, ratings, and available buyer media."
      : "Preliminary signals from visible text, ratings, and buyer media. Connect the model API for research-grade results.";
    root.querySelector(".ac-market-rating").innerHTML = Number.isFinite(marketplace) ? `${marketplace.toFixed(1)} <em>★</em>` : "Not found";
    root.querySelector(".ac-verified-rating").innerHTML = Number.isFinite(verified) ? `${verified.toFixed(1)} <em>★</em>` : "—";
    root.querySelector(".ac-delta").textContent = delta == null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}`;
    root.querySelector(".ac-trigger-score").textContent = confidence;
    root.querySelector(".ac-trigger-copy strong").textContent = `${confidence}% look authentic`;
    root.querySelector(".ac-trigger-copy small").textContent = `${total} visible review${total === 1 ? "" : "s"} analyzed`;

    renderOverview(result, total, payload);
    renderInsights(result);
    renderReviews(result);
  }

  function renderOverview(result, total, payload) {
    const p = key => percent(result.counts[key] || 0, total);
    const positive = sentimentPercent(result, "positive");
    const neutral = sentimentPercent(result, "neutral");
    const negative = sentimentPercent(result, "negative");
    root.querySelector('[data-page="overview"]').innerHTML = `
      <div class="ac-title"><div><h3>Review quality</h3><p>How currently visible reviews were classified</p></div><span class="ac-lang">Tagalog + Taglish</span></div>
      <div class="ac-card"><div class="ac-quality-total"><div><strong>${result.counts.authentic || 0}</strong><span>Authentic reviews</span></div><b>${p("authentic")}%</b></div>
        <div class="ac-stack"><i class="ac-q-auth" style="width:${p("authentic")}%"></i><i class="ac-q-liv" style="width:${p("liv")}%"></i><i class="ac-q-irrel" style="width:${p("irrelevant")}%"></i><i class="ac-q-decep" style="width:${p("deceptive")}%"></i></div>
        <div class="ac-quality-grid">
          ${qualityRow("ac-q-auth","Authentic",result.counts.authentic,p("authentic"))}
          ${qualityRow("ac-q-liv","Low-value",result.counts.liv,p("liv"))}
          ${qualityRow("ac-q-irrel","Irrelevant",result.counts.irrelevant,p("irrelevant"))}
          ${qualityRow("ac-q-decep","Potential mismatch",result.counts.deceptive,p("deceptive"))}
        </div>
      </div>
      <div class="ac-title" style="margin-top:16px"><div><h3>Authentic sentiment</h3><p>Based on reviews classified as authentic</p></div></div>
      <div class="ac-card ac-sentiment"><div class="ac-donut" style="--positive:${positive};--neutral:${neutral}"><div><strong>${positive}%</strong><span>positive</span></div></div><div class="ac-legend"><div><i class="positive"></i><strong>Positive</strong><b>${positive}%</b></div><div><i class="neutral"></i><strong>Neutral</strong><b>${neutral}%</b></div><div><i class="negative"></i><strong>Negative</strong><b>${negative}%</b></div></div></div>
      <div class="ac-title" style="margin-top:16px"><div><h3>Input coverage</h3><p>Data prepared for the analysis pipeline</p></div></div>
      <div class="ac-card ac-coverage"><div><strong>${payload.reviews.length}</strong><span>Visible reviews</span></div><div><strong>${payload.reviews.filter(review => review.rating).length}</strong><span>With ratings</span></div><div><strong>${payload.reviews.filter(review => review.imageUrls?.length).length}</strong><span>With buyer media</span></div></div>
      <div class="ac-note">${shield}<p><strong>${result.mode === "api" ? "Model-connected analysis." : "Preliminary local estimate."}</strong> Only content already visible in your browser is read. No account or checkout information is collected.</p></div>`;
  }

  function qualityRow(colorClass, title, count = 0, pct = 0) {
    return `<div class="ac-quality"><i class="${colorClass}"></i><div><strong>${title}</strong><small>${count} review${count === 1 ? "" : "s"}</small></div><b>${pct}%</b></div>`;
  }

  function renderInsights(result) {
    const aspects = result.aspects.length ? result.aspects : [{ name:"No aspects found", mentions:0, positivePercent:0 }];
    root.querySelector('[data-page="insights"]').innerHTML = `
      <div class="ac-title"><div><h3>Product aspects</h3><p>What authentic visible reviews mention</p></div></div>
      <div class="ac-aspects">${aspects.map((aspect,index) => `<div class="ac-aspect"><span class="ac-aspect-icon">${["✦","◎","₱","◴","◇"][index] || "•"}</span><div><div class="ac-aspect-head"><strong>${escapeHtml(aspect.name)}</strong><b>${aspect.positivePercent}% positive</b></div><span class="ac-bar"><i style="width:${aspect.positivePercent}%"></i></span><p>${aspect.mentions} authentic mention${aspect.mentions === 1 ? "" : "s"} found in the loaded reviews.</p></div></div>`).join("")}</div>
      <div class="ac-takeaway"><span>✦</span><div><strong>Scope note</strong><p>Aspect coverage grows as the marketplace loads more reviews. Scroll further and rescan for a broader sample.</p></div></div>`;
  }

  function renderReviews(result) {
    root.querySelector('[data-page="reviews"]').innerHTML = `
      <div class="ac-title"><div><h3>Review evidence</h3><p>Signals behind each visible classification</p></div></div>
      <div class="ac-review-tools"><input class="ac-review-search" type="search" placeholder="Search visible review text" aria-label="Search visible review text"><select class="ac-review-filter" aria-label="Filter review classification"><option value="all">All classifications</option><option value="authentic">Authentic</option><option value="liv">Low-value</option><option value="irrelevant">Irrelevant</option><option value="deceptive">Potential mismatch</option></select></div>
      <p class="ac-review-count"></p><div class="ac-review-list"></div>`;
    const search = root.querySelector(".ac-review-search");
    const filter = root.querySelector(".ac-review-filter");
    const update = () => {
      const query = search.value.trim().toLowerCase();
      const selected = filter.value;
      const visible = result.reviews.filter(review => (selected === "all" || review.label === selected) && (!query || review.text.toLowerCase().includes(query))).slice(0, 50);
      root.querySelector(".ac-review-count").textContent = `${visible.length} of ${result.reviews.length} reviews shown`;
      root.querySelector(".ac-review-list").innerHTML = visible.length ? visible.map((review,index) => reviewCard(review,index)).join("") : `<div class="ac-no-match">No reviews match this filter.</div>`;
    };
    search.addEventListener("input", update);
    filter.addEventListener("change", update);
    update();
  }

  function reviewCard(review, index) {
    const media = review.imageUrls?.length ? `<span>▣ ${review.imageUrls.length} media</span>` : "";
    return `<article class="ac-review"><header><div class="ac-review-user"><span class="ac-avatar">${String(index+1).padStart(2,"0")}</span><div><strong>Visible review</strong><span class="ac-stars">${"★".repeat(Math.max(0,Math.min(5,review.rating || 0)))}</span></div></div><span class="ac-verdict ${escapeHtml(review.label)}">${escapeHtml(verdictTitle(review.label))}</span></header><p>${escapeHtml(review.text)}</p><div class="ac-signals">${(review.signals || []).map(signal => `<span class="${review.label === "authentic" ? "" : "warn"}">${review.label === "authentic" ? "✓" : "!"} ${escapeHtml(signal)}</span>`).join("")}${media}</div></article>`;
  }

  function exportAnalysis() {
    if (!lastPayload || !lastResult) return;
    const exportData = { schemaVersion: "0.2", exportedAt: new Date().toISOString(), input: lastPayload, analysis: lastResult };
    const url = URL.createObjectURL(new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `authenticheck-${adapter?.id || "analysis"}-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function observePage() {
    const observer = new MutationObserver(() => {
      if (location.href !== currentUrl) {
        currentUrl = location.href;
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(syncPageContext, 1800);
      } else if (settings.autoAnalyze && root && adapter?.isProductPage()) {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(() => {
          const count = adapter.extractReviews().length;
          if (count !== lastPayload?.reviews?.length) analyzePage();
        }, 2200);
      }
    });
    observer.observe(document.body, { childList:true, subtree:true });
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type === "AUTHENTICHECK_RESCAN") analyzePage();
    if (message?.type === "AUTHENTICHECK_STATUS") {
      adapter = globalThis.AuthentiCheckExtractors?.getActive();
      sendResponse({
        productPage: Boolean(adapter?.isProductPage()),
        platform: adapter?.name || null,
        reviewCount: lastPayload?.reviews?.length || 0
      });
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
