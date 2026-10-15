import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Exercise event handlers and rendered content without a live marketplace or model.
const nodes = new Map();
function node(selector) {
  if (!nodes.has(selector)) {
    const classes = new Set();
    nodes.set(selector, {
      value: "", textContent: "", innerHTML: "", attributes: {}, listeners: {}, dataset: {}, hidden: false, isConnected: true,
      classList: { add: key => classes.add(key), remove: key => classes.delete(key),
        contains: key => classes.has(key), toggle(key, on) { on ? classes.add(key) : classes.delete(key); } },
      setAttribute(key, value) { this.attributes[key] = value; },
      getAttribute(key) { return this.attributes[key]; },
      addEventListener(key, handler) { this.listeners[key] = handler; },
      focus() { this.focused = true; }, scrollIntoView() {},
      style: { setProperty() {} }
    });
  }
  return nodes.get(selector);
}
const names = ["overview", "insights", "reviews", "performance", "testing"];
const tabs = names.map(name => Object.assign(node(`tab-${name}`), { dataset: { tab: name } }));
const pages = names.map(name => Object.assign(node(`[data-page="${name}"]`), { dataset: { page: name } }));
const root = {
  querySelector: node,
  querySelectorAll(selector) {
    if (selector === ".ac-tab") return tabs;
    if (selector === ".ac-page") return pages;
    if (selector === ".ac-review") return Array.from({ length: (node(".ac-review-list").innerHTML.match(/<article /g) || []).length }, (_, index) => node(`review-${index}`));
    if (selector === ".ac-quality") return ["authentic", "liv", "irrelevant", "deceptive"].map(label => Object.assign(node(`quality-${label}`), { dataset: { filter: label } }));
    if (selector === ".ac-review-expand") return [...node(".ac-review-list").innerHTML.matchAll(/<button class="ac-review-expand" data-review-index="(\d+)"/g)].map(match => Object.assign(node(`expand-${match[1]}`), { dataset: { reviewIndex: match[1] }, closest: () => ({ querySelector: () => node(`text-${match[1]}`) }) }));
    if (selector === ".ac-signals-details") return [...node(".ac-review-list").innerHTML.matchAll(/<details class="ac-signals-details" data-review-index="(\d+)"[^>]*>/g)].map(match => Object.assign(node(`reasons-${match[1]}`), { dataset: { reviewIndex: match[1] }, open: / open/.test(match[0]) }));
    return [];
  }
};
const escape = text => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const context = {
  window: {}, location: { href: "https://shopee.ph/product/1" }, Date, testRoot: root, clearInterval() {}, setInterval: () => 1, setTimeout: callback => { callback(); return 1; },
  document: { addEventListener() {}, createElement() { return { set textContent(value) { this.innerHTML = escape(value); } }; } },
  chrome: { runtime: { onMessage: { addListener() {} } }, storage: { onChanged: { addListener() {} } } }
};
vm.createContext(context);
const source = readFileSync("extension/content.js", "utf8").replace("  start();", `
  root = globalThis.testRoot;
  globalThis.check = { renderReviews, renderOverview, renderInsights, highlightText, aspectEvidence,
    renderError, renderLoading, renderResult, reviewCard, reviewExcerpt, sortReviews, compareScans, showPage, updateFreshness, errorPresentation,
    analyzePage, useLocalPreview, getPageStatus, handlePanelKeydown,
    configure(options, activeAdapter) { settings = { ...settings, ...options }; adapter = activeAdapter; },
    setResult(result, payload) { lastResult = result; displayedPayload = payload; },
    setFreshness(at, changed) { lastAnalyzedAt = at; reviewsChanged = changed; } };
`);
vm.runInContext(source, context);
const check = context.check;
const reviews = Array.from({ length: 65 }, (_, index) => ({
  id: String(index), username: `reviewer_${index}`, label: index === 64 ? "unavailable" : ["authentic", "liv", "irrelevant", "deceptive"][index % 4],
  text: index === 64 ? "" : `Review ${index}: quality [x] <img src=x onerror=alert(1)>`, rating: 5, signals: [], imageUrls: []
}));
const result = { mode: "local", reviews, counts: { authentic: 16, liv: 16, irrelevant: 16, deceptive: 16 }, sentimentCounts: { positive: 0, negative: 0, neutral: 0 }, aspects: [{ name: "Quality", mentions: 16, positivePercent: 0 }] };
check.setResult(result);
check.renderReviews(result);
assert.equal(root.querySelectorAll(".ac-review").length, 20);
node(".ac-load-more").listeners.click();
assert.equal(root.querySelectorAll(".ac-review").length, 40);
node(".ac-load-more").listeners.click();
node(".ac-load-more").listeners.click();
assert.equal(root.querySelectorAll(".ac-review").length, 65);
assert.equal(node(".ac-load-more").hidden, true);
node(".ac-review-search").value = "Review 61:";
node(".ac-review-search").listeners.input();
assert.match(node(".ac-review-list").innerHTML, /<mark>Review 61:<\/mark>/);
assert.match(node(".ac-review-list").innerHTML, /<strong>reviewer_61<\/strong>/, "Username stays attached to its review when filtered");
assert.doesNotMatch(node(".ac-review-list").innerHTML, /<img/);
assert.match(node(".ac-review-count").textContent, /1 of 1 matching reviews · 65 loaded/);
node(".ac-clear-filters").listeners.click();
assert.equal(root.querySelectorAll(".ac-review").length, 20);
assert.equal(node(".ac-clear-filters").hidden, true);
node(".ac-review-search").value = "no such review";
node(".ac-review-search").listeners.input();
assert.match(node(".ac-review-list").innerHTML, /No reviews match/);
assert.equal(node(".ac-load-more").hidden, true);
assert.equal(check.highlightText("[x] <b>[x]</b>", "[x]"), "<mark>[x]</mark> &lt;b&gt;<mark>[x]</mark>&lt;/b&gt;");

check.renderOverview(result, 64, { reviews, extraction: { sellerResponsesExcluded: 2 } });
const overview = node('[data-page="overview"]').innerHTML;
assert.match(overview, /Scan coverage/);
assert.doesNotMatch(overview, /Extraction coverage|Input coverage|ac-donut/);
assert.match(overview, /Buyer photos are optional/);
node("quality-liv").listeners.click();
assert.equal(node(".ac-review-filter").value, "liv");
assert.equal(node(".ac-panel").attributes["data-view"], "reviews");
assert.equal(root.querySelectorAll(".ac-review").length, 16);
assert.equal(node(".ac-review-search").value, "");
node(".ac-show-skipped").listeners.click();
assert.equal(node(".ac-review-filter").value, "unavailable");
assert.equal(root.querySelectorAll(".ac-review").length, 1);

assert.equal(check.aspectEvidence(result, result.aspects[0]).length, 2);
const apiResult = { ...result, mode: "api" };
assert.equal(check.aspectEvidence(apiResult, result.aspects[0]).length, 0, "Do not invent model evidence from keyword matches");
const apiReview = { ...reviews[0], aspectSentiment: [{ category: "product_quality", sentiment: "positive" }] };
assert.equal(check.aspectEvidence({ mode: "api", reviews: [apiReview] }, { name: "product_quality" }).length, 1);
check.renderInsights(result);
assert.match(node('[data-page="insights"]').innerHTML, /Supporting reviews/);
assert.doesNotMatch(node('[data-page="insights"]').innerHTML, /<img/);

check.setFreshness(Date.now() - 120000, false);
check.updateFreshness();
assert.match(node(".ac-freshness").textContent, /2 minutes ago/);
check.setFreshness(Date.now(), true);
check.updateFreshness();
assert.equal(node(".ac-scan-new").hidden, false);
assert.match(node(".ac-freshness").textContent, /out of date/);
check.setFreshness(Date.now(), false);
check.updateFreshness();
assert.equal(node(".ac-scan-new").hidden, true);
assert.match(node(".ac-freshness").textContent, /just now/);

check.renderError("Model API returned HTTP 503. Stage 1 input contract mismatch; retrain");
assert.equal(node(".ac-error h2").textContent, "Model update required");
assert.doesNotMatch(node(".ac-error-message").textContent, /HTTP 503/);
assert.match(node(".ac-error-details p").textContent, /HTTP 503/);
assert.equal(node(".ac-error-details").open, false);
check.showPage("performance");
assert.equal(node(".ac-panel").attributes["data-state"], "error");
assert.equal(node(".ac-panel").attributes["data-view"], "performance");
assert.equal(tabs[3].attributes["aria-selected"], "true");
check.showPage("testing");
assert.equal(node(".ac-panel").attributes["data-view"], "testing");
assert.equal(tabs[4].attributes["aria-selected"], "true");
assert.equal(check.errorPresentation("Model API timed out").title, "The model took too long");
assert.equal(check.errorPresentation("AuthentiCheck was reloaded or updated").retry, "Refresh page");

// Sorting remains stable, keeps unrated reviews last, and never mutates the source.
const sortable = [{ id: "a", rating: 3, text: "short" }, { id: "b", rating: null, text: "very long text" }, { id: "c", rating: 1, text: "tiny" }, { id: "d", rating: 5, text: "a longer review" }, { id: "e", rating: 3, text: "other" }];
const sortedIds = order => Array.from(check.sortReviews(sortable, order), review => review.id);
assert.deepEqual(sortedIds("lowest"), ["c", "a", "e", "d", "b"]);
assert.deepEqual(sortedIds("highest"), ["d", "a", "e", "c", "b"]);
assert.deepEqual(sortedIds("longest"), ["d", "b", "a", "e", "c"]);
assert.deepEqual(sortable.map(review => review.id), ["a", "b", "c", "d", "e"]);

// A match deep in a long review stays visible before expansion.
const longText = "opening ".repeat(100) + "needle <script>alert(1)</script> " + "tail ".repeat(100);
assert.ok(check.reviewExcerpt(longText, "needle").includes("needle"));
const longReview = { ...reviews[0], id: "long", text: longText, signals: ["Product detail", "Stars agree with text"] };
const longResult = { ...result, reviews: [longReview] };
check.setResult(longResult);
check.renderReviews(longResult);
node(".ac-clear-filters").listeners.click();
assert.match(node('[data-page="reviews"]').innerHTML, /Authentic \(1\)/);
assert.match(node(".ac-review-list").innerHTML, /Read full review/);
assert.doesNotMatch(node(".ac-review-list").innerHTML, /<script>/);
node("expand-0").listeners.click();
assert.equal(node("expand-0").attributes["aria-expanded"], "true");
assert.match(node("text-0").innerHTML, /needle &lt;script&gt;/);
node("reasons-0").open = true;
node("reasons-0").listeners.toggle();
check.renderReviews(longResult);
assert.match(node(".ac-review-list").innerHTML, /Show less/);
assert.match(node(".ac-review-list").innerHTML, /class="ac-signals-details" data-review-index="0" open/);
assert.doesNotMatch(node(".ac-review-list").innerHTML, /Stage 2/);

// Each view restores its own scroll position.
check.showPage("reviews");
node(".ac-scroll").scrollTop = 740;
check.showPage("insights");
node(".ac-scroll").scrollTop = 230;
check.showPage("reviews");
assert.equal(node(".ac-scroll").scrollTop, 740);
check.showPage("insights");
assert.equal(node(".ac-scroll").scrollTop, 230);

const payload = { url: context.location.href, productTitle: "Test product", reviews: [longReview], extraction: {}, marketplaceRating: 4.5 };
check.setResult(longResult, payload);
node(".ac-review-sort").value = "longest";
node(".ac-review-sort").listeners.change();
node(".ac-review-search").value = "needle";
node(".ac-review-search").listeners.input();
check.renderLoading();
assert.equal(node(".ac-panel").attributes["data-state"], "updating");
assert.equal(node(".ac-panel").attributes["data-has-results"], "true");
assert.equal(node(".ac-results").classList.contains("hidden"), false);
assert.equal(node(".ac-export").disabled, true);
assert.match(node(".ac-freshness").textContent, /Showing your previous scan/);
node(".ac-scroll").scrollTop = 360;
check.renderResult(longResult, payload);
assert.equal(node(".ac-scroll").scrollTop, 360);
assert.equal(node(".ac-review-sort").value, "longest");
assert.equal(node(".ac-review-search").value, "needle");
assert.match(node(".ac-review-list").innerHTML, /Show less/);
assert.equal(node(".ac-results").attributes["aria-busy"], "false");
assert.equal(node(".ac-export").disabled, false);
assert.equal(check.compareScans(longResult, payload, { ...longResult, reviews: [longReview, { ...longReview, id: "new" }] }, payload), "1 additional review analyzed");
assert.equal(check.compareScans(longResult, payload, longResult, { ...payload, url: "other-product" }), "");
assert.equal(check.compareScans(longResult, payload, { ...longResult, mode: "api" }, payload), "");
assert.match(check.reviewCard({ ...longReview, username: "m***a" }, 0), /<strong>m\*\*\*a<\/strong>/);
assert.match(check.reviewCard({ ...longReview, username: "" }, 0), /Reviewer name unavailable/);
assert.doesNotMatch(check.reviewCard({ ...longReview, username: "<img src=x onerror=alert(1)>" }, 0), /<img/);
assert.doesNotMatch(source, /ac-market-rating|ac-verified-rating|Authenticity-adjusted rating|ratingExplanation/);

// Exercise actual scans, not just render helpers, through failure and recovery.
const apiResponse = JSON.parse(readFileSync("data/samples/extension_api_response.json", "utf8"));
const input = { ...reviews[0], id: apiResponse.reviews[0].id, analysisEligible: true, hasImage: false };
let loaded = [input];
let extractionFails = false;
const adapter = { id: "shopee", name: "Shopee", isProductPage: () => true,
  getProductTitle: () => "Recovery fixture", getMarketplaceRating: () => null,
  extractReviews() { if (extractionFails) throw new Error("Changed markup"); return { reviews: loaded, stats: { analyzableReviews: loaded.length } }; }
};
check.configure({ enabled: true, useApi: true }, adapter);
const requests = [];
context.chrome.runtime.sendMessage = () => new Promise(resolve => requests.push(resolve));
const failedScan = check.analyzePage();
assert.equal(check.getPageStatus().analysisState, "updating");
requests.shift()({ ok: false, error: "Model API returned HTTP 503. Stage 1 input contract mismatch; retrain" });
await failedScan;
assert.equal(check.getPageStatus().analysisState, "error");
assert.equal(check.getPageStatus().resultMode, null, "Previous results must not be reported as current after failure");
assert.equal(node(".ac-results").classList.contains("hidden"), true);
assert.equal(node(".ac-export").disabled, true);
check.useLocalPreview();
assert.equal(check.getPageStatus().resultMode, "local", "Local preview must be an explicit choice");
assert.equal(node(".ac-mode").textContent, "Local estimate");
const retry = check.analyzePage();
requests.shift()({ ok: true, result: apiResponse });
await retry;
assert.equal(check.getPageStatus().analysisState, "ready");
assert.equal(check.getPageStatus().resultMode, "api");
assert.equal(node(".ac-export").disabled, false);
assert.match(node(".ac-status-text").textContent, /1 of 1 loaded review analyzed/);
loaded = [];
await check.analyzePage();
assert.equal(check.getPageStatus().analysisState, "empty");
assert.equal(check.getPageStatus().resultMode, null);
extractionFails = true;
await assert.doesNotReject(check.analyzePage());
assert.equal(node(".ac-error h2").textContent, "Could not read the loaded reviews");
assert.equal(node(".ac-local-preview").disabled, true, "Do not preview stale inputs when extraction failed");
extractionFails = false;
loaded = [input];
const oldScan = check.analyzePage();
const newerScan = check.analyzePage();
requests[1]({ ok: true, result: { ...apiResponse, modelVersion: "newest" } });
await newerScan;
requests[0]({ ok: true, result: { ...apiResponse, modelVersion: "old" } });
await oldScan;
assert.equal(check.getPageStatus().modelVersion, "newest", "An older slow response must not overwrite a newer completed scan");

// Keyboard navigation wraps within the open dialog and skips hidden controls.
const firstControl = { tabIndex: 0, disabled: false, closest: () => null, getClientRects: () => [{}], focus() { this.focused = true; } };
const lastControl = { ...firstControl };
const hiddenControl = { ...firstControl, getClientRects: () => [] };
node(".ac-panel").querySelectorAll = () => [firstControl, hiddenControl, lastControl];
node(".ac-panel").classList.add("open");
root.activeElement = lastControl;
let prevented = false;
check.handlePanelKeydown({ key: "Tab", shiftKey: false, preventDefault() { prevented = true; } });
assert.equal(prevented, true);
assert.equal(firstControl.focused, true);
root.activeElement = firstControl;
check.handlePanelKeydown({ key: "Tab", shiftKey: true, preventDefault() {} });
assert.equal(lastControl.focused, true);
check.handlePanelKeydown({ key: "Escape", preventDefault() {}, stopPropagation() {} });
assert.equal(node(".ac-panel").classList.contains("open"), false);
assert.equal(node(".ac-trigger").focused, true);
console.log("Frontend reading controls, scan recovery/races, actual mode reporting, and keyboard containment checks passed.");
