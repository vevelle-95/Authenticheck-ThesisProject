import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const read = path => readFileSync(path, "utf8");
const context = {
  window: {}, location: { href: "https://shopee.ph/" },
  chrome: { runtime: { onMessage: { addListener() {} } }, storage: { onChanged: { addListener() {} } } },
  document: { addEventListener() {} }
};
vm.createContext(context);
vm.runInContext(read("extension/content.js").replace("  start();", "  globalThis.check = { normalizeApiResult, applyEnabledState, closePanel };"), context);
assert.doesNotThrow(() => context.check.applyEnabledState());
assert.doesNotThrow(() => context.check.closePanel());

// Closed panels must not leave invisible controls in the keyboard sequence.
const panelNodes = new Map();
function panelNode(selector) {
  if (!panelNodes.has(selector)) {
    const classes = new Set();
    panelNodes.set(selector, { attributes: {}, focused: false,
      classList: { add: name => classes.add(name), remove: name => classes.delete(name),
        contains: name => classes.has(name),
        toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
      setAttribute(name, value) { this.attributes[name] = value; },
      focus() { this.focused = true; }
    });
  }
  return panelNodes.get(selector);
}
const tabs = ["overview", "insights", "reviews", "performance", "testing"].map(name => Object.assign(panelNode(name), { dataset: { tab: name } }));
const pages = tabs.map(tab => ({ dataset: { page: tab.dataset.tab }, classList: { toggle() {} } }));
context.testRoot = { querySelector: panelNode, querySelectorAll: selector => selector === ".ac-tab" ? tabs : pages };
context.window = {};
vm.runInContext(read("extension/content.js").replace("  start();", "  root = globalThis.testRoot; globalThis.panelCheck = { openPanel, closePanel, showPage };"), context);
context.panelCheck.openPanel();
assert.equal(panelNode(".ac-panel").inert, false);
assert.equal(panelNode(".ac-close").focused, true);
assert.equal(panelNode(".ac-trigger").attributes["aria-expanded"], "true");
context.panelCheck.showPage("reviews");
assert.deepEqual(tabs.map(tab => tab.tabIndex), [-1, -1, 0, -1, -1]);
assert.deepEqual(tabs.map(tab => tab.attributes["aria-selected"]), ["false", "false", "true", "false", "false"]);
context.panelCheck.showPage("performance");
assert.deepEqual(tabs.map(tab => tab.tabIndex), [-1, -1, -1, 0, -1]);
assert.equal(tabs[3].attributes["aria-selected"], "true");
context.panelCheck.showPage("testing");
assert.deepEqual(tabs.map(tab => tab.tabIndex), [-1, -1, -1, -1, 0]);
context.panelCheck.closePanel();
assert.equal(panelNode(".ac-panel").inert, true);
assert.equal(panelNode(".ac-trigger").inert, false);
assert.equal(panelNode(".ac-trigger").focused, true);
assert.equal(panelNode(".ac-trigger").attributes["aria-expanded"], "false");

const request = JSON.parse(read("data/samples/extension_api_payload.json"));
const response = JSON.parse(read("data/samples/extension_api_response.json"));
const payload = { ...request, reviews: request.reviews.map(review => ({ ...review, analysisEligible: true })) };
assert.equal(context.check.normalizeApiResult(response, payload).mode, "api");
const namedPayload = { ...payload, reviews: payload.reviews.map(review => ({ ...review, username: "m***a" })) };
const renamedResponse = { ...response, reviews: response.reviews.map(review => ({ ...review, username: "API must not rename buyers" })) };
assert.equal(context.check.normalizeApiResult(renamedResponse, namedPayload).reviews[0].username, "m***a");
const unratedPayload = { ...payload, reviews: [...payload.reviews,
  { id: "unrated", text: "Detailed feedback without stars", rating: null,
    analysisEligible: false, missingFields: ["rating", "image"] }] };
const unrated = context.check.normalizeApiResult(response, unratedPayload).reviews[1];
assert.equal(unrated.label, "unavailable");
assert.match(unrated.signals[0], /missing rating/);
assert.doesNotMatch(unrated.signals[0], /image/);
for (const value of [null, "", false, "100", -1, 101]) {
  assert.throws(() => context.check.normalizeApiResult({ ...response, authenticShare: value }, payload));
}
assert.throws(() => context.check.normalizeApiResult({ ...response, counts: { ...response.counts, authentic: 0 } }, payload));
assert.throws(() => context.check.normalizeApiResult({ ...response, authenticShare: 50 }, payload));
assert.throws(() => context.check.normalizeApiResult({ ...response, reviews: [...response.reviews, ...response.reviews] }, payload));
assert.throws(() => context.check.normalizeApiResult({ ...response, reviews: [{ id: "unexpected", label: "authentic" }] }, payload));

let abortTimer;
let timerCleared = false;
const background = {
  URL, AbortController,
  setTimeout(callback) { abortTimer = callback; return 1; },
  clearTimeout() { timerCleared = true; },
  chrome: {
    storage: { sync: { get: async defaults => ({ ...defaults, useApi: true }) } },
    runtime: { onInstalled: { addListener() {} }, onMessage: { addListener() {} } }
  },
  fetch: async (_url, options) => ({
    ok: true,
    json: () => new Promise((_resolve, reject) => {
      assert.equal(timerCleared, false, "Timeout must remain active while reading the body");
      options.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
      abortTimer();
    })
  })
};
vm.createContext(background);
vm.runInContext(read("extension/background.js"), background);
await assert.rejects(background.analyzeWithConfiguredService(request), /timed out/);
assert.equal(timerCleared, true);
background.fetch = async () => ({ ok: true, json: async () => response });
assert.equal((await background.analyzeWithConfiguredService(request)).schemaVersion, "1.0");
background.fetch = async () => ({ ok: false, status: 503, json: async () => ({ detail: "Model unavailable" }) });
await assert.rejects(background.analyzeWithConfiguredService(request), /HTTP 503.*Model unavailable/);
background.fetch = async () => ({ ok: true, json: async () => { throw new SyntaxError("Invalid JSON"); } });
await assert.rejects(background.analyzeWithConfiguredService(request), /valid JSON/);
console.log("Extension runtime regression checks passed.");
