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

const request = JSON.parse(read("data/samples/extension_api_payload.json"));
const response = JSON.parse(read("data/samples/extension_api_response.json"));
const payload = { ...request, reviews: request.reviews.map(review => ({ ...review, analysisEligible: true })) };
assert.equal(context.check.normalizeApiResult(response, payload).mode, "api");
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
