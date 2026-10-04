import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

let onMessage;
let requestedUrl;
let requestedBody;
const chrome = {
  runtime: {
    onInstalled: { addListener() {} },
    onMessage: { addListener(handler) { onMessage = handler; } }
  },
  storage: {
    sync: {
      async get() {
        return { useApi: true, apiEndpoint: "http://127.0.0.1:8000/analyze" };
      },
      async set() {}
    }
  },
  tabs: { sendMessage() { return Promise.resolve(); } }
};
const fetch = async (url, options) => {
  requestedUrl = url;
  requestedBody = JSON.parse(options.body);
  return { ok: true, async json() { return { taxonomy: ["seller_service"], engines: {} }; } };
};
vm.runInNewContext(readFileSync("extension/background.js", "utf8"), {
  chrome, fetch, URL, AbortController, setTimeout, clearTimeout
});
const payload = {
  product_description: "Wireless headphones",
  reviews: [{ id: "r1", text: "Late delivery", star_rating: 2, image_urls: [] }]
};
const result = await new Promise(resolve => {
  const keepOpen = onMessage({ type: "AUTHENTICHECK_COMPARE", payload }, {}, resolve);
  assert.equal(keepOpen, true);
});
assert.equal(requestedUrl, "http://127.0.0.1:8000/api/compare/lu-et-al");
assert.deepEqual(requestedBody, payload);
assert.equal(result.ok, true);
console.log("Extension comparison message reaches the local API route.");
