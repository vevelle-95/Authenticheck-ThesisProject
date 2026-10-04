import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const readJson = path => JSON.parse(readFileSync(path, "utf8"));
const manifest = readJson("extension/manifest.json");
const request = readJson("data/samples/extension_api_payload.json");
const response = readJson("data/samples/extension_api_response.json");
const scripts = manifest.content_scripts[0].js;

assert.equal(manifest.manifest_version, 3);
assert.ok(scripts.includes("extractors/buyer-reviews.js"));
assert.ok(scripts.indexOf("extractors/buyer-reviews.js") < scripts.indexOf("extractors/shopee.js"));
assert.equal(request.schemaVersion, "1.0");
assert.ok(["shopee", "lazada"].includes(request.platform));
assert.ok(request.reviews.length >= 1 && request.reviews.length <= 20);
for (const review of request.reviews) {
  assert.equal(typeof review.id, "string");
  assert.equal(typeof review.text, "string");
  assert.ok(review.imageUrls.length <= 5);
  assert.ok(review.rating == null || (review.rating >= 1 && review.rating <= 5));
}

assert.equal(response.schemaVersion, "1.0");
assert.equal(typeof response.modelVersion, "string");
assert.deepEqual(Object.keys(response.counts).sort(), ["authentic", "deceptive", "irrelevant", "liv"]);
assert.equal(response.reviews.length, request.reviews.length);
for (const review of response.reviews) {
  assert.ok(["authentic", "deceptive", "liv", "irrelevant"].includes(review.label));
  assert.ok(request.reviews.some(source => source.id === review.id));
}

console.log("Extension/API sample contract is valid.");
