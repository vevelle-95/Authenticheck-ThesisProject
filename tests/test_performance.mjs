import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import vm from "node:vm";

const dir = mkdtempSync(join(tmpdir(), "authenticheck-performance-"));
try {
  const source = join(dir, "report.json"), output = join(dir, "performance.js");
  // Synthetic metrics for regression testing only, never published to the extension.
  const quality = { macro_f1: 0.5, macro_precision: 0.5, macro_recall: 0.5, accuracy: 0.5,
    confusion_matrix: [[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] };
  const report = { partition: "test", review_count: 4, product_count: 2, quality_xgboost: quality,
    filtered_reviews: [{ text: "PRIVATE REVIEW MUST NOT BE PUBLISHED" }] };
  const publish = () => spawnSync(process.execPath, ["scripts/publish_performance.mjs", "--own", source,
    "--bundle", "test-only", "--dataset", "test-data", "--split", "test-split",
    "--evaluated-at", "2026-10-15", "--output", output], { encoding: "utf8" });
  writeFileSync(source, JSON.stringify(report));
  const run = publish();
  assert.equal(run.status, 0, run.stderr);
  const published = readFileSync(output, "utf8");
  assert.ok(!published.includes("PRIVATE REVIEW"));
  const context = { window: {}, location: {},
    chrome: { runtime: { onMessage: { addListener() {} } }, storage: { onChanged: { addListener() {} } } },
    document: { addEventListener() {}, createElement() { return { set textContent(v) { this.innerHTML = String(v).replaceAll("<", "&lt;").replaceAll(">", "&gt;"); } }; } } };
  vm.createContext(context);
  vm.runInContext(published, context);
  const metrics = context.AuthentiCheckPerformance.models[0].tasks[0].metrics;
  assert.equal(metrics.per_class[0].support, 3);
  assert.equal(metrics.per_class[0].precision, 1);
  assert.equal(metrics.per_class[0].recall, 2 / 3);
  assert.equal(metrics.per_class[2].f1, 0);
  const summary = {}, body = {};
  context.testRoot = { querySelector(selector) { return selector === ".ac-performance-score" ? summary : body; } };
  const content = readFileSync("extension/content.js", "utf8").replace("  start();", "  root = globalThis.testRoot; globalThis.check = { renderPerformance, metricScore };");
  vm.runInContext(content, context);
  context.check.renderPerformance();
  assert.match(summary.textContent, /50.00%/);
  assert.match(body.innerHTML, /Confusion matrix/);
  assert.match(body.innerHTML, /different scopes/);
  assert.match(body.innerHTML, /CLIP-CA-CG/);
  assert.match(body.innerHTML, /Not evaluated/);
  assert.equal(context.check.metricScore(null), "Not evaluated");
  assert.equal(context.check.metricScore(0), "0.00%");
  context.AuthentiCheckPerformance = { models: [] };
  context.check.renderPerformance();
  assert.match(summary.textContent, /Not evaluated/);
  assert.ok(!body.innerHTML.includes("50.00%"));
  writeFileSync(source, JSON.stringify({ ...report, partition: "validation" }));
  assert.notEqual(publish().status, 0);
  writeFileSync(source, JSON.stringify({ ...report, quality_xgboost: { macro_f1: 55 } }));
  assert.notEqual(publish().status, 0);
  console.log("Performance publishing and rendering checks passed.");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
