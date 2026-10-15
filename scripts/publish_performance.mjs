// Publish compact evaluation summaries; never include review text or predictions.
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!args[i].startsWith("--") || !args[i + 1]) throw new Error("Expected --option value pairs");
  options[args[i].slice(2)] = args[i + 1];
}
for (const name of ["own", "bundle", "dataset", "split", "evaluated-at"]) {
  if (!options[name]) throw new Error(`Missing --${name}`);
}
if (!Number.isFinite(Date.parse(options["evaluated-at"]))) throw new Error("Invalid evaluation date");
const labels = ["Authentic", "Deceptive", "Low-information", "Irrelevant"];
const sentiments = ["Negative", "Neutral", "Positive"];
function load(path) {
  const raw = readFileSync(path, "utf8");
  const report = JSON.parse(raw);
  if (report.partition !== "test") throw new Error(`${path}: only held-out test reports may be published`);
  return { report, sha256: createHash("sha256").update(raw).digest("hex") };
}
function metrics(value, names = []) {
  if (!value) return null;
  const result = {};
  for (const key of ["macro_f1", "macro_precision", "macro_recall", "accuracy"]) {
    const score = value[key];
    if (score != null && (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1)) throw new Error(`Invalid ${key}`);
    result[key] = score ?? null;
  }
  if (value.confusion_matrix) {
    const matrix = value.confusion_matrix;
    if (matrix.length !== names.length || matrix.some(row => row.length !== names.length || row.some(n => !Number.isInteger(n) || n < 0))) throw new Error("Invalid confusion matrix");
    result.labels = names;
    result.confusion_matrix = matrix;
    result.per_class = names.map((name, i) => {
      const tp = matrix[i][i], support = matrix[i].reduce((a, b) => a + b, 0);
      const predicted = matrix.reduce((sum, row) => sum + row[i], 0);
      const precision = predicted ? tp / predicted : 0, recall = support ? tp / support : 0;
      return { name, precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0, support };
    });
  }
  return result;
}
const own = load(options.own);
const provenance = { dataset: options.dataset, split: options.split, evaluatedAt: options["evaluated-at"] };
const models = [{
  name: "AuthentiCheck", version: options.bundle, ...provenance, reportSha256: own.sha256,
  reviewCount: own.report.review_count, productCount: own.report.product_count,
  tasks: [
    { name: "Review quality · RoBERTa + XGBoost", scope: "Four quality classes", metrics: metrics(own.report.quality_xgboost, labels) },
    { name: "Review quality · RoBERTa only", scope: "Same test reviews; text-only ablation", metrics: metrics(own.report.quality_text_only, labels) },
    { name: "Aspect detection", scope: "Ten aspects · gold Authentic reviews", metrics: metrics(own.report.category_detection) },
    { name: "Sentiment", scope: "Gold categories · gold Authentic reviews", metrics: metrics(own.report.gold_category_sentiment, sentiments) }
  ]
}];
if (options.baseline) {
  if (!options["baseline-version"]) throw new Error("Missing --baseline-version");
  const baseline = load(options.baseline);
  models.push({ name: "CLIP-CA-CG", version: options["baseline-version"], ...provenance, reportSha256: baseline.sha256,
    reviewCount: baseline.report.review_count, productCount: baseline.report.product_count,
    tasks: [
      { name: "Aspect detection", scope: "Six aspects · gold Authentic reviews; different scope from AuthentiCheck", metrics: metrics(baseline.report.metrics?.category_detection) },
      { name: "Sentiment", scope: "Six-aspect gold categories; different scope from AuthentiCheck", metrics: metrics(baseline.report.metrics?.gold_category_sentiment, sentiments) }
    ] });
}
const output = resolve(options.output || "extension/performance-data.js");
writeFileSync(output, `// Generated held-out evaluation summaries. No live review scores.\nglobalThis.AuthentiCheckPerformance = ${JSON.stringify({ schemaVersion: 1, models }, null, 2)};\n`);
console.log(`Published ${models.length} model report(s) to ${output}`);
