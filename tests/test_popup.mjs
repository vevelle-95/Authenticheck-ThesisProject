import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const elements = new Map();
function element(selector) {
  if (!elements.has(selector)) {
    const classes = new Set();
    elements.set(selector, {
      type: ["#enabled", "#autoAnalyze", "#useApi"].includes(selector) ? "checkbox" : "url",
      value: "", checked: false, attributes: {}, listeners: {},
      classList: { add: name => classes.add(name), remove: name => classes.delete(name),
        toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } },
      addEventListener(name, callback) {
        const previous = this.listeners[name];
        this.listeners[name] = previous ? (...args) => { previous(...args); return callback(...args); } : callback;
      },
      setAttribute(name, value) { this.attributes[name] = value; },
      querySelector: name => element(`${selector} ${name}`),
      focus() { this.focused = true; }
    });
  }
  return elements.get(selector);
}
let writes = 0;
let failSave = false;
let failMessage = false;
let activeUrl = "https://shopee.ph/product/1";
let savedStore;
let rescanRequests = 0;
let pageState = { productPage: true, platform: "shopee", reviewCount: 3, analyzableCount: 2, analysisState: "ready", resultMode: "local" };
const context = {
  URL, setTimeout: () => 1, clearTimeout() {},
  document: { querySelector: element },
  chrome: {
    storage: { sync: {
      get: async defaults => ({ ...defaults, ...savedStore }),
      set: async values => { if (failSave) throw new Error("Storage unavailable"); writes++; savedStore = values; }
    } },
    tabs: {
      query: async () => [{ id: 1, url: activeUrl }],
      sendMessage: async (_id, message) => {
        if (failMessage) throw new Error("Disconnected");
        if (message.type === "AUTHENTICHECK_RESCAN") rescanRequests++;
        return message.type === "AUTHENTICHECK_STATUS" ? pageState : { requested: true };
      }
    }
  }
};
vm.createContext(context);
vm.runInContext(readFileSync("extension/popup.js", "utf8"), context);
await context.initialize();
const save = () => element("#save").listeners.click();
const rescan = () => element("#rescan").listeners.click();
assert.equal(element("#save").disabled, true);
assert.equal(element("#currentMode").textContent, "Local estimate (heuristics)");
element("#useApi").checked = true;
element("#useApi").listeners.change();
assert.match(element("#settingsState").textContent, /Unsaved changes/);
assert.equal(element("#rescan").disabled, true);
await rescan();
assert.equal(rescanRequests, 0, "Do not rescan using old saved settings while edits are pending");
assert.match(element("#notice").textContent, /Save settings before rescanning/);
for (const url of ["", "not a URL", "https://localhost/analyze", "http://example.com/analyze", "http://localhost.example.com/analyze"]) {
  element("#apiEndpoint").value = url;
  await save();
  assert.equal(writes, 0, `Must reject ${url}`);
  assert.equal(element("#apiEndpoint").attributes["aria-invalid"], "true");
  assert.equal(element("#endpointError").hidden, false);
}
for (const url of ["http://127.0.0.1:8000/analyze", "http://localhost:8000/analyze"]) {
  element("#apiEndpoint").value = url;
  await save();
  assert.equal(element("#notice").textContent, "Settings saved");
}
assert.equal(writes, 2);
assert.match(element("#savedMode").textContent, /Model API/);
assert.equal(element("#currentMode").textContent, "Local estimate (heuristics)", "Show the mode that produced the result, not the saved preference");
assert.equal(element("#save").disabled, true);
failSave = true;
element("#autoAnalyze").checked = false;
element("#autoAnalyze").listeners.change();
await save();
assert.match(element("#notice").textContent, /Could not save/);
assert.equal(element("#save").disabled, false);
assert.match(element("#settingsState").textContent, /Unsaved changes/);
failSave = false;
await save();
failMessage = true;
await rescan();
assert.match(element("#notice").textContent, /Could not connect/);
assert.match(element("#supportState span").textContent, /Reload/);
assert.equal(element("#rescan").disabled, true);
assert.equal(element("#useApi").checked, true, "Status refresh must preserve field values");
failMessage = false;
await context.initialize(false);
await rescan();
assert.match(element("#notice").textContent, /Scan requested/);
assert.equal(context.resultModeLabel({ productPage: true, analysisState: "error", resultMode: "api" }), "Scan failed · No current result");
assert.equal(context.resultModeLabel({ productPage: true, analysisState: "loading" }), "Analysis in progress");
assert.equal(context.resultModeLabel({ productPage: true, analysisState: "updating", resultMode: "local" }), "Updating · Previous result: Local estimate (heuristics)");
assert.equal(context.resultModeLabel({ productPage: true, analysisState: "ready", resultMode: "api", modelVersion: "development-test" }), "Development model");
element("#enabled").checked = false;
element("#enabled").listeners.change();
pageState = { ...pageState, analysisState: "disabled" };
await save();
assert.equal(element("#rescan").disabled, true);
assert.equal(element("#currentMode").textContent, "Extension hidden");
activeUrl = "https://evilshopee.ph/product/1";
await context.initialize(false);
assert.equal(element("#rescan").disabled, true);
console.log("Popup unsaved changes, actual result mode, validation, save failure, and rescan recovery checks passed.");
