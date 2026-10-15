import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync("extension/content.js", "utf8").replace("  start();", "  root = globalThis.testRoot; globalThis.check = { startLoadingTips, stopLoadingTips };");
const storage = new Map();
function page(blockStorage = false) {
  const tip = {};
  let callback, interval, cleared = 0;
  const context = {
    window: {}, location: {}, testRoot: { querySelector: () => tip },
    document: { addEventListener() {} },
    chrome: { runtime: { onMessage: { addListener() {} } }, storage: { onChanged: { addListener() {} } } },
    sessionStorage: {
      getItem(key) { if (blockStorage) throw new Error("Blocked"); return storage.get(key) ?? null; },
      setItem(key, value) { if (blockStorage) throw new Error("Blocked"); storage.set(key, value); }
    },
    setInterval(fn, ms) { callback = fn; interval = ms; return 1; },
    clearInterval() { cleared++; }
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return { start: () => context.check.startLoadingTips(), stop: () => context.check.stopLoadingTips(), tick: () => callback(), tip, interval: () => interval, cleared: () => cleared };
}
const first = page();
first.start();
const initial = first.tip.textContent;
assert.equal(first.interval(), 7500);
const reload = page();
reload.start();
assert.notEqual(reload.tip.textContent, initial, "Reload advances from the last displayed tip");
const second = reload.tip.textContent;
reload.tick();
assert.notEqual(reload.tip.textContent, second);
const afterRotation = page();
afterRotation.start();
assert.notEqual(afterRotation.tip.textContent, reload.tip.textContent, "Reload follows the most recently rotated tip");
for (let i = 0; i < 10; i++) {
  const previous = afterRotation.tip.textContent;
  afterRotation.tick();
  assert.notEqual(afterRotation.tip.textContent, previous);
}
afterRotation.stop();
assert.ok(afterRotation.cleared() >= 2);
const blocked = page(true);
assert.doesNotThrow(() => blocked.start());
assert.doesNotThrow(() => blocked.tick());
console.log("Review tips advance across reloads and rotate every 7.5 seconds.");
