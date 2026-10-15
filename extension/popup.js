const DEFAULTS = {
  enabled: true,
  autoAnalyze: true,
  useApi: false,
  apiEndpoint: "http://127.0.0.1:8000/analyze"
};

const fields = {
  enabled: document.querySelector("#enabled"),
  autoAnalyze: document.querySelector("#autoAnalyze"),
  useApi: document.querySelector("#useApi"),
  apiEndpoint: document.querySelector("#apiEndpoint")
};

let noticeTimer;
let savedSettings = null;
let saving = false;
let activeProduct = false;

function readFields() {
  return Object.fromEntries(Object.entries(fields).map(([key, field]) => [key,
    field.type === "checkbox" ? field.checked : field.value.trim()
  ]));
}

function hasUnsavedChanges() {
  const values = readFields();
  return savedSettings != null && Object.keys(fields).some(key => values[key] !== savedSettings[key]);
}

function updateSettingsState() {
  const dirty = hasUnsavedChanges();
  document.querySelector("#settingsState").textContent = saving ? "Saving settings…"
    : savedSettings == null ? "Settings unavailable. Reopen the popup to try again."
    : dirty ? "Unsaved changes · Save before rescanning." : "All settings saved";
  document.querySelector("#settingsState").classList.toggle("dirty", dirty);
  document.querySelector("#savedMode").textContent = savedSettings == null ? "Saved preference unavailable"
    : `Saved preference: ${!savedSettings.enabled ? "Extension hidden" : savedSettings.useApi ? "Model API" : "Local estimate (heuristics)"}`;
  document.querySelector("#save").disabled = saving || !savedSettings || !dirty;
  document.querySelector("#rescan").disabled = saving || !savedSettings || !activeProduct || dirty || !savedSettings.enabled;
  document.querySelector("#rescan").title = dirty ? "Save settings before rescanning" : !savedSettings?.enabled ? "Enable the extension and save to scan" : "Scan loaded reviews using saved settings";
}

function resultModeLabel(status) {
  if (!status?.productPage) return "No product analysis";
  const mode = status.resultMode === "api" ? String(status.modelVersion || "").startsWith("development-") ? "Development model" : "Model API"
    : status.resultMode === "local" ? "Local estimate (heuristics)" : status.resultMode === "coverage" ? "Coverage only" : "No completed scan";
  if (status.analysisState === "disabled") return "Extension hidden";
  if (status.analysisState === "error") return "Scan failed · No current result";
  if (status.analysisState === "empty") return "No loaded reviews to analyze";
  if (status.analysisState === "loading") return "Analysis in progress";
  if (status.analysisState === "updating") return `Updating · Previous result: ${mode}`;
  return status.analysisState === "ready" ? mode : "Refresh product page to update status";
}
function showNotice(message, error = false) {
  const notice = document.querySelector("#notice");
  clearTimeout(noticeTimer);
  notice.textContent = message;
  notice.classList.toggle("error", error);
  notice.classList.add("show");
  if (!error) noticeTimer = setTimeout(() => notice.classList.remove("show"), 3500);
}

function validateEndpoint() {
  let valid = !fields.useApi.checked;
  if (!valid) {
    try {
      const url = new URL(fields.apiEndpoint.value.trim());
      valid = url.protocol === "http:" && ["localhost", "127.0.0.1"].includes(url.hostname);
    } catch { /* Show the inline explanation below. */ }
  }
  fields.apiEndpoint.setAttribute("aria-invalid", String(!valid));
  const error = document.querySelector("#endpointError");
  error.hidden = valid;
  error.textContent = valid ? "" : "Enter a local HTTP address, such as http://127.0.0.1:8000/analyze.";
  return valid;
}

async function initialize(loadSettings = true) {
  if (loadSettings) {
    const settings = await chrome.storage.sync.get(DEFAULTS);
    Object.entries(fields).forEach(([key, field]) => {
      if (field.type === "checkbox") field.checked = settings[key];
      else field.value = settings[key];
    });
    savedSettings = readFields();
    syncEndpointVisibility();
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let supported = false;
  try {
    const url = new URL(tab?.url);
    supported = url.protocol === "https:" && /(^|\.)(shopee\.ph|lazada\.com\.ph)$/i.test(url.hostname);
  } catch { /* Internal browser pages are unsupported. */ }
  const state = document.querySelector("#supportState");
  let pageStatus = null;
  if (supported && tab?.id) {
    pageStatus = await chrome.tabs.sendMessage(tab.id, { type: "AUTHENTICHECK_STATUS" }).catch(() => null);
  }
  activeProduct = Boolean(pageStatus?.productPage);
  state.className = `support ${activeProduct ? "ok" : "unsupported"}`;
  const reviewStatus = pageStatus?.reviewCount
    ? `${pageStatus.reviewCount} loaded · ${pageStatus.analyzableCount} eligible`
    : pageStatus?.candidateCount
      ? `0 accepted · ${pageStatus.candidateCount} candidate elements`
      : "0 loaded reviews";
  state.querySelector("span").textContent = activeProduct
    ? `Active on this ${pageStatus.platform} product page · ${reviewStatus}`
    : supported ? pageStatus ? "Open a product page to analyze reviews" : "Reload this marketplace tab to connect AuthentiCheck" : "Open a Shopee or Lazada product page";
  document.querySelector("#currentMode").textContent = supported && !pageStatus ? "Page connection unavailable" : resultModeLabel(pageStatus);
  updateSettingsState();
}

function syncEndpointVisibility() {
  document.querySelector("#endpointWrap").classList.toggle("visible", fields.useApi.checked);
  validateEndpoint();
}

fields.useApi.addEventListener("change", syncEndpointVisibility);
fields.apiEndpoint.addEventListener("input", validateEndpoint);
Object.values(fields).forEach(field => field.addEventListener(field.type === "checkbox" ? "change" : "input", () => {
  document.querySelector("#notice").classList.remove("show");
  updateSettingsState();
}));

document.querySelector("#save").addEventListener("click", async () => {
  if (saving || savedSettings == null) return;
  if (!hasUnsavedChanges()) { updateSettingsState(); return; }
  if (!validateEndpoint()) {
    fields.apiEndpoint.focus();
    return;
  }
  const button = document.querySelector("#save");
  const next = readFields();
  let didSave = false;
  saving = true;
  Object.values(fields).forEach(field => { field.disabled = true; });
  updateSettingsState();
  button.textContent = "Saving…";
  try {
    await chrome.storage.sync.set(next);
    savedSettings = next;
    didSave = true;
    showNotice("Settings saved");
  } catch {
    showNotice("Could not save settings. Please try again.", true);
  } finally {
    saving = false;
    Object.values(fields).forEach(field => { field.disabled = false; });
    button.textContent = "Save settings";
    updateSettingsState();
  }
  if (didSave) await initialize(false).catch(() => showNotice("Settings saved, but page status is unavailable. Refresh the product page.", true));
});

document.querySelector("#rescan").addEventListener("click", async () => {
  if (hasUnsavedChanges()) {
    showNotice("Save settings before rescanning.", true);
    return;
  }
  if (saving || !savedSettings?.enabled || !activeProduct) return;
  const button = document.querySelector("#rescan");
  button.disabled = true;
  button.textContent = "Requesting…";
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error("No active tab");
    await chrome.tabs.sendMessage(tab.id, { type: "AUTHENTICHECK_RESCAN" });
    showNotice("Scan requested. Open the panel to see results.");
  } catch {
    showNotice("Could not connect. Reload the product page and try again.", true);
  } finally {
    button.textContent = "Rescan page";
    await initialize(false).catch(() => showNotice("Could not refresh page status. Reopen this popup.", true));
  }
});

initialize().catch(() => {
  document.querySelector("#currentMode").textContent = "Page status unavailable";
  updateSettingsState();
  showNotice("Could not load settings or page status. Reopen this popup to try again.", true);
});
