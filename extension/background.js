const DEFAULTS = {
  enabled: true,
  autoAnalyze: true,
  useApi: false,
  apiEndpoint: "http://127.0.0.1:8000/analyze",
  apiTimeoutMs: 20000
};

chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.sync.get(DEFAULTS);
  await chrome.storage.sync.set(current);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "AUTHENTICHECK_ANALYZE") {
    analyzeWithConfiguredService(message.payload)
      .then(result => sendResponse({ ok: true, result }))
      .catch(error => sendResponse({ ok: false, error: error.message }));
    return true;
  }

  if (message?.type === "AUTHENTICHECK_RESCAN") {
    chrome.tabs.sendMessage(message.tabId, { type: "AUTHENTICHECK_RESCAN" }).catch(() => {});
    sendResponse({ ok: true });
  }
});

async function analyzeWithConfiguredService(payload) {
  const settings = await chrome.storage.sync.get(DEFAULTS);
  if (!settings.useApi) return null;

  const endpoint = new URL(settings.apiEndpoint);
  const allowedLocal = endpoint.hostname === "127.0.0.1" || endpoint.hostname === "localhost";
  if (!allowedLocal || endpoint.protocol !== "http:") {
    throw new Error("Version 0.2 supports local HTTP AuthentiCheck API endpoints only.");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Math.max(3000, settings.apiTimeoutMs || 20000));
  let response;
  try {
    response = await fetch(endpoint.href, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("The model API timed out. Check that the local inference service is running.");
    throw new Error("The model API could not be reached. Check the endpoint and local inference service.");
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) throw new Error(`Model API returned HTTP ${response.status}.`);
  const result = await response.json().catch(() => { throw new Error("The model API did not return valid JSON."); });
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("The model API returned an invalid result object.");
  }
  const supportedFields = ["authenticShare", "confidence", "verifiedRating", "counts", "sentimentCounts", "aspects", "reviews"];
  if (!supportedFields.some(field => Object.hasOwn(result, field))) {
    throw new Error("The model API response does not contain any supported analysis fields.");
  }
  return result;
}
