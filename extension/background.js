const DEFAULTS = {
  enabled: true,
  autoAnalyze: true,
  useApi: false,
  apiEndpoint: "http://127.0.0.1:8000/analyze"
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
  if (!allowedLocal) {
    throw new Error("Version 0.1 supports local AuthentiCheck API endpoints only.");
  }

  const response = await fetch(endpoint.href, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) throw new Error(`Analyzer returned HTTP ${response.status}`);
  return response.json();
}
