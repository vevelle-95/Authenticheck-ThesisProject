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

async function initialize() {
  const settings = await chrome.storage.sync.get(DEFAULTS);
  Object.entries(fields).forEach(([key, field]) => {
    if (field.type === "checkbox") field.checked = settings[key];
    else field.value = settings[key];
  });
  syncEndpointVisibility();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const supported = /^https:\/\/[^/]*(shopee\.ph|lazada\.com\.ph)\//i.test(tab?.url || "");
  const state = document.querySelector("#supportState");
  let pageStatus = null;
  if (supported && tab?.id) {
    pageStatus = await chrome.tabs.sendMessage(tab.id, { type: "AUTHENTICHECK_STATUS" }).catch(() => null);
  }
  const activeProduct = Boolean(pageStatus?.productPage);
  state.className = `support ${activeProduct ? "ok" : "unsupported"}`;
  state.querySelector("span").textContent = activeProduct
    ? `Active on this ${pageStatus.platform} product page · ${pageStatus.reviewCount} visible reviews`
    : supported ? "Supported marketplace, but this is not a product page" : "Open a Shopee or Lazada product page";
  document.querySelector("#rescan").disabled = !activeProduct;
}

function syncEndpointVisibility() {
  document.querySelector("#endpointWrap").classList.toggle("visible", fields.useApi.checked);
}

fields.useApi.addEventListener("change", syncEndpointVisibility);

document.querySelector("#save").addEventListener("click", async () => {
  const next = Object.fromEntries(Object.entries(fields).map(([key, field]) => [
    key,
    field.type === "checkbox" ? field.checked : field.value.trim()
  ]));
  await chrome.storage.sync.set(next);
  const notice = document.querySelector("#notice");
  notice.textContent = "Settings saved";
  notice.classList.add("show");
  setTimeout(() => notice.classList.remove("show"), 1600);
});

document.querySelector("#rescan").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  await chrome.tabs.sendMessage(tab.id, { type: "AUTHENTICHECK_RESCAN" }).catch(() => {});
  const notice = document.querySelector("#notice");
  notice.textContent = "Page rescan requested";
  notice.classList.add("show");
  setTimeout(() => notice.classList.remove("show"), 1600);
});

initialize();
