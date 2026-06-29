(function () {
  const shell = document.querySelector(".page-shell");
  const pageName = document.getElementById("page-name");
  const pageDescription = document.getElementById("page-description");
  const pageContent = document.getElementById("page-content");
  const navLinks = Array.from(document.querySelectorAll(".site-nav a[data-page]"));
  const parser = new DOMParser();
  const cache = new Map();
  const cacheTtlMs = 5000;
  const pageRefreshMs = 5000;
  const overviewLiveRefreshMs = 3000;
  const overviewPageKey = "overview";
  const fullRefreshPages = new Set(["events"]);
  let refreshTimerId = 0;
  let refreshInFlight = false;

  if (!shell || !pageName || !pageContent || !navLinks.length || !window.history.pushState) {
    return;
  }

  function now() {
    return Date.now();
  }

  function getCurrentKey() {
    return location.pathname + location.search;
  }

  function canHandlePath(pathname) {
    return pathname === "/" || pathname === "/events" || pathname === "/api";
  }

  function normalizeUrl(urlLike) {
    const url = new URL(urlLike, location.href);
    if (url.origin !== location.origin || !canHandlePath(url.pathname)) {
      return null;
    }
    return url.pathname + url.search;
  }

  function readState(doc) {
    const nextPageName = doc.getElementById("page-name");
    const nextPageDescription = doc.getElementById("page-description");
    const nextPageContent = doc.getElementById("page-content");
    if (!nextPageName || !nextPageContent) {
      return null;
    }

    return {
      title: doc.title,
      pageName: nextPageName.textContent || "",
      pageDescription: nextPageDescription ? nextPageDescription.textContent || "" : "",
      activePage: nextPageContent.dataset.pageKey || "",
      contentHtml: nextPageContent.innerHTML,
    };
  }

  function remember(urlKey, state) {
    if (!state) {
      return null;
    }
    cache.set(urlKey, {
      expiresAt: now() + cacheTtlMs,
      state,
    });
    return state;
  }

  function readFromCache(urlKey) {
    const entry = cache.get(urlKey);
    if (!entry) {
      return null;
    }
    if (entry.expiresAt < now()) {
      cache.delete(urlKey);
      return null;
    }
    return entry.state;
  }

  function updateActiveNav(activePage) {
    for (const link of navLinks) {
      const isActive = link.dataset.page === activePage;
      link.classList.toggle("active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }

  function render(state, { scroll = true } = {}) {
    document.title = state.title;
    pageName.textContent = state.pageName;
    if (pageDescription) {
      pageDescription.textContent = state.pageDescription;
    }
    pageContent.innerHTML = state.contentHtml;
    pageContent.dataset.pageKey = state.activePage;
    shell.dataset.pageKey = state.activePage;
    updateActiveNav(state.activePage);
    syncLiveRefresh();
    if (scroll) {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    }
  }

  async function fetchState(urlKey, { force = false } = {}) {
    const response = await fetch(urlKey, {
      headers: {
        Accept: "text/html",
      },
      cache: force ? "no-store" : "default",
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Navigation failed with ${response.status}`);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html")) {
      throw new Error("Expected an HTML response for internal navigation.");
    }

    const nextDocument = parser.parseFromString(await response.text(), "text/html");
    const nextState = readState(nextDocument);
    if (!nextState) {
      throw new Error("Fetched page is missing the expected content anchors.");
    }
    return remember(urlKey, nextState);
  }

  function getOverviewLivePanel() {
    return pageContent.querySelector("[data-live-stream-panel]");
  }

  function getOverviewLiveLimit() {
    const panel = getOverviewLivePanel();
    const parsed = Number.parseInt(panel ? panel.dataset.liveLimit || "" : "", 10);
    return Number.isFinite(parsed) ? parsed : 10;
  }

  function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString("en-US") : String(value == null ? 0 : value);
  }

  function updateOverviewSummary(summary) {
    if (!summary) {
      return;
    }
    for (const node of pageContent.querySelectorAll("[data-live-summary]")) {
      const key = node.dataset.liveSummary || "";
      if (Object.prototype.hasOwnProperty.call(summary, key)) {
        node.textContent = formatNumber(summary[key]);
      }
    }
  }

  function decisionClass(value) {
    const normalized = String(value || "unknown").toLowerCase();
    if (normalized === "accept" || normalized === "review" || normalized === "reject") {
      return normalized;
    }
    return "unknown";
  }

  function confidenceTone(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "low";
    }
    if (number >= 0.85) {
      return "high";
    }
    if (number >= 0.6) {
      return "mid";
    }
    return "low";
  }

  function getStreamId(item) {
    if (item && item.id) {
      return String(item.id);
    }
    return `${item.event_uuid || "event"}:${item.object_id || item.label || "object"}:${item.received_at || ""}`;
  }

  function formatConfidence(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : "-";
  }

  function formatRelativeTime(value) {
    if (!value) {
      return "-";
    }
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed)) {
      return String(value);
    }

    const diffSeconds = Math.floor((Date.now() - parsed) / 1000);
    if (diffSeconds < 5) {
      return "just now";
    }
    if (diffSeconds < 60) {
      return `${diffSeconds}s ago`;
    }
    if (diffSeconds < 3600) {
      return `${Math.floor(diffSeconds / 60)}m ago`;
    }
    if (diffSeconds < 86400) {
      return `${Math.floor(diffSeconds / 3600)}h ago`;
    }
    return new Intl.DateTimeFormat(undefined, {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(parsed));
  }

  function updateLiveTimes(root) {
    for (const node of root.querySelectorAll("[data-live-time]")) {
      node.textContent = formatRelativeTime(node.getAttribute("datetime") || node.dataset.liveTime || "");
    }
  }

  function createChip(className, text) {
    const node = document.createElement("span");
    node.className = className;
    node.textContent = text;
    return node;
  }

  function createLiveStreamRow(item, isNew) {
    const row = document.createElement("div");
    row.className = "live-row";
    if (isNew) {
      row.classList.add("is-new");
    }
    row.dataset.streamId = getStreamId(item);

    const main = document.createElement("div");
    main.className = "live-row-main";
    main.appendChild(createChip("chip device", item.device_id || "-"));

    const label = document.createElement("strong");
    label.textContent = item.label || "Unknown";
    main.appendChild(label);

    const right = document.createElement("div");
    right.className = "live-row-right";
    const decision = item.decision || "Unknown";
    right.appendChild(createChip(`badge ${decisionClass(decision)}`, decision));
    right.appendChild(createChip(`chip confidence ${confidenceTone(item.confidence)}`, formatConfidence(item.confidence)));

    const time = document.createElement("time");
    const rawTime = item.received_at || item.timestamp || "";
    time.dateTime = rawTime;
    time.title = rawTime;
    time.dataset.liveTime = rawTime;
    time.textContent = formatRelativeTime(rawTime);
    right.appendChild(time);

    row.appendChild(main);
    row.appendChild(right);
    return row;
  }

  function renderLiveStream(items) {
    const list = pageContent.querySelector("[data-live-stream-list]");
    if (!list) {
      return;
    }

    const normalizedItems = Array.isArray(items) ? items : [];
    const existingRows = Array.from(list.querySelectorAll("[data-stream-id]"));
    const existingIds = existingRows.map((row) => row.dataset.streamId || "");
    const nextIds = normalizedItems.map(getStreamId);
    if (existingIds.length === nextIds.length && existingIds.every((id, index) => id === nextIds[index])) {
      updateLiveTimes(list);
      return;
    }

    const previousIds = new Set(existingIds);
    list.replaceChildren();
    if (!normalizedItems.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.dataset.liveEmpty = "";
      empty.textContent = "NO RESULTS";
      list.appendChild(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const item of normalizedItems) {
      fragment.appendChild(createLiveStreamRow(item, previousIds.size > 0 && !previousIds.has(getStreamId(item))));
    }
    list.appendChild(fragment);
  }

  function updateLiveUpdated() {
    const node = pageContent.querySelector("[data-live-updated]");
    if (node) {
      node.textContent = "JUST NOW";
    }
  }

  function readOpenDeviceIds(container) {
    return new Set(
      Array.from(container.querySelectorAll("details[data-device-id][open]"))
        .map((node) => node.dataset.deviceId || "")
        .filter(Boolean),
    );
  }

  function restoreOpenDeviceIds(container, openIds) {
    if (!openIds.size) {
      return;
    }
    for (const node of container.querySelectorAll("details[data-device-id]")) {
      if (openIds.has(node.dataset.deviceId || "")) {
        node.open = true;
      }
    }
  }

  function replaceHtmlFragment(selector, html, { preserveOpenDevices = false } = {}) {
    if (typeof html !== "string") {
      return;
    }
    const container = pageContent.querySelector(selector);
    if (!container || container.innerHTML === html) {
      return;
    }

    const openIds = preserveOpenDevices ? readOpenDeviceIds(container) : new Set();
    container.innerHTML = html;
    if (preserveOpenDevices) {
      restoreOpenDeviceIds(container, openIds);
    }
  }

  function renderOverviewDeviceSections(payload) {
    replaceHtmlFragment("[data-live-device-accordion]", payload.devices_html, { preserveOpenDevices: true });
    replaceHtmlFragment("[data-live-recent-devices]", payload.recent_devices_html);
  }

  async function fetchOverviewLiveData() {
    const limit = getOverviewLiveLimit();
    const response = await fetch(`/api/overview/live?limit=${encodeURIComponent(limit)}`, {
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Live stream refresh failed with ${response.status}`);
    }
    return response.json();
  }

  async function navigate(urlLike, { replace = false, scroll = true } = {}) {
    const urlKey = normalizeUrl(urlLike);
    if (!urlKey || urlKey === getCurrentKey()) {
      return false;
    }

    shell.classList.add("is-loading");
    try {
      const nextState = readFromCache(urlKey) || await fetchState(urlKey);
      render(nextState, { scroll });
      const historyMethod = replace ? "replaceState" : "pushState";
      window.history[historyMethod]({ urlKey }, "", urlKey);
      remember(urlKey, nextState);
      return true;
    } catch (error) {
      window.location.assign(urlKey);
      return false;
    } finally {
      shell.classList.remove("is-loading");
    }
  }

  function prefetch(urlLike) {
    const urlKey = normalizeUrl(urlLike);
    if (!urlKey || readFromCache(urlKey)) {
      return;
    }
    fetchState(urlKey).catch(() => {});
  }

  function clearLiveRefresh() {
    if (refreshTimerId) {
      window.clearInterval(refreshTimerId);
      refreshTimerId = 0;
    }
  }

  function syncLiveRefresh() {
    clearLiveRefresh();
    const activePage = pageContent.dataset.pageKey || "";
    if (activePage === overviewPageKey && getOverviewLivePanel()) {
      updateLiveTimes(pageContent);
      refreshTimerId = window.setInterval(() => {
        if (document.hidden || refreshInFlight || shell.classList.contains("is-loading")) {
          return;
        }
        void refreshOverviewLiveData();
      }, overviewLiveRefreshMs);
      return;
    }
    if (!fullRefreshPages.has(activePage)) {
      return;
    }
    refreshTimerId = window.setInterval(() => {
      if (document.hidden || refreshInFlight || shell.classList.contains("is-loading")) {
        return;
      }
      void refreshCurrentPage();
    }, pageRefreshMs);
  }

  async function refreshCurrentPage() {
    const urlKey = getCurrentKey();
    const activePage = pageContent.dataset.pageKey || "";
    if (!fullRefreshPages.has(activePage)) {
      return;
    }

    refreshInFlight = true;
    try {
      const nextState = await fetchState(urlKey, { force: true });
      if (getCurrentKey() !== urlKey) {
        return;
      }
      render(nextState, { scroll: false });
    } catch (error) {
      // Ignore transient demo refresh errors and keep the current page visible.
    } finally {
      refreshInFlight = false;
    }
  }

  async function refreshOverviewLiveData() {
    if ((pageContent.dataset.pageKey || "") !== overviewPageKey || !getOverviewLivePanel()) {
      return;
    }

    refreshInFlight = true;
    try {
      const payload = await fetchOverviewLiveData();
      if ((pageContent.dataset.pageKey || "") !== overviewPageKey) {
        return;
      }
      updateOverviewSummary(payload.summary);
      renderLiveStream(payload.items);
      renderOverviewDeviceSections(payload);
      updateLiveUpdated();
    } catch (error) {
      // Keep the last rendered rows visible when a local demo refresh races with ingestion.
    } finally {
      refreshInFlight = false;
    }
  }

  function refreshActivePage() {
    const activePage = pageContent.dataset.pageKey || "";
    if (activePage === overviewPageKey) {
      return refreshOverviewLiveData();
    }
    return refreshCurrentPage();
  }

  remember(getCurrentKey(), readState(document));
  updateActiveNav(pageContent.dataset.pageKey || "");
  syncLiveRefresh();

  document.addEventListener("click", (event) => {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }

    const link = event.target instanceof Element ? event.target.closest("a[href]") : null;
    if (!link || !(link instanceof HTMLAnchorElement) || (link.target && link.target !== "_self")) {
      return;
    }
    if (link.hasAttribute("download")) {
      return;
    }

    const urlKey = normalizeUrl(link.href);
    if (!urlKey) {
      return;
    }

    event.preventDefault();
    void navigate(urlKey);
  });

  document.addEventListener(
    "pointerenter",
    (event) => {
      const link = event.target instanceof Element ? event.target.closest("a[href]") : null;
      if (!link || !(link instanceof HTMLAnchorElement)) {
        return;
      }
      prefetch(link.href);
    },
    true,
  );

  const idleCallback = window.requestIdleCallback || ((callback) => window.setTimeout(callback, 150));
  idleCallback(() => {
    for (const link of navLinks) {
      prefetch(link.href);
    }
  });

  window.addEventListener("popstate", () => {
    const urlKey = getCurrentKey();
    const cachedState = readFromCache(urlKey);
    if (cachedState) {
      render(cachedState, { scroll: false });
      return;
    }
    shell.classList.add("is-loading");
    fetchState(urlKey)
      .then((state) => {
        render(state, { scroll: false });
      })
      .catch(() => {
        window.location.reload();
      })
      .finally(() => {
        shell.classList.remove("is-loading");
      });
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      void refreshActivePage();
    }
  });
})();
