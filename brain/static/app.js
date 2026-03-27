(function () {
  const shell = document.querySelector(".page-shell");
  const pageName = document.getElementById("page-name");
  const pageDescription = document.getElementById("page-description");
  const pageContent = document.getElementById("page-content");
  const navLinks = Array.from(document.querySelectorAll(".site-nav a[data-page]"));
  const parser = new DOMParser();
  const cache = new Map();
  const cacheTtlMs = 5000;

  if (!shell || !pageName || !pageDescription || !pageContent || !navLinks.length || !window.history.pushState) {
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
    if (!nextPageName || !nextPageDescription || !nextPageContent) {
      return null;
    }

    return {
      title: doc.title,
      pageName: nextPageName.textContent || "",
      pageDescription: nextPageDescription.textContent || "",
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
    pageDescription.textContent = state.pageDescription;
    pageContent.innerHTML = state.contentHtml;
    pageContent.dataset.pageKey = state.activePage;
    shell.dataset.pageKey = state.activePage;
    updateActiveNav(state.activePage);
    if (scroll) {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    }
  }

  async function fetchState(urlKey) {
    const response = await fetch(urlKey, {
      headers: {
        Accept: "text/html",
      },
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

  remember(getCurrentKey(), readState(document));
  updateActiveNav(pageContent.dataset.pageKey || "");

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
})();
