// Backgammon service worker.
// Strategy: cache-first for same-origin GET requests. The app shell is added on
// install; everything else (hashed asset bundles, weights) is cached on first
// fetch. After one online load, the app runs offline.

const CACHE_NAME = "backgammon-v1";
const APP_SHELL = ["/", "/index.html", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL).catch(() => undefined)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(req);
      if (cached) {
        // Fire-and-forget revalidation
        fetch(req)
          .then((resp) => {
            if (resp && resp.ok) {
              caches.open(CACHE_NAME).then((cache) => cache.put(req, resp.clone()));
            }
          })
          .catch(() => undefined);
        return cached;
      }
      try {
        const resp = await fetch(req);
        if (resp && resp.ok) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return resp;
      } catch {
        // Offline and uncached: fall back to shell so SPA works
        const fallback = await caches.match("/");
        if (fallback) return fallback;
        return new Response("offline", { status: 503, statusText: "offline" });
      }
    })(),
  );
});
