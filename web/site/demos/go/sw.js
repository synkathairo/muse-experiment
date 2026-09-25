'use strict';

/* Offline support for the weiqi demo. Registered from index.html; precaches
 * the page, the WASM engine, the manifest and the current weight files so the
 * demo works offline after the first visit. No UI for this — it simply kicks
 * in when the network is gone. */

const CACHE_NAME = 'weiqi-go-v3';

const PRECACHE = [
  './',
  './index.html',
  './manifest.json',
  './pkg/weiqi_wasm.js',
  './pkg/weiqi_wasm_bg.wasm',
  './weights/selfplay-clean-100k.bin',
  './weights/selfplay-clean-300k.bin',
  './weights/selfplay-clean-1M.bin',
  './weights/selfplay-clean-3M.bin',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

function isFreshDoc(pathname) {
  // The page and the manifest change when checkpoints are added; always try
  // the network first for these so online users see updates immediately.
  return /(^|\/)(index\.html|manifest\.json)$/.test(pathname) || pathname.endsWith('/');
}

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // cross-origin: leave alone
  const scopePath = new URL(self.registration.scope).pathname;
  if (!url.pathname.startsWith(scopePath)) return; // outside this demo

  if (isFreshDoc(url.pathname)) {
    // Network-first, fall back to cache when offline.
    event.respondWith(
      fetch(req).then(res => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(req, copy));
        return res;
      }).catch(() => caches.match(req))
    );
  } else {
    // Cache-first, fall back to network (caching the result for next time,
    // so future checkpoints added to the manifest also work offline).
    event.respondWith(
      caches.match(req).then(hit => hit || fetch(req).then(res => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(req, copy));
        return res;
      }))
    );
  }
});
