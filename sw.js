// 名刺管理 Service Worker
// アプリ本体とOCRエンジンをキャッシュしてオフライン動作させる。
// 個人データ（名刺情報・画像）は一切扱わない（IndexedDB にのみ存在）。
const VERSION = 'meishi-v1';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icons/icon-180.png', './icons/icon-192.png', './icons/icon-512.png'];
const ENGINE = [
  './vendor/tesseract/tesseract.min.js',
  './vendor/tesseract/worker.min.js',
  './vendor/tesseract/tesseract-core-simd-lstm.wasm.js',
  './vendor/tesseract/tesseract-core-lstm.wasm.js',
  './vendor/tesseract/lang/jpn.traineddata.gz',
  './vendor/tesseract/lang/eng.traineddata.gz',
];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    await c.addAll(SHELL);
    // OCRエンジンは大きいので個別に取得（失敗しても初回OCR時に再取得される）
    await Promise.all(ENGINE.map((u) => c.add(u).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return; // 外部へのリクエストには関与しない
  if (req.mode === 'navigate' || url.pathname.endsWith('/index.html')) {
    // アプリ本体: ネット優先（更新を反映）、オフライン時はキャッシュ
    e.respondWith((async () => {
      try {
        const res = await fetch(req, { cache: 'no-store' });
        if (res.ok) (await caches.open(VERSION)).put('./index.html', res.clone());
        return res;
      } catch {
        return (await caches.match('./index.html')) || Response.error();
      }
    })());
    return;
  }
  // その他（エンジン・アイコン）: キャッシュ優先
  e.respondWith((async () => {
    const hit = await caches.match(req, { ignoreSearch: true });
    if (hit) return hit;
    const res = await fetch(req);
    if (res.ok && url.pathname.includes('/vendor/')) (await caches.open(VERSION)).put(req, res.clone());
    return res;
  })());
});
