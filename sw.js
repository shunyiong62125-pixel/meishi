// 名刺管理 Service Worker
// アプリ本体とOCRエンジンをキャッシュしてオフライン動作させる。
// 個人データ（名刺情報・画像）は一切扱わない（IndexedDB にのみ存在）。
const VERSION = 'meishi-v2';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icons/icon-180.png', './icons/icon-192.png', './icons/icon-512.png'];
// OCRエンジン（vendor/ 以下）は大きいので、初めて使ったときに保存する（下の fetch 処理）

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    await c.addAll(SHELL);
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    // 古い版のキャッシュから、OCRエンジンなど今も使うファイルを引き継いでから削除（再ダウンロードを避ける）
    const cur = await caches.open(VERSION);
    for (const k of await caches.keys()) {
      if (k === VERSION) continue;
      const old = await caches.open(k);
      for (const req of await old.keys()) if (new URL(req.url).pathname.includes('/vendor/') && !(await cur.match(req))) await cur.put(req, await old.match(req));
      await caches.delete(k);
    }
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
