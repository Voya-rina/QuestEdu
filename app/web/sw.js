/* QuestEdu 学问 · service worker
 *
 * 策略分两类，故意的：
 *   - 页面（navigate）：网络优先。改完前端刷新就是新的，不会被旧缓存咬住
 *   - 静态资源（/static）：缓存优先。KaTeX 字体 26 个文件，断网也要能排版公式
 *   - 接口（/api）：完全不碰。数学答案不能缓存，会话状态更不行
 */
const CACHE = 'questedu-v2';

const PRECACHE = [
  '/',
  '/manifest.json',
  '/static/vendor/katex/katex.min.css',
  '/static/vendor/katex/katex.min.js',
  '/static/vendor/katex/contrib/auto-render.min.js',
  '/static/vendor/chartjs/chart.umd.js',
  '/static/icons/icon-192.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => Promise.allSettled(PRECACHE.map(u => c.add(u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/api/')) return;         // 接口永不缓存

  if (req.mode === 'navigate') {
    e.respondWith(fetch(req).catch(() => caches.match('/')));
    return;
  }

  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.match(req).then(hit => hit || fetch(req).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {});
        return res;
      }))
    );
  }
});
