// PP-OCRv5（mobile）をブラウザ内で実行する Web Worker
// ・モデル／実行環境はすべてこのサイト内のファイル。画像はどこにも送信しない
// ・検出（DB）→ 行ごとに切り出し → 認識（CTC）
import * as ort from './ort.wasm.min.mjs';

const BASE = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = BASE;
ort.env.wasm.numThreads = 1; // GitHub Pages ではマルチスレッド（COOP/COEP）が使えない
ort.env.wasm.proxy = false;

const DET = { minSide: 736, maxSide: 1600, thresh: 0.3, boxThresh: 0.5, unclip: 1.6, minSize: 3 };
const REC = { h: 48, minW: 320, maxW: 3200, minScore: 0.45 };

let det = null, rec = null, chars = null, allowed = null, initP = null;

const post = (m) => self.postMessage(m);

async function fetchBytes(name, stage) {
  const res = await fetch(BASE + name);
  if (!res.ok) throw new Error(`${name} の読み込みに失敗（${res.status}）`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(); const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (total) post({ type: 'progress', stage, p: Math.min(1, got / total) });
  }
  const out = new Uint8Array(got); let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function init() {
  initP ||= (async () => {
    const dict = await (await fetch(BASE + 'dict.json')).json();
    chars = dict.chars;
    const detBytes = await fetchBytes('det.onnx', 'det');
    const recBytes = await fetchBytes('rec.onnx', 'rec');
    post({ type: 'progress', stage: 'init', p: 0 });
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    det = await ort.InferenceSession.create(detBytes, opt);
    rec = await ort.InferenceSession.create(recBytes, opt);
    // 出力クラス = [blank, 辞書..., 空白]。日本語モード：Shift_JIS(CP932)で表せない文字（簡体字など）を候補から外す
    allowed = new Uint8Array(chars.length + 2);
    allowed[0] = 1; allowed[chars.length + 1] = 1;
    for (let i = 0; i < chars.length; i++) allowed[i + 1] = dict.ja[i] === '1' ? 1 : 0;
  })();
  initP.catch(() => { initP = null; });
  return initP;
}

// ── 検出 ─────────────────────────────────────────
async function detect(img) {
  const { width: W, height: H, data } = img;
  const t = new Float32Array(3 * W * H), n = W * H;
  // PaddleOCRの前処理：BGR順、ImageNetの平均・標準偏差で正規化
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < n; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    t[i] = (b - mean[0]) / std[0]; t[n + i] = (g - mean[1]) / std[1]; t[2 * n + i] = (r - mean[2]) / std[2];
  }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, H, W]) });
  const prob = out[det.outputNames[0]].data;
  // 二値化＋膨張（2x2）
  const bin = new Uint8Array(n);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (prob[i] > DET.thresh || (x > 0 && prob[i - 1] > DET.thresh) || (y > 0 && prob[i - W] > DET.thresh) || (x > 0 && y > 0 && prob[i - W - 1] > DET.thresh)) bin[i] = 1;
  }
  // 連結成分 → 外接矩形
  const seen = new Uint8Array(n), stack = new Int32Array(n), boxes = [];
  for (let s = 0; s < n; s++) {
    if (!bin[s] || seen[s]) continue;
    let sp = 0, x0 = W, y0 = H, x1 = 0, y1 = 0, cnt = 0, sum = 0;
    stack[sp++] = s; seen[s] = 1;
    while (sp) {
      const i = stack[--sp], x = i % W, y = (i / W) | 0;
      cnt++; sum += prob[i];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && bin[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; stack[sp++] = i - 1; }
      if (x < W - 1 && bin[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; stack[sp++] = i + 1; }
      if (y > 0 && bin[i - W] && !seen[i - W]) { seen[i - W] = 1; stack[sp++] = i - W; }
      if (y < H - 1 && bin[i + W] && !seen[i + W]) { seen[i + W] = 1; stack[sp++] = i + W; }
    }
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (Math.min(w, h) < DET.minSize || cnt < 10) continue;
    const score = sum / cnt;
    if (score < DET.boxThresh) continue;
    const d = (w * h * DET.unclip) / (2 * (w + h)); // DBNetの縮小領域を元の文字領域へ戻す
    boxes.push({ x0: x0 - d, y0: y0 - d, x1: x1 + 1 + d, y1: y1 + 1 + d, score });
  }
  return boxes;
}

// ── 認識 ─────────────────────────────────────────
function sample(img, fx, fy, out) {
  const { width: W, height: H, data } = img;
  fx = Math.min(W - 1, Math.max(0, fx)); fy = Math.min(H - 1, Math.max(0, fy));
  const x0 = fx | 0, y0 = fy | 0, x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1), ax = fx - x0, ay = fy - y0;
  for (let c = 0; c < 3; c++) {
    const a = data[(y0 * W + x0) * 4 + c], b = data[(y0 * W + x1) * 4 + c], d = data[(y1 * W + x0) * 4 + c], e = data[(y1 * W + x1) * 4 + c];
    out[c] = (a * (1 - ax) + b * ax) * (1 - ay) + (d * (1 - ax) + e * ax) * ay;
  }
}
async function recognize(img, b) {
  const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
  const vertical = bh >= bw * 1.5; // 縦書きは90°回転して読む（PaddleOCRと同じ）
  const ratio = vertical ? bh / bw : bw / bh;
  const H = REC.h, tw = Math.max(1, Math.min(REC.maxW, Math.ceil(H * ratio))), W = Math.max(REC.minW, tw);
  const t = new Float32Array(3 * H * W), n = H * W, px = [0, 0, 0];
  for (let y = 0; y < H; y++) for (let x = 0; x < tw; x++) {
    let sx, sy;
    if (!vertical) { sx = b.x0 + ((x + 0.5) * bw) / tw - 0.5; sy = b.y0 + ((y + 0.5) * bh) / H - 0.5; }
    else { sy = b.y0 + ((x + 0.5) * bh) / tw - 0.5; sx = b.x1 - ((y + 0.5) * bw) / H - 0.5; }
    sample(img, sx, sy, px);
    const i = y * W + x; // BGR、(x/255-0.5)/0.5
    t[i] = px[2] / 127.5 - 1; t[n + i] = px[1] / 127.5 - 1; t[2 * n + i] = px[0] / 127.5 - 1;
  }
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, H, W]) });
  const o = out[rec.outputNames[0]], [, T, C] = o.dims, p = o.data;
  // CTCデコード（各文字の確信度と位置を記録）
  const seq = []; let prev = -1;
  for (let s = 0; s < T; s++) {
    let best = -Infinity, bi = 0; const base = s * C;
    for (let c = 0; c < C; c++) { const v = p[base + c]; if (v > best && allowed[c]) { best = v; bi = c; } }
    if (bi !== 0 && bi !== prev) seq.push({ ch: bi === C - 1 ? ' ' : chars[bi - 1] ?? '', p: best, t: s });
    prev = bi;
  }
  // 確信度の低い文字（空白部分のノイズなど）を除く
  const kept = seq.filter((c) => c.ch === ' ' || c.p >= 0.35);
  // 文字間の空きが極端に大きい所に空白を入れる（名刺の「姓　名」の全角スペースなど）
  const steps = []; for (let i = 1; i < kept.length; i++) if (kept[i].ch !== ' ' && kept[i - 1].ch !== ' ') steps.push(kept[i].t - kept[i - 1].t);
  const med = steps.length >= 2 ? [...steps].sort((x, y) => x - y)[Math.floor(steps.length / 2)] : 0;
  let text = '', sum = 0;
  kept.forEach((c, i) => {
    if (i > 0 && med && c.ch !== ' ' && kept[i - 1].ch !== ' ' && c.t - kept[i - 1].t >= Math.max(4, med * 2.2)) text += ' ';
    text += c.ch; sum += c.p;
  });
  return { text: text.replace(/\s+/g, ' ').trim(), score: kept.length ? sum / kept.length : 0, vertical };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') { await init(); post({ type: 'ready' }); return; }
    if (m.type === 'ocr') {
      await init();
      const t0 = performance.now();
      const boxes = await detect(m.det);
      const sx = m.full.width / m.det.width, sy = m.full.height / m.det.height;
      const lines = [];
      boxes.sort((a, b) => a.y0 - b.y0 || a.x0 - b.x0);
      for (let i = 0; i < boxes.length; i++) {
        const b0 = boxes[i];
        const b = { x0: Math.max(0, b0.x0 * sx), y0: Math.max(0, b0.y0 * sy), x1: Math.min(m.full.width, b0.x1 * sx), y1: Math.min(m.full.height, b0.y1 * sy) };
        let r = await recognize(m.full, b);
        // メール・URL・英数字の行は1文字の誤りが致命的なので、切り出しの余白を変えて読み直し、最も確信度の高い結果を採用
        const ascii = r.text.replace(/\s/g, '');
        if (!r.vertical && ascii.length >= 5 && (/@|www\.|https?:/i.test(ascii) || (ascii.match(/[\x21-\x7e]/g) || []).length / ascii.length > 0.85)) {
          const h = b.y1 - b.y0;
          for (const [px, py] of [[0.15, 0.08], [-0.05, -0.06], [0.3, 0.15]]) {
            const v = await recognize(m.full, { x0: Math.max(0, b.x0 - h * px), y0: Math.max(0, b.y0 - h * py), x1: Math.min(m.full.width, b.x1 + h * px), y1: Math.min(m.full.height, b.y1 + h * py) });
            if (v.text && v.score > r.score) r = v;
          }
        }
        post({ type: 'progress', id: m.id, stage: 'rec', p: (i + 1) / boxes.length });
        if (r.text && r.score >= REC.minScore) lines.push({ text: r.text, score: r.score, vertical: r.vertical, bbox: b });
      }
      post({ type: 'result', id: m.id, lines, ms: Math.round(performance.now() - t0) });
    }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
