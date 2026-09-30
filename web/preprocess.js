// Browser port of src/data/preprocessing.py (the parts the final model uses):
// adaptive black-border crop, resize to the network input size, ImageNet
// normalisation, plus the image-quality checks and the heatmap colouring.
// Images are {width, height, data} with RGBA bytes, like canvas ImageData.
"use strict";

const DRPre = (() => {
  // Same integer formula as OpenCV's (vectorised) RGB -> grey conversion:
  // 15-bit fixed-point coefficients for 0.299 R + 0.587 G + 0.114 B.
  // Verified bit-exact against cv2.cvtColor on random images.
  function grey(img) {
    const n = img.width * img.height, g = new Uint8Array(n), d = img.data;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      g[i] = (d[j] * 9798 + d[j + 1] * 19235 + d[j + 2] * 3735 + 16384) >> 15;
    }
    return g;
  }

  // numpy.percentile (linear interpolation) for 0-255 integer values.
  function percentile(g, q) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < g.length; i++) hist[g[i]]++;
    const pos = (q / 100) * (g.length - 1), lo = Math.floor(pos), frac = pos - lo;
    let cum = 0, vLo = -1, vHi = -1;
    for (let v = 0; v < 256; v++) {
      cum += hist[v];
      if (vLo < 0 && cum > lo) vLo = v;
      if (vHi < 0 && cum > lo + 1) { vHi = v; break; }
    }
    if (vHi < 0) vHi = vLo;
    return vLo + frac * (vHi - vLo);
  }

  // retina_threshold(): 20% of the bright level, at least minThr.
  function retinaThreshold(g, minThr) {
    return Math.max(minThr, 0.2 * percentile(g, 90));
  }

  // crop_black_border(): bounding box of rows / columns with >1% retina pixels.
  function cropBox(img, g, minThr) {
    const { width: w, height: h } = img, thr = retinaThreshold(g, minThr);
    const rows = new Uint32Array(h), cols = new Uint32Array(w);
    let total = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (g[y * w + x] > thr) { rows[y]++; cols[x]++; total++; }
    }
    const full = { x0: 0, y0: 0, x1: w - 1, y1: h - 1 };
    if (total < 0.05 * w * h) return full;
    let y0 = -1, y1 = -1, x0 = -1, x1 = -1;
    for (let y = 0; y < h; y++) if (rows[y] > 0.01 * w) { if (y0 < 0) y0 = y; y1 = y; }
    for (let x = 0; x < w; x++) if (cols[x] > 0.01 * h) { if (x0 < 0) x0 = x; x1 = x; }
    return y0 < 0 || x0 < 0 ? full : { x0, y0, x1, y1 };
  }

  function crop(img, b) {
    const w = b.x1 - b.x0 + 1, h = b.y1 - b.y0 + 1, out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      const src = ((b.y0 + y) * img.width + b.x0) * 4;
      out.set(img.data.subarray(src, src + w * 4), y * w * 4);
    }
    return { width: w, height: h, data: out };
  }

  // OpenCV INTER_AREA for shrinking: each output pixel is the area-weighted
  // mean of the source pixels it covers (separable, fractional overlaps).
  function areaWeights(src, dst) {
    const scale = src / dst, taps = [];
    for (let i = 0; i < dst; i++) {
      const f0 = i * scale, f1 = f0 + scale, t = [];
      for (let s = Math.floor(f0); s < Math.min(Math.ceil(f1), src); s++) {
        const wgt = Math.min(f1, s + 1) - Math.max(f0, s);
        if (wgt > 1e-9) t.push([s, wgt / scale]);
      }
      taps.push(t);
    }
    return taps;
  }

  // OpenCV INTER_CUBIC for enlarging: Keys cubic (a = -0.75), pixel-centre
  // alignment, replicated borders.
  // OpenCV computes the scale as 1 / (dst / src) in double precision and
  // casts the fractional offsets to float32; this is copied exactly because
  // it decides the source pixel at boundaries (e.g. row 140 of 200 -> 224).
  function cvScale(src, dst) { return 1 / (dst / src); }

  function cubicWeights(src, dst) {
    const scale = cvScale(src, dst), taps = [], A = -0.75;
    const k = (x) => {
      x = Math.abs(x);
      if (x <= 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
      if (x < 2) return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
      return 0;
    };
    for (let i = 0; i < dst; i++) {
      const fx = Math.fround((i + 0.5) * scale - 0.5), ix = Math.floor(fx), t = Math.fround(fx - ix), row = [];
      for (let m = -1; m <= 2; m++) {
        row.push([Math.min(Math.max(ix + m, 0), src - 1), k(m - t)]);
      }
      taps.push(row);
    }
    return taps;
  }

  // OpenCV INTER_AREA when one side shrinks but the other grows: OpenCV then
  // uses 2-tap linear weights with its own "area" offset formula on both axes.
  function areaLinearWeights(src, dst) {
    const scale = cvScale(src, dst), inv = dst / src, taps = [];
    for (let i = 0; i < dst; i++) {
      let sx = Math.floor(i * scale);
      let fx = Math.fround((i + 1) - (sx + 1) * inv);
      fx = fx <= 0 ? 0 : Math.fround(fx - Math.floor(fx));
      if (sx < 0) { sx = 0; fx = 0; }
      if (sx >= src - 1) { sx = src - 1; fx = 0; }
      taps.push(fx === 0 ? [[sx, 1]] : [[sx, 1 - fx], [sx + 1, fx]]);
    }
    return taps;
  }

  function resizeSeparable(img, size, weightsFn) {
    const { width: w, height: h, data } = img;
    const wx = weightsFn(w, size), wy = weightsFn(h, size);
    const tmp = new Float32Array(size * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0;
      for (const [s, wt] of wx[x]) { const j = (y * w + s) * 4; r += data[j] * wt; g += data[j + 1] * wt; b += data[j + 2] * wt; }
      const o = (y * size + x) * 3; tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
    const out = new Uint8ClampedArray(size * size * 4);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0;
      for (const [s, wt] of wy[y]) { const j = (s * size + x) * 3; r += tmp[j] * wt; g += tmp[j + 1] * wt; b += tmp[j + 2] * wt; }
      const o = (y * size + x) * 4;
      out[o] = Math.round(r); out[o + 1] = Math.round(g); out[o + 2] = Math.round(b); out[o + 3] = 255;
    }
    return { width: size, height: size, data: out };
  }

  // resize(): as in training, OpenCV INTER_AREA when the longer side shrinks
  // and INTER_CUBIC otherwise. INTER_AREA is a true area average only when
  // both sides shrink (or stay equal); otherwise OpenCV uses linear weights.
  function resize(img, size) {
    if (Math.max(img.width, img.height) <= size) return resizeSeparable(img, size, cubicWeights);
    const bothShrink = img.width >= size && img.height >= size;
    return resizeSeparable(img, size, bothShrink ? areaWeights : areaLinearWeights);
  }

  // Full model-input preprocessing for the final (no-enhancement) model.
  function preprocess(img, meta) {
    const p = meta.preprocessing;
    if ((p.denoise && p.denoise !== "none") || p.clahe || p.ben_graham) {
      throw new Error("This model needs image enhancement steps that the browser version does not implement.");
    }
    let out = img;
    if (p.crop_black_border !== false) out = crop(img, cropBox(img, grey(img), p.border_threshold ?? 4));
    return resize(out, meta.image_size);
  }

  // (x / 255 - mean) / std, as a CHW float tensor.
  function toTensor(img, mean, std) {
    const n = img.width * img.height, t = new Float32Array(3 * n), d = img.data;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      t[c * n + i] = (d[i * 4 + c] / 255 - mean[c]) / std[c];
    }
    return t;
  }

  // ---------- quality checks (same limits as api/inference.py) ----------
  const LIMITS = { minBrightness: 35, maxBrightness: 215, minSharpness: 25,
                   minRetinaFraction: 0.15, maxCornerBrightness: 40, lowConfidence: 0.5 };

  function erodeSquare(mask, w, h, r) {
    const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let ok = 1;
      for (let dx = -r; dx <= r && ok; dx++) { const xx = x + dx; if (xx < 0 || xx >= w || !mask[y * w + xx]) ok = 0; }
      tmp[y * w + x] = ok;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let ok = 1;
      for (let dy = -r; dy <= r && ok; dy++) { const yy = y + dy; if (yy < 0 || yy >= h || !tmp[yy * w + x]) ok = 0; }
      out[y * w + x] = ok;
    }
    return out;
  }

  // Checks on the uploaded photo (is it a fundus photo?) and on the model
  // input (dark / overexposed / blurred, with limits from the training data).
  function qualityWarnings(original, modelInput, minThr) {
    const g = grey(original), w = original.width, h = original.height;
    const thr = retinaThreshold(g, minThr);
    let bright = 0;
    for (let i = 0; i < g.length; i++) if (g[i] > thr) bright++;
    if (bright / g.length < LIMITS.minRetinaFraction) {
      return ["No retina found. Is this a colour fundus photograph?"];
    }
    const k = Math.max(2, Math.floor(Math.min(w, h) / 12)), corners = [];
    for (const [cx, cy] of [[0, 0], [w - k, 0], [0, h - k], [w - k, h - k]]) {
      let s = 0;
      for (let y = cy; y < cy + k; y++) for (let x = cx; x < cx + k; x++) s += g[y * w + x];
      corners.push(s / (k * k));
    }
    corners.sort((a, b) => a - b);
    if ((corners[1] + corners[2]) / 2 > LIMITS.maxCornerBrightness) {
      return ["This does not look like a fundus photograph (no dark border around the retina). The result is not meaningful."];
    }

    const mg = grey(modelInput), s = modelInput.width;
    const mthr = retinaThreshold(mg, minThr), raw = new Uint8Array(mg.length);
    for (let i = 0; i < mg.length; i++) raw[i] = mg[i] > mthr ? 1 : 0;
    const mask = erodeSquare(raw, s, s, Math.max(1, Math.floor(s * 0.08)));
    let n = 0, sum = 0, lsum = 0, lsq = 0;
    const at = (x, y) => mg[Math.min(Math.max(y, 0), s - 1) * s + Math.min(Math.max(x, 0), s - 1)];
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
      if (!mask[y * s + x]) continue;
      const lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * at(x, y);
      n++; sum += mg[y * s + x]; lsum += lap; lsq += lap * lap;
    }
    const warnings = [];
    if (n === 0) return warnings;
    const brightness = sum / n, sharpness = lsq / n - (lsum / n) ** 2;
    if (brightness < LIMITS.minBrightness) warnings.push("Image is very dark; lesions may be hidden.");
    else if (brightness > LIMITS.maxBrightness) warnings.push("Image is overexposed; retinal detail may be washed out.");
    if (sharpness < LIMITS.minSharpness) warnings.push("Image looks blurred or out of focus.");
    return warnings;
  }

  // ---------- Grad-CAM from feature maps and classifier weights ----------
  // For global-average-pool + linear heads this equals Grad-CAM exactly
  // (see src/deployment/export_onnx.py).
  function cam(features, dims, weightRow, size) {
    const [c, fh, fw] = dims, n = fh * fw, small = new Float32Array(n);
    for (let k = 0; k < c; k++) {
      const wk = weightRow[k], off = k * n;
      for (let i = 0; i < n; i++) small[i] += wk * features[off + i];
    }
    for (let i = 0; i < n; i++) small[i] = Math.max(small[i], 0);
    // Bilinear upsampling with pixel-centre alignment (OpenCV INTER_LINEAR).
    const out = new Float32Array(size * size), sy = fh / size, sx = fw / size;
    for (let y = 0; y < size; y++) {
      let fy = (y + 0.5) * sy - 0.5; fy = Math.min(Math.max(fy, 0), fh - 1);
      const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, fh - 1), ty = fy - y0;
      for (let x = 0; x < size; x++) {
        let fx = (x + 0.5) * sx - 0.5; fx = Math.min(Math.max(fx, 0), fw - 1);
        const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, fw - 1), tx = fx - x0;
        out[y * size + x] =
          (small[y0 * fw + x0] * (1 - tx) + small[y0 * fw + x1] * tx) * (1 - ty) +
          (small[y1 * fw + x0] * (1 - tx) + small[y1 * fw + x1] * tx) * ty;
      }
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of out) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let i = 0; i < out.length; i++) out[i] = hi > lo ? (out[i] - lo) / (hi - lo) : 0;
    return out;
  }

  // Jet colour map (as OpenCV COLORMAP_JET) blended onto the model input;
  // the black background stays black.
  function overlay(img, heat, alpha = 0.45) {
    const out = new Uint8ClampedArray(img.data.length);
    const ch = (v) => Math.min(Math.max(1.5 - Math.abs(v), 0), 1) * 255;
    for (let i = 0; i < heat.length; i++) {
      const j = i * 4, v = heat[i] * 4;
      const r = ch(v - 3), g = ch(v - 2), b = ch(v - 1);
      const d = img.data;
      if (d[j] === 0 && d[j + 1] === 0 && d[j + 2] === 0) { out[j + 3] = 255; continue; }
      out[j] = d[j] * (1 - alpha) + r * alpha;
      out[j + 1] = d[j + 1] * (1 - alpha) + g * alpha;
      out[j + 2] = d[j + 2] * (1 - alpha) + b * alpha;
      out[j + 3] = 255;
    }
    return { width: img.width, height: img.height, data: out };
  }

  return { grey, percentile, retinaThreshold, cropBox, crop, resize, preprocess, toTensor,
           qualityWarnings, cam, overlay, LIMITS };
})();
