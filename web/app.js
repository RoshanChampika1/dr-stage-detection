// Diabetic retinopathy grading in the browser with ONNX Runtime Web.
// The photograph is processed on this device only; nothing is uploaded.
"use strict";

const MODEL_URL = "model/dr_model.onnx";
const META_URL = "model/model_meta.json";
const COLORS = ["var(--s0)", "var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
const ADVICE = [
  "No signs of diabetic retinopathy detected. Continue routine annual screening.",
  "Mild non-proliferative DR (microaneurysms only). Re-screen in 6-12 months.",
  "Moderate non-proliferative DR. Refer to an ophthalmologist.",
  "Severe non-proliferative DR. Urgent referral to an ophthalmologist.",
  "Proliferative DR. Urgent referral: sight-threatening, needs treatment.",
];
const $ = (id) => document.getElementById(id);

// The advice follows the referral decision. "No DR" can be the single most
// likely stage while the combined probability of stages 1-4 is still above
// the referral threshold; the screening rule then wins.
function adviceFor(stage, refer, pDR) {
  if (stage === 0 && refer) {
    return `No DR is the single most likely stage, but the combined probability of retinopathy ` +
      `(${Math.round(pDR * 100)}%) is above the referral threshold, so this image should be ` +
      `checked by a human grader.`;
  }
  return ADVICE[stage];
}

let session = null, meta = null, photo = null;
let images = { photo: null, input: null, cam: null };

// ---------- model loading ----------
function setStatus(state, text) {
  $("server").dataset.state = state;
  $("server-text").textContent = text;
}

async function loadModel() {
  setStatus("waking", "Loading the model (16 MB, only on the first visit)");
  try {
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const [m, s] = await Promise.all([
      fetch(META_URL).then((r) => { if (!r.ok) throw new Error("model information missing"); return r.json(); }),
      ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] }),
    ]);
    meta = m; session = s;
    setStatus("ready", "Model ready. Photos stay on this device.");
    showModelInfo(meta);
    updateGradeButton();
  } catch (err) {
    console.error(err);
    setStatus("down", "The model could not be loaded. Reload the page to try again.");
  }
}

function showModelInfo(m) {
  const t = m.test_metrics || {};
  if (t.accuracy === undefined) return;
  const pct = (v) => `${(100 * v).toFixed(1)}%`;
  $("metrics").textContent =
    `On 5,270 held-out test images from patients not seen in training, this model reached ` +
    `${pct(t.accuracy)} accuracy, a quadratic weighted kappa of ${t.qwk.toFixed(2)}, ` +
    `and at the referral threshold ${pct(t.screening_sensitivity)} sensitivity with ` +
    `${pct(t.screening_specificity)} specificity for any retinopathy. It detects mild ` +
    `retinopathy poorly, because its lesions are tiny at this image size.`;
}

// ---------- analysis (also usable from the console: await drGrade(imageData)) ----------
async function drGrade(img) {
  const t0 = performance.now();
  const input = DRPre.preprocess(img, meta);
  const size = meta.image_size;
  const tensor = new ort.Tensor("float32", DRPre.toTensor(input, meta.mean, meta.std), [1, 3, size, size]);
  const out = await session.run({ image: tensor });
  const probs = Array.from(out.probabilities.data);
  const stage = probs.indexOf(Math.max(...probs));
  const f = out.features;
  const heat = DRPre.cam(f.data, f.dims.slice(1), meta.classifier_weight[stage], size);
  const ms = performance.now() - t0;

  const { gradable, warnings } = DRPre.qualityWarnings(img, input, meta.preprocessing.border_threshold ?? 4);
  if (gradable && probs[stage] < DRPre.LIMITS.lowConfidence) warnings.push("Low confidence: the model is unsure between stages.");
  const pDR = 1 - probs[0];
  const refer = pDR >= meta.screening_threshold;
  return {
    stage, label: meta.class_names[stage], confidence: probs[stage], probabilities: probs,
    dr_probability: pDR, screening_threshold: meta.screening_threshold,
    refer, advice: adviceFor(stage, refer, pDR), warnings, gradable,
    input, overlay: DRPre.overlay(input, heat), ms,
  };
}
window.drGrade = drGrade;

// ---------- choosing a photo ----------
function toDataURL(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  c.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return c.toDataURL("image/png");
}

async function pickFile(f) {
  if (!f) return;
  if (!f.type.startsWith("image/")) { showError("Choose an image file (JPG or PNG)."); return; }
  if (f.size > 20 * 1024 * 1024) { showError("Choose an image smaller than 20 MB."); return; }
  try {
    const bmp = await createImageBitmap(f);
    const c = document.createElement("canvas");
    c.width = bmp.width; c.height = bmp.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(bmp, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height);
    photo = { width: d.width, height: d.height, data: d.data };
    images = { photo: c.toDataURL("image/jpeg", 0.92), input: null, cam: null };
  } catch (_) {
    showError("This file could not be read as an image. Use JPG or PNG.");
    return;
  }
  $("views").hidden = true;
  showView("photo");
  $("empty").hidden = true;
  resetResult();
  updateGradeButton();
}

function updateGradeButton() { $("grade").disabled = !(photo && session); }

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); }
});
$("choose").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

// ---------- image views ----------
const VIEW_NOTES = {
  photo: "",
  input: "What the network sees: cropped to the retina and resized to 224 × 224 pixels.",
  cam: "Red areas raised the score of the predicted stage the most (Grad-CAM).",
};
const VIEW_ALT = { photo: "Chosen fundus photograph", input: "Model input image", cam: "Grad-CAM heatmap over the retina" };
function showView(name) {
  const img = $("view");
  img.src = images[name];
  img.alt = VIEW_ALT[name];
  img.hidden = false;
  document.querySelectorAll(".views button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.view === name)));
  $("view-note").textContent = VIEW_NOTES[name];
  $("view-note").hidden = !VIEW_NOTES[name];
}
document.querySelectorAll(".views button").forEach((b) =>
  b.addEventListener("click", () => showView(b.dataset.view)));

// ---------- grading ----------
function resetResult() {
  $("out").hidden = true; $("error").hidden = true; $("idle").hidden = false;
}
function showError(msg) {
  $("busy").hidden = true; $("idle").hidden = true;
  $("error").textContent = msg; $("error").hidden = false;
}

$("grade").addEventListener("click", async () => {
  if (!photo || !session) return;
  $("idle").hidden = true; $("out").hidden = true; $("error").hidden = true; $("busy").hidden = false;
  $("grade").disabled = true;
  await new Promise((r) => setTimeout(r, 30)); // let the spinner appear
  try {
    const result = await drGrade(photo);
    if (!result.gradable) {
      showError(result.warnings.join(" "));
      return;
    }
    render(result);
  } catch (err) {
    console.error(err);
    showError(err.message || "The photo could not be analysed.");
  } finally {
    updateGradeButton();
  }
});

function render(d) {
  images.input = toDataURL(d.input);
  images.cam = toDataURL(d.overlay);
  $("views").hidden = false;
  showView("cam");

  $("stage-number").textContent = `Stage ${d.stage} of 4`;
  $("stage-name").textContent = d.label;
  $("stage-name").style.color = COLORS[d.stage];
  $("confidence").textContent = `${Math.round(d.confidence * 100)}% probability for this stage`;

  const bars = $("bars");
  bars.innerHTML = "";
  document.querySelectorAll(".bar-labels").forEach((el) => el.remove());
  const labels = document.createElement("div");
  labels.className = "bar-labels";
  d.probabilities.forEach((p, i) => {
    const chosen = i === d.stage;
    const bar = document.createElement("div");
    bar.className = `bar${chosen ? " chosen" : ""}`;
    bar.style.setProperty("--c", COLORS[i]);
    bar.style.setProperty("--h", "0%");
    bar.innerHTML = `<span class="bar-value">${Math.round(p * 100)}%</span><div class="bar-fill"></div>`;
    bars.appendChild(bar);
    const lab = document.createElement("div");
    lab.className = `bar-label${chosen ? " chosen" : ""}`;
    lab.textContent = `${i} ${meta.class_names[i]}`;
    labels.appendChild(lab);
    requestAnimationFrame(() => requestAnimationFrame(() =>
      bar.style.setProperty("--h", `${Math.max(p * 100, 1)}%`)));
  });
  bars.after(labels);

  const dec = $("decision");
  dec.textContent = d.refer ? "Refer: signs of diabetic retinopathy likely" : "No referable retinopathy detected";
  dec.className = `decision ${d.refer ? "refer" : "clear"}`;
  $("meter-tick").style.left = `calc(${d.screening_threshold * 100}% - 1px)`;
  $("meter-fill").style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() =>
    ($("meter-fill").style.width = `${d.dr_probability * 100}%`)));
  $("meter-note").textContent =
    `Probability of any retinopathy ${Math.round(d.dr_probability * 100)}%. ` +
    `The red mark at ${Math.round(d.screening_threshold * 100)}% is the referral threshold.`;

  $("advice").textContent = d.advice;
  const w = $("warnings");
  w.innerHTML = "";
  d.warnings.forEach((t) => { const li = document.createElement("li"); li.textContent = t; w.appendChild(li); });
  w.hidden = d.warnings.length === 0;
  $("timing").textContent = `Analysed in ${Math.round(d.ms)} ms on this device.`;

  $("busy").hidden = true;
  $("out").hidden = false;
}

loadModel();
