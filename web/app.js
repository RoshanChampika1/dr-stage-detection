// Front end for the DR grading API: upload, call /predict, show the result.

// When the page is served by the API itself (/app/ on the Space or locally),
// use the same origin; otherwise use the address from config.js.
const API = location.pathname.startsWith("/app")
  ? location.origin
  : (window.DR_API_URL || "").replace(/\/$/, "");

const COLORS = ["var(--s0)", "var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
const $ = (id) => document.getElementById(id);

let file = null;
let images = { photo: null, input: null, cam: null };
let serverReady = false;

// ---------- server status (the free Space sleeps when idle) ----------
function setServer(state, text) {
  $("server").dataset.state = state;
  $("server-text").textContent = text;
}

async function wakeServer() {
  setServer("waking", "Connecting to the model server");
  const started = Date.now();
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const res = await fetch(`${API}/health`, { cache: "no-store" });
      if (res.ok) {
        const info = await res.json();
        serverReady = true;
        setServer("ready", "Model ready");
        showModelInfo(info.model);
        updateGradeButton();
        return;
      }
    } catch (_) { /* server still starting */ }
    if (Date.now() - started > 8000) {
      setServer("waking", "Starting the model server (up to a minute after a quiet period)");
    }
    await new Promise((r) => setTimeout(r, 4000));
  }
  setServer("down", "Model server unavailable. Reload the page to try again.");
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

// ---------- choosing a photo ----------
function pickFile(f) {
  if (!f) return;
  if (!f.type.startsWith("image/")) {
    showError("Choose an image file (JPG or PNG).");
    return;
  }
  file = f;
  const reader = new FileReader();
  reader.onload = () => {
    images = { photo: reader.result, input: null, cam: null };
    $("views").hidden = true;
    $("view-note").hidden = true;
    showView("photo");
    $("empty").hidden = true;
    resetResult();
  };
  reader.readAsDataURL(f);
  updateGradeButton();
}

function updateGradeButton() {
  $("grade").disabled = !(file && serverReady);
}

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
function showView(name) {
  const img = $("view");
  img.src = images[name];
  img.alt = { photo: "Uploaded fundus photograph", input: "Model input image",
              cam: "Grad-CAM heatmap over the retina" }[name];
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
  $("out").hidden = true;
  $("error").hidden = true;
  $("idle").hidden = false;
}
function showError(msg) {
  $("busy").hidden = true;
  $("error").textContent = msg;
  $("error").hidden = false;
}

$("grade").addEventListener("click", async () => {
  if (!file) return;
  $("idle").hidden = true;
  $("out").hidden = true;
  $("error").hidden = true;
  $("busy").hidden = false;
  $("grade").disabled = true;
  const slow = setTimeout(() => {
    $("busy-text").textContent = "Still working, the server may be starting up";
  }, 8000);
  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(`${API}/predict`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `The server returned an error (${res.status}).`);
    render(data);
  } catch (err) {
    showError(err.message === "Failed to fetch"
      ? "Could not reach the model server. Check your connection and try again."
      : err.message);
  } finally {
    clearTimeout(slow);
    $("busy-text").textContent = "Grading the photo";
    updateGradeButton();
  }
});

function render(d) {
  images.input = `data:image/png;base64,${d.model_input_png}`;
  images.cam = `data:image/png;base64,${d.gradcam_png}`;
  $("views").hidden = false;
  showView("cam");

  $("stage-number").textContent = `Stage ${d.stage} of 4`;
  $("stage-name").textContent = d.label;
  $("stage-name").style.color = COLORS[d.stage];
  $("confidence").textContent = `${Math.round(d.confidence * 100)}% probability for this stage`;

  const bars = $("bars");
  bars.innerHTML = "";
  const labels = document.createElement("div");
  labels.className = "bar-labels";
  d.probabilities.forEach((p) => {
    const chosen = p.stage === d.stage;
    const bar = document.createElement("div");
    bar.className = `bar${chosen ? " chosen" : ""}`;
    bar.style.setProperty("--c", COLORS[p.stage]);
    bar.style.setProperty("--h", "0%");
    bar.innerHTML = `<span class="bar-value">${Math.round(p.probability * 100)}%</span><div class="bar-fill"></div>`;
    bars.appendChild(bar);
    const lab = document.createElement("div");
    lab.className = `bar-label${chosen ? " chosen" : ""}`;
    lab.textContent = `${p.stage} ${p.label}`;
    labels.appendChild(lab);
    requestAnimationFrame(() => requestAnimationFrame(() =>
      bar.style.setProperty("--h", `${Math.max(p.probability * 100, 1)}%`)));
  });
  bars.after(labels);
  document.querySelectorAll(".bar-labels").forEach((el, i, all) => { if (i < all.length - 1) el.remove(); });

  const dec = $("decision");
  dec.textContent = d.refer ? "Refer: signs of diabetic retinopathy likely"
                            : "No referable retinopathy detected";
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
  $("timing").textContent = `Analysed in ${Math.round(d.inference_ms)} ms on the server.`;

  $("busy").hidden = true;
  $("out").hidden = false;
}

wakeServer();
