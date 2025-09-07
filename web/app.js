const API = window.location.origin;

const $ = (id) => document.getElementById(id);
const imgEl = $("imgPreview");
const cv = $("overlay");
const ctx = cv.getContext("2d");
let lastData = null;

// ===== Utility =====
function escapeHtml(s = "") {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
function clamp01(x) { return Math.max(0, Math.min(1, Number(x) || 0)); }
function toast(msg) {
  const d = document.createElement("div");
  d.className = "toast";
  d.textContent = msg;
  document.body.appendChild(d);
  setTimeout(() => d.classList.add("show"));
  setTimeout(() => d.classList.remove("show"), 2200);
  setTimeout(() => d.remove(), 2600);
}

// ===== Disclaimer gating =====
const ACK_KEY = "clai_disclaimer_ack";
const ack = localStorage.getItem(ACK_KEY) === "1";
if (!ack) $("banner").classList.remove("hidden");
$("agree").checked = ack;
setButtonsEnabled(ack);

$("agree").addEventListener("change", (e) => {
  const ok = !!e.target.checked;
  setButtonsEnabled(ok);
  localStorage.setItem(ACK_KEY, ok ? "1" : "0");
});
$("dismiss").onclick = () => {
  $("banner").classList.add("hidden");
};

function setButtonsEnabled(ok) {
  $("btnUpload").disabled = !ok;
  $("btnJson").disabled = !ok;
  $("btnUpload").classList.toggle("btn-disabled", !ok);
  $("btnJson").classList.toggle("btn-disabled", !ok);
}

// ===== Tabs =====
$("tabUpload").onclick = () => switchTab("upload");
$("tabUrl").onclick = () => switchTab("url");
function switchTab(which) {
  const upload = which === "upload";
  $("tabUpload").classList.toggle("active", upload);
  $("tabUrl").classList.toggle("active", !upload);
  $("formUpload").classList.toggle("hidden", !upload);
  $("formUrl").classList.toggle("hidden", upload);
}
switchTab("upload");

// ===== Preview Sizing =====
$("file").addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  imgEl.src = URL.createObjectURL(f);
  await waitImageLoad();
  fitCanvasToImage();
  clearOverlay();
});
function waitImageLoad() {
  return new Promise((res) => {
    if (imgEl.complete) return res();
    imgEl.onload = () => res();
    imgEl.onerror = () => res();
  });
}
function fitCanvasToImage() {
  const rect = imgEl.getBoundingClientRect();
  cv.width = rect.width || 600;
  cv.height = rect.height || 400;
}
window.addEventListener("resize", () => {
  if (!imgEl.src) return;
  fitCanvasToImage();
  if (lastData) drawOverlayFromScenario(lastData.scenarios?.[0]);
});
function clearOverlay() { ctx.clearRect(0, 0, cv.width, cv.height); }

// ===== Analyze (Upload) =====
$("formUpload").addEventListener("submit", async (e) => {
  e.preventDefault();
  if ($("btnUpload").disabled) return toast("Please accept the disclaimer first.");
  const file = $("file").files?.[0];
  if (!file) return toast("Please choose an image.");

  $("busy").classList.remove("hidden");
  try {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("timeframe", $("timeframeU").value);
    fd.append("strategy", $("strategyU").value);
    if ($("intentionU").value) fd.append("intention", $("intentionU").value);
    if ($("questionU").value) fd.append("question", $("questionU").value);

    const resp = await fetch(`${API}/analyze/upload`, { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || data?.error || "Analyze failed");
    renderOutput(data);
  } catch (err) {
    console.error(err);
    toast(err.message || "Error");
  } finally {
    $("busy").classList.add("hidden");
  }
});

// ===== Analyze (URL) =====
$("formUrl").addEventListener("submit", async (e) => {
  e.preventDefault();
  if ($("btnJson").disabled) return toast("Please accept the disclaimer first.");
  const image_url = $("imageUrl").value.trim();
  if (!image_url) return toast("Enter an image URL.");

  $("busy2").classList.remove("hidden");
  try {
    const body = {
      image_url,
      timeframe: $("timeframeJ").value,
      strategy: $("strategyJ").value,
      intention: $("intentionJ").value || undefined,
      question: $("questionJ").value || undefined,
    };
    const resp = await fetch(`${API}/analyze/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || data?.error || "Analyze failed");
    imgEl.src = image_url;
    await waitImageLoad();
    fitCanvasToImage();
    renderOutput(data);
  } catch (err) {
    console.error(err);
    toast(err.message || "Error");
  } finally {
    $("busy2").classList.add("hidden");
  }
});

// ===== Render =====
function renderOutput(data) {
  lastData = data;
  const box = $("scenarios");
  box.innerHTML = "";
  const list = data.scenarios || [];
  if (!Array.isArray(list) || list.length === 0) {
    box.innerHTML = `<p class="text-sm text-slate-400">No scenarios returned.</p>`;
    clearOverlay();
    return;
  }

  list.forEach((s, idx) => {
    const side = (s.side || "").toLowerCase();
    const color = side === "sell" ? "border-rose-500/60" : "border-emerald-500/60";
    const chip = side === "sell" ? "bg-rose-500/15 text-rose-300" : "bg-emerald-500/15 text-emerald-300";
    const r = s.r_multiple || {};
    const t1 = r.t1 ?? "–", t2 = r.t2 ?? "–", t3 = r.t3 ?? "–";
    const conf = s.confidence != null ? (Math.round(s.confidence * 100) / 100) : "–";
    const maxWait = s.max_time_wait_min != null ? s.max_time_wait_min : "–";

    const confluence = (s.confluence || []).map(c => `<span class="pill">${escapeHtml(c)}</span>`).join("");

    const card = document.createElement("div");
    card.className = `scenario-card border ${color}`;
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="pill ${chip} capitalize">${escapeHtml(side || "n/a")}</span>
          <h4 class="font-semibold">${escapeHtml(s.name || s.id || "Scenario " + (idx+1))}</h4>
          <span class="badge badge-subtle">AI</span>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn btn-sm" data-idx="${idx}" data-action="overlay">Overlay</button>
          <button class="btn btn-sm" data-idx="${idx}" data-action="copy">Copy</button>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-3 mt-3">
        <div>
          <div class="k">Entry</div>
          <div class="v">${escapeHtml(s.entry?.type || "n/a")} · ${escapeHtml(s.entry?.price_text || s.entry?.zone_text || "")}</div>
        </div>
        <div>
          <div class="k">Stop</div>
          <div class="v">${escapeHtml(s.stop?.price_text || "n/a")}</div>
        </div>
        <div>
          <div class="k">Targets</div>
          <div class="v">${(s.targets || []).map(t => `<span class="pill">${escapeHtml(`${t.label || ""} ${t.price_text || ""}`.trim())}</span>`).join(" ") || "–"}</div>
        </div>
        <div>
          <div class="k">R Multiple</div>
          <div class="v">T1 ${t1} · T2 ${t2} · T3 ${t3}</div>
        </div>
        <div>
          <div class="k">Max wait (min)</div>
          <div class="v">${maxWait}</div>
        </div>
        <div>
          <div class="k">Confidence</div>
          <div class="v">${conf}</div>
        </div>
      </div>

      <div class="mt-3">${confluence || ""}</div>
      <p class="text-sm text-slate-300 mt-2">${escapeHtml(s.explanation || "")}</p>
    `;
    box.appendChild(card);

    card.querySelectorAll("button[data-idx]").forEach(btn => {
      btn.onclick = async () => {
        const action = btn.getAttribute("data-action");
        if (action === "overlay") drawOverlayFromScenario(s);
        if (action === "copy") {
          await navigator.clipboard.writeText(JSON.stringify(s, null, 2));
          toast("Scenario copied");
        }
      };
    });
  });

  drawOverlayFromScenario(list[0]);
}

function drawOverlayFromScenario(s) {
  clearOverlay();
  if (!imgEl.src) return;
  if (!s?.draw_commands) return;

  const H = cv.height;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.9;

  s.draw_commands.forEach(cmd => {
    const op = (cmd.op || "").toLowerCase();
    const y1 = clamp01(cmd.y1_rel ?? 0.5) * H;
    const y2 = clamp01(cmd.y2_rel ?? cmd.y1_rel ?? 0.5) * H;

    if (op === "line") {
      ctx.beginPath();
      ctx.moveTo(8, y1);
      ctx.lineTo(cv.width - 8, y2);
      ctx.strokeStyle = "#22d3ee";
      ctx.stroke();
    } else if (op === "box") {
      const top = Math.min(y1, y2);
      const h = Math.abs(y2 - y1);
      ctx.strokeStyle = "#818cf8";
      ctx.strokeRect(12, top, cv.width - 24, h || 2);
    } else if (op === "label") {
      ctx.fillStyle = "#eab308";
      ctx.font = "12px ui-sans-serif";
      ctx.fillText(cmd.note || "label", 12, y1);
    }
  });
}

// Download merged image+overlay
$("btnDownloadOverlay").onclick = () => {
  if (!imgEl.src) return;
  const tmp = document.createElement("canvas");
  tmp.width = cv.width; tmp.height = cv.height;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(imgEl, 0, 0, tmp.width, tmp.height);
  tctx.drawImage(cv, 0, 0);
  const url = tmp.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url; a.download = "sagemaker-overlay.png";
  a.click();
};
