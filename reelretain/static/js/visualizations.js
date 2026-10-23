const colors = ["#38bdf8", "#6592ff", "#53d6b0"];

export function renderMetrics(container, estimates) {
  const specs = [
    ["watched_percentage_synthetic", "Estimated watched", "A synthetic-model estimate", "◉"],
    ["average_watch_time_sec_synthetic", "Estimated watch time", "Constrained to video duration", "◷"],
    ["early_dropoff_rate_pct_synthetic", "Estimated early drop-off", "A synthetic-model estimate", "⌁"],
  ];
  container.replaceChildren(...specs.map(([key, label, detail, icon], index) => {
    const result = estimates[key];
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<span class="metric-label"><i class="metric-dot" style="background:${colors[index]}"></i>${label}</span>
      <strong class="metric-value" data-value="${finite(result?.estimate)}"><span class="count-value">0</span>&nbsp;<small>${escapeHtml(result?.unit || "unavailable")}</small></strong>
      <span class="metric-sub">${detail}</span>
      ${result?.constrained ? '<div class="metric-constraint">Display value constrained from raw model output</div>' : ""}
      <div class="metric-model">${result?.selected_model ? `Model · ${escapeHtml(result.selected_model.replaceAll("_", " "))}` : "Model unavailable"}</div>`;
    return card;
  }));
  container.querySelectorAll(".metric-value").forEach((value) => animateCount(value));
}

export function renderMetricChart(canvas, legend, estimates, duration) {
  const data = [
    ["watched_percentage_synthetic", "Watched percentage", "%", 100],
    ["average_watch_time_sec_synthetic", "Average watch time", "sec", duration],
    ["early_dropoff_rate_pct_synthetic", "Early drop-off", "%", 100],
  ].map(([key, label, unit, max]) => ({
    key, label, unit, value: finite(estimates[key]?.estimate), max: finite(max),
  }));
  const dpr = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(bounds.width, 280);
  const height = 185;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  const left = 105;
  const right = width - 39;
  const usable = Math.max(100, right - left);
  const top = 20;
  const row = 48;
  context.font = "10px Segoe UI, sans-serif";
  data.forEach((item, index) => {
    const y = top + index * row;
    context.fillStyle = "#a8b4c7";
    context.textAlign = "right";
    context.fillText(item.label, left - 11, y + 12);
    context.fillStyle = "#28364a";
    roundedRect(context, left, y, usable, 11, 5);
    const ratio = item.max > 0 ? Math.max(0, Math.min(1, item.value / item.max)) : 0;
    context.fillStyle = colors[index];
    roundedRect(context, left, y, usable * ratio, 11, 5);
    context.fillStyle = "#e6edf6";
    context.textAlign = "left";
    const formatted = item.value === null ? "—" : `${formatNumber(item.value)} ${item.unit}`;
    context.fillText(formatted, Math.min(left + usable * ratio + 7, width - 50), y + 10);
    context.fillStyle = "#74859a";
    context.fillText(`scale 0–${formatNumber(item.max)} ${item.unit}`, left, y + 28);
  });
  legend.replaceChildren(...data.map((item, index) => {
    const line = document.createElement("li");
    line.innerHTML = `<i class="legend-dot" style="background:${colors[index]}"></i>${escapeHtml(item.label)}: <b>${item.value === null ? "unavailable" : `${formatNumber(item.value)} ${item.unit}`}</b>`;
    return line;
  }));
}

export function renderFeatures(container, features) {
  const labels = [
    ["video_duration_sec", "Duration", (value) => `${formatNumber(value)} sec`],
    ["scene_changes", "Scene changes", (value) => formatNumber(value, 0)],
    ["motion_intensity", "Motion intensity", (value) => formatNumber(value, 3)],
    ["audio_energy", "Audio energy · RMS", (value) => formatNumber(value, 3)],
    ["silence_duration_sec", "Silence · thresholded", (value) => `${formatNumber(value)} sec`],
    ["caption_word_count", "Caption words", (value) => formatNumber(value, 0)],
    ["hashtag_count", "Hashtags", (value) => formatNumber(value, 0)],
    ["frame_rate_fps", "Frame rate", (value) => `${formatNumber(value, 1)} fps`],
    ["width_px", "Resolution width", (value) => `${formatNumber(value, 0)} px`],
    ["height_px", "Resolution height", (value) => `${formatNumber(value, 0)} px`],
  ];
  container.replaceChildren(...labels.filter(([key]) => Object.hasOwn(features, key)).map(([key, label, format]) => {
    const item = document.createElement("div");
    item.className = "snapshot-item";
    const value = features[key];
    item.innerHTML = `<span>${label}</span><b>${value === null || !Number.isFinite(Number(value)) ? "Unavailable" : format(Number(value))}</b>`;
    return item;
  }));
}

export function renderSignals(container, description, features) {
  const rows = [
    ["Scene changes", finite(features.scene_changes), Math.max(1, finite(features.video_duration_sec) * 1.5), "count"],
    ["Motion intensity", finite(features.motion_intensity), 1, "ratio"],
    ["Audio energy", finite(features.audio_energy), 1, "ratio"],
    ["Silence duration", finite(features.silence_duration_sec), finite(features.video_duration_sec), "seconds"],
    ["Caption words", finite(features.caption_word_count), 40, "count"],
    ["Hashtags", finite(features.hashtag_count), 15, "count"],
  ];
  const available = rows.filter(([, value]) => value !== null);
  container.replaceChildren(...available.map(([label, value, max, unit]) => {
    const row = document.createElement("div");
    row.className = "signal-row";
    const percent = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
    row.innerHTML = `<span class="signal-label">${label}</span><div class="signal-track" role="img" aria-label="${label}: ${formatNumber(value)} ${unit}"><div class="signal-bar" style="width:${percent}%"></div></div><span class="signal-value">${formatNumber(value)} ${unit}</span>`;
    return row;
  }));
  description.textContent = "Bars use simple display scales to make extracted feature values easier to scan. They do not represent viewer retention over time.";
}

export function renderInsights(container, features) {
  const suggestions = [];
  const duration = finite(features.video_duration_sec);
  const silence = finite(features.silence_duration_sec);
  const scenes = finite(features.scene_changes);
  const words = finite(features.caption_word_count);
  if (duration !== null && silence !== null && duration > 0 && silence / duration >= 0.2) {
    suggestions.push(["neutral", `Detected quiet spans total about ${formatNumber(silence)} sec (${formatNumber((silence / duration) * 100, 0)}% of duration). Review those moments if they are unintended.`]);
  }
  if (duration !== null && scenes !== null && duration > 0) {
    const rate = scenes / duration * 60;
    if (rate < 2) suggestions.push(["neutral", "Few sampled scene changes were detected per minute. Consider reviewing visual variety and pacing by hand."]);
    else if (rate > 20) suggestions.push(["neutral", "Many sampled scene changes were detected per minute. Check that the pacing feels intentional and easy to follow."]);
  }
  if (words === 0) suggestions.push(["neutral", "No caption words were supplied. Add the intended caption to include word and hashtag counts in this analysis."]);
  if (suggestions.length === 0) {
    suggestions.push(["positive", "No review prompt was triggered by the available thresholds. That is not a prediction of audience response."]);
  }
  container.replaceChildren(...suggestions.map(([kind, text]) => {
    const item = document.createElement("div");
    item.className = `insight-item ${kind}`;
    item.innerHTML = `<span aria-hidden="true">${kind === "positive" ? "✦" : "↗"}</span><span>${text}</span>`;
    return item;
  }));
}

export function buildReport(result, fileName) {
  return {
    title: "ReelRetain AI analysis report",
    file_name: fileName,
    generated_at: new Date().toISOString(),
    disclaimer: result.disclaimer,
    request_id: result.request_id,
    estimates: result.estimates,
    recommendations: result.recommendations,
    content_analysis: result.content_analysis,
    audience_hypotheses: result.audience_hypotheses || [],
    references: result.references || [],
    methodology: result.methodology,
    extracted_features: result.features,
    warnings: result.warnings || [],
    note: "Experimental estimates from synthetic training data only; not validated real-world audience analytics.",
  };
}

function animateCount(element) {
  const target = Number(element.dataset.value);
  if (!Number.isFinite(target)) {
    element.querySelector(".count-value").textContent = "—";
    return;
  }
  const output = element.querySelector(".count-value");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    output.textContent = formatNumber(target);
    return;
  }
  const start = performance.now();
  const duration = 600;
  const tick = (now) => {
    const progress = Math.min(1, (now - start) / duration);
    const eased = 1 - (1 - progress) ** 3;
    output.textContent = formatNumber(target * eased);
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function roundedRect(context, x, y, width, height, radius) {
  if (width <= 0) return;
  context.beginPath();
  context.roundRect(x, y, width, height, Math.min(radius, width / 2, height / 2));
  context.fill();
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function formatNumber(value, digits = 1) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}
