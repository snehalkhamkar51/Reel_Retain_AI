import { analyzeVideo, loadModelInfo } from "./api.js";
import {
  buildReport,
  renderFeatures,
  renderInsights,
  renderMetricChart,
  renderMetrics,
  renderSignals,
} from "./visualizations.js";

const ACCEPTED_EXTENSIONS = new Set([".mp4", ".mov", ".m4v", ".webm", ".avi"]);
const elements = {
  nav: document.querySelector("#site-nav"),
  menu: document.querySelector(".menu-toggle"),
  dropzone: document.querySelector("#dropzone"),
  input: document.querySelector("#video-input"),
  empty: document.querySelector("#dropzone-empty"),
  selected: document.querySelector("#selected-video"),
  preview: document.querySelector("#video-preview"),
  fileName: document.querySelector("#file-name"),
  fileDetails: document.querySelector("#file-details"),
  remove: document.querySelector("#remove-video"),
  caption: document.querySelector("#caption"),
  captionCount: document.querySelector("#caption-count"),
  limit: document.querySelector("#upload-limit"),
  formatChips: document.querySelector("#format-chips"),
  submit: document.querySelector("#analyze-button"),
  error: document.querySelector("#upload-error"),
  progress: document.querySelector("#progress-panel"),
  progressTitle: document.querySelector("#progress-title"),
  progressMessage: document.querySelector("#progress-message"),
  progressFill: document.querySelector("#progress-fill"),
  progressPercent: document.querySelector("#progress-percent"),
  results: document.querySelector("#results"),
  resultFile: document.querySelector("#results-file"),
  disclaimer: document.querySelector("#result-disclaimer"),
  metrics: document.querySelector("#metrics-grid"),
  chart: document.querySelector("#metric-chart"),
  legend: document.querySelector("#chart-legend"),
  summary: document.querySelector("#summary-text"),
  insights: document.querySelector("#insights-list"),
  features: document.querySelector("#feature-snapshot"),
  signals: document.querySelector("#signal-profile"),
  signalDescription: document.querySelector("#signal-description"),
  warnings: document.querySelector("#warning-list"),
  requestId: document.querySelector("#request-id"),
  download: document.querySelector("#download-report"),
  analyzeAnother: document.querySelector("#analyze-another"),
  toast: document.querySelector("#toast"),
};

let selectedFile = null;
let previewUrl = null;
let latestAnalysis = null;
let latestModelInfo = null;
let maxUploadBytes = 100 * 1024 * 1024;
let toastTimeout;

elements.menu.addEventListener("click", toggleNavigation);
elements.nav.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeNavigation();
});

elements.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button, video, input")) return;
  elements.input.click();
});
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.target !== elements.dropzone) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.input.click();
  }
});
elements.input.addEventListener("change", () => {
  const [file] = elements.input.files;
  if (file) setSelectedFile(file);
});
elements.remove.addEventListener("click", (event) => {
  event.stopPropagation();
  clearSelectedFile();
});
elements.caption.addEventListener("input", () => {
  elements.captionCount.textContent = `${elements.caption.value.length} / 2200`;
});
elements.submit.addEventListener("click", submitAnalysis);
elements.analyzeAnother.addEventListener("click", resetForAnother);
elements.download.addEventListener("click", downloadReport);

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragging");
  });
}
elements.dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) setSelectedFile(file);
});

window.addEventListener("resize", () => {
  if (latestAnalysis) renderMetricChart(
    elements.chart,
    elements.legend,
    latestAnalysis.estimates,
    latestAnalysis.features.video_duration_sec,
  );
});

loadPageInfo();
observeNavigation();

async function loadPageInfo() {
  try {
    latestModelInfo = await loadModelInfo();
    const upload = latestModelInfo.upload || {};
    if (Number.isFinite(upload.max_bytes) && upload.max_bytes > 0) {
      maxUploadBytes = upload.max_bytes;
      elements.limit.textContent = `Up to ${formatBytes(maxUploadBytes)} · temporary local processing`;
    } else {
      elements.limit.textContent = "Maximum upload size set by the local server";
    }
    const extensions = Array.isArray(upload.extensions)
      ? upload.extensions.filter((extension) => ACCEPTED_EXTENSIONS.has(extension))
      : [];
    if (extensions.length) {
      elements.input.accept = extensions.join(",");
      elements.formatChips.replaceChildren(...extensions.map((extension) => {
        const chip = document.createElement("span");
        chip.textContent = extension.slice(1).toUpperCase();
        return chip;
      }));
    }
  } catch {
    elements.limit.textContent = "Server upload limit unavailable · checked again on submit";
  }
}

function setSelectedFile(file) {
  clearError();
  const extension = file.name.includes(".")
    ? `.${file.name.split(".").pop().toLowerCase()}`
    : "";
  if (!ACCEPTED_EXTENSIONS.has(extension)) {
    showError("Choose a supported video file: MP4, MOV, M4V, WEBM, or AVI.");
    elements.input.value = "";
    return;
  }
  if (file.size === 0) {
    showError("This file is empty. Choose a video that contains data.");
    elements.input.value = "";
    return;
  }
  if (file.size > maxUploadBytes) {
    showError(`This file is larger than the current ${formatBytes(maxUploadBytes)} upload limit.`);
    elements.input.value = "";
    return;
  }
  if (file.type && !file.type.startsWith("video/") && file.type !== "application/octet-stream") {
    showError("The selected file does not appear to be a video. Choose a supported video file.");
    elements.input.value = "";
    return;
  }
  releasePreviewUrl();
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.fileName.textContent = file.name;
  elements.fileDetails.textContent = `${formatBytes(file.size)} · Reading video metadata…`;
  elements.preview.onloadedmetadata = () => {
    const duration = Number.isFinite(elements.preview.duration)
      ? `${formatDuration(elements.preview.duration)} · `
      : "";
    const dimensions = elements.preview.videoWidth && elements.preview.videoHeight
      ? `${elements.preview.videoWidth} × ${elements.preview.videoHeight} · `
      : "";
    elements.fileDetails.textContent = `${formatBytes(file.size)} · ${duration}${dimensions}ready to analyze`;
  };
  elements.preview.onerror = () => {
    elements.fileDetails.textContent = `${formatBytes(file.size)} · Preview unavailable; server will validate the video`;
  };
  elements.empty.classList.add("hidden");
  elements.selected.classList.remove("hidden");
  elements.submit.disabled = false;
  elements.results.classList.add("hidden");
}

function clearSelectedFile() {
  selectedFile = null;
  releasePreviewUrl();
  elements.input.value = "";
  elements.preview.removeAttribute("src");
  elements.preview.load();
  elements.empty.classList.remove("hidden");
  elements.selected.classList.add("hidden");
  elements.submit.disabled = true;
  elements.results.classList.add("hidden");
  clearError();
  elements.progress.classList.add("hidden");
}

function releasePreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

async function submitAnalysis() {
  if (!selectedFile || elements.submit.disabled) return;
  clearError();
  setBusy(true);
  elements.results.classList.add("hidden");
  showProgress("Uploading video", "Sending the selected video to the local analysis service…", 0, false);
  try {
    const response = await analyzeVideo(selectedFile, elements.caption.value, updateProgress);
    validateResponse(response);
    latestAnalysis = response;
    elements.progress.classList.add("hidden");
    elements.results.classList.remove("hidden");
    renderResult(response, selectedFile.name);
    elements.results.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
    window.setTimeout(() => document.querySelector("#results-title").focus({ preventScroll: true }), 100);
  } catch (error) {
    elements.progress.classList.add("hidden");
    showError(error instanceof Error ? error.message : "Analysis failed. Please try again.");
  } finally {
    setBusy(false);
  }
}

function updateProgress(progress) {
  if (progress.stage === "upload") {
    showProgress("Uploading video", "Sending the selected video to the local analysis service…", progress.percent, false);
    return;
  }
  showProgress("Analyzing your video", "Extracting features and preparing experimental estimates…", null, true);
}

function showProgress(title, message, percent, indeterminate) {
  elements.progress.classList.remove("hidden");
  elements.progressTitle.textContent = title;
  elements.progressMessage.textContent = message;
  elements.progressFill.classList.toggle("indeterminate", indeterminate);
  if (percent === null || percent === undefined) {
    elements.progressFill.style.width = "35%";
    elements.progressPercent.textContent = "";
  } else {
    elements.progressFill.style.width = `${percent}%`;
    elements.progressPercent.textContent = `${percent}%`;
  }
}

function setBusy(busy) {
  elements.submit.disabled = busy || !selectedFile;
  elements.remove.disabled = busy;
  elements.input.disabled = busy;
  elements.caption.disabled = busy;
  elements.submit.querySelector(".button-label").textContent = busy ? "Analyzing…" : "Analyze my video";
}

function validateResponse(response) {
  if (!response || typeof response !== "object"
      || !response.features || typeof response.features !== "object"
      || Array.isArray(response.features)
      || !response.estimates || typeof response.estimates !== "object"
      || Array.isArray(response.estimates)) {
    throw new Error("The analysis service response is missing video features or estimates.");
  }
  const required = [
    "watched_percentage_synthetic",
    "average_watch_time_sec_synthetic",
    "early_dropoff_rate_pct_synthetic",
  ];
  for (const key of required) {
    const value = response.estimates[key]?.estimate;
    if (value !== null && value !== undefined && !Number.isFinite(Number(value))) {
      throw new Error(`The analysis service returned an invalid estimate for ${key.replaceAll("_", " ")}.`);
    }
  }
}

function renderResult(response, fileName) {
  elements.resultFile.textContent = fileName;
  elements.requestId.textContent = response.request_id || "unavailable";
  elements.disclaimer.querySelector("p").textContent = response.disclaimer
    || "Experimental estimates from synthetic-data models only; not validated real-world audience analytics.";
  renderMetrics(elements.metrics, response.estimates);
  const duration = Number(response.features.video_duration_sec);
  renderMetricChart(elements.chart, elements.legend, response.estimates, Number.isFinite(duration) ? duration : 0);
  renderFeatures(elements.features, response.features);
  renderSignals(elements.signals, elements.signalDescription, response.features);
  renderInsights(elements.insights, response.features);
  renderWarnings(response.warnings);
  elements.summary.textContent = makeSummary(response.estimates, response.features);
}

function makeSummary(estimates, features) {
  const watched = estimates.watched_percentage_synthetic?.estimate;
  const watchTime = estimates.average_watch_time_sec_synthetic?.estimate;
  const duration = features.video_duration_sec;
  if (![watched, watchTime, duration].every(isFiniteValue)) {
    return "Some summary values were unavailable in the service response. Review the feature snapshot and individual estimate fields.";
  }
  return `The model returned an experimental watched estimate of ${formatNumber(Number(watched))}% and an average watch-time estimate of ${formatNumber(Number(watchTime))} seconds for a ${formatNumber(Number(duration))}-second video. These outputs come from synthetic training data and are not measured audience behavior.`;
}

function renderWarnings(warnings) {
  elements.warnings.replaceChildren();
  if (!Array.isArray(warnings) || warnings.length === 0) {
    elements.warnings.classList.add("hidden");
    return;
  }
  elements.warnings.classList.remove("hidden");
  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning-item";
    item.textContent = String(warning);
    elements.warnings.append(item);
  }
}

function downloadReport() {
  if (!latestAnalysis || !selectedFile) return;
  const report = buildReport(latestAnalysis, selectedFile.name);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${safeFileStem(selectedFile.name)}-reelretain-report.json`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast("Your analysis report has been downloaded.");
}

function resetForAnother() {
  latestAnalysis = null;
  clearSelectedFile();
  elements.caption.value = "";
  elements.captionCount.textContent = "0 / 2200";
  document.querySelector("#analyze").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth" });
  elements.input.click();
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("hidden");
}

function clearError() {
  elements.error.textContent = "";
  elements.error.classList.add("hidden");
}

function showToast(message) {
  window.clearTimeout(toastTimeout);
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  toastTimeout = window.setTimeout(() => elements.toast.classList.add("hidden"), 3200);
}

function toggleNavigation() {
  const expanded = elements.menu.getAttribute("aria-expanded") === "true";
  elements.menu.setAttribute("aria-expanded", String(!expanded));
  elements.menu.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
  elements.nav.classList.toggle("is-open", !expanded);
}

function closeNavigation() {
  elements.menu.setAttribute("aria-expanded", "false");
  elements.menu.setAttribute("aria-label", "Open navigation");
  elements.nav.classList.remove("is-open");
}

function observeNavigation() {
  if (!("IntersectionObserver" in window)) return;
  const sections = ["home", "analyze", "about"].map((id) => document.getElementById(id));
  const links = [...document.querySelectorAll(".nav-link")];
  const observer = new IntersectionObserver((entries) => {
    const active = entries.filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!active) return;
    for (const link of links) {
      const selected = link.getAttribute("href") === `#${active.target.id}`;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }, { rootMargin: "-25% 0px -60% 0px", threshold: [0, 0.15, 0.4] });
  sections.forEach((section) => observer.observe(section));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown size";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 10 * 1024 * 1024 ? 0 : 1)} MB`;
}

function formatDuration(seconds) {
  const rounded = Math.floor(seconds);
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(Number(value))
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(Number(value))
    : "unavailable";
}

function isFiniteValue(value) {
  return value !== null && value !== undefined && value !== ""
    && Number.isFinite(Number(value));
}

function safeFileStem(name) {
  return name.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "").slice(0, 64) || "video";
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
