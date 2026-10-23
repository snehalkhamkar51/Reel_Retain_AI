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
  topic: document.querySelector("#topic"),
  geminiConsent: document.querySelector("#gemini-consent"),
  geminiAvailability: document.querySelector("#gemini-availability"),
  referenceUrls: document.querySelector("#reference-urls"),
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
  recommendationContext: document.querySelector("#recommendation-context"),
  recommendations: document.querySelector("#recommendation-list"),
  contentAnalysisState: document.querySelector("#content-analysis-state"),
  contentAnalysisDetails: document.querySelector("#content-analysis-details"),
  analysisTopic: document.querySelector("#analysis-topic"),
  analysisDescription: document.querySelector("#analysis-description"),
  analysisFields: document.querySelector("#analysis-fields"),
  improvementPlan: document.querySelector("#improvement-plan"),
  geminiHooks: document.querySelector("#gemini-hooks"),
  geminiCaptions: document.querySelector("#gemini-captions"),
  analysisMoments: document.querySelector("#analysis-moments"),
  analysisOverlaysStoryboard: document.querySelector("#analysis-overlays-storyboard"),
  analysisUncertainties: document.querySelector("#analysis-uncertainties"),
  audienceState: document.querySelector("#audience-state"),
  audienceList: document.querySelector("#audience-list"),
  referenceList: document.querySelector("#reference-list"),
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
elements.recommendations.addEventListener("click", copyRecommendation);
elements.geminiHooks.addEventListener("click", copyRecommendation);
elements.geminiCaptions.addEventListener("click", copyRecommendation);
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
    const contentAnalysis = latestModelInfo.content_analysis || {};
    elements.geminiConsent.disabled = !contentAnalysis.available;
    elements.geminiAvailability.textContent = contentAnalysis.available
      ? `Optional Gemini analysis is ready (${contentAnalysis.model}); your video is sent to Google only when you check the consent box.`
      : "Gemini analysis is unavailable until the app owner configures a Google Gemini API key.";
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
    const response = await analyzeVideo(
      selectedFile,
      elements.caption.value,
      elements.topic.value,
      elements.geminiConsent.checked,
      elements.referenceUrls.value,
      updateProgress,
    );
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
  showProgress(
    "Analyzing your video",
    elements.geminiConsent.checked
      ? "Extracting measurements and requesting consented Gemini content analysis…"
      : "Extracting measurements and preparing local estimates and suggestions…",
    null,
    true,
  );
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
  elements.topic.disabled = busy;
  elements.geminiConsent.disabled = busy || !latestModelInfo?.content_analysis?.available;
  elements.referenceUrls.disabled = busy;
  elements.submit.querySelector(".button-label").textContent = busy ? "Analyzing…" : "Analyze my video";
}

function validateResponse(response) {
  if (!response || typeof response !== "object"
      || !response.features || typeof response.features !== "object"
      || Array.isArray(response.features)
      || !response.estimates || typeof response.estimates !== "object"
      || Array.isArray(response.estimates)
      || !Array.isArray(response.recommendations?.items)
      || response.recommendations.items.length === 0) {
    throw new Error("The analysis service response is missing features, estimates, or recommendations.");
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
    || "Experimental retention estimates: this model was trained on synthetic data and has not been validated against real audience-retention analytics. Predictions are not actual platform analytics or guarantees.";
  renderRecommendations(response.recommendations);
  renderContentAnalysis(response.content_analysis);
  renderAudienceHypotheses(response.audience_hypotheses || []);
  renderReferences(response.references || []);
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

function renderRecommendations(recommendations) {
  elements.recommendationContext.textContent = recommendations.context
    || (recommendations.mode === "personalized"
      ? "Personalized using the supplied context and available measured features."
      : "General advice based on available measured features.");
  elements.recommendations.replaceChildren(...recommendations.items.map((recommendation) => {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    const header = document.createElement("div");
    header.className = "recommendation-card-head";
    const title = document.createElement("h4");
    title.textContent = recommendation.category || "Practical suggestion";
    const priority = document.createElement("span");
    priority.className = `recommendation-priority ${String(recommendation.priority || "Low").toLowerCase()}`;
    priority.textContent = `${recommendation.priority || "Low"} priority`;
    header.append(title, priority);
    card.append(header);
    for (const [label, value] of [
      ["What to improve", recommendation.what],
      ["Why it could help", recommendation.why],
      ["Try this", recommendation.action],
    ]) {
      if (!value) continue;
      const detail = document.createElement("p");
      detail.className = "recommendation-detail";
      const strong = document.createElement("b");
      strong.textContent = `${label}: `;
      detail.append(strong, document.createTextNode(value));
      card.append(detail);
    }
    if (recommendation.example) {
      const example = document.createElement("p");
      example.className = "recommendation-example";
      example.textContent = `Example: ${recommendation.example}`;
      card.append(example);
    }
    if (recommendation.copy_text) {
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "button button-secondary copy-recommendation";
      copy.dataset.copyText = recommendation.copy_text;
      copy.dataset.copyLabel = recommendation.category === "Opening hook" ? "hook" : "caption";
      copy.textContent = `Copy ${copy.dataset.copyLabel}`;
      copy.setAttribute("aria-label", `Copy ${copy.dataset.copyLabel}`);
      card.append(copy);
    }
    return card;
  }));
}

function renderContentAnalysis(contentAnalysis = {}) {
  const status = contentAnalysis.status || "not_requested";
  const messages = {
    not_requested: "Optional Gemini analysis was not requested. Your local measurements, model estimates, and feature-based recommendations are still available.",
    unavailable: "Gemini is not configured. Configure the API key to enable optional video understanding.",
    error: contentAnalysis.message || "Gemini could not analyze this video. Local results remain available.",
  };
  const analysis = contentAnalysis.analysis;
  if (status !== "complete" || !analysis) {
    elements.contentAnalysisState.textContent = messages[status] || messages.error;
    elements.contentAnalysisDetails.classList.add("hidden");
    return;
  }
  elements.contentAnalysisState.textContent = `AI-generated content analysis · ${contentAnalysis.model || "configured Gemini model"}. Verify observations and timestamps against your video.`;
  elements.contentAnalysisDetails.classList.remove("hidden");
  elements.analysisTopic.textContent = analysis.main_topic_and_niche;
  elements.analysisDescription.textContent = analysis.description;
  elements.analysisFields.replaceChildren(
    labeledDetail("Opening hook", `${analysis.opening_hook} ${analysis.hook_clarity}`),
    labeledDetail("Main message", analysis.main_message),
    labeledDetail("Structure", analysis.content_structure),
    labeledDetail("Visual presentation", analysis.visual_presentation),
    labeledDetail("Audio observations", analysis.audio_observations || "Unavailable or not sufficiently clear."),
    labeledDetail("Ending and payoff", analysis.ending_and_payoff),
  );
  elements.improvementPlan.replaceChildren(...analysis.editing_changes.map((edit) => {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    card.append(
      detailHeading(edit.what_to_change, edit.priority),
      labeledDetail("Where", displayTimestamp(edit.timestamp_seconds)),
      labeledDetail("Change", `${edit.operation}: ${edit.example}`),
      labeledDetail("Why it may help", edit.viewer_rationale),
      labeledDetail("Evidence", "Gemini content analysis; confirm against the video."),
    );
    return card;
  }));
  elements.geminiHooks.replaceChildren(...analysis.alternative_hooks.map((hook) => copyExample(hook, "Copy hook")));
  elements.geminiCaptions.replaceChildren(...analysis.caption_alternatives.map((item) => {
    const card = document.createElement("article");
    card.className = "hook-card";
    card.append(
      labeledDetail("Caption", item.caption),
      labeledDetail("CTA", item.cta),
      copyButton(`${item.caption} ${item.cta}`, "Copy caption"),
    );
    return card;
  }));
  elements.analysisMoments.replaceChildren(
    labeledDetail("Useful moments · Gemini observations", analysis.useful_moments.length
      ? analysis.useful_moments.map((moment) => `${displayTimestamp(moment.timestamp_seconds)} — ${moment.observation} Suggested: ${moment.suggested_action}`).join("\n")
      : "No specific moments were identified."),
  );
  elements.analysisOverlaysStoryboard.replaceChildren(
    labeledDetail("Suggested on-screen text", analysis.on_screen_text.join("\n") || "No overlay text suggested."),
    labeledDetail("Subtitle recommendations", analysis.subtitle_recommendations.join("\n") || "No subtitle suggestions were returned."),
    labeledDetail("Mini storyboard", analysis.storyboard.join("\n") || "No storyboard supplied."),
  );
  elements.analysisUncertainties.replaceChildren(
    labeledDetail("Uncertainties", analysis.uncertainties.join("\n") || "No additional uncertainties reported."),
    labeledDetail("Not reliably determined", analysis.unavailable_observations.join("\n") || "None reported."),
  );
}

function renderAudienceHypotheses(hypotheses) {
  elements.audienceList.replaceChildren();
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    elements.audienceState.textContent = "No audience hypotheses are available. Opt in to Gemini content analysis to request topic-based hypotheses; viewer demographics are never inferred from the video alone.";
    return;
  }
  elements.audienceState.textContent = "Potential audience hypotheses only—not measured viewers, age analytics, or demographic predictions.";
  elements.audienceList.replaceChildren(...hypotheses.map((hypothesis) => {
    const card = document.createElement("article");
    card.className = "audience-card";
    card.append(
      detailHeading(hypothesis.age_band || "Age not specified"),
      labeledDetail("Why it may be relevant", hypothesis.relevance),
      labeledDetail("Need or interest", hypothesis.need_or_interest),
      labeledDetail("Accessible presentation", hypothesis.accessible_presentation),
      labeledDetail("Reasoning", hypothesis.reasoning),
      labeledDetail("Evidence", "Gemini content analysis · hypothesis only"),
    );
    return card;
  }));
}

function renderReferences(references) {
  elements.referenceList.replaceChildren();
  if (!Array.isArray(references) || references.length === 0) {
    const empty = document.createElement("p");
    empty.className = "reference-empty";
    empty.textContent = "No reference URLs supplied. This app does not search or scrape video platforms; no example metrics are available.";
    elements.referenceList.append(empty);
    return;
  }
  elements.referenceList.replaceChildren(...references.map((reference) => {
    const card = document.createElement("article");
    card.className = "reference-card";
    const title = document.createElement("b");
    title.textContent = `${reference.platform} · User-provided URL`;
    const link = document.createElement("a");
    link.href = reference.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = reference.url;
    const notice = document.createElement("p");
    notice.textContent = reference.notice;
    const metrics = document.createElement("p");
    metrics.textContent = "Public engagement metrics and publication details: not retrieved.";
    card.append(title, link, notice, metrics);
    return card;
  }));
}

function labeledDetail(label, text) {
  const item = document.createElement("p");
  item.className = "analysis-detail";
  const heading = document.createElement("b");
  heading.textContent = `${label}: `;
  item.append(heading, document.createTextNode(String(text || "Unavailable.")));
  return item;
}

function detailHeading(text, priority) {
  const header = document.createElement("div");
  header.className = "recommendation-card-head";
  const title = document.createElement("h4");
  title.textContent = text;
  header.append(title);
  if (priority) {
    const badge = document.createElement("span");
    badge.className = `recommendation-priority ${priority.toLowerCase()}`;
    badge.textContent = `${priority} priority`;
    header.append(badge);
  }
  return header;
}

function copyExample(text, label) {
  const card = document.createElement("article");
  card.className = "hook-card";
  const example = document.createElement("p");
  example.className = "recommendation-example";
  example.textContent = text;
  card.append(example, copyButton(text, label));
  return card;
}

function copyButton(text, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-secondary copy-recommendation";
  button.dataset.copyText = text;
  button.dataset.copyLabel = label.replace(/^Copy\s+/i, "").toLowerCase();
  button.textContent = label;
  return button;
}

function displayTimestamp(seconds) {
  if (seconds === null || seconds === undefined || seconds === ""
      || !Number.isFinite(Number(seconds)) || Number(seconds) < 0) {
    return "No verified timestamp available";
  }
  const whole = Math.floor(Number(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")} (Gemini suggested; verify)`;
}

async function copyRecommendation(event) {
  const button = event.target.closest("button[data-copy-text]");
  if (!button) return;
  try {
    await navigator.clipboard.writeText(button.dataset.copyText);
    button.textContent = "Copied";
    showToast(`Copied ${button.dataset.copyLabel || "text"} to clipboard.`);
    window.setTimeout(() => {
      if (button.isConnected) button.textContent = `Copy ${button.dataset.copyLabel || "text"}`;
    }, 1600);
  } catch {
    showToast("Clipboard access is unavailable. Select the example text and copy it manually.");
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
  elements.topic.value = "";
  elements.geminiConsent.checked = false;
  elements.referenceUrls.value = "";
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
