export async function loadModelInfo() {
  const response = await fetch("/api/model-info", {
    headers: { Accept: "application/json" },
    signal: AbortSignal.timeout(10000),
  });
  const body = await readResponse(response);
  if (!response.ok) throw new Error(body.error?.message || `Service error (${response.status})`);
  return body;
}

export function analyzeVideo(file, caption, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const form = new FormData();
    form.append("video", file, file.name);
    form.append("caption", caption);
    request.open("POST", "/api/analyze");
    request.timeout = 180000;
    request.setRequestHeader("Accept", "application/json");
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress({ stage: "upload", percent: Math.round((event.loaded / event.total) * 100) });
      }
    });
    request.upload.addEventListener("load", () => onProgress({ stage: "processing" }));
    request.addEventListener("load", () => {
      let body;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        reject(new Error("The analysis service returned an unreadable response. Please try again."));
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(body.error?.message || `Analysis failed (${request.status}).`));
        return;
      }
      resolve(body);
    });
    request.addEventListener("error", () => reject(new Error("Could not reach the analysis service. Check that the Flask app is running and try again.")));
    request.addEventListener("timeout", () => reject(new Error("Analysis took too long. Try a smaller or shorter video.")));
    request.addEventListener("abort", () => reject(new Error("The analysis request was cancelled.")));
    request.send(form);
  });
}

async function readResponse(response) {
  try {
    return await response.json();
  } catch {
    throw new Error("The analysis service returned an unreadable response.");
  }
}
