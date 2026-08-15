export type ProbabilityItem = {
  condition: string;
  probability: number;
};

export type SignalPoint = {
  x: number;
  y: number;
};

export type PredictionResult = {
  id: string;
  timestamp: string;
  predictedClass: string;
  confidence: number;
  riskLevel: string;
  probabilities: ProbabilityItem[];
  clinicalSummary: string;
  recommendation: string;
  inferenceTime: number;
  modelVersion: string;
  heartRateBpm: number | null;
  rPeakCount: number;
  signalPreview: SignalPoint[];
  scalogramImage: string;
  gradcamImage: string | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function readApiError(response: Response) {
  try {
    const body = await response.json();
    return body.detail || body.message || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function analyzeFiles(files: FileList | File[]) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as PredictionResult;
}

export async function analyzeSample() {
  const response = await fetch(`${API_BASE_URL}/api/sample`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as PredictionResult;
}

export function savePredictionResult(result: PredictionResult) {
  if (typeof window === "undefined") {
    return;
  }

  sessionStorage.setItem("cardiovision:lastPrediction", JSON.stringify(result));
}

export function loadPredictionResult() {
  if (typeof window === "undefined") {
    return null;
  }

  const raw = sessionStorage.getItem("cardiovision:lastPrediction");
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as PredictionResult;
  } catch {
    return null;
  }
}
