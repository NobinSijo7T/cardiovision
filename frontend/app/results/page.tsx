"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import PillNav from "@/components/PillNav";
import { loadPredictionResult, type PredictionResult, type SignalPoint } from "@/lib/api";

const navItems = [
  { href: "/", label: "Home" },
  { href: "#upload", label: "Upload" },
  { href: "#research", label: "Research" },
  { href: "#about", label: "About" },
  { href: "https://github.com", label: "GitHub" },
];

function ConfidenceGauge({ confidence }: { confidence: number }) {
  const gaugeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (gaugeRef.current) {
      gsap.fromTo(
        gaugeRef.current,
        { rotate: -90 },
        {
          rotate: -90 + (confidence / 100) * 180,
          duration: 1.5,
          ease: "power2.out",
        }
      );
    }
  }, [confidence]);

  return (
    <div className="relative mx-auto h-48 w-48">
      {/* Background arc */}
      <svg className="h-full w-full -rotate-90" viewBox="0 0 200 200">
        <circle
          cx="100"
          cy="100"
          r="80"
          fill="none"
          stroke="#ebebeb"
          strokeWidth="16"
          strokeDasharray="251.2 251.2"
          strokeLinecap="round"
        />
        <circle
          cx="100"
          cy="100"
          r="80"
          fill="none"
          stroke="#0070f3"
          strokeWidth="16"
          strokeDasharray={`${(confidence / 100) * 251.2} 251.2`}
          strokeLinecap="round"
          className="transition-all duration-1000"
        />
      </svg>
      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <p className="text-4xl font-semibold text-ink">{confidence}%</p>
        <p className="text-sm text-mute">Confidence</p>
      </div>
    </div>
  );
}

function WaveformViewer({ points }: { points: SignalPoint[] }) {
  const polyline = useMemo(() => {
    if (points.length === 0) {
      return "";
    }

    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    const yRange = maxY - minY || 1;
    const xRange = Math.max(points.length - 1, 1);

    return points
      .map((point, index) => {
        const x = (index / xRange) * 800;
        const y = 360 - ((point.y - minY) / yRange) * 320;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [points]);

  return (
    <div className="card-elevated rounded-xl border border-hairline p-6">
      <h3
        className="mb-4 text-lg font-semibold tracking-tight text-ink"
        style={{ letterSpacing: "-0.4px" }}
      >
        ECG Waveform
      </h3>
      <div className="aspect-[2/1] w-full rounded-lg bg-hairline-soft">
        <svg
          className="h-full w-full"
          viewBox="0 0 800 400"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Grid lines */}
          <defs>
            <pattern
              id="grid"
              width="20"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 20 0 L 0 0 0 20"
                fill="none"
                stroke="#e0e0e0"
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect width="800" height="400" fill="url(#grid)" />
          
          {polyline ? (
            <polyline points={polyline} stroke="#0070f3" strokeWidth="2" fill="none" />
          ) : null}
        </svg>
      </div>
      <p className="mt-3 text-sm text-body">Filtered ECG preview - primary lead</p>
    </div>
  );
}

function ScalogramView({ image }: { image: string }) {
  return (
    <div className="card-elevated rounded-xl border border-hairline p-6">
      <h3
        className="mb-4 text-lg font-semibold tracking-tight text-ink"
        style={{ letterSpacing: "-0.4px" }}
      >
        CWT Scalogram
      </h3>
      <img src={image} alt="CWT scalogram" className="aspect-square w-full rounded-lg object-cover" />
      <p className="mt-3 text-sm text-body">
        Continuous Wavelet Transform representation
      </p>
    </div>
  );
}

function GradCAMHeatmap({ image }: { image: string | null }) {
  return (
    <div className="card-elevated rounded-xl border border-hairline p-6">
      <h3
        className="mb-4 text-lg font-semibold tracking-tight text-ink"
        style={{ letterSpacing: "-0.4px" }}
      >
        Attention Heatmap (Grad-CAM)
      </h3>
      {image ? (
        <img src={image} alt="Grad-CAM attention heatmap" className="aspect-square w-full rounded-lg object-cover" />
      ) : (
        <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-hairline-soft">
          <p className="text-sm text-body">Grad-CAM unavailable for this result</p>
        </div>
      )}
      <p className="mt-3 text-sm text-body">
        Regions influencing the prediction highlighted
      </p>
    </div>
  );
}

export default function ResultsPage() {
  const [showToast, setShowToast] = useState(false);
  const [predictionData] = useState<PredictionResult | null>(() => loadPredictionResult());

  const handleDownloadPDF = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const handleDownloadJSON = () => {
    if (!predictionData) {
      return;
    }

    const dataStr = JSON.stringify(predictionData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${predictionData.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!predictionData) {
    return (
      <div className="relative min-h-screen bg-canvas">
        <PillNav
          logo="/logo.svg"
          logoAlt="Cardiovision"
          items={navItems}
          baseColor="#171717"
          pillColor="#ffffff"
          hoveredPillTextColor="#ffffff"
          pillTextColor="#171717"
        />
        <div className="mx-auto flex min-h-screen max-w-2xl flex-col items-center justify-center px-6 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-ink">No analysis result found</h1>
          <p className="mt-4 text-sm leading-relaxed text-body">
            Upload a matching WFDB .hea and .dat record, or run the sample ECG analysis from the home page.
          </p>
          <Link
            href="/"
            className="mt-8 inline-flex h-10 items-center rounded-full bg-ink px-6 text-base font-medium text-on-primary transition-opacity hover:opacity-90"
          >
            Back to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-canvas">
      <PillNav
        logo="/logo.svg"
        logoAlt="Cardiovision"
        items={navItems}
        baseColor="#171717"
        pillColor="#ffffff"
        hoveredPillTextColor="#ffffff"
        pillTextColor="#171717"
      />

      <div className="mx-auto max-w-7xl px-6 pt-32 pb-16 lg:px-8">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-mono text-xs font-medium uppercase tracking-wider text-mute">
              Analysis Result
            </p>
            <h1
              className="mt-2 text-3xl font-semibold tracking-tight text-ink lg:text-4xl"
              style={{ letterSpacing: "-1.28px" }}
            >
              ECG Diagnostic Assessment
            </h1>
            <p className="mt-2 text-sm text-body">
              {predictionData.id} • {predictionData.timestamp}
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex h-10 items-center rounded-full border border-hairline bg-canvas-elevated px-6 text-base font-medium text-ink transition-colors hover:bg-hairline-soft"
          >
            ← Back to Home
          </Link>
        </div>

        {/* Primary Result Cards */}
        <div className="mb-8 grid gap-6 md:grid-cols-3">
          <div className="card-elevated rounded-xl border border-hairline p-6">
            <p className="text-sm text-mute">Predicted Condition</p>
            <p className="mt-3 text-2xl font-semibold text-ink">
              {predictionData.predictedClass}
            </p>
          </div>
          <div className="card-elevated rounded-xl border border-hairline p-6">
            <p className="text-sm text-mute">Confidence Score</p>
            <p className="mt-3 text-2xl font-semibold text-ink">
              {predictionData.confidence}%
            </p>
          </div>
          <div className="card-elevated rounded-xl border border-hairline p-6">
            <p className="text-sm text-mute">Risk Level</p>
            <p className="mt-3 text-2xl font-semibold text-error">
              {predictionData.riskLevel}
            </p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left Column */}
          <div className="space-y-8">
            {/* Confidence Gauge */}
            <div className="card-elevated rounded-xl border border-hairline p-8">
              <h3
                className="mb-6 text-center text-lg font-semibold tracking-tight text-ink"
                style={{ letterSpacing: "-0.4px" }}
              >
                Model Confidence
              </h3>
              <ConfidenceGauge confidence={predictionData.confidence} />
            </div>

            {/* Probability Distribution */}
            <div className="card-elevated rounded-xl border border-hairline p-6">
              <h3
                className="mb-4 text-lg font-semibold tracking-tight text-ink"
                style={{ letterSpacing: "-0.4px" }}
              >
                Probability Distribution
              </h3>
              <div className="space-y-4">
                {predictionData.probabilities.map((item) => (
                  <div key={item.condition}>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="text-body">{item.condition}</span>
                      <span className="font-mono text-ink">
                        {item.probability.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-hairline-soft">
                      <div
                        className="h-full rounded-full bg-link transition-all duration-500"
                        style={{ width: `${item.probability}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Clinical Summary */}
            <div className="card-elevated rounded-xl border border-hairline p-6">
              <h3
                className="mb-4 text-lg font-semibold tracking-tight text-ink"
                style={{ letterSpacing: "-0.4px" }}
              >
                Clinical Summary
              </h3>
              <p className="mb-4 text-sm leading-relaxed text-body">
                {predictionData.clinicalSummary}
              </p>
              <div className="rounded-lg border border-warning/20 bg-warning-soft p-4">
                <p className="text-sm font-medium text-warning-deep">
                  Medical Recommendation
                </p>
                <p className="mt-2 text-sm leading-relaxed text-body">
                  {predictionData.recommendation}
                </p>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <WaveformViewer points={predictionData.signalPreview} />
            <ScalogramView image={predictionData.scalogramImage} />
            <GradCAMHeatmap image={predictionData.gradcamImage} />
          </div>
        </div>

        {/* Actions */}
        <div className="mt-8 flex flex-wrap gap-4">
          <button
            onClick={handleDownloadPDF}
            className="inline-flex h-12 items-center rounded-full bg-ink px-8 text-base font-medium text-on-primary transition-opacity hover:opacity-90"
          >
            Download PDF Report
          </button>
          <button
            onClick={handleDownloadJSON}
            className="inline-flex h-12 items-center rounded-full border border-hairline bg-canvas-elevated px-8 text-base font-medium text-ink transition-colors hover:bg-hairline-soft"
          >
            Download JSON Data
          </button>
          <Link
            href="/"
            className="inline-flex h-12 items-center rounded-full border border-hairline bg-canvas-elevated px-8 text-base font-medium text-ink transition-colors hover:bg-hairline-soft"
          >
            Analyze Another ECG
          </Link>
        </div>

        {/* Metadata Footer */}
        <div className="mt-12 flex flex-wrap gap-6 border-t border-hairline pt-6 text-sm text-mute">
          <div>
            <span className="font-mono">Inference Time:</span> {predictionData.inferenceTime}s
          </div>
          <div>
            <span className="font-mono">Model:</span> {predictionData.modelVersion}
          </div>
          <div>
            <span className="font-mono">R-Peaks:</span> {predictionData.rPeakCount}
          </div>
          <div>
            <span className="font-mono">Heart Rate:</span>{" "}
            {predictionData.heartRateBpm ? `${predictionData.heartRateBpm} BPM` : "Unavailable"}
          </div>
          <div>
            <span className="font-mono">Dataset:</span> PTB-XL (21,837 records)
          </div>
        </div>
      </div>

      {/* Toast Notification */}
      {showToast && (
        <div className="fixed bottom-6 right-6 card-floating rounded-xl border border-hairline px-6 py-4">
          <p className="text-sm font-medium text-ink">
            PDF report generation is not yet implemented
          </p>
        </div>
      )}
    </div>
  );
}
