"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import PillNav from "@/components/PillNav";

const navItems = [
  { href: "/", label: "Home" },
  { href: "#upload", label: "Upload" },
  { href: "#research", label: "Research" },
  { href: "#about", label: "About" },
  { href: "https://github.com", label: "GitHub" },
];

// Mock prediction data
const predictionData = {
  id: "ECG-2026-001",
  timestamp: "August 14, 2026 at 3:42 PM",
  predictedClass: "Myocardial Infarction",
  confidence: 92.4,
  riskLevel: "High",
  probabilities: [
    { condition: "Normal ECG", probability: 2.1 },
    { condition: "Myocardial Infarction", probability: 92.4 },
    { condition: "Cardiac Arrhythmia", probability: 3.2 },
    { condition: "Left Ventricular Hypertrophy", probability: 1.8 },
    { condition: "ST/T Wave Abnormalities", probability: 0.5 },
  ],
  clinicalSummary:
    "The AI model has detected patterns consistent with myocardial infarction with high confidence. Key features include ST-segment elevation and abnormal Q-waves in multiple leads.",
  recommendation:
    "Immediate medical attention recommended. This prediction should be reviewed by a qualified cardiologist. Do not use as sole basis for clinical decisions.",
  inferenceTime: 2.34,
  modelVersion: "CardioViT-v1.2.0",
};

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

function WaveformViewer() {
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
          
          {/* ECG waveform - Lead I */}
          <path
            d="M0 200 L50 200 L60 180 L70 200 L80 220 L90 200 L100 190 L110 200 L150 200 L160 180 L170 200 L180 220 L190 200 L200 190 L210 200 L250 200 L260 180 L270 200 L280 220 L290 200 L300 190 L310 200 L350 200 L360 180 L370 200 L380 220 L390 200 L400 190 L410 200 L450 200 L460 180 L470 200 L480 220 L490 200 L500 190 L510 200 L550 200 L560 180 L570 200 L580 220 L590 200 L600 190 L610 200 L650 200 L660 180 L670 200 L680 220 L690 200 L700 190 L710 200 L750 200 L760 180 L770 200 L780 220 L790 200 L800 200"
            stroke="#0070f3"
            strokeWidth="2"
            fill="none"
          />
        </svg>
      </div>
      <p className="mt-3 text-sm text-body">12-lead ECG - Lead I shown</p>
    </div>
  );
}

function ScalogramView() {
  return (
    <div className="card-elevated rounded-xl border border-hairline p-6">
      <h3
        className="mb-4 text-lg font-semibold tracking-tight text-ink"
        style={{ letterSpacing: "-0.4px" }}
      >
        CWT Scalogram
      </h3>
      <div className="aspect-square w-full overflow-hidden rounded-lg bg-gradient-to-br from-violet via-cyan to-pink">
        {/* Placeholder for actual scalogram image */}
        <div className="flex h-full items-center justify-center">
          <p className="text-sm text-white/70">Scalogram visualization</p>
        </div>
      </div>
      <p className="mt-3 text-sm text-body">
        Continuous Wavelet Transform representation
      </p>
    </div>
  );
}

function GradCAMHeatmap() {
  return (
    <div className="card-elevated rounded-xl border border-hairline p-6">
      <h3
        className="mb-4 text-lg font-semibold tracking-tight text-ink"
        style={{ letterSpacing: "-0.4px" }}
      >
        Attention Heatmap (Grad-CAM)
      </h3>
      <div className="aspect-square w-full overflow-hidden rounded-lg bg-gradient-to-br from-error/20 via-warning/30 to-link/20">
        {/* Placeholder for actual Grad-CAM heatmap */}
        <div className="flex h-full items-center justify-center">
          <p className="text-sm text-body">Explainability visualization</p>
        </div>
      </div>
      <p className="mt-3 text-sm text-body">
        Regions influencing the prediction highlighted
      </p>
    </div>
  );
}

export default function ResultsPage() {
  const [showToast, setShowToast] = useState(false);

  const handleDownloadPDF = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const handleDownloadJSON = () => {
    const dataStr = JSON.stringify(predictionData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${predictionData.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

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
                        {item.probability}%
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
                  ⚠️ Medical Recommendation
                </p>
                <p className="mt-2 text-sm leading-relaxed text-body">
                  {predictionData.recommendation}
                </p>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <WaveformViewer />
            <ScalogramView />
            <GradCAMHeatmap />
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
