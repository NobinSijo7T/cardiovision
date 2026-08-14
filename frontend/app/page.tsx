"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import PillNav from "@/components/PillNav";

const navItems = [
  { href: "/", label: "Home" },
  { href: "#upload", label: "Upload" },
  { href: "#research", label: "Research" },
  { href: "#about", label: "About" },
  { href: "https://github.com", label: "GitHub" },
];

const supportedConditions = [
  "Normal ECG",
  "Myocardial Infarction",
  "Cardiac Arrhythmia",
  "Left Ventricular Hypertrophy",
  "ST/T Wave Abnormalities",
];

const pipelineSteps = [
  { step: "Upload ECG", detail: "WFDB (.hea/.dat) or image formats" },
  { step: "Signal Preprocessing", detail: "Validate and clean signal" },
  { step: "Continuous Wavelet Transform", detail: "Generate scalogram" },
  { step: "Vision Transformer", detail: "CardioViT inference" },
  { step: "Prediction", detail: "5-class probability distribution" },
  { step: "Clinical Report", detail: "Downloadable results + Grad-CAM" },
];

const modelPerformance = [
  { metric: "Accuracy", value: "91.2%" },
  { metric: "Macro F1", value: "88.7%" },
  { metric: "ROC-AUC", value: "0.94" },
  { metric: "Dataset", value: "PTB-XL" },
];





function UploadCard() {
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const router = useRouter();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFileName(e.dataTransfer.files[0].name);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFileName(e.target.files[0].name);
    }
  };

  const handleAnalyze = () => {
    router.push("/analyzing");
  };

  const handleUseSample = () => {
    setFileName("sample-ecg-001.dat");
  };

  return (
    <div
      className={`card-elevated rounded-xl border-2 border-dashed p-12 text-center transition-all ${
        dragActive ? "upload-drag-active" : ""
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <div className="mx-auto max-w-md space-y-6">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-hairline-soft">
          <svg className="h-8 w-8 text-body" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <div>
          <h3 className="text-xl font-semibold tracking-tight text-ink">
            {fileName ? fileName : "Drop your ECG file here"}
          </h3>
          <p className="mt-2 text-sm text-body">
            Supports .hea, .dat, .mat, .csv, .png, .jpg, .jpeg formats
          </p>
        </div>
        <div className="flex items-center justify-center gap-3">
          <label htmlFor="file-upload" className="cursor-pointer">
            <span className="inline-flex h-10 items-center rounded-full bg-ink px-6 text-base font-medium text-on-primary transition-opacity hover:opacity-90">
              Choose File
            </span>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept=".hea,.dat,.mat,.csv,.png,.jpg,.jpeg"
              onChange={handleChange}
            />
          </label>
          <button 
            onClick={handleUseSample}
            className="inline-flex h-10 items-center rounded-full border border-hairline bg-canvas-elevated px-6 text-base font-medium text-ink transition-colors hover:bg-hairline-soft"
          >
            Use Sample ECG
          </button>
        </div>
        {fileName && (
          <button 
            onClick={handleAnalyze}
            className="mt-4 inline-flex h-10 items-center rounded-full bg-link px-6 text-base font-medium text-on-primary transition-opacity hover:opacity-90"
          >
            Analyze ECG
          </button>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="relative min-h-screen bg-canvas">
      <PillNav
        logo="/1.png"
        logoAlt="Cardiovision"
        items={navItems}
        baseColor="#171717"
        pillColor="#ffffff"
        hoveredPillTextColor="#ffffff"
        pillTextColor="#171717"
      />
      
      {/* Hero Section with Mesh Gradient */}
      <section id="upload" className="hero-mesh-gradient border-b border-hairline pt-32 pb-24 lg:pt-40 lg:pb-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-3xl text-center">
            <div className="flex items-center justify-center gap-4">
              <img src="/1.png" alt="CardioVision Logo" className="h-24 w-24 lg:h-32 lg:w-32" />
              <h1 className="text-5xl font-semibold leading-tight tracking-tight text-ink lg:text-6xl" style={{ letterSpacing: '-2.4px' }}>
                CardioVision
              </h1>
            </div>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-body lg:text-lg">
              Upload an ECG record or cardiac image to receive an AI-powered diagnostic assessment in seconds.
            </p>
          </div>
          <div className="mx-auto mt-12 max-w-4xl">
            <UploadCard />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="about" className="border-b border-hairline py-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <p className="font-mono text-xs font-medium uppercase tracking-wider text-mute">How the AI Works</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-ink lg:text-4xl" style={{ letterSpacing: '-1.28px' }}>
              From signal to diagnosis in six steps
            </h2>
          </div>
          <div className="mx-auto mt-16 grid max-w-5xl gap-6 md:grid-cols-2 lg:grid-cols-3">
            {pipelineSteps.map((item, index) => (
              <div key={index} className="card-elevated rounded-xl p-6">
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-md bg-hairline-soft text-base font-semibold text-ink">
                  {index + 1}
                </div>
                <h3 className="text-lg font-semibold tracking-tight text-ink" style={{ letterSpacing: '-0.4px' }}>
                  {item.step}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-body">{item.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Conditions Section */}
      <section className="border-b border-hairline py-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="grid gap-16 lg:grid-cols-2 lg:gap-24">
            <div>
              <p className="font-mono text-xs font-medium uppercase tracking-wider text-mute">Supported Conditions</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight text-ink lg:text-4xl" style={{ letterSpacing: '-1.28px' }}>
              Five cardiac diagnoses detected by CardioViT
              </h2>
              <p className="mt-6 text-base leading-relaxed text-body">
                The model was trained on the PTB-XL dataset with 21,837 ECG recordings and can identify five major cardiovascular conditions with high accuracy.
              </p>
            </div>
            <div className="space-y-3">
              {supportedConditions.map((condition, index) => (
                <div key={index} className="card-elevated rounded-xl border border-hairline p-5">
                  <div className="flex items-center justify-between">
                    <span className="text-base font-medium text-ink">{condition}</span>
                    <span className="font-mono text-sm text-mute">Class {index + 1}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Model Performance Section */}
      <section id="research" className="border-b border-hairline py-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <p className="font-mono text-xs font-medium uppercase tracking-wider text-mute">Model Performance</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-ink lg:text-4xl" style={{ letterSpacing: '-1.28px' }}>
              Clinical-grade accuracy on PTB-XL
            </h2>
          </div>
          <div className="mx-auto mt-16 grid max-w-4xl gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {modelPerformance.map((item, index) => (
              <div key={index} className="card-elevated rounded-lg p-6 text-center">
                <p className="text-sm text-body">{item.metric}</p>
                <p className="mt-3 text-3xl font-semibold tracking-tight text-ink">{item.value}</p>
              </div>
            ))}
          </div>
          <div className="mx-auto mt-12 max-w-3xl card-elevated rounded-xl p-8">
            <h3 className="text-lg font-semibold tracking-tight text-ink" style={{ letterSpacing: '-0.4px' }}>
              About the Model
            </h3>
            <p className="mt-4 text-sm leading-relaxed text-body">
              CardioViT is a Vision Transformer architecture trained on continuous wavelet transform (CWT) scalograms generated from 12-lead ECG signals. The model uses Grad-CAM for explainability, highlighting the regions of the scalogram that most influence each prediction.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <span className="inline-flex items-center rounded-full border border-hairline bg-canvas px-4 py-1.5 text-sm font-medium text-ink">
                Vision Transformer
              </span>
              <span className="inline-flex items-center rounded-full border border-hairline bg-canvas px-4 py-1.5 text-sm font-medium text-ink">
                PTB-XL Dataset
              </span>
              <span className="inline-flex items-center rounded-full border border-hairline bg-canvas px-4 py-1.5 text-sm font-medium text-ink">
                Grad-CAM Explainability
              </span>
            </div>
          </div>
        </div>
      </section>


    </div>
  );
}
