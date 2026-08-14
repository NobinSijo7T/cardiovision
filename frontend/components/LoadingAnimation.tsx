"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";

interface LoadingAnimationProps {
  currentStep: number;
}

const steps = [
  "Reading ECG",
  "Cleaning Signal",
  "Generating Scalogram",
  "Running CardioViT",
  "Computing Prediction",
  "Generating Report",
];

export default function LoadingAnimation({ currentStep }: LoadingAnimationProps) {
  const heartbeatRef = useRef<SVGPathElement>(null);
  const pulseRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (heartbeatRef.current) {
      // Animate ECG heartbeat
      gsap.fromTo(
        heartbeatRef.current,
        { strokeDashoffset: 1000 },
        {
          strokeDashoffset: 0,
          duration: 2,
          ease: "none",
          repeat: -1,
        }
      );
    }

    if (pulseRef.current) {
      // Pulse animation
      gsap.to(pulseRef.current, {
        scale: 1.1,
        opacity: 0.5,
        duration: 1,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      });
    }
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center bg-canvas">
      <div className="w-full max-w-2xl space-y-12 px-6">
        {/* Animated ECG Heartbeat */}
        <div className="relative mx-auto h-32 w-full">
          <div
            ref={pulseRef}
            className="absolute inset-0 rounded-xl bg-link/10"
          />
          <svg
            className="h-full w-full"
            viewBox="0 0 800 150"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              ref={heartbeatRef}
              d="M0 75 L200 75 L220 45 L240 105 L260 30 L280 75 L800 75"
              stroke="#0070f3"
              strokeWidth="3"
              strokeDasharray="1000"
              strokeDashoffset="1000"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Progress Steps */}
        <div className="space-y-4">
          <div className="mb-6 text-center">
            <h2
              className="text-2xl font-semibold tracking-tight text-ink"
              style={{ letterSpacing: "-0.4px" }}
            >
              Analyzing ECG
            </h2>
            <p className="mt-2 text-sm text-body">
              This may take a few moments...
            </p>
          </div>

          <div className="space-y-3">
            {steps.map((step, index) => {
              const isCompleted = index < currentStep;
              const isCurrent = index === currentStep;
              const isPending = index > currentStep;

              return (
                <div
                  key={step}
                  className={`card-elevated flex items-center gap-4 rounded-xl border p-4 transition-all ${
                    isCurrent
                      ? "border-link bg-link-soft"
                      : isCompleted
                      ? "border-hairline bg-canvas-elevated"
                      : "border-hairline-soft bg-hairline-soft opacity-50"
                  }`}
                >
                  <div
                    className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
                      isCompleted
                        ? "bg-link text-on-primary"
                        : isCurrent
                        ? "bg-link text-on-primary"
                        : "bg-hairline text-mute"
                    }`}
                  >
                    {isCompleted ? (
                      <svg
                        className="h-5 w-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <span className="text-sm font-semibold">{index + 1}</span>
                    )}
                  </div>
                  <div className="flex-1">
                    <p
                      className={`text-base font-medium ${
                        isPending ? "text-mute" : "text-ink"
                      }`}
                    >
                      {step}
                    </p>
                  </div>
                  {isCurrent && (
                    <div className="flex gap-1">
                      <span className="h-2 w-2 animate-bounce rounded-full bg-link [animation-delay:-0.3s]" />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-link [animation-delay:-0.15s]" />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-link" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="h-2 overflow-hidden rounded-full bg-hairline-soft">
            <div
              className="h-full rounded-full bg-link transition-all duration-500"
              style={{
                width: `${((currentStep + 1) / steps.length) * 100}%`,
              }}
            />
          </div>
          <p className="text-center font-mono text-xs text-mute">
            {Math.round(((currentStep + 1) / steps.length) * 100)}% complete
          </p>
        </div>
      </div>
    </div>
  );
}
