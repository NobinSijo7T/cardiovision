"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingAnimation from "@/components/LoadingAnimation";

export default function AnalyzingPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const router = useRouter();

  useEffect(() => {
    // Simulate the analysis pipeline
    const stepDuration = 1500; // 1.5 seconds per step
    const totalSteps = 6;

    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < totalSteps - 1) {
          return prev + 1;
        } else {
          clearInterval(timer);
          // Navigate to results page after completion
          setTimeout(() => {
            router.push("/results");
          }, 1000);
          return prev;
        }
      });
    }, stepDuration);

    return () => clearInterval(timer);
  }, [router]);

  return <LoadingAnimation currentStep={currentStep} />;
}
