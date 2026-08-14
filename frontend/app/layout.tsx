import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Geist } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

// Using Inter as fallback for Geist Sans (Geist Sans can be added via CDN or local files)
const geist = Geist({subsets:['latin'],variable:'--font-sans'});

// Using JetBrains Mono as fallback for Geist Mono
const jetBrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Cardiovision",
  description:
    "Upload an ECG record or cardiac image to receive an AI-powered diagnostic assessment in seconds.",
  icons: {
    icon: [
      { url: "/favicon/favicon-96x96.png", sizes: "96x96", type: "image/png" },
      { url: "/favicon/favicon.svg", type: "image/svg+xml" },
    ],
    shortcut: "/favicon/favicon.ico",
    apple: "/favicon/apple-touch-icon.png",
  },
  manifest: "/favicon/site.webmanifest",
  appleWebApp: {
    title: "CardioVision",
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={cn("antialiased", jetBrainsMono.variable, "font-sans", geist.variable)}
    >
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
