import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import MoleculeBackdrop from "../components/MoleculeBackdrop";
import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "PEARL Experience",
  description:
    "Predict molecular properties -- enzyme inhibition, cardiotoxicity, permeability, and more -- with PEARL's benchmarked models.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen bg-white font-sans text-slate-900 antialiased">
        {/* Rotating molecule silhouettes, furthest back. */}
        <MoleculeBackdrop />

        {/* Faint background tints, not a bold aurora -- strong glows read as
            "premium depth" on near-black but as a color cast on white. */}
        <div className="pointer-events-none fixed inset-0 -z-20 overflow-hidden">
          <div className="absolute -top-40 left-1/4 h-[32rem] w-[32rem] rounded-full bg-accent/[0.07] blur-[120px]" />
          <div className="absolute -top-20 right-1/4 h-[28rem] w-[28rem] rounded-full bg-violet-400/[0.05] blur-[120px]" />
        </div>

        <div className="fixed inset-0 -z-10 bg-grid-pattern bg-grid opacity-60 [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,black_40%,transparent_100%)]" />
        {children}
      </body>
    </html>
  );
}
