"use client";

import Script from "next/script";
import { useEffect, useRef, useState } from "react";

// 3Dmol.js attaches itself to window.$3Dmol -- loaded via the official CDN build
// rather than an npm wrapper, which is the library's own documented integration
// pattern and keeps this component framework-agnostic about the underlying API.
declare global {
  interface Window {
    $3Dmol?: {
      createViewer: (element: HTMLElement, config?: Record<string, unknown>) => Any3DmolViewer;
    };
  }
}

interface Any3DmolViewer {
  addModel: (data: string, format: string) => unknown;
  setStyle: (sel: Record<string, unknown>, style: Record<string, unknown>) => void;
  zoomTo: () => void;
  render: () => void;
  spin: (axis: string, speed: number) => void;
  resize: () => void;
}

interface MoleculeViewer3DProps {
  molBlock: string | null;
}

// Mid-slate rather than white/near-white: Jmol's element colors include a fair
// amount of light gray (carbon) and would wash out against a very light canvas.
// This keeps atom colors readable while still sitting comfortably in a light page.
const VIEWER_BACKGROUND = "#e2e8f0";

export default function MoleculeViewer3D({ molBlock }: MoleculeViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Any3DmolViewer | null>(null);
  const [scriptReady, setScriptReady] = useState(false);

  useEffect(() => {
    if (!scriptReady || !molBlock || !containerRef.current || !window.$3Dmol) return;

    const viewer = window.$3Dmol.createViewer(containerRef.current, {
      backgroundColor: VIEWER_BACKGROUND,
    });
    viewerRef.current = viewer;

    viewer.addModel(molBlock, "mol");
    viewer.setStyle(
      {},
      {
        stick: { radius: 0.15, colorscheme: "Jmol" },
        sphere: { scale: 0.28, colorscheme: "Jmol" },
      }
    );
    viewer.zoomTo();
    viewer.spin("y", 0.4);
    viewer.render();

    const handleResize = () => viewerRef.current?.resize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [scriptReady, molBlock]);

  return (
    <>
      <Script
        src="https://3Dmol.org/build/3Dmol-min.js"
        strategy="afterInteractive"
        onReady={() => setScriptReady(true)}
      />
      <div
        className="relative h-72 w-full overflow-hidden rounded-2xl border border-slate-200 shadow-card transition-shadow hover:shadow-card-hover sm:h-96"
        style={{ backgroundColor: VIEWER_BACKGROUND }}
      >
        <div ref={containerRef} className="h-full w-full" />

        {/* Vignette overlay, not a change to the WebGL canvas itself -- 3Dmol's
            own backgroundColor stays solid/opaque (reliable, proven), this just
            adds depth around the edges without touching canvas rendering. */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background: "radial-gradient(ellipse at center, transparent 55%, rgba(71,85,105,0.18) 100%)",
          }}
        />

        <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-600">3D Structure</span>
        </div>
        {molBlock && (
          <div className="pointer-events-none absolute bottom-4 right-4 rounded-full border border-slate-300 bg-white/85 px-2.5 py-1 text-[10px] font-medium text-slate-600">
            Drag to rotate · Scroll to zoom
          </div>
        )}

        {!molBlock && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-600">
            No 3D structure available
          </div>
        )}
      </div>
    </>
  );
}
