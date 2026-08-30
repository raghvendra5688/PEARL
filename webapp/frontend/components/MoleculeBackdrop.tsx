// Purely decorative background motifs -- three distinct molecule/ring shapes,
// very light grey, slowly rotating at different speeds/directions so the motion
// reads as ambient rather than as an obvious spinner. pointer-events-none and
// aria-hidden throughout: this carries no information, so it must never intercept
// clicks or be announced to assistive tech.

function RingMolecule({ className }: { className?: string }) {
  // A benzene-style ring with two substituent branches -- the simplest, most
  // immediately recognizable "molecule" silhouette.
  return (
    <svg viewBox="0 0 200 200" className={className} fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M100 30 168 65 168 135 100 170 32 135 32 65Z" />
      <circle cx="100" cy="30" r="5" fill="currentColor" stroke="none" />
      <circle cx="168" cy="65" r="5" fill="currentColor" stroke="none" />
      <circle cx="168" cy="135" r="5" fill="currentColor" stroke="none" />
      <circle cx="100" cy="170" r="5" fill="currentColor" stroke="none" />
      <circle cx="32" cy="135" r="5" fill="currentColor" stroke="none" />
      <circle cx="32" cy="65" r="5" fill="currentColor" stroke="none" />
      <path d="M168 65 210 45M32 135 -10 155" />
      <circle cx="210" cy="45" r="5" fill="currentColor" stroke="none" />
      <circle cx="-10" cy="155" r="5" fill="currentColor" stroke="none" />
    </svg>
  );
}

function BicyclicMolecule({ className }: { className?: string }) {
  // Two fused rings with a short branch -- more structurally complex, for variety.
  return (
    <svg viewBox="0 0 260 160" className={className} fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M70 20 120 45 120 95 70 120 20 95 20 45Z" />
      <path d="M120 45 170 20 220 45 220 95 170 120 120 95" />
      <path d="M170 20 170 -20" />
      {[
        [70, 20], [120, 45], [120, 95], [70, 120], [20, 95], [20, 45],
        [170, 20], [220, 45], [220, 95], [170, 120], [170, -20],
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="4.5" fill="currentColor" stroke="none" />
      ))}
    </svg>
  );
}

function ChainMolecule({ className }: { className?: string }) {
  // A simple zigzag chain -- reads as a fatty-acid-like backbone, breaks up the
  // repetition of two ring shapes.
  return (
    <svg viewBox="0 0 220 80" className={className} fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10 60 50 20 90 60 130 20 170 60 210 20" />
      {[[10, 60], [50, 20], [90, 60], [130, 20], [170, 60], [210, 20]].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="4.5" fill="currentColor" stroke="none" />
      ))}
    </svg>
  );
}

export default function MoleculeBackdrop() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-30 overflow-hidden text-slate-300/70">
      <RingMolecule className="animate-spin-slow absolute -left-24 -top-16 h-[26rem] w-[26rem]" />
      <BicyclicMolecule className="animate-spin-slow-reverse absolute -right-32 top-1/4 h-[22rem] w-[36rem]" />
      <ChainMolecule className="animate-spin-slower absolute -bottom-10 left-1/3 h-[18rem] w-[28rem] opacity-70" />
    </div>
  );
}
