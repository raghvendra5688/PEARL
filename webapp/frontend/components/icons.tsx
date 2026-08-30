// Minimal, hand-drawn inline SVG icon set -- no icon library dependency. Every icon
// is stroke-based, 20x20, currentColor, so it inherits color from its text-color
// utility class rather than carrying a hardcoded fill.

type IconProps = { className?: string };
const base = { width: 20, height: 20, viewBox: "0 0 20 20", fill: "none", strokeWidth: 1.6 };

export function IconScale({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3v14M5 6h10M5 6l-2.5 5a2.5 2.5 0 0 0 5 0L5 6ZM15 6l-2.5 5a2.5 2.5 0 0 0 5 0L15 6ZM6.5 17h7" />
    </svg>
  );
}

export function IconDroplet({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 2.5c2.8 3.4 5 6.4 5 9.2a5 5 0 0 1-10 0c0-2.8 2.2-5.8 5-9.2Z" />
    </svg>
  );
}

export function IconWaves({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2.5 7.5c1.5-1.5 3-1.5 4.5 0s3 1.5 4.5 0 3-1.5 4.5 0M2.5 12.5c1.5-1.5 3-1.5 4.5 0s3 1.5 4.5 0 3-1.5 4.5 0" />
    </svg>
  );
}

export function IconArrowDown({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3v12M5 11l5 5 5-5" />
    </svg>
  );
}

export function IconArrowUp({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 17V5M5 9l5-5 5 5" />
    </svg>
  );
}

export function IconBond({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="5" cy="15" r="2" />
      <circle cx="15" cy="5" r="2" />
      <path d="M6.8 13.2 13.2 6.8" />
    </svg>
  );
}

export function IconHexagon({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 2.5 17 6.5v7L10 17.5 3 13.5v-7Z" />
    </svg>
  );
}

export function IconCheckCircle({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="7.5" />
      <path d="m6.8 10.2 2.2 2.2 4.2-4.8" />
    </svg>
  );
}

export function IconAlertTriangle({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 3.2 18 16.5H2L10 3.2Z" />
      <path d="M10 8.3v3.6" />
      <circle cx="10" cy="14.3" r="0.15" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconAtom({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="1.4" fill="currentColor" stroke="none" />
      <ellipse cx="10" cy="10" rx="8" ry="3.2" />
      <ellipse cx="10" cy="10" rx="8" ry="3.2" transform="rotate(60 10 10)" />
      <ellipse cx="10" cy="10" rx="8" ry="3.2" transform="rotate(120 10 10)" />
    </svg>
  );
}

export function IconFlask({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2.5h4M8.5 2.5v4.8L4 14.8a1.8 1.8 0 0 0 1.6 2.7h8.8a1.8 1.8 0 0 0 1.6-2.7l-4.5-7.5V2.5" />
      <path d="M6 12.5h8" />
    </svg>
  );
}

export function IconCompass({ className }: IconProps) {
  return (
    <svg {...base} className={className} stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10" cy="10" r="7.5" />
      <path d="m12.6 7.4-1.4 4-4 1.4 1.4-4 4-1.4Z" />
    </svg>
  );
}
