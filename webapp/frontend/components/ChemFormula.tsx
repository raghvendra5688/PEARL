// Renders a molecular formula ("C9H8O4") with real digit subscripts instead of
// plain inline digits, matching how it's written in chemistry -- C9H8O4 -> C₉H₈O₄.
export default function ChemFormula({ formula, className }: { formula: string; className?: string }) {
  const parts = formula.split(/(\d+)/).filter(Boolean);
  return (
    <span className={className}>
      {parts.map((part, i) =>
        /^\d+$/.test(part) ? <sub key={i}>{part}</sub> : <span key={i}>{part}</span>
      )}
    </span>
  );
}
