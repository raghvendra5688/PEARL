"use client";

import type { ResolveCandidate } from "../lib/types";

interface DisambiguationPickerProps {
  candidates: ResolveCandidate[];
  onPick: (smiles: string) => void;
  onCancel: () => void;
}

export default function DisambiguationPicker({ candidates, onPick, onCancel }: DisambiguationPickerProps) {
  return (
    <div className="w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-5 shadow-card">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-sm text-slate-700">
          That formula matches {candidates.length} known compounds. Which one did you mean?
        </p>
        <button onClick={onCancel} className="text-xs text-slate-500 hover:text-slate-700">
          Cancel
        </button>
      </div>
      <ul className="max-h-72 space-y-1 overflow-y-auto">
        {candidates.map((c) => (
          <li key={c.cid}>
            <button
              onClick={() => onPick(c.smiles)}
              className="flex w-full flex-col rounded-xl border border-transparent px-3 py-2.5 text-left transition-colors hover:border-accent/40 hover:bg-accent/5"
            >
              <span className="text-sm font-medium text-slate-900">{c.name ?? "Unnamed compound"}</span>
              <span className="font-mono text-xs text-slate-500">{c.smiles}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
