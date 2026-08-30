"use client";

import { useState, type FormEvent } from "react";

interface MoleculeInputProps {
  onSubmit: (inputType: "smiles" | "formula", value: string) => void;
  isLoading: boolean;
}

export default function MoleculeInput({ onSubmit, isLoading }: MoleculeInputProps) {
  const [mode, setMode] = useState<"smiles" | "formula">("smiles");
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    onSubmit(mode, trimmed);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-5 shadow-card"
    >
      <div className="mb-3 flex gap-2">
        {(["smiles", "formula"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded-full border px-4 py-1.5 font-mono text-xs font-medium uppercase tracking-widest transition-colors ${
              mode === m
                ? "border-accent bg-accent/10 text-accent-dim"
                : "border-slate-300 text-slate-600 hover:border-slate-400 hover:text-slate-800"
            }`}
          >
            {m === "smiles" ? "SMILES" : "Formula"}
          </button>
        ))}
      </div>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={mode === "smiles" ? "e.g. CC(=O)Oc1ccccc1C(=O)O" : "e.g. C9H8O4"}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          className="flex-1 rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 font-mono text-sm text-slate-900 placeholder:text-slate-400 transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
        />
        <button
          type="submit"
          disabled={isLoading || !value.trim()}
          className="rounded-xl bg-accent px-6 py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-accent-bright disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isLoading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
    </form>
  );
}
