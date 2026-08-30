"use client";

import { motion } from "framer-motion";
import ChemFormula from "./ChemFormula";
import type { Neighbour } from "../lib/types";

interface NeighbourPanelProps {
  neighbours: Neighbour[] | null;
}

export default function NeighbourPanel({ neighbours }: NeighbourPanelProps) {
  if (!neighbours || neighbours.length === 0) return null;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card transition-shadow hover:shadow-card-hover">
      <div className="mb-1 flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-accent" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-600">
          Nearest Known Compounds
        </h3>
      </div>
      <p className="mb-4 text-xs text-slate-500">Closest structural matches in ZINC-250k, by Tanimoto similarity.</p>

      {/* Capped at 3 columns, not 5 -- chemical names (esp. IUPAC-style ones like
          "2-[(2,4-Dichlorophenyl)carbamoyl]phenyl acetate") run long, and a
          narrower card was clipping them well before 2 full lines of text. */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {neighbours.map((n, i) => (
          <motion.div
            key={`${n.smiles}-${i}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.04 }}
            className="flex flex-col rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-3 transition-colors hover:border-accent/40"
          >
            <div className="mb-2 flex items-start justify-between gap-2">
              <span className="font-mono-tabular font-mono text-xs text-accent-dim">
                {(n.similarity * 100).toFixed(0)}%
              </span>
            </div>

            <p className="line-clamp-3 min-h-[3.75rem] text-sm font-medium leading-tight text-slate-900">
              {n.name ?? "Unnamed compound"}
            </p>
            {n.formula && (
              <ChemFormula formula={n.formula} className="mt-1 font-mono text-xs text-slate-500" />
            )}

            <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-accent/15">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${n.similarity * 100}%` }}
                transition={{ duration: 0.6, ease: "easeOut", delay: i * 0.05 }}
                className="h-full rounded-full bg-accent"
              />
            </div>

            <p className="mt-3 truncate font-mono text-[10px] text-slate-500" title={n.smiles}>
              {n.smiles}
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
