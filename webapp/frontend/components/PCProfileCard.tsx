"use client";

import type { ReactNode } from "react";
import type { PCProfile } from "../lib/types";
import { IconArrowDown, IconArrowUp, IconBond, IconDroplet, IconHexagon, IconScale, IconWaves } from "./icons";

interface PCProfileCardProps {
  profile: PCProfile | null;
}

interface Stat {
  label: string;
  value: string;
  icon: ReactNode;
}

export default function PCProfileCard({ profile }: PCProfileCardProps) {
  if (!profile) return null;

  const stats: Stat[] = [
    { label: "Molecular weight", value: `${profile.molecular_weight}`, icon: <IconScale /> },
    { label: "LogP", value: `${profile.logp}`, icon: <IconDroplet /> },
    { label: "TPSA", value: `${profile.tpsa} Å²`, icon: <IconWaves /> },
    { label: "H-bond donors", value: `${profile.h_bond_donors}`, icon: <IconArrowUp /> },
    { label: "H-bond acceptors", value: `${profile.h_bond_acceptors}`, icon: <IconArrowDown /> },
    { label: "Rotatable bonds", value: `${profile.rotatable_bonds}`, icon: <IconBond /> },
    { label: "Aromatic rings", value: `${profile.aromatic_rings}`, icon: <IconHexagon /> },
  ];

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card transition-shadow hover:shadow-card-hover">
      <div className="mb-4 flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-accent" />
        <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-600">
          Physicochemical Profile
        </h3>
      </div>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
        {stats.map((s) => (
          <div
            key={s.label}
            className="group rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 transition-colors hover:border-accent/40"
          >
            <div className="mb-1.5 flex items-center gap-1.5 text-slate-500 transition-colors group-hover:text-accent-dim">
              {s.icon}
            </div>
            <div className="text-[11px] leading-tight text-slate-600">{s.label}</div>
            <div className="mt-0.5 font-mono text-base font-medium text-slate-900">{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
