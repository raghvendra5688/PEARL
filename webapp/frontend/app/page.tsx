"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";

import { getProfile, resolveMolecule, streamPredictions } from "../lib/api";
import { DATASET_CATALOG } from "../lib/datasets";
import type { ProfileResponse, ResolveCandidate } from "../lib/types";

import ChemFormula from "../components/ChemFormula";
import DatasetResultCard, { type ResultCardState } from "../components/DatasetResultCard";
import DisambiguationPicker from "../components/DisambiguationPicker";
import { IconAtom } from "../components/icons";
import MoleculeInput from "../components/MoleculeInput";
import MoleculeViewer3D from "../components/MoleculeViewer3D";
import NeighbourPanel from "../components/NeighbourPanel";
import PCProfileCard from "../components/PCProfileCard";
import PropertySelector from "../components/PropertySelector";

function SectionLabel({ text }: { text: string }) {
  return (
    <div className="mb-5 flex items-center gap-3">
      <span className="h-1.5 w-1.5 rounded-full bg-accent" />
      <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-600">{text}</h2>
      <span className="h-px flex-1 bg-slate-200" />
    </div>
  );
}

function SkeletonLine({ className }: { className?: string }) {
  return <div className={`animate-pulse rounded-full bg-slate-200 ${className ?? ""}`} />;
}

export default function Home() {
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ResolveCandidate[] | null>(null);

  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [results, setResults] = useState<Record<string, ResultCardState>>({});

  async function loadProfile(smiles: string) {
    setProfileLoading(true);
    setResolveError(null);
    setResults({});
    try {
      const p = await getProfile(smiles);
      setProfile(p);
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : "Could not build a profile for that molecule.");
      setProfile(null);
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleInputSubmit(inputType: "smiles" | "formula", value: string) {
    setResolving(true);
    setResolveError(null);
    setCandidates(null);
    setProfile(null);

    try {
      const res = await resolveMolecule(inputType, value);
      if (res.status === "disambiguation" && res.candidates) {
        setCandidates(res.candidates);
      } else if (res.status === "resolved" && res.smiles) {
        await loadProfile(res.smiles);
      }
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : "Could not resolve that input.");
    } finally {
      setResolving(false);
    }
  }

  async function handlePredict() {
    if (!profile || selectedDatasets.length === 0) return;
    setPredicting(true);

    const pending: Record<string, ResultCardState> = {};
    for (const key of selectedDatasets) {
      const entry = DATASET_CATALOG.find((d) => d.key === key);
      pending[key] = { status: "pending", displayName: entry?.displayName ?? key };
    }
    setResults(pending);

    try {
      for await (const event of streamPredictions(profile.smiles, selectedDatasets)) {
        if (event.type === "result") {
          setResults((prev) => ({ ...prev, [event.data.dataset]: { status: "done", result: event.data } }));
        } else if (event.type === "error") {
          const entry = DATASET_CATALOG.find((d) => d.key === event.data.dataset);
          setResults((prev) => ({
            ...prev,
            [event.data.dataset]: {
              status: "error",
              displayName: entry?.displayName ?? event.data.dataset,
              message: event.data.error,
            },
          }));
        }
      }
    } catch {
      // Stream itself failed to start or was interrupted mid-way -- whatever cards
      // are still "pending" get caught by the sweep below rather than hanging on
      // "Computing..." forever. Kept generic here per ARCHITECTURE.md #9.
    } finally {
      // Defensive sweep: if the connection dropped (a proxy, a network blip) before
      // every selected dataset got a terminal event, don't leave those cards stuck
      // mid-spinner indefinitely -- surface it as a retryable error instead.
      setResults((prev) => {
        const next = { ...prev };
        for (const key of selectedDatasets) {
          if (next[key]?.status === "pending") {
            const entry = DATASET_CATALOG.find((d) => d.key === key);
            next[key] = {
              status: "error",
              displayName: entry?.displayName ?? key,
              message: "No response received -- try predicting again.",
            };
          }
        }
        return next;
      });
      setPredicting(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col items-center px-6 pb-24">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="py-6"
      >
        <IconAtom className="h-7 w-7 text-accent-dim" />
      </motion.div>

      <header className="mb-12 mt-2 text-center">
        {/* Hero treatment for the paper's title: a per-letter reveal into a
            teal-to-violet gradient, then a slow shimmer sweep -- pearl luster. */}
        <h1
          aria-label="PEARL"
          className="flex justify-center text-6xl font-bold tracking-tight sm:text-7xl"
        >
          {"PEARL".split("").map((letter, i) => (
            <motion.span
              key={i}
              aria-hidden="true"
              initial={{ opacity: 0, y: 36, rotateX: -70 }}
              animate={{ opacity: 1, y: 0, rotateX: 0 }}
              transition={{ duration: 0.7, delay: 0.15 + i * 0.09, ease: [0.16, 1, 0.3, 1] }}
              style={{ transformPerspective: 500 }}
              className="inline-block animate-shimmer bg-[length:200%_auto] bg-gradient-to-r from-accent-dim via-violet-400 via-50% to-accent-dim bg-clip-text text-transparent [text-shadow:0_0_50px_rgba(45,212,191,0.3)]"
            >
              {letter}
            </motion.span>
          ))}
        </h1>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.85, ease: "easeOut" }}
          className="mt-2 font-mono text-xs uppercase tracking-[0.35em] text-slate-500"
        >
          Molecular Property Prediction
        </motion.p>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1, ease: "easeOut" }}
          className="mx-auto mt-4 max-w-xl text-sm leading-relaxed text-slate-600"
        >
          Enter a molecule as SMILES or a chemical formula. Get an instant structural profile, then predict
          properties with PEARL&apos;s benchmarked models.
        </motion.p>
      </header>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4, ease: "easeOut" }}
        className="w-full max-w-2xl"
      >
        <MoleculeInput onSubmit={handleInputSubmit} isLoading={resolving} />
      </motion.div>

      {resolveError && <p className="mt-4 text-sm text-rose-600">{resolveError}</p>}

      <AnimatePresence>
        {candidates && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-6 w-full"
          >
            <DisambiguationPicker
              candidates={candidates}
              onPick={(smiles) => {
                setCandidates(null);
                loadProfile(smiles);
              }}
              onCancel={() => setCandidates(null)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {profileLoading && (
        <div className="mt-12 w-full max-w-5xl space-y-6">
          <div className="flex flex-col items-center gap-2">
            <SkeletonLine className="h-6 w-40" />
            <SkeletonLine className="h-3 w-56" />
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <SkeletonLine className="h-72 w-full rounded-2xl sm:h-96" />
            <div className="space-y-6">
              <SkeletonLine className="h-40 w-full rounded-2xl" />
              <SkeletonLine className="h-40 w-full rounded-2xl" />
            </div>
          </div>
        </div>
      )}

      <AnimatePresence>
        {profile && !profileLoading && (
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-12 w-full space-y-10"
          >
            <div className="text-center">
              {profile.formula && (
                <ChemFormula
                  formula={profile.formula}
                  className="font-mono text-2xl font-medium text-slate-900 sm:text-3xl"
                />
              )}
              <p className="mt-1 text-sm text-slate-600">{profile.name ?? "Unnamed compound"}</p>
              <p className="mx-auto mt-2 max-w-xl truncate font-mono text-xs text-slate-500">{profile.smiles}</p>
            </div>

            <div className="space-y-6">
              <SectionLabel text="Instant Structural Profile" />
              {/* 3D viewer and PC profile are close enough in natural height to sit
                  side by side; the neighbour list is not (5 cards vs. 7 stat tiles)
                  -- CSS Grid doesn't reflow content between columns of different
                  heights, so pairing it here left a large dead gap under the
                  shorter column. It gets its own full-width section below instead. */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:items-stretch">
                <MoleculeViewer3D molBlock={profile.mol_block} />
                <PCProfileCard profile={profile.pc_profile} />
              </div>
              <NeighbourPanel neighbours={profile.neighbours} />
            </div>

            <div className="border-t border-slate-200 pt-10">
              <SectionLabel text="Choose Properties to Predict" />
              <PropertySelector
                selected={selectedDatasets}
                onChange={setSelectedDatasets}
                onPredict={handlePredict}
                isPredicting={predicting}
              />
            </div>

            {Object.keys(results).length > 0 && (
              <div>
                <SectionLabel text="Prediction Results" />
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(results).map(([key, state]) => (
                    <DatasetResultCard key={key} state={state} />
                  ))}
                </div>
              </div>
            )}
          </motion.section>
        )}
      </AnimatePresence>

      <footer className="mt-20 max-w-lg border-t border-slate-200 pt-6 text-center text-[11px] leading-relaxed text-slate-500">
        Predictions come from models benchmarked in the PEARL study and are provided for research purposes only
        -- not a substitute for experimental validation or clinical judgment.
        <p className="mt-2 text-slate-400">
          PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning for Molecular Property Prediction
        </p>
      </footer>
    </main>
  );
}
