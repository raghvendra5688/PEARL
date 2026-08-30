"use client";

import { motion } from "framer-motion";
import { DATASET_CATALOG } from "../lib/datasets";
import type { PredictResult } from "../lib/types";
import { IconAlertTriangle, IconCheckCircle } from "./icons";

export type ResultCardState =
  | { status: "pending"; displayName: string }
  | { status: "done"; result: PredictResult }
  | { status: "error"; displayName: string; message: string };

// A single ratio against a limit, rendered as a meter (fill + same-hue track at
// low opacity), not a donut/gauge -- per the dataviz skill's form guidance.
function Meter({ value, colorClass, trackTint }: { value: number; colorClass: string; trackTint: string }) {
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full ${trackTint}`}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${Math.round(value * 100)}%` }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className={`h-full rounded-full ${colorClass}`}
      />
    </div>
  );
}

function CredibilityLine({ result }: { result: PredictResult }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-200 pt-3 text-[11px] text-slate-500">
      <span className="text-slate-600">{result.method_family}</span>
      {result.credibility.map((c) => (
        <span key={c.metric} className="font-mono-tabular font-mono">
          {c.metric} <span className="text-slate-700">{c.mean.toFixed(3)}</span>
          <span className="text-slate-500"> ± {c.se.toFixed(3)}</span>
        </span>
      ))}
    </div>
  );
}

function ClassificationBody({ result }: { result: PredictResult }) {
  const confidence = result.confidence ?? 0;
  const isPositive = result.predicted_class === 1;

  // Status color carries meaning, always paired with an icon + label. The label
  // names the predicted class directly (e.g. "BACE1 Inhibitor") rather than a
  // generic "Likely"/"Unlikely".
  const entry = DATASET_CATALOG.find((d) => d.key === result.dataset);
  const label = entry?.classLabels
    ? isPositive
      ? entry.classLabels.positive
      : entry.classLabels.negative
    : isPositive
      ? "Likely"
      : "Unlikely";

  // Solid-fill badge, not colored text: status.warning sits too close to white in
  // lightness for text to clear contrast, even with a tinted background.
  const badgeClass = isPositive ? "bg-status-warning text-slate-900" : "bg-status-good text-white";
  const fillClass = isPositive ? "bg-status-warning" : "bg-status-good";
  const trackTint = isPositive ? "bg-status-warning/15" : "bg-status-good/15";
  const Icon = isPositive ? IconAlertTriangle : IconCheckCircle;

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold leading-tight ${badgeClass}`}>
          <Icon className="h-4 w-4 shrink-0" />
          {label}
        </span>
        <span className="shrink-0 pt-1 font-mono-tabular font-mono text-sm text-slate-600">
          {(confidence * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-3">
        <Meter value={confidence} colorClass={fillClass} trackTint={trackTint} />
      </div>
    </div>
  );
}

// Multiclass (Flavor, 5 taste categories): "emphasis" form -- one predicted class
// is the point, the rest are context, so it's 1 hue + gray rather than a full
// categorical palette. Ranked by probability, each row direct-labeled.
function MulticlassBody({ result }: { result: PredictResult }) {
  const labels = result.class_labels ?? [];
  const probabilities = result.probabilities ?? [];
  const rows = labels
    .map((label, i) => ({ label, prob: probabilities[i] ?? 0 }))
    .sort((a, b) => b.prob - a.prob);
  const top = rows[0];

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-accent px-3 py-1 text-sm font-semibold capitalize leading-tight text-white">
          {top?.label}
        </span>
        <span className="shrink-0 pt-1 font-mono-tabular font-mono text-sm text-slate-600">
          {((top?.prob ?? 0) * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-3 space-y-2">
        {rows.map((row, i) => (
          <div key={row.label} className="flex items-center gap-2">
            <span className="w-16 shrink-0 text-xs capitalize text-slate-600">{row.label}</span>
            <Meter
              value={row.prob}
              colorClass={i === 0 ? "bg-accent" : "bg-slate-300"}
              trackTint={i === 0 ? "bg-accent/15" : "bg-slate-100"}
            />
            <span className="w-9 shrink-0 text-right font-mono-tabular font-mono text-xs text-slate-500">
              {(row.prob * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RegressionBody({ result }: { result: PredictResult }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="font-mono-tabular font-mono text-2xl font-semibold text-accent-dim">
        {result.value?.toFixed(3)}
      </span>
      {result.unit_label && <span className="text-sm text-slate-600">{result.unit_label}</span>}
    </div>
  );
}

export default function DatasetResultCard({ state }: { state: ResultCardState }) {
  const displayName = state.status === "done" ? state.result.display_name : state.displayName;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card transition-shadow hover:shadow-card-hover"
    >
      <h4 className="mb-3 text-sm font-medium text-slate-900">{displayName}</h4>

      {state.status === "pending" && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="pulse-ring h-2 w-2 rounded-full bg-accent" />
          Computing…
        </div>
      )}

      {state.status === "error" && (
        <p className="flex items-center gap-1.5 text-sm text-rose-600">
          <IconAlertTriangle className="h-4 w-4 shrink-0" />
          {state.message}
        </p>
      )}

      {state.status === "done" && (
        <>
          {state.result.task === "classification" ? (
            state.result.class_labels && state.result.class_labels.length > 0 ? (
              <MulticlassBody result={state.result} />
            ) : (
              <ClassificationBody result={state.result} />
            )
          ) : (
            <RegressionBody result={state.result} />
          )}
          <CredibilityLine result={state.result} />
        </>
      )}
    </motion.div>
  );
}
