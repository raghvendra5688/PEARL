"use client";

import { DATASET_CATALOG } from "../lib/datasets";
import { IconCheckCircle } from "./icons";

interface PropertySelectorProps {
  selected: string[];
  onChange: (selected: string[]) => void;
  onPredict: () => void;
  isPredicting: boolean;
}

export default function PropertySelector({ selected, onChange, onPredict, isPredicting }: PropertySelectorProps) {
  const enabledKeys = DATASET_CATALOG.filter((d) => d.enabled).map((d) => d.key);
  const allSelected = enabledKeys.length > 0 && enabledKeys.every((k) => selected.includes(k));

  function toggle(key: string) {
    onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  }

  function toggleAll() {
    onChange(allSelected ? [] : enabledKeys);
  }

  return (
    <div className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-xs text-slate-600">
          {selected.length} of {enabledKeys.length} available properties selected
        </p>
        <button onClick={toggleAll} className="text-xs font-medium text-accent-dim hover:underline">
          {allSelected ? "Clear all" : "Select all"}
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {DATASET_CATALOG.map((d) => {
          const isSelected = selected.includes(d.key);
          return (
            <button
              key={d.key}
              type="button"
              disabled={!d.enabled}
              onClick={() => toggle(d.key)}
              className={`relative rounded-xl border p-4 text-left transition-shadow ${
                !d.enabled
                  ? "cursor-not-allowed border-slate-200 bg-slate-50 opacity-60"
                  : isSelected
                    ? "border-accent bg-accent/10 shadow-card"
                    : "border-slate-200 bg-white shadow-card hover:shadow-card-hover"
              }`}
            >
              {isSelected && <IconCheckCircle className="absolute right-3.5 top-3.5 h-4 w-4 text-accent-dim" />}
              <div className="mb-1.5 flex items-center gap-2 pr-5">
                <span className="text-sm font-medium text-slate-900">{d.displayName}</span>
              </div>
              <div className="mb-2 flex items-center gap-1.5">
                <span className="rounded-full border border-slate-300 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-600">
                  {d.task === "classification" ? "Classification" : "Regression"}
                </span>
                {!d.enabled && (
                  <span className="rounded-full bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-600">
                    Coming soon
                  </span>
                )}
              </div>
              <p className="text-xs leading-relaxed text-slate-600">{d.description}</p>
            </button>
          );
        })}
      </div>

      <button
        onClick={onPredict}
        disabled={selected.length === 0 || isPredicting}
        className="mt-5 w-full rounded-lg bg-accent py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-accent-bright disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto sm:px-8"
      >
        {isPredicting
          ? "Predicting…"
          : `Predict ${selected.length || ""} ${selected.length === 1 ? "Property" : "Properties"}`}
      </button>
    </div>
  );
}
