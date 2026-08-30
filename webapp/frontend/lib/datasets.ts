// Plain-language dataset catalog for the property selector. Mirrors PLAN.md §3.
// Purely display metadata (no model wiring, no internal naming) -- safe to hardcode
// here rather than round-tripping through the API. BACE/HERG/CACO2/DILI/HALF_LIFE/
// FLAVOR are enabled and verified end-to-end against real predictions. BBBP's
// backend code exists but stays disabled -- its RAFE feature vector is a fixed 3
// columns short of what its XGBoost model expects, failing every call, still
// under investigation -- see ARCHITECTURE.md. ClinTox is excluded entirely, not
// "coming soon" -- its model was never uploaded (PLAN.md §4).

export interface DatasetCatalogEntry {
  key: string;
  displayName: string;
  description: string;
  task: "classification" | "regression";
  enabled: boolean;
  // Binary classification only -- names the predicted class itself (e.g. "BACE1
  // Inhibitor") rather than a generic "Likely"/"Unlikely", so a result card is
  // self-explanatory without the reader having to cross-reference the card title.
  classLabels?: { positive: string; negative: string };
}

export const DATASET_CATALOG: DatasetCatalogEntry[] = [
  {
    key: "BACE",
    displayName: "BACE1 Enzyme Inhibition",
    description: "Whether this molecule inhibits BACE1, a target implicated in Alzheimer's disease.",
    task: "classification",
    enabled: true,
    classLabels: { positive: "BACE1 Inhibitor", negative: "Not a BACE1 Inhibitor" },
  },
  {
    key: "HERG",
    displayName: "Cardiotoxicity (hERG Channel Inhibition)",
    description: "Risk of blocking the hERG cardiac ion channel -- a major cardiotoxicity liability.",
    task: "classification",
    enabled: true,
    classLabels: { positive: "hERG Blocker", negative: "Not a hERG Blocker" },
  },
  {
    key: "CACO2",
    displayName: "Intestinal Permeability",
    description: "Predicted Caco-2 cell permeability, a standard proxy for oral drug absorption.",
    task: "regression",
    enabled: true,
  },
  {
    key: "BBBP",
    displayName: "Blood-Brain Barrier Permeability",
    description: "Whether this molecule is likely to cross the blood-brain barrier.",
    task: "classification",
    enabled: false,
    classLabels: { positive: "Crosses BBB", negative: "Does Not Cross BBB" },
    // Stays disabled -- see the file header comment above.
  },
  {
    key: "DILI",
    displayName: "Drug-Induced Liver Injury Risk",
    description: "Risk of causing drug-induced liver injury.",
    task: "classification",
    enabled: true,
    classLabels: { positive: "Liver Injury Risk", negative: "Low Liver Injury Risk" },
  },
  {
    key: "HALF_LIFE",
    displayName: "Pharmacokinetic Half-Life",
    description: "Predicted elimination half-life.",
    task: "regression",
    enabled: true,
  },
  {
    key: "FLAVOR",
    displayName: "Taste Profile",
    description: "Predicted taste category -- bitter, sour, sweet, umami, or undefined.",
    task: "classification",
    enabled: true,
    // Multiclass, not binary -- classLabels' positive/negative shape doesn't apply.
  },
];
