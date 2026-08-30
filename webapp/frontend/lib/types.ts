export interface ResolveCandidate {
  cid: number;
  smiles: string;
  name: string | null;
}

export interface ResolveResponse {
  status: "resolved" | "disambiguation";
  smiles: string | null;
  candidates: ResolveCandidate[] | null;
}

export interface PCProfile {
  molecular_weight: number;
  logp: number;
  tpsa: number;
  h_bond_donors: number;
  h_bond_acceptors: number;
  rotatable_bonds: number;
  aromatic_rings: number;
}

export interface Neighbour {
  smiles: string;
  formula: string | null;
  name: string | null;
  similarity: number;
  logp: number;
  qed: number;
  sas: number;
}

export interface ProfileResponse {
  smiles: string;
  formula: string | null;
  name: string | null;
  mol_block: string | null;
  pc_profile: PCProfile | null;
  neighbours: Neighbour[] | null;
}

export interface Credibility {
  metric: string;
  mean: number;
  se: number;
}

export type DatasetKey = "BACE" | "HERG" | "CACO2" | "BBBP" | "DILI" | "HALF_LIFE" | "FLAVOR";

export interface PredictResult {
  key: DatasetKey;
  dataset: DatasetKey;
  display_name: string;
  task: "classification" | "regression";
  method_family: string;
  unit_label: string | null;
  class_labels: string[] | null;
  credibility: Credibility[];
  predicted_class?: number;
  confidence?: number;
  value?: number;
  // Present for multiclass results (Flavor) -- class_labels[i] names probabilities[i].
  probabilities?: number[];
}

export interface PredictErrorEvent {
  dataset: DatasetKey;
  error: string;
}
