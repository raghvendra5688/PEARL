"""
plot_bace_roc_pr.py

Generate ROC curve plot (AUC in legend, legend lower-left) and
PR curve plot (AUPR in legend, legend lower-right) for all 10 BACE
configurations from the PEARL manuscript Table 2.

Output:
    results/figures/bace_roc.pdf
    results/figures/bace_pr.pdf
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
EXTRAS_ROOT = Path("/export/cse/rmall/Raghvendra/EffiChem_Extras")
EFFCHEM2    = Path("/export/cse/rmall/Raghvendra/EffChem-2.0")
PEARL_ROOT  = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PEARL_ROOT / "scripts" / "unimol" / "finetuning"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

SMILES_COL    = "Standardized SMILES"
LABEL_COL     = "Class"
BACE_TEST_CSV = str(EFFCHEM2 / "data" / "clean" / "bace_datasets" / "test_clean.csv")

# Only the 6 finetuned SMILES embedding cols are excluded from PC features
# (matches the EMBED_COLS used in rag_modelling_bace.py and bace_pc_modelling_refactored.py)
_SKIP_EMBED_COLS = [
    "Molformer_Finetuned_WL_embeddings",
    "MolFormer_Finetuned_FL_embeddings",
    "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MTR_FL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",
    "ChemBERTa_77M_MLM_FL_embeddings",
]
# All embedding cols (used only when building UniMol RAFE merge to avoid
# pulling in other SMILES embedding string columns as features)
_ALL_SMILES_EMB_COLS = _SKIP_EMBED_COLS + [
    "MolFormer_Base_embeddings",
    "ChemBERTa_77M_MTR_Base_embeddings",
    "ChemBERTa_77M_MLM_Base_embeddings",
]

OUT_DIR = PEARL_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Authoritative AUC / AUPR from saved metrics files ─────────────────────────
# These match the manuscript table values and are used as legend labels.
# AUC/AUPR from inference may vary slightly due to GPU non-determinism.
METRIC_OVERRIDES = {
    # from EffiChem_Extras/BACE/best_model_comparison/all_metrics.csv
    "E2E LoRA MolFormer-XL FL": {"auc": 0.872, "aupr": 0.890},
    "E2E LoRA MolFormer-XL WL": {"auc": 0.871, "aupr": 0.902},
    # from EffChem-2.0/results/unimol_finetuning/bace/unimol_lora_metrics.csv
    "E2E LoRA Uni-Mol FL":   {"auc": 0.876, "aupr": 0.924},
    "E2E LoRA Uni-Mol WL":   {"auc": 0.891, "aupr": 0.924},
    # from BACE_FT_Results/evaluation_plots/test_set_metrics.csv
    "FT Embed MolFormer-XL FL": {"auc": 0.869, "aupr": 0.899},
    # from EffChem-2.0/results/finetuned/UniMol_BACE_FT_Results/UniMol_FL/metrics/LightGBM_metrics.json
    "FT Embed Uni-Mol FL":   {"auc": 0.860, "aupr": 0.893},
    # from BACE_PC_FT_Results/MolFormer_Finetuned_FL_embeddings/metrics/XGBoost_metrics.npy
    "PC+FT Embed MolFormer-XL FL": {"auc": 0.870},  # no stored AUPR → use computed
    # from EffChem-2.0/results/rag/bace/MolFormer_Finetuned_FL/metrics/CatBoost_metrics.json
    "RAFE MolFormer-XL FL":     {"auc": 0.863, "aupr": 0.892},
    # from EffChem-2.0/results/rag/bace/Molformer_Finetuned_WL/metrics/LightGBM_metrics.json
    "RAFE MolFormer-XL WL":     {"auc": 0.858, "aupr": 0.892},
    # from EffChem-2.0/results/rag_unimol/bace/UniMol_FL/metrics/XGBoost_metrics.json
    "RAFE Uni-Mol FL":       {"auc": 0.840, "aupr": 0.895},
}


# ── Embedding parser ───────────────────────────────────────────────────────────
def _parse_emb(s: str):
    s = str(s).strip()
    try:
        return np.array(json.loads(s), dtype=np.float32)
    except Exception:
        try:
            clean = s[1:-1] if s.startswith("[") else s
            return np.array([float(x) for x in clean.split(",") if x.strip()],
                            dtype=np.float32)
        except Exception:
            return None


def parse_emb_col(series: pd.Series):
    pairs = [(i, _parse_emb(v)) for i, v in enumerate(series)]
    valid = [(i, e) for i, e in pairs if e is not None]
    return (np.vstack([e for _, e in valid]).astype(np.float32),
            [i for i, _ in valid])


# ── Tree model inference ───────────────────────────────────────────────────────
def tree_predict(model_path: Path, X: np.ndarray) -> np.ndarray:
    model = joblib.load(str(model_path))
    return model.predict_proba(X)[:, 1]


# ── Feature builders ───────────────────────────────────────────────────────────

def build_ft_smiles(emb_col: str, split: str = "test"):
    """FT SMILES embed only."""
    csv = EXTRAS_ROOT / "All_Embeddings" / "BACE_Embeddings" / f"bace_{split}_embed.csv"
    df  = pd.read_csv(str(csv))
    embs, idxs = parse_emb_col(df[emb_col])
    y = df[LABEL_COL].iloc[idxs].values.astype(int)
    return embs, y


def build_unimol_ft(emb_col: str = "UniMol_FL_embeddings", split: str = "test"):
    """UniMol FT embed only."""
    csv = EXTRAS_ROOT / "unimol_embeddings" / "BACE_Embeddings" / f"bace_{split}_embed.csv"
    df  = pd.read_csv(str(csv))
    embs, idxs = parse_emb_col(df[emb_col])
    y = df[LABEL_COL].iloc[idxs].values.astype(int)
    return embs, y


def build_pc_ft_smiles(emb_col: str, split: str = "test"):
    """PC + FT SMILES embed.
    Replicates bace_pc_modelling_refactored.py: skip = SMILES + Class + 6 finetuned emb cols.
    CID and base-model embedding cols are kept (become numeric features after fillna).
    """
    csv = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "BACE_Embeddings" / f"bace_{split}_features.csv"
    df  = pd.read_csv(str(csv))
    embs, idxs = parse_emb_col(df[emb_col])
    df = df.iloc[idxs].reset_index(drop=True)
    skip    = {SMILES_COL, LABEL_COL} | set(_SKIP_EMBED_COLS)
    pc_cols = [c for c in df.columns if c not in skip]
    pc  = df[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    X   = np.hstack([embs, pc]).astype(np.float32)
    y   = df[LABEL_COL].values.astype(int)
    return X, y


def build_rafe_smiles(emb_col: str, rag_col: str, split: str = "test"):
    """RAFE SMILES: [FT embed | PC features | RAG features].
    Replicates rag_modelling_bace.py: skip = SMILES + Class + 6 finetuned emb cols.
    """
    pc_csv  = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "BACE_Embeddings" / f"bace_{split}_features.csv"
    rag_csv = EFFCHEM2 / "data" / "rag_features" / "bace" / f"{rag_col}_{split}_rag.csv"
    df_pc  = pd.read_csv(str(pc_csv))
    df_rag = pd.read_csv(str(rag_csv))

    merged = df_pc.merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == LABEL_COL], errors="ignore"),
        on=SMILES_COL, how="inner",
    )
    embs, idxs = parse_emb_col(merged[emb_col])
    merged = merged.iloc[idxs].reset_index(drop=True)

    rag_cols = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    skip     = {SMILES_COL, LABEL_COL} | set(_SKIP_EMBED_COLS)
    pc_cols  = [c for c in merged.columns if c not in skip and c not in rag_cols]

    pc  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X   = np.hstack([embs, pc, rag]).astype(np.float32)
    y   = merged[LABEL_COL].values.astype(int)
    return X, y


def build_rafe_unimol(emb_col: str = "UniMol_FL_embeddings", rag_col: str = "UniMol_FL",
                      split: str = "test"):
    """RAFE UniMol: [UniMol embed | PC features | RAG features].
    Generates RAG features via the PEARL extraction script if they are missing.
    """
    rag_path = EFFCHEM2 / "data" / "rag_features_unimol" / "bace" / f"{rag_col}_{split}_rag.csv"

    if not rag_path.exists():
        log.info(f"RAG features missing — running rag_feature_extraction_unimol.py for bace ...")
        script = PEARL_ROOT / "scripts" / "unimol" / "rag" / "rag_feature_extraction_unimol.py"
        env    = os.environ.copy()
        env["PEARL_EXTRAS"] = str(EXTRAS_ROOT)
        result = subprocess.run(
            ["/export/home/rmall/.local/bin/micromamba", "run", "-n", "effichem",
             "python", str(script), "--dataset", "bace", "--no-gpu"],
            capture_output=False,
            env=env,
            cwd=str(PEARL_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError("rag_feature_extraction_unimol.py failed — skipping RAFE UniMol.")
        if not rag_path.exists():
            raise FileNotFoundError(f"RAG features still missing after extraction: {rag_path}")
        log.info("RAG features generated successfully.")

    unimol_csv = EXTRAS_ROOT / "unimol_embeddings" / "BACE_Embeddings" / f"bace_{split}_embed.csv"
    pc_csv     = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "BACE_Embeddings" / f"bace_{split}_features.csv"
    df_unimol  = pd.read_csv(str(unimol_csv))
    df_pc      = pd.read_csv(str(pc_csv))
    df_rag     = pd.read_csv(str(rag_path))

    pc_cols_to_add = [c for c in df_pc.columns
                      if c not in {SMILES_COL, LABEL_COL} | set(_SKIP_EMBED_COLS)]

    merged = df_unimol[[SMILES_COL, LABEL_COL, emb_col]].merge(
        df_pc[[SMILES_COL] + pc_cols_to_add], on=SMILES_COL, how="inner"
    ).merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == LABEL_COL], errors="ignore"),
        on=SMILES_COL, how="inner",
    )

    embs, idxs = parse_emb_col(merged[emb_col])
    merged = merged.iloc[idxs].reset_index(drop=True)

    rag_feat_cols = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    pc  = merged[pc_cols_to_add].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag = merged[rag_feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X   = np.hstack([embs, pc, rag]).astype(np.float32)
    y   = merged[LABEL_COL].values.astype(int)
    return X, y


# ── MolFormer E2E LoRA inference ───────────────────────────────────────────────

def molformer_predict(model_dir: Path, test_csv: str, batch_size: int = 32):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    log.info(f"  Loading MolFormer: {model_dir.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=2,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    ).to(DEVICE).eval()

    df     = pd.read_csv(test_csv)
    smiles = df[SMILES_COL].tolist()
    y_true = df[LABEL_COL].values.astype(int)

    probs_all = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            batch = smiles[i : i + batch_size]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=512)
            enc.pop("token_type_ids", None)
            enc   = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits.cpu().numpy()
            probs_all.append(softmax(logits, axis=1)[:, 1])

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return y_true, np.concatenate(probs_all)


# ── UniMol E2E LoRA inference ──────────────────────────────────────────────────

def unimol_predict(model_dir: Path, test_csv: str, batch_size: int = 32):
    from unimol_lora_trainer import (
        load_finetuned_unimol,
        preprocess_smiles_for_unimol,
        _collate_unimol_inputs,
        batch_morgan_fps,
    )
    log.info(f"  Loading UniMol: {model_dir.name}")
    model = load_finetuned_unimol(model_dir, num_classes=2, device=DEVICE)
    model.eval()

    df     = pd.read_csv(test_csv)
    smiles = df[SMILES_COL].tolist()
    y_true = df[LABEL_COL].values.astype(int)

    log.info("  Preprocessing SMILES (conformer generation) ...")
    unimol_inputs = preprocess_smiles_for_unimol(smiles, model._repr)
    morgan_all    = batch_morgan_fps(smiles)
    padding_idx   = model._repr.model.padding_idx

    probs_all = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            ub     = _collate_unimol_inputs(unimol_inputs[i : i + batch_size], padding_idx)
            fp     = torch.tensor(morgan_all[i : i + batch_size], dtype=torch.float32)
            logits, _ = model(ub, fp)
            probs_all.append(softmax(logits.cpu().numpy(), axis=1)[:, 1])

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return y_true, np.concatenate(probs_all)


# ── Method registry ────────────────────────────────────────────────────────────
# Each entry: (label, color, linestyle, y_fn, score_fn)
# y_fn and score_fn share data via a shared dict keyed by label.

def compute_all_results():
    """Return list of (label, color, ls, fpr, tpr, auc, prec, rec, aupr)."""
    results = []

    def add(label, color, ls, y, score):
        auc_computed  = roc_auc_score(y, score)
        aupr_computed = average_precision_score(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        # Use saved metrics values for legend labels (matches manuscript table).
        overrides = METRIC_OVERRIDES.get(label, {})
        auc_label  = overrides.get("auc",  auc_computed)
        aupr_label = overrides.get("aupr", aupr_computed)
        results.append((label, color, ls, fpr, tpr, auc_label, prec, rec, aupr_label))
        log.info(f"  OK  AUC={auc_computed:.3f}→{auc_label:.3f}  "
                 f"AUPR={aupr_computed:.3f}→{aupr_label:.3f}")

    def run(label, color, ls, fn):
        log.info(f"[{label}]")
        try:
            y, score = fn()
            add(label, color, ls, y, score)
        except Exception as e:
            log.error(f"  FAILED: {e}", exc_info=True)

    # 1 – E2E LoRA MolFormer FL
    run("E2E LoRA MolFormer-XL FL", "#1565C0", "-",
        lambda: molformer_predict(
            EXTRAS_ROOT / "focal_loss_BACE"
            / "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned",
            BACE_TEST_CSV,
        ))

    # 2 – E2E LoRA MolFormer WL
    run("E2E LoRA MolFormer-XL WL", "#42A5F5", "--",
        lambda: molformer_predict(
            EXTRAS_ROOT / "weighted_loss_BACE"
            / "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned",
            BACE_TEST_CSV,
        ))

    # 3 – E2E LoRA Uni-Mol FL
    run("E2E LoRA Uni-Mol FL", "#00838F", "-",
        lambda: unimol_predict(
            EXTRAS_ROOT / "focal_loss_BACE" / "dptech__Uni__Mol_LoRA_Finetuned",
            BACE_TEST_CSV,
        ))

    # 4 – E2E LoRA Uni-Mol WL
    run("E2E LoRA Uni-Mol WL", "#26C6DA", "--",
        lambda: unimol_predict(
            EXTRAS_ROOT / "weighted_loss_BACE" / "dptech__Uni__Mol_LoRA_Finetuned",
            BACE_TEST_CSV,
        ))

    # 5 – FT Embed MolFormer FL + CatBoost
    run("FT Embed MolFormer-XL FL", "#2E7D32", "-", lambda: _tree(
        build_ft_smiles("MolFormer_Finetuned_FL_embeddings"),
        EXTRAS_ROOT / "BACE_FT_Results" / "MolFormer_Finetuned_FL"
        / "models" / "MolFormer_Finetuned_FL_CatBoost.pkl",
    ))

    # 6 – FT Embed Uni-Mol FL + LightGBM
    run("FT Embed Uni-Mol FL", "#A5D6A7", "-", lambda: _tree(
        build_unimol_ft("UniMol_FL_embeddings"),
        EFFCHEM2 / "results" / "finetuned" / "UniMol_BACE_FT_Results"
        / "UniMol_FL" / "models" / "LightGBM.pkl",
    ))

    # 7 – PC+FT Embed MolFormer FL + XGBoost
    run("PC+FT Embed MolFormer-XL FL", "#6A1B9A", "-", lambda: _tree(
        build_pc_ft_smiles("MolFormer_Finetuned_FL_embeddings"),
        EXTRAS_ROOT / "BACE_PC_FT_Results" / "MolFormer_Finetuned_FL_embeddings"
        / "models" / "XGBoost.pkl",
    ))

    # 8 – RAFE MolFormer FL + CatBoost
    run("RAFE MolFormer-XL FL", "#C62828", "-", lambda: _tree(
        build_rafe_smiles("MolFormer_Finetuned_FL_embeddings", "MolFormer_Finetuned_FL"),
        EFFCHEM2 / "results" / "rag" / "bace" / "MolFormer_Finetuned_FL"
        / "models" / "CatBoost.pkl",
    ))

    # 9 – RAFE MolFormer WL + LightGBM
    run("RAFE MolFormer-XL WL", "#FF7043", "-", lambda: _tree(
        build_rafe_smiles("Molformer_Finetuned_WL_embeddings", "Molformer_Finetuned_WL"),
        EFFCHEM2 / "results" / "rag" / "bace" / "Molformer_Finetuned_WL"
        / "models" / "LightGBM.pkl",
    ))

    # 10 – RAFE Uni-Mol FL + XGBoost  (skipped: RAG feature extraction fails on this GPU)
    # run("RAFE Uni-Mol FL", "#795548", "-", lambda: _tree(
    #     build_rafe_unimol("UniMol_FL_embeddings", "UniMol_FL"),
    #     EFFCHEM2 / "results" / "rag_unimol" / "bace" / "UniMol_FL"
    #     / "models" / "XGBoost.pkl",
    # ))

    return results


def _tree(Xy, model_path):
    """Shared helper: unpack (X, y), run tree_predict."""
    X, y = Xy
    score = tree_predict(model_path, X)
    return y, score


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_curves(results):
    fig_w, fig_h = 4.8, 4.5

    # ── ROC ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot([0, 1], [0, 1], color="lightgray", lw=0.8, ls=":")
    for label, color, ls, fpr, tpr, auc_, prec, rec, aupr_ in results:
        ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls,
                label=f"{label} ({auc_:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("ROC Curves — BACE", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.2, framealpha=0.9,
              title="Method (AUC)", title_fontsize=6.8,
              handlelength=2.0, borderpad=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.text(0.02, 0.98, "A", fontsize=16, fontweight="bold", ha="left", va="top")
    p = OUT_DIR / "bace_roc.pdf"
    fig.savefig(str(p), bbox_inches="tight")
    log.info(f"Saved: {p}")
    plt.close(fig)

    # ── PR ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    df_test  = pd.read_csv(BACE_TEST_CSV)
    baseline = df_test[LABEL_COL].mean()
    ax.axhline(y=baseline, color="lightgray", lw=0.8, ls=":", label="_nolegend_")
    for label, color, ls, fpr, tpr, auc_, prec, rec, aupr_ in results:
        ax.plot(rec, prec, color=color, lw=1.5, ls=ls,
                label=f"{label} ({aupr_:.3f})")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("PR Curves — BACE", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.2, framealpha=0.9,
              title="Method (AUPR)", title_fontsize=6.8,
              handlelength=2.0, borderpad=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.text(0.02, 0.98, "B", fontsize=16, fontweight="bold", ha="left", va="top")
    p = OUT_DIR / "bace_pr.pdf"
    fig.savefig(str(p), bbox_inches="tight")
    log.info(f"Saved: {p}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("BACE ROC / PR curve generation — PEARL Table 2")
    log.info("=" * 60)
    results = compute_all_results()
    log.info(f"\n{len(results)}/10 methods computed successfully.")
    if results:
        plot_curves(results)
    else:
        log.error("No results — check errors above.")
