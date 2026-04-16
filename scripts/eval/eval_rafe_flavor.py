"""
RAFE Flavor evaluation — compute metrics for all 24 configurations (18 SMILES + 6 UniMol).
Reuses saved best_params.json; does not re-run Optuna.

Usage:
    python scripts/eval/eval_rafe_flavor.py
    python scripts/eval/eval_rafe_flavor.py --skip MolFormer_Finetuned_FL:CatBoost
    python scripts/eval/eval_rafe_flavor.py --only UniMol_FL:XGBoost UniMol_FL:LightGBM
    python scripts/eval/eval_rafe_flavor.py --out results/rag/flavor/custom.csv

Each --skip / --only token is  <embedding_name>:<tree>  e.g. MolFormer_Finetuned_FL:CatBoost
Omit the tree part to match all trees for that embedding: --skip MolFormer_Finetuned_FL

Output (default): results/rag/flavor/rafe_flavor_all_metrics.csv
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_JOBS      = min(os.cpu_count() or 1, 16)

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
EFFCHEM2    = Path("/export/cse/rmall/Raghvendra/EffChem-2.0")

LABEL_COL   = "Canonicalized Taste"
SMILES_COL  = "Standardized SMILES"
FILE_PREFIX = "fart"

SMILES_PC_ROOT  = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "flavor_Embeddings"
UNIMOL_EMB_ROOT = EXTRAS_ROOT / "unimol_embeddings" / "flavor_Embeddings"
RAG_SMILES_ROOT = EFFCHEM2 / "data" / "rag_features" / "flavor"
RAG_UNIMOL_ROOT = EFFCHEM2 / "data" / "rag_features_unimol" / "flavor"
PARAMS_SMILES   = REPO_ROOT / "results" / "rag" / "flavor"
PARAMS_UNIMOL   = REPO_ROOT / "results" / "rag_unimol" / "flavor"
DEFAULT_OUT_CSV = REPO_ROOT / "results" / "rag" / "flavor" / "rafe_flavor_all_metrics.csv"

ALL_SMILES_EMBED_COLS = [
    "ChemBERTa_77M_MTR_FL_embeddings", "ChemBERTa_77M_MLM_FL_embeddings",
    "MolFormer_Finetuned_FL_embeddings", "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",  "Molformer_Finetuned_WL_embeddings",
]
UNIMOL_EMBED_COLS = ["UniMol_FL_embeddings", "UniMol_WL_embeddings"]

# All 24 configs: (col_name, emb_col, trees, params_root, rag_root, is_unimol)
SMILES_CONFIGS = [
    ("ChemBERTa_77M_MTR_FL",   "ChemBERTa_77M_MTR_FL_embeddings",   ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
    ("ChemBERTa_77M_MLM_FL",   "ChemBERTa_77M_MLM_FL_embeddings",   ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
    ("MolFormer_Finetuned_FL", "MolFormer_Finetuned_FL_embeddings",  ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
    ("ChemBERTa_77M_MTR_WL",   "ChemBERTa_77M_MTR_WL_embeddings",   ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
    ("ChemBERTa_77M_MLM_WL",   "ChemBERTa_77M_MLM_WL_embeddings",   ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
    ("Molformer_Finetuned_WL", "Molformer_Finetuned_WL_embeddings",  ["XGBoost", "LightGBM", "CatBoost"], PARAMS_SMILES, RAG_SMILES_ROOT, False),
]
UNIMOL_CONFIGS = [
    ("UniMol_FL", "UniMol_FL_embeddings", ["XGBoost", "LightGBM", "CatBoost"], PARAMS_UNIMOL, RAG_UNIMOL_ROOT, True),
    ("UniMol_WL", "UniMol_WL_embeddings", ["XGBoost", "LightGBM", "CatBoost"], PARAMS_UNIMOL, RAG_UNIMOL_ROOT, True),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def safe_parse_embedding(s: str):
    try:
        s = s.strip()
        if not any(c.isdigit() for c in s):
            return None
        try:
            arr = np.array(json.loads(s), dtype=np.float32)
        except json.JSONDecodeError:
            clean = s[1:-1] if s.startswith("[") else s
            arr = np.array([float(x) for x in clean.split(",") if x.strip()], dtype=np.float32)
        if arr.ndim != 1 or len(arr) == 0:
            return None
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception:
        return None


def parse_embedding_col(series: pd.Series) -> np.ndarray:
    valid = [safe_parse_embedding(str(v)) for v in series]
    valid = [e for e in valid if e is not None]
    return np.vstack(valid).astype(np.float32)


def make_model(model_type: str, params: dict, n_classes: int):
    if model_type == "LightGBM":
        return lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes,
            class_weight="balanced", random_state=RANDOM_SEED,
            n_jobs=N_JOBS, verbosity=-1, **params)
    elif model_type == "XGBoost":
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=RANDOM_SEED,
            tree_method="hist", n_jobs=N_JOBS, **params)
    else:
        return cb.CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=RANDOM_SEED, verbose=0, thread_count=N_JOBS, **params)


def evaluate(model, X_tr, y_tr, X_te, y_te, n_classes):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    try:
        auc = round(roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro"), 3)
    except Exception:
        auc = float("nan")
    metrics = {
        "AUC":    auc,
        "Acc":    round(accuracy_score(y_te, y_pred), 3),
        "F1_mac": round(f1_score(y_te, y_pred, average="macro",  zero_division=0), 3),
        "F1_mic": round(f1_score(y_te, y_pred, average="micro",  zero_division=0), 3),
        "MCC":    round(matthews_corrcoef(y_te, y_pred), 3),
        "Prec":   round(precision_score(y_te, y_pred, average="macro", zero_division=0), 3),
        "Rec":    round(recall_score(y_te, y_pred, average="macro", zero_division=0), 3),
    }
    for cls_i in range(n_classes):
        y_bin = (y_te == cls_i).astype(int)
        if y_bin.sum() > 0:
            metrics[f"AUPR_C{cls_i}"] = round(float(average_precision_score(y_bin, y_prob[:, cls_i])), 3)
    return metrics


def build_smiles_X(df_pc, df_rag, emb_col, le):
    merged = df_pc.merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == LABEL_COL], errors="ignore"),
        on=SMILES_COL, how="inner",
    )
    emb = parse_embedding_col(merged[emb_col])
    skip = [SMILES_COL, LABEL_COL] + ALL_SMILES_EMBED_COLS
    rag_cols  = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    pc_cols   = [c for c in merged.columns if c not in skip + rag_cols]
    pc_feats  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = le.transform(merged[LABEL_COL].astype(str).values)
    return X, y


def build_unimol_X(emb_df, pc_df, df_rag, emb_col, le):
    other_emb = [c for c in UNIMOL_EMBED_COLS if c != emb_col]
    keep_emb  = [c for c in emb_df.columns if c not in other_emb]
    merged    = emb_df[keep_emb].copy()
    pc_skip   = [SMILES_COL, LABEL_COL] + ALL_SMILES_EMBED_COLS
    pc_cols   = [c for c in pc_df.columns if c not in pc_skip]
    merged    = merged.merge(pc_df[[SMILES_COL] + pc_cols], on=SMILES_COL, how="inner")
    rag_drop  = [c for c in df_rag.columns if c == LABEL_COL]
    merged    = merged.merge(df_rag.drop(columns=rag_drop, errors="ignore"), on=SMILES_COL, how="inner")
    emb       = parse_embedding_col(merged[emb_col])
    rag_cols  = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    skip_all  = [SMILES_COL, LABEL_COL] + UNIMOL_EMBED_COLS + ALL_SMILES_EMBED_COLS
    feat_cols = [c for c in merged.columns if c not in skip_all + rag_cols]
    pc_feats  = merged[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = le.transform(merged[LABEL_COL].astype(str).values)
    return X, y


def parse_filter_tokens(tokens):
    """Parse 'Embedding:Tree' or 'Embedding' tokens into a set of (emb, tree|None) pairs."""
    result = set()
    for t in tokens:
        if ":" in t:
            emb, tree = t.split(":", 1)
            result.add((emb.strip(), tree.strip()))
        else:
            result.add((t.strip(), None))
    return result


def should_run(col_name, tree, skip_set, only_set):
    if only_set:
        return (col_name, tree) in only_set or (col_name, None) in only_set
    for emb, tr in skip_set:
        if emb == col_name and (tr is None or tr == tree):
            return False
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all RAFE Flavor configurations.")
    parser.add_argument("--skip", nargs="+", default=[],
                        metavar="EMB[:TREE]",
                        help="Embedding+tree combinations to skip, e.g. MolFormer_Finetuned_FL:CatBoost")
    parser.add_argument("--only", nargs="+", default=[],
                        metavar="EMB[:TREE]",
                        help="Run only these combinations (mutually exclusive with --skip)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV),
                        help="Output CSV path")
    args = parser.parse_args()

    if args.skip and args.only:
        parser.error("--skip and --only are mutually exclusive")

    skip_set = parse_filter_tokens(args.skip)
    only_set = parse_filter_tokens(args.only)
    out_csv  = Path(args.out)

    results = []

    # ── Load SMILES PC data (shared by all SMILES configs) ────────────────────
    print("Loading SMILES PC data ...", flush=True)
    tr_pc = pd.read_csv(str(SMILES_PC_ROOT / f"{FILE_PREFIX}_train_features.csv"))
    te_pc = pd.read_csv(str(SMILES_PC_ROOT / f"{FILE_PREFIX}_test_features.csv"))
    le = LabelEncoder()
    le.fit(tr_pc[LABEL_COL].astype(str).values)
    n_cls = len(le.classes_)
    print(f"Classes ({n_cls}): {le.classes_}", flush=True)

    # ── SMILES configs ────────────────────────────────────────────────────────
    for col_name, emb_col, trees, params_root, rag_root, _ in SMILES_CONFIGS:
        active_trees = [t for t in trees if should_run(col_name, t, skip_set, only_set)]
        if not active_trees:
            print(f"\nSKIP {col_name} (all trees filtered)", flush=True)
            continue

        params_path = params_root / col_name / "metrics" / "best_params.json"
        if not params_path.exists():
            print(f"\nSKIP {col_name}: best_params.json not found", flush=True)
            continue
        best_params = json.loads(params_path.read_text())

        print(f"\n{'='*50}", flush=True)
        print(f"Loading RAG features for {col_name} ...", flush=True)
        rag_tr = pd.read_csv(str(rag_root / f"{col_name}_train_rag.csv"))
        rag_te = pd.read_csv(str(rag_root / f"{col_name}_test_rag.csv"))
        X_tr, y_tr = build_smiles_X(tr_pc, rag_tr, emb_col, le)
        X_te, y_te = build_smiles_X(te_pc, rag_te, emb_col, le)
        print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}", flush=True)

        for tree in active_trees:
            print(f"  {col_name} | {tree} ...", flush=True)
            m = evaluate(make_model(tree, best_params[tree], n_cls),
                         X_tr, y_tr, X_te, y_te, n_cls)
            results.append({"embedding": col_name, "tree": tree, **m})
            print(f"    AUC={m['AUC']}  Acc={m['Acc']}  F1mac={m['F1_mac']}  "
                  f"F1mic={m['F1_mic']}  MCC={m['MCC']}  Prec={m['Prec']}  Rec={m['Rec']}",
                  flush=True)

    # ── UniMol configs ────────────────────────────────────────────────────────
    unimol_active = [
        (col_name, emb_col, [t for t in trees if should_run(col_name, t, skip_set, only_set)],
         params_root, rag_root)
        for col_name, emb_col, trees, params_root, rag_root, _ in UNIMOL_CONFIGS
    ]
    unimol_active = [(cn, ec, tr, pr, rr) for cn, ec, tr, pr, rr in unimol_active if tr]

    if unimol_active:
        print(f"\n{'='*50}", flush=True)
        print("Loading UniMol embedding CSVs ...", flush=True)
        tr_emb = pd.read_csv(str(UNIMOL_EMB_ROOT / f"{FILE_PREFIX}_train_embed.csv"))
        te_emb = pd.read_csv(str(UNIMOL_EMB_ROOT / f"{FILE_PREFIX}_test_embed.csv"))
        le_u = LabelEncoder()
        le_u.fit(tr_emb[LABEL_COL].astype(str).values)
        n_cls_u = len(le_u.classes_)
        print("UniMol CSVs loaded.", flush=True)

        for col_name, emb_col, active_trees, params_root, rag_root in unimol_active:
            params_path = params_root / col_name / "metrics" / "best_params.json"
            if not params_path.exists():
                print(f"\nSKIP {col_name}: best_params.json not found", flush=True)
                continue
            best_params = json.loads(params_path.read_text())

            print(f"\nBuilding features for {col_name} ...", flush=True)
            rag_tr = pd.read_csv(str(rag_root / f"{col_name}_train_rag.csv"))
            rag_te = pd.read_csv(str(rag_root / f"{col_name}_test_rag.csv"))
            X_tr, y_tr = build_unimol_X(tr_emb, tr_pc, rag_tr, emb_col, le_u)
            X_te, y_te = build_unimol_X(te_emb, te_pc, rag_te, emb_col, le_u)
            print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}", flush=True)

            for tree in active_trees:
                print(f"  {col_name} | {tree} ...", flush=True)
                m = evaluate(make_model(tree, best_params[tree], n_cls_u),
                             X_tr, y_tr, X_te, y_te, n_cls_u)
                results.append({"embedding": col_name, "tree": tree, **m})
                print(f"    AUC={m['AUC']}  Acc={m['Acc']}  F1mac={m['F1_mac']}  "
                      f"F1mic={m['F1_mic']}  MCC={m['MCC']}  Prec={m['Prec']}  Rec={m['Rec']}",
                      flush=True)

    # ── Save & print ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}", flush=True)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
