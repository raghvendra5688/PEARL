"""
PC-Only Baseline Modeling Pipeline

Addresses editor comment (see manuscript/editor_response_suggestions.md, Section 2.2):
PEARL's tree-classifier stage is only ever evaluated on [CLM embedding + PC features]
or [CLM embedding + PC features + RAFE]; it never reports what the 473-dim engineered
physicochemical (PC) feature vector alone -- 148 filtered RDKit descriptors, ~30 NetworkX
graph features, 128-bit Morgan fingerprints, 167-bit MACCS keys -- can achieve with no
CLM embedding at all. This script trains that missing baseline directly from the cleaned
SMILES splits in data/clean/, independent of any finetuned CLM checkpoint, so it can run
without the externally-hosted PEARL_EXTRAS artifacts.

Covers all four PEARL datasets:
- BACE    (binary,   label: Class)
- BBBP    (binary,   label: p_np)
- ClinTox (binary,   label: FDA_APPROVED)
- Flavor  (multiclass, label: Canonicalized Taste)

Usage:
    python pc_only_modelling.py --dataset {bace,bbbp,clintox,flavor,all}
"""

import os
import json
import logging
import argparse
import contextlib
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, precision_recall_curve, average_precision_score,
    make_scorer,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_JOBS = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

CLEAN_ROOT = BASE_DIR / "data" / "clean"
FEATURE_CACHE_ROOT = BASE_DIR / "data" / "pc_only_features"
OUTPUT_ROOT = BASE_DIR / "results" / "pc_only"

DATASET_CONFIG = {
    "bace": {
        "clean_dir": CLEAN_ROOT / "bace_datasets",
        "file_prefix": "",
        "smiles_col": "Standardized SMILES",
        "label_col": "Class",
        "task": "binary",
    },
    "bbbp": {
        "clean_dir": CLEAN_ROOT / "bbbp_datasets",
        "file_prefix": "",
        "smiles_col": "Standardized SMILES",
        "label_col": "p_np",
        "task": "binary",
    },
    "clintox": {
        "clean_dir": CLEAN_ROOT / "clintox_datasets",
        "file_prefix": "",
        "smiles_col": "Standardized SMILES",
        "label_col": "FDA_APPROVED",
        "task": "binary",
    },
    "flavor": {
        "clean_dir": CLEAN_ROOT / "flavor_datasets",
        "file_prefix": "",
        "smiles_col": "Standardized SMILES",
        "label_col": "Canonicalized Taste",
        "task": "multiclass",
    },
}

SPLITS = ["train", "valid", "test"]

MODEL_COLORS = {"XGBoost": "tab:blue", "LightGBM": "tab:green", "CatBoost": "tab:red"}

# Same exclusion list used in pc_feature_extraction_ft_model_refactored.py, kept in
# sync so the PC-only baseline and the CLM+PC pipelines use an identical descriptor set.
USELESS_COLS = [
    'MaxPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI',
    'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
    'BCUT2D_MRLOW', 'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9',
    'fr_barbitur', 'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide',
    'fr_isothiocyan', 'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd',
    'fr_thiocyan', 'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt',
    'NumValenceElectrons', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
    'Chi1v', 'Chi2n', 'Kappa1', 'LabuteASA', 'HeavyAtomCount', 'MolMR',
    'Chi3n', 'BertzCT', 'Chi2v', 'Chi4n', 'HallKierAlpha', 'Chi3v',
    'Chi4v', 'MinAbsPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
    'FpDensityMorgan2', 'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile',
    'SlogP_VSA6', 'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene',
    'VSA_EState6', 'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors',
    'fr_amide', 'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_COO2', 'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]
RD_DESC_NAMES = [d[0] for d in Descriptors.descList if d[0] not in USELESS_COLS]

_DUMMY_GRAPH_KEYS: Optional[List[str]] = None


def setup_logging(log_dir: Path, log_name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_dir / log_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        logging.error(f"Error parsing SMILES '{smiles}': {e}")
        return None


def compute_rdkit_descriptors(mol: Chem.Mol) -> List[float]:
    try:
        return [func(mol) for name, func in Descriptors.descList if name not in USELESS_COLS]
    except Exception as e:
        logging.error(f"Error computing RDKit descriptors: {e}")
        return [np.nan] * len(RD_DESC_NAMES)


def compute_graph_features(mol: Chem.Mol) -> Dict[str, float]:
    try:
        adj = rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        n = G.number_of_nodes()
        e = G.number_of_edges()
        feats: Dict[str, float] = {}

        feats["graph_diameter"] = nx.diameter(G) if nx.is_connected(G) and n > 1 else 0
        feats["avg_shortest_path"] = (
            nx.average_shortest_path_length(G) if nx.is_connected(G) and n > 1 else 0
        )
        feats["num_cycles"] = len(nx.cycle_basis(G))
        feats["num_chains"] = len(list(nx.chain_decomposition(G)))
        feats["clustering_coefficients"] = nx.average_clustering(G) if n > 1 else 0
        feats["wiener_index"] = nx.wiener_index(G) if n > 1 else 0
        feats["max_degree"] = max(dict(G.degree()).values()) if n > 0 else 0

        dc = list(nx.degree_centrality(G).values())
        feats["avg_degree_centrality"] = np.mean(dc) if dc else 0

        bc = np.array(list(nx.betweenness_centrality(G).values()))
        bc = bc[np.isfinite(bc)]
        feats["avg_betweenness_centrality"] = bc.mean() if bc.size else 0
        feats["betweenness_mean"] = bc.mean() if bc.size else 0
        feats["betweenness_std"] = bc.std() if bc.size > 1 else 0

        lc = np.array(list(nx.load_centrality(G).values()))
        lc = lc[np.isfinite(lc)]
        feats["avg_load_centrality"] = lc.mean() if lc.size else 0

        if nx.is_connected(G):
            cc = list(nx.closeness_centrality(G).values())
            feats["closeness_mean"] = np.mean(cc) if cc else 0
        else:
            feats["closeness_mean"] = 0

        try:
            ev = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
            ev = ev[np.isfinite(ev)]
            feats["eigenvector_mean"] = ev.mean() if ev.size else 0
            feats["avg_eigen_centrality"] = ev.mean() if ev.size else 0
        except Exception:
            feats["eigenvector_mean"] = 0
            feats["avg_eigen_centrality"] = 0

        try:
            kz = np.array(list(nx.katz_centrality(G, max_iter=1000).values()))
            kz = kz[np.isfinite(kz)]
            feats["katz_centrality_std"] = kz.std() if kz.size > 1 else 0
        except Exception:
            feats["katz_centrality_std"] = 0

        cycles = nx.cycle_basis(G)
        cycle_lengths = [len(c) for c in cycles]
        for k in [1, 2, 3, 4, 5]:
            feats[f"ring_{k}"] = sum(1 for l in cycle_lengths if l == k)

        try:
            aromatic = [c for c in cycles if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in c)]
            feats["num_aromatic_rings"] = len(aromatic)
        except Exception:
            feats["num_aromatic_rings"] = 0

        try:
            non_aromatic = [c for c in cycles if not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in c)]
            feats["num_non_aromatic_rings"] = len(non_aromatic)
        except Exception:
            feats["num_non_aromatic_rings"] = 0

        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        cnt = Counter(atoms)
        feats["heteroatom_ratio"] = (n - cnt.get("C", 0)) / n if n > 0 else 0
        feats["average_carbon"] = cnt.get("C", 0) / n if n > 0 else 0
        feats["average_oxygen"] = cnt.get("O", 0) / n if n > 0 else 0
        feats["average_nitrogen"] = cnt.get("N", 0) / n if n > 0 else 0
        feats["average_sulphur"] = cnt.get("S", 0) / n if n > 0 else 0

        single = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
        double = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        feats["num_single_bonds"] = single / e if e > 0 else 0
        feats["num_double_bonds"] = double / e if e > 0 else 0

        return feats
    except Exception as e:
        logging.error(f"Error computing graph features: {e}")
        return {}


def get_dummy_graph_keys() -> List[str]:
    global _DUMMY_GRAPH_KEYS
    if _DUMMY_GRAPH_KEYS is None:
        dummy_mol = Chem.MolFromSmiles("CC")
        _DUMMY_GRAPH_KEYS = list(compute_graph_features(dummy_mol).keys())
    return _DUMMY_GRAPH_KEYS


def compute_fingerprints(mol: Chem.Mol, morgan_gen: Any) -> np.ndarray:
    try:
        morgan_fp = np.array(morgan_gen.GetFingerprint(mol))
        maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        return np.concatenate([morgan_fp, maccs_fp])
    except Exception as e:
        logging.error(f"Error computing fingerprints: {e}")
        return np.zeros(295)


def extract_pc_features(smiles: pd.Series) -> pd.DataFrame:
    """Compute the 473-dim PC feature vector (RDKit + graph + Morgan + MACCS) from raw SMILES."""
    mols = smiles.apply(safe_mol_from_smiles)
    morgan_gen = GetMorganGenerator(radius=2, fpSize=128)
    dummy_graph_keys = get_dummy_graph_keys()

    rdkit_desc, graph_desc, fps = [], [], []
    for idx, mol in enumerate(mols):
        if mol is None:
            rdkit_desc.append([np.nan] * len(RD_DESC_NAMES))
            graph_desc.append({k: np.nan for k in dummy_graph_keys})
            fps.append(np.zeros(295))
            continue
        rdkit_desc.append(compute_rdkit_descriptors(mol))
        graph_desc.append(compute_graph_features(mol))
        fps.append(compute_fingerprints(mol, morgan_gen))
        if (idx + 1) % 2000 == 0:
            logging.info(f"  featurized {idx + 1}/{len(mols)} molecules")

    rd_df = pd.DataFrame(rdkit_desc, columns=RD_DESC_NAMES)
    graph_df = pd.DataFrame(graph_desc)
    fp_df = pd.DataFrame(fps, columns=[f"FP_{i}" for i in range(295)])
    feat_df = pd.concat([rd_df, graph_df, fp_df], axis=1)
    logging.info(f"PC feature matrix: {feat_df.shape} (target 473-dim: 148 RDKit + ~30 graph + 128 Morgan + 167 MACCS)")
    return feat_df


def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)
    return X


def load_or_compute_features(dataset: str, split: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load cleaned SMILES+label split and attach cached (or freshly computed) PC features."""
    clean_path = cfg["clean_dir"] / f"{split}_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean split: {clean_path}")

    raw_df = pd.read_csv(clean_path)
    raw_df = raw_df.dropna(subset=[cfg["smiles_col"], cfg["label_col"]]).reset_index(drop=True)

    cache_dir = FEATURE_CACHE_ROOT / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_pc_features.csv"

    if cache_path.exists():
        logging.info(f"[{dataset}|{split}] Loading cached PC features: {cache_path}")
        feat_df = pd.read_csv(cache_path)
        if len(feat_df) != len(raw_df):
            logging.warning(f"[{dataset}|{split}] Cache size mismatch, recomputing features")
            feat_df = extract_pc_features(raw_df[cfg["smiles_col"]])
            feat_df.to_csv(cache_path, index=False)
    else:
        logging.info(f"[{dataset}|{split}] Computing PC features for {len(raw_df)} molecules")
        feat_df = extract_pc_features(raw_df[cfg["smiles_col"]])
        feat_df.to_csv(cache_path, index=False)

    feat_df = sanitize_features(feat_df)
    feat_df = pd.concat([feat_df, raw_df[[cfg["label_col"]]].reset_index(drop=True)], axis=1)
    return feat_df


def optimize_model(
    trial: optuna.Trial, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
    sample_weights: np.ndarray, task: str, n_classes: int,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    }

    try:
        if model_type == "xgb":
            objective = "multi:softprob" if task == "multiclass" else "binary:logistic"
            extra = {"num_class": n_classes} if task == "multiclass" else {}
            model = xgb.XGBClassifier(
                objective=objective, eval_metric="mlogloss" if task == "multiclass" else "logloss",
                random_state=RANDOM_SEED, tree_method="hist", n_jobs=N_JOBS, **extra, **params,
            )
        elif model_type == "lgb":
            objective = "multiclass" if task == "multiclass" else "binary"
            extra = {"num_class": n_classes} if task == "multiclass" else {}
            model = lgb.LGBMClassifier(
                objective=objective, class_weight="balanced", random_state=RANDOM_SEED,
                n_jobs=N_JOBS, verbosity=-1, **extra, **params,
            )
        else:
            loss_fn = "MultiClass" if task == "multiclass" else "Logloss"
            model = cb.CatBoostClassifier(
                loss_function=loss_fn, auto_class_weights="Balanced", random_seed=RANDOM_SEED,
                verbose=0, thread_count=N_JOBS, **params,
            )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        mcc_scorer = make_scorer(matthews_corrcoef)
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            w_fold = sample_weights[train_idx]
            model.fit(X_train[train_idx], y_train[train_idx], sample_weight=w_fold)
            y_pred = model.predict(X_train[val_idx])
            fold_scores.append(matthews_corrcoef(y_train[val_idx], y_pred))
        return float(np.mean(fold_scores))

    except Exception as e:
        logging.error(f"Error in trial optimization for {model_type}: {e}")
        raise optuna.exceptions.TrialPruned()


def run_optimization(model_type: str, X_train, y_train, sample_weights, task, n_classes) -> Dict[str, Any]:
    logging.info(f"Running Optuna optimization for {model_type} ({OPTUNA_TRIALS} trials)")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: optimize_model(t, model_type, X_train, y_train, sample_weights, task, n_classes),
        n_trials=OPTUNA_TRIALS,
    )
    logging.info(f"{model_type} optimization complete. Best MCC: {study.best_value:.4f}")
    return study.best_params


def build_model(name: str, task: str, n_classes: int, params: Dict[str, Any]):
    if name == "XGBoost":
        objective = "multi:softprob" if task == "multiclass" else "binary:logistic"
        extra = {"num_class": n_classes} if task == "multiclass" else {}
        return xgb.XGBClassifier(
            objective=objective, eval_metric="mlogloss" if task == "multiclass" else "logloss",
            random_state=RANDOM_SEED, tree_method="hist", n_jobs=N_JOBS, **extra, **params,
        )
    if name == "LightGBM":
        objective = "multiclass" if task == "multiclass" else "binary"
        extra = {"num_class": n_classes} if task == "multiclass" else {}
        return lgb.LGBMClassifier(
            objective=objective, class_weight="balanced", random_state=RANDOM_SEED,
            n_jobs=N_JOBS, verbosity=-1, **extra, **params,
        )
    loss_fn = "MultiClass" if task == "multiclass" else "Logloss"
    return cb.CatBoostClassifier(
        loss_function=loss_fn, auto_class_weights="Balanced", random_seed=RANDOM_SEED,
        verbose=0, thread_count=N_JOBS, **params,
    )


def train_and_evaluate(
    name: str, model: Any, X_train, y_train, X_test, y_test, sample_weights,
    task: str, n_classes: int, out_dir: Path, dataset: str,
) -> np.ndarray:
    logging.info(f"Training {name} for {dataset} (PC-only)")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    if task == "multiclass":
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    else:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "AUC": round(auc, 3),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 3),
        "F1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
        "F1_micro": round(f1_score(y_test, y_pred, average="micro"), 3),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 3),
    }
    logging.info(f"[{dataset} | PC-only] {name} metrics: {metrics}")

    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics" / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import joblib
    joblib.dump(model, out_dir / "models" / f"{name}.pkl")

    return y_proba if task == "multiclass" else y_proba[:, 1]


@contextlib.contextmanager
def plot_context():
    try:
        yield
    finally:
        plt.close('all')


def plot_binary_curves(predictions: Dict[str, np.ndarray], y_test: np.ndarray, dataset: str, out_dir: Path) -> None:
    with plot_context():
        plt.figure(figsize=(8, 6))
        for name, prob in predictions.items():
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=MODEL_COLORS[name], linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.title(f"{dataset.upper()} PC-Only ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        (out_dir / "ROC_Curves").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "ROC_Curves" / "roc_all_models.pdf", dpi=300, bbox_inches='tight')

    with plot_context():
        plt.figure(figsize=(8, 6))
        for name, prob in predictions.items():
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=MODEL_COLORS[name], linewidth=2)
        plt.title(f"{dataset.upper()} PC-Only Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        (out_dir / "PR_Curves").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "PR_Curves" / "pr_all_models.pdf", dpi=300, bbox_inches='tight')


def run_dataset(dataset: str) -> Dict[str, Dict[str, Any]]:
    cfg = DATASET_CONFIG[dataset]
    task = cfg["task"]
    out_dir = OUTPUT_ROOT / f"{dataset.upper()}_PC_Only_Results"
    setup_logging(out_dir / "logs", "pc_only_modelling.log")

    logging.info("=" * 80)
    logging.info(f"PC-Only baseline: {dataset} (task={task})")
    logging.info("=" * 80)

    splits = {s: load_or_compute_features(dataset, s, cfg) for s in SPLITS}
    train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]

    label_encoder = None
    if task == "multiclass":
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([train_df[cfg["label_col"]], valid_df[cfg["label_col"]], test_df[cfg["label_col"]]]))
        y_train = label_encoder.transform(train_df[cfg["label_col"]])
        y_test = label_encoder.transform(test_df[cfg["label_col"]])
        n_classes = len(label_encoder.classes_)
        logging.info(f"Classes: {list(label_encoder.classes_)}")
    else:
        y_train = train_df[cfg["label_col"]].astype(int).values
        y_test = test_df[cfg["label_col"]].astype(int).values
        n_classes = 2

    feature_cols = [c for c in train_df.columns if c != cfg["label_col"]]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    logging.info(f"Feature matrix shapes | Train={X_train.shape}, Test={X_test.shape}")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    best_params = {
        "XGBoost": run_optimization("xgb", X_train, y_train, sample_weights, task, n_classes),
        "LightGBM": run_optimization("lgb", X_train, y_train, sample_weights, task, n_classes),
        "CatBoost": run_optimization("cb", X_train, y_train, sample_weights, task, n_classes),
    }
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics" / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    predictions = {}
    all_metrics = {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        model = build_model(name, task, n_classes, best_params[name])
        proba = train_and_evaluate(
            name, model, X_train, y_train, X_test, y_test, sample_weights,
            task, n_classes, out_dir, dataset,
        )
        predictions[name] = proba
        with open(out_dir / "metrics" / f"{name}_metrics.json") as f:
            all_metrics[name] = json.load(f)

    if task == "binary":
        plot_binary_curves(predictions, y_test, dataset, out_dir)

    logging.info(f"PC-Only baseline complete for {dataset}: {all_metrics}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="PC-only (no CLM embedding) tree-classifier baseline")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    summary = {}
    for dataset in datasets:
        try:
            summary[dataset] = run_dataset(dataset)
        except Exception as e:
            logging.error(f"Failed dataset {dataset}: {e}")
            raise

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, model_metrics in summary.items():
        for model_name, metrics in model_metrics.items():
            rows.append({"Dataset": dataset, "Model": model_name, **metrics})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_ROOT / "pc_only_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
