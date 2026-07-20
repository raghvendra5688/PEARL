"""
PyTorch Geometric GIN Baseline

Addresses editor comment (see manuscript revision notes, Section 2 of
editor_response_suggestions.md): a second in-house graph-based baseline
alongside Chemprop's D-MPNN, so the "does a graph model beat/match the CLM
pipeline" comparison isn't resting on a single GNN architecture. Uses GIN
(Graph Isomorphism Network) -- a standard, strong message-passing baseline --
on the same data/clean/ splits used by every other PEARL baseline.

Pipeline per dataset:
1. RDKit SMILES -> PyG Data objects (atom features: atomic number, degree,
   formal charge, hybridization, aromaticity, H count, ring membership; bonds
   are unfeaturized/binary edges, consistent with a plain GIN's edge-agnostic
   message passing).
2. Optuna search (default 20 trials, matching every other baseline's rigor)
   over num_layers/hidden_dim/dropout/lr.
3. Final training run with the best config and more epochs, early-stopped on
   validation loss, evaluated on the held-out test split.
4. Metrics computed with the *same* sklearn/scipy functions used by
   pc_only_modelling.py / chemprop_baseline.py, plus a bootstrapped 95% CI
   (scripts/common/bootstrap_ci.py) matching PEARL_paper.tex's own protocol,
   so comparisons across all baselines are on equal statistical footing.

Usage:
    python gcn_baseline.py --dataset {bace,bbbp,clintox,flavor,herg,dili,caco2,half_life,all}
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    matthews_corrcoef, mean_absolute_error, r2_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "common"))
from bootstrap_ci import bootstrap_ci  # noqa: E402

CLEAN_ROOT = BASE_DIR / "data" / "clean"
OUTPUT_ROOT = BASE_DIR / "results" / "gnn" / "gcn"
# Same convention as chemprop_baseline.py: the original PEARL_EXTRAS path was
# found read-only from this host; new GNN-baseline artifacts go here instead.
PEARL_EXTRAS = Path(os.getenv("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
OPTUNA_TRIALS = int(os.getenv("GCN_OPTUNA_TRIALS", "20"))
SEARCH_EPOCHS = int(os.getenv("GCN_SEARCH_EPOCHS", "30"))
FINAL_EPOCHS = int(os.getenv("GCN_FINAL_EPOCHS", "100"))
PATIENCE = int(os.getenv("GCN_PATIENCE", "15"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS = ["train", "valid", "test"]

DATASET_CONFIG = {
    "bace": {"clean_dir": CLEAN_ROOT / "bace_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "Class", "task": "binary"},
    "bbbp": {"clean_dir": CLEAN_ROOT / "bbbp_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "p_np", "task": "binary"},
    "clintox": {"clean_dir": CLEAN_ROOT / "clintox_datasets", "smiles_col": "Standardized SMILES",
                "label_col": "FDA_APPROVED", "task": "binary"},
    "flavor": {"clean_dir": CLEAN_ROOT / "flavor_datasets", "smiles_col": "Standardized SMILES",
               "label_col": "Canonicalized Taste", "task": "multiclass"},
    "herg": {"clean_dir": CLEAN_ROOT / "herg_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "hERG_Inhib", "task": "binary"},
    "dili": {"clean_dir": CLEAN_ROOT / "dili_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "DILI_Label", "task": "binary"},
    "caco2": {"clean_dir": CLEAN_ROOT / "caco2_datasets", "smiles_col": "Standardized SMILES",
              "label_col": "Caco2_LogPapp", "task": "regression", "target_transform": None},
    "half_life": {"clean_dir": CLEAN_ROOT / "half_life_datasets", "smiles_col": "Standardized SMILES",
                  "label_col": "Half_Life_Hours", "task": "regression", "target_transform": "log1p"},
}

ATOM_LIST = list(range(1, 101))  # atomic numbers 1..100, covers all organic + common heteroatoms
HYBRIDIZATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                  Chem.HybridizationType.SP3D2]
NUM_ATOM_FEATURES = (
    (len(ATOM_LIST) + 1) + (len(list(range(6))) + 1) + (len(HYBRIDIZATIONS) + 1)
    + (len(list(range(6))) + 1) + 1 + 1 + 1
)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_dir / "gcn_baseline.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def one_hot(value, choices) -> List[int]:
    return [1 if value == c else 0 for c in choices] + [1 if value not in choices else 0]


def atom_features(atom: Chem.Atom) -> List[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + one_hot(atom.GetDegree(), list(range(6)))
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + one_hot(atom.GetTotalNumHs(), list(range(6)))
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0]
        + [float(atom.GetFormalCharge())]
    )


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    if not edges:  # single-atom molecule, add a self-loop so message passing doesn't crash
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def apply_target_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.log1p(y) if transform == "log1p" else y


def invert_target_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.expm1(y) if transform == "log1p" else y


def build_dataset(df: pd.DataFrame, cfg: Dict[str, Any], label_encoder: Optional[LabelEncoder]) -> List[Data]:
    graphs = []
    labels = df[cfg["label_col"]]
    if cfg["task"] == "multiclass":
        y = label_encoder.transform(labels)
    elif cfg["task"] == "regression":
        y = apply_target_transform(labels.astype(float).values, cfg.get("target_transform"))
    else:
        y = labels.astype(int).values

    for smiles, target in zip(df[cfg["smiles_col"]], y):
        g = smiles_to_graph(smiles)
        if g is None:
            continue
        g.y = torch.tensor([target], dtype=torch.float if cfg["task"] != "multiclass" else torch.long)
        graphs.append(g)
    return graphs


class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer in range(num_layers):
            d_in = in_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


def task_loss_fn(task: str):
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def run_epoch(model, loader, optimizer, loss_fn, task: str, train: bool) -> float:
    model.train(mode=train)
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(batch)
            if task == "multiclass":
                loss = loss_fn(out, batch.y)
            else:
                loss = loss_fn(out.squeeze(-1), batch.y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, task: str) -> np.ndarray:
    model.eval()
    outs = []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        if task == "multiclass":
            outs.append(F.softmax(out, dim=-1).cpu().numpy())
        elif task == "binary":
            outs.append(torch.sigmoid(out.squeeze(-1)).cpu().numpy())
        else:
            outs.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_model(train_graphs, valid_graphs, cfg: Dict[str, Any], hparams: Dict[str, Any],
                 n_classes: int, epochs: int, patience: int):
    out_dim = n_classes if cfg["task"] == "multiclass" else 1
    model = GIN(NUM_ATOM_FEATURES, hparams["hidden_dim"], hparams["num_layers"], out_dim, hparams["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    loss_fn = task_loss_fn(cfg["task"])

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=128, shuffle=False)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        run_epoch(model, train_loader, optimizer, loss_fn, cfg["task"], train=True)
        val_loss = run_epoch(model, valid_loader, optimizer, loss_fn, cfg["task"], train=False)
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def optuna_objective(trial: optuna.Trial, train_graphs, valid_graphs, cfg: Dict[str, Any], n_classes: int) -> float:
    hparams = {
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }
    _, best_val = train_model(train_graphs, valid_graphs, cfg, hparams, n_classes, SEARCH_EPOCHS, patience=8)
    return best_val


def move_large_files_to_extras(save_dir: Path, dataset: str) -> None:
    for f in save_dir.rglob("*"):
        if not f.is_file() or f.stat().st_size <= LARGE_FILE_THRESHOLD_BYTES:
            continue
        size_mb = f.stat().st_size / 1e6
        try:
            dest_dir = PEARL_EXTRAS / "gnn_checkpoints" / "gcn" / dataset
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f.name
            shutil.move(str(f), str(dest))
            f.symlink_to(dest)
            logging.info(f"Moved large file ({size_mb:.1f}MB) to {dest}, left symlink at {f}")
        except OSError as e:
            logging.warning(
                f"Could not move large file ({size_mb:.1f}MB) at {f} to PEARL_EXTRAS "
                f"({PEARL_EXTRAS}): {e}. Left in place under results/."
            )


def spearman_metric(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


def compute_metrics_and_ci(task: str, y_test_orig: np.ndarray, y_pred: np.ndarray,
                            n_classes: int, target_transform: Optional[str]) -> Dict[str, Any]:
    if task == "regression":
        y_pred_orig = invert_target_transform(y_pred, target_transform)
        point = {
            "MAE": round(float(mean_absolute_error(y_test_orig, y_pred_orig)), 4),
            "R2": round(float(r2_score(y_test_orig, y_pred_orig)), 4),
            "Spearman": round(float(spearman_metric(y_test_orig, y_pred_orig)), 4),
        }
        ci = {
            "R2": bootstrap_ci(y_test_orig, y_pred_orig, r2_score, stratified=False),
            "Spearman": bootstrap_ci(y_test_orig, y_pred_orig, spearman_metric, stratified=False),
        }
        return {"point": point, "ci": ci}

    if task == "multiclass":
        y_pred_class = y_pred.argmax(axis=1)
        auc = roc_auc_score(y_test_orig, y_pred, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    else:
        y_pred_class = (y_pred >= 0.5).astype(int)
        auc = roc_auc_score(y_test_orig, y_pred)

    point = {
        "Accuracy": round(accuracy_score(y_test_orig, y_pred_class), 3),
        "AUC": round(float(auc), 3),
        "Precision": round(precision_score(y_test_orig, y_pred_class, average="macro", zero_division=0), 3),
        "Recall": round(recall_score(y_test_orig, y_pred_class, average="macro", zero_division=0), 3),
        "F1_macro": round(f1_score(y_test_orig, y_pred_class, average="macro"), 3),
        "F1_micro": round(f1_score(y_test_orig, y_pred_class, average="micro"), 3),
        "MCC": round(matthews_corrcoef(y_test_orig, y_pred_class), 3),
    }
    ci = {"MCC": bootstrap_ci(y_test_orig, y_pred_class, matthews_corrcoef, stratified=True)}
    if task == "binary":
        ci["AUC"] = bootstrap_ci(y_test_orig, y_pred, roc_auc_score, stratified=True)
    return {"point": point, "ci": ci}


def run_dataset(dataset: str) -> Dict[str, Any]:
    cfg = DATASET_CONFIG[dataset]
    out_dir = OUTPUT_ROOT / f"{dataset.upper()}_GCN_Results"
    setup_logging(out_dir / "logs")

    logging.info("=" * 80)
    logging.info(f"GIN baseline: {dataset} (task={cfg['task']}, device={DEVICE})")
    logging.info("=" * 80)

    raw = {s: pd.read_csv(cfg["clean_dir"] / f"{s}_clean.csv") for s in SPLITS}

    label_encoder, n_classes = None, 0
    if cfg["task"] == "multiclass":
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([raw[s][cfg["label_col"]] for s in SPLITS]))
        n_classes = len(label_encoder.classes_)

    graphs = {s: build_dataset(raw[s], cfg, label_encoder) for s in SPLITS}
    logging.info(f"Graphs built: " + ", ".join(f"{s}={len(graphs[s])}" for s in SPLITS))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: optuna_objective(t, graphs["train"], graphs["valid"], cfg, n_classes),
        n_trials=OPTUNA_TRIALS,
    )

    best_hparams = study.best_params
    logging.info(f"Best hyperparameters: {best_hparams} (val loss {study.best_value:.4f})")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics" / "best_params.json", "w") as f:
        json.dump(best_hparams, f, indent=2)

    model, _ = train_model(graphs["train"], graphs["valid"], cfg, best_hparams, n_classes,
                            FINAL_EPOCHS, patience=PATIENCE)

    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")

    test_loader = DataLoader(graphs["test"], batch_size=128, shuffle=False)
    y_pred = predict(model, test_loader, cfg["task"])
    y_test_orig = raw["test"][cfg["label_col"]].values
    if cfg["task"] == "multiclass":
        y_test_orig = label_encoder.transform(y_test_orig)
    elif cfg["task"] == "regression":
        y_test_orig = y_test_orig.astype(float)
    else:
        y_test_orig = y_test_orig.astype(int)

    results = compute_metrics_and_ci(cfg["task"], y_test_orig, y_pred, n_classes, cfg.get("target_transform"))
    logging.info(f"[{dataset} | GIN] point metrics: {results['point']}")
    logging.info(f"[{dataset} | GIN] CI: {results['ci']}")
    with open(out_dir / "metrics" / "test_metrics.json", "w") as f:
        json.dump(results["point"], f, indent=2)
    with open(out_dir / "metrics" / "ci_metrics.json", "w") as f:
        json.dump(results["ci"], f, indent=2)

    move_large_files_to_extras(final_dir, dataset)

    return results["point"]


def main():
    parser = argparse.ArgumentParser(description="PyG GIN baseline, in-house on PEARL splits")
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
    rows = [{"Dataset": dataset, **metrics} for dataset, metrics in summary.items()]
    new_df = pd.DataFrame(rows)
    summary_path = OUTPUT_ROOT / "gcn_summary.csv"
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        existing_df = existing_df[~existing_df["Dataset"].isin(datasets)]
        summary_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        summary_df = new_df
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
