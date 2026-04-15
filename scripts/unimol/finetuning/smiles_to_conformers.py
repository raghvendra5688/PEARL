"""
SMILES to 3D Conformer Preprocessing — Additional Modality for Uni-Mol

Generates RDKit ETKDGv3 + MMFF94 optimised 3D conformers from SMILES for all
EffiChem-2.0 datasets and ZINC-250k. This produces the ADDITIONAL MODALITY
that Uni-Mol uses as input: explicit 3D atomic coordinates derived from SMILES.

ChemBERTa and MolFormer only see SMILES token sequences.
Uni-Mol additionally sees:
  1. Atom types (element symbols)
  2. 3D Cartesian coordinates (x, y, z per atom) from RDKit ETKDGv3 + MMFF94
  3. Inter-atomic distances (derived internally by Uni-Mol)

Outputs:
  data/conformers/{dataset}/{prefix}_{split}_conformers.pkl
      dict: {smiles -> {"atoms": [...], "coordinates": np.ndarray(N_atoms, 3)}}
  EffiChem_Extras/zinc_conformers/zinc250k_conformers.pkl  (large — outside git)

Usage:
    python smiles_to_conformers.py --dataset bace
    python smiles_to_conformers.py --dataset zinc250k
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

# Suppress RDKit C++ invariant violation messages printed to stderr
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT        = Path(__file__).resolve().parent.parent.parent.parent
CLEAN_ROOT       = REPO_ROOT / "data" / "clean"
CONFORM_ROOT     = REPO_ROOT / "data" / "conformers"
ZINC_CSV         = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT      = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
ZINC_CONFORM_DIR = EXTRAS_ROOT / "zinc_conformers"
LOG_DIR          = REPO_ROOT / "logs"

SMILES_COL  = "Standardized SMILES"
RANDOM_SEED = 42

DATASET_CFG = {
    "bace": {
        "clean_dir":   "bace_datasets",
        "file_prefix": "bace",
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "bbbp": {
        "clean_dir":   "bbbp_datasets",
        "file_prefix": "bbbp",
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "clintox": {
        "clean_dir":   "clintox_datasets",
        "file_prefix": "clintox",
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "flavor": {
        "clean_dir":   "flavor_datasets",
        "file_prefix": "fart",
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
}

ATOMIC_SYMBOLS = {
    1: "H",  6: "C",  7: "N",  8: "O",  9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


# ── Conformer generation ───────────────────────────────────────────────────────

def smiles_to_unimol_input(
    smiles: str,
    remove_hs: bool = True,
    seed: int = RANDOM_SEED,
) -> Optional[Dict]:
    """
    Convert a SMILES string to Uni-Mol input format.

    Returns a dict with:
        atoms       : List[str]      — element symbols, e.g. ["C", "O", "N", ...]
        coordinates : np.ndarray     — shape (N_atoms, 3), float32
        smiles      : str            — original SMILES

    Returns None if conformer generation fails entirely.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)

    result = -1

    # ETKDGv3 — disable useExpTorsionAnglePrefs to avoid BFGS invariant
    # violation crash in molecules with unusual torsion geometries
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed                = seed
        params.numThreads                = 1
        params.useSmallRingTorsions      = True
        params.useMacrocycleTorsions     = True
        params.useExpTorsionAnglePrefs   = False   # prevents BFGS crash
        params.useBasicKnowledge         = True
        result = AllChem.EmbedMolecule(mol_h, params)
    except Exception:
        result = -1

    if result == -1:
        # Fallback: ETKDGv2 (no experimental torsion preferences)
        try:
            params2 = AllChem.ETKDGv2()
            params2.randomSeed = seed
            params2.numThreads = 1
            result = AllChem.EmbedMolecule(mol_h, params2)
        except Exception:
            result = -1

    if result == -1:
        # Fallback: random distance geometry (most permissive)
        try:
            params3 = AllChem.EmbedParameters()
            params3.randomSeed = seed
            result = AllChem.EmbedMolecule(mol_h, params3)
        except Exception:
            result = -1

    if result == -1:
        return None

    # MMFF94 geometry optimisation (wrapped — can fail for exotic molecules)
    try:
        ff_result = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94", maxIters=2000)
        if ff_result == -1:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
    except Exception:
        pass  # Use unoptimised ETKDG coordinates

    if remove_hs:
        mol_h = Chem.RemoveHs(mol_h)

    conf = mol_h.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=np.float32)
    atoms  = [mol_h.GetAtomWithIdx(i).GetSymbol() for i in range(mol_h.GetNumAtoms())]

    return {"atoms": atoms, "coordinates": coords, "smiles": smiles}


def batch_smiles_to_unimol(
    smiles_list: List[str],
    remove_hs: bool = True,
    log_every: int = 500,
) -> Tuple[Dict[str, Dict], int]:
    """
    Process a list of SMILES into Uni-Mol input format.

    Returns:
        conformer_dict : {smiles -> unimol_input_dict}  (None values for failures)
        n_failed       : int
    """
    result = {}
    n_failed = 0

    for i, smi in enumerate(smiles_list):
        data = smiles_to_unimol_input(smi, remove_hs=remove_hs)
        result[smi] = data
        if data is None:
            n_failed += 1

        if (i + 1) % log_every == 0:
            logging.info(
                f"  {i+1}/{len(smiles_list)} processed "
                f"(success={i+1-n_failed}, failed={n_failed})"
            )

    return result, n_failed


# ── Main ───────────────────────────────────────────────────────────────────────

def setup_logging(tag: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"conformer_gen_{tag}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def process_dataset(dataset_key: str) -> None:
    cfg     = DATASET_CFG[dataset_key]
    out_dir = CONFORM_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    for out_split, clean_split in cfg["splits"].items():
        input_csv = CLEAN_ROOT / cfg["clean_dir"] / f"{clean_split}_clean.csv"
        if not input_csv.exists():
            logging.warning(f"Not found: {input_csv}")
            continue

        out_pkl = out_dir / f"{cfg['file_prefix']}_{out_split}_conformers.pkl"
        if out_pkl.exists():
            logging.info(f"Already exists, skipping: {out_pkl.name}")
            continue

        df          = pd.read_csv(str(input_csv))
        smiles_list = df[SMILES_COL].astype(str).tolist()

        logging.info(
            f"\nDataset={dataset_key} split={out_split} "
            f"n={len(smiles_list)} → {out_pkl.name}"
        )

        conformers, n_failed = batch_smiles_to_unimol(smiles_list)

        with open(str(out_pkl), "wb") as f:
            pickle.dump(conformers, f, protocol=4)

        n_ok = len(smiles_list) - n_failed
        logging.info(
            f"  Saved {out_pkl.name} | 3D success={n_ok} | "
            f"2D fallback/failed={n_failed}"
        )


def process_zinc(batch_start: int, batch_end: int) -> None:
    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"ZINC CSV not found: {ZINC_CSV}")

    df          = pd.read_csv(str(ZINC_CSV))
    smiles_list = df["smiles"].astype(str).tolist()

    if batch_end < 0:
        batch_end = len(smiles_list)

    batch_smiles = smiles_list[batch_start:batch_end]
    ZINC_CONFORM_DIR.mkdir(parents=True, exist_ok=True)

    tag     = f"{batch_start}_{batch_end}"
    out_pkl = ZINC_CONFORM_DIR / f"zinc250k_{tag}_conformers.pkl"

    logging.info(
        f"\nZINC-250k [{batch_start}:{batch_end}] "
        f"n={len(batch_smiles)} → {out_pkl.name}"
    )

    conformers, n_failed = batch_smiles_to_unimol(batch_smiles, log_every=2000)

    with open(str(out_pkl), "wb") as f:
        pickle.dump(conformers, f, protocol=4)

    n_ok = len(batch_smiles) - n_failed
    logging.info(f"  Saved | 3D success={n_ok} | failed={n_failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D conformers (Uni-Mol additional modality)"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["bace", "bbbp", "clintox", "flavor", "zinc250k"],
    )
    parser.add_argument(
        "--zinc-start", type=int, default=0,
        help="Start index for ZINC-250k batch (default: 0)",
    )
    parser.add_argument(
        "--zinc-end", type=int, default=-1,
        help="End index for ZINC-250k batch (-1 = full dataset)",
    )
    args = parser.parse_args()

    setup_logging(args.dataset)
    logging.info("=" * 60)
    logging.info(f"Conformer generation | dataset={args.dataset}")
    logging.info(
        "Additional modality: 3D structure (ETKDGv3 + MMFF94) from SMILES"
    )
    logging.info("=" * 60)

    if args.dataset == "zinc250k":
        process_zinc(args.zinc_start, args.zinc_end)
    else:
        process_dataset(args.dataset)


if __name__ == "__main__":
    main()
