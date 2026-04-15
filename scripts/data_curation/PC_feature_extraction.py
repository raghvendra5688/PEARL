"""
Full feature extraction pipeline.

This script extracts:
1. Filtered RDKit descriptors (Descriptors.descList - useless_cols)
2. Graph-theoretic features using NetworkX
3. Morgan fingerprints (radius=2, 128 bits)
4. MACCS fingerprints (167 bits)

The script operates on already cleaned datasets and preserves label alignment.
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = str(REPO_ROOT / "data" / "clean")
FEATURE_ROOT = str(REPO_ROOT / "data" / "features")
LOG_DIR = str(REPO_ROOT / "logs")


DATASET_CONFIG = {
    "bbbp_datasets": ["p_np"],
    "clintox_datasets": ["FDA_APPROVED", "CT_TOX"],
    "flavor_datasets": ["Canonicalized Taste"],
    "bace_datasets": ["Class"]
}

SPLITS = ["train", "test", "valid"]

os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "full_feature_extraction.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Descriptor exclusion list

useless_cols = [
    'MaxPartialCharge','BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI',
    'BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI',
    'BCUT2D_MRLOW','NumRadicalElectrons','SMR_VSA8','SlogP_VSA9',
    'fr_barbitur','fr_benzodiazepine','fr_dihydropyridine','fr_epoxide',
    'fr_isothiocyan','fr_lactam','fr_nitroso','fr_prisulfonamd',
    'fr_thiocyan','MaxEStateIndex','HeavyAtomMolWt','ExactMolWt',
    'NumValenceElectrons','Chi0','Chi0n','Chi0v','Chi1','Chi1n',
    'Chi1v','Chi2n','Kappa1','LabuteASA','HeavyAtomCount','MolMR',
    'Chi3n','BertzCT','Chi2v','Chi4n','HallKierAlpha','Chi3v',
    'Chi4v','MinAbsPartialCharge','MinPartialCharge','MaxAbsPartialCharge',
    'FpDensityMorgan2','FpDensityMorgan3','Phi','Kappa3','fr_nitrile',
    'SlogP_VSA6','NumAromaticCarbocycles','NumAromaticRings','fr_benzene',
    'VSA_EState6','NOCount','fr_C_O','fr_C_O_noCOO','NumHDonors',
    'fr_amide','fr_Nhpyrrole','fr_phenol','fr_phenol_noOrthoHbond',
    'fr_COO2','fr_halogen','fr_diazo','fr_nitro_arom','fr_phos_ester'
]

RD_DESC_NAMES = [d[0] for d in Descriptors.descList if d[0] not in useless_cols]

# Feature functions
def compute_rdkit_descriptors(mol):
    return [func(mol) for name, func in Descriptors.descList if name not in useless_cols]


def compute_graph_features(mol):
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    n = G.number_of_nodes()
    e = G.number_of_edges()

    feats = {}

    # Global graph properties
    feats["graph_diameter"] = nx.diameter(G) if nx.is_connected(G) and n > 1 else 0
    feats["avg_shortest_path"] = (
        nx.average_shortest_path_length(G) if nx.is_connected(G) and n > 1 else 0
    )
    feats["num_cycles"] = len(nx.cycle_basis(G))
    feats["num_chains"] = len(list(nx.chain_decomposition(G)))
    feats["clustering_coefficients"] = nx.average_clustering(G) if n > 1 else 0
    feats["wiener_index"] = nx.wiener_index(G) if n > 1 else 0
    feats["max_degree"] = max(dict(G.degree()).values()) if n > 0 else 0

    # Degree centrality
    dc = list(nx.degree_centrality(G).values())
    feats["avg_degree_centrality"] = np.mean(dc) if dc else 0

    # Betweenness / load
    bc = np.array(list(nx.betweenness_centrality(G).values()))
    bc = bc[np.isfinite(bc)]

    feats["avg_betweenness_centrality"] = bc.mean() if bc.size else 0
    feats["betweenness_mean"] = bc.mean() if bc.size else 0
    feats["betweenness_std"] = bc.std() if bc.size > 1 else 0

    lc = np.array(list(nx.load_centrality(G).values()))
    lc = lc[np.isfinite(lc)]
    feats["avg_load_centrality"] = lc.mean() if lc.size else 0

    # Closeness
    if nx.is_connected(G):
        cc = list(nx.closeness_centrality(G).values())
        feats["closeness_mean"] = np.mean(cc) if cc else 0
    else:
        feats["closeness_mean"] = 0

    # Eigen / Katz
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

    # Ring analysis
    cycles = nx.cycle_basis(G)
    cycle_lengths = [len(c) for c in cycles]

    for k in [1, 2, 3, 4, 5]:
        feats[f"ring_{k}"] = sum(1 for l in cycle_lengths if l == k)

    try:
        aromatic = [
            c for c in cycles
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in c)
        ]
        feats["num_aromatic_rings"] = len(aromatic)
    except Exception:
        feats["num_aromatic_rings"] = 0

    try:
        non_aromatic = [
            c for c in cycles
            if not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in c)
        ]
        feats["num_non_aromatic_rings"] = len(non_aromatic)
    except Exception:
        feats["num_non_aromatic_rings"] = 0

    # Atom composition
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    cnt = Counter(atoms)

    feats["heteroatom_ratio"] = (n - cnt.get("C", 0)) / n if n > 0 else 0
    feats["average_carbon"] = cnt.get("C", 0) / n if n > 0 else 0
    feats["average_oxygen"] = cnt.get("O", 0) / n if n > 0 else 0
    feats["average_nitrogen"] = cnt.get("N", 0) / n if n > 0 else 0
    feats["average_sulphur"] = cnt.get("S", 0) / n if n > 0 else 0

    # Bond statistics
    single = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
    double = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)

    feats["num_single_bonds"] = single / e if e > 0 else 0
    feats["num_double_bonds"] = double / e if e > 0 else 0

    return feats

# Main execution when triggered
def extract_features(df, label_cols):
    mols = df["Standardized SMILES"].apply(Chem.MolFromSmiles)

    rdkit_desc = []
    graph_desc = []
    fps = []

    morgan = GetMorganGenerator(radius=2, fpSize=128)

    for mol in mols:
        if mol is None:
            rdkit_desc.append([np.nan] * len(RD_DESC_NAMES))
            graph_desc.append({k: np.nan for k in compute_graph_features(Chem.MolFromSmiles("CC")).keys()})
            fps.append(np.zeros(295))
            continue

        rdkit_desc.append(compute_rdkit_descriptors(mol))
        graph_desc.append(compute_graph_features(mol))

        fp = np.concatenate([
            np.array(morgan.GetFingerprint(mol)),
            np.array(MACCSkeys.GenMACCSKeys(mol))
        ])
        fps.append(fp)

    rd_df = pd.DataFrame(rdkit_desc, columns=RD_DESC_NAMES)
    graph_df = pd.DataFrame(graph_desc)
    fp_df = pd.DataFrame(fps, columns=[f"FP_{i}" for i in range(295)])

    return pd.concat(
    [
        df[["Standardized SMILES"]].reset_index(drop=True),
        rd_df,
        graph_df,
        fp_df,
        df[label_cols].reset_index(drop=True),
    ],
    axis=1
)

def main():
    logging.info("Full feature extraction pipeline started")

    for dataset_name, label_cols in DATASET_CONFIG.items():
        input_dir = os.path.join(CLEAN_ROOT, dataset_name)
        output_dir = os.path.join(FEATURE_ROOT, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Processing dataset: {dataset_name}")

        for split in SPLITS:
            input_path = os.path.join(input_dir, f"{split}_clean.csv")
            output_path = os.path.join(output_dir, f"{split}_features.csv")

            if not os.path.exists(input_path):
                logging.warning(f"Missing file: {input_path}")
                continue

            df = pd.read_csv(input_path)

            logging.info(
                f"{dataset_name} | {split} | Input shape: {df.shape}"
            )

            feature_df = extract_features(df, label_cols)

            logging.info(
                f"{dataset_name} | {split} | Output shape: {feature_df.shape}"
            )

            feature_df.to_csv(output_path, index=False)

            logging.info(
                f"{dataset_name} | {split} | Saved: {output_path}"
            )

    logging.info("Full feature extraction pipeline finished")


if __name__ == "__main__":
    main()
