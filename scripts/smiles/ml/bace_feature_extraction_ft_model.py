"""
BACE Feature Extraction Pipeline

This script extracts chemical and molecular features for the BACE dataset:
1. Filtered RDKit descriptors (Descriptors.descList - useless_cols)
2. Graph-theoretic features using NetworkX
3. Morgan fingerprints (radius=2, 128 bits)
4. MACCS fingerprints (167 bits)

Security improvements:
- Path validation to prevent directory traversal
- Comprehensive error handling
- Safe molecule parsing

Performance improvements:
- Cache dummy molecule keys for invalid molecules (major optimization)
- Efficient feature computation
- Memory-efficient concatenation

Code quality improvements:
- Type hints throughout
- Environment variable configuration
- Both file and console logging
- Modular, reusable functions
"""

import os
import logging
import contextlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class Config:
    """Configuration class with path validation and environment variable support."""

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Data paths
        self.BASE_ROOT = self._validate_path(
            Path(os.getenv('PC_BASE_ROOT', self.BASE_DIR / "data" / "finetuned_embeddings"))
        )
        self.FEATURE_ROOT = self._validate_path(
            Path(os.getenv('PC_FEATURE_ROOT', self.BASE_DIR / "data" / "finetuned_pc_embeddings")),
            create=True
        )
        self.LOG_DIR = self._validate_path(
            self.FEATURE_ROOT / "logs",
            create=True
        )

        # Dataset configuration
        self.DATASET_CONFIG = {
            "BACE_Embeddings": ["Class"],
        }

        self.dataset_mapping = {
            "BACE_Embeddings": "bace",
        }

        self.smiles_mapping = {
            "BACE_Embeddings": "Standardized SMILES",
        }

        self.SPLITS = ["train", "test", "eval"]

        # Descriptor exclusion list
        self.useless_cols = [
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

        self.RD_DESC_NAMES = [d[0] for d in Descriptors.descList if d[0] not in self.useless_cols]

        # Cache for dummy molecule graph feature keys (major performance optimization)
        self._dummy_graph_keys: Optional[List[str]] = None

    def _validate_path(self, path: Path, create: bool = False) -> Path:
        """Validate path is within base directory and optionally create it."""
        path = Path(path).resolve()
        try:
            path.relative_to(self.BASE_DIR)
        except ValueError:
            raise ValueError(f"Path {path} is outside base directory {self.BASE_DIR}")

        if create:
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_dummy_graph_keys(self) -> List[str]:
        """
        Get graph feature keys from a dummy molecule.
        This is cached to avoid recreating the dummy molecule every time.
        Major performance optimization from the original code.
        """
        if self._dummy_graph_keys is None:
            dummy_mol = Chem.MolFromSmiles("CC")
            if dummy_mol is None:
                raise ValueError("Failed to create dummy molecule for graph feature keys")
            self._dummy_graph_keys = list(compute_graph_features(dummy_mol).keys())
        return self._dummy_graph_keys


def setup_logging(config: Config) -> None:
    """Setup both file and console logging."""
    log_file = config.LOG_DIR / "bace_feature_extraction.log"

    # Create formatters
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Safely parse SMILES string into RDKit molecule object."""
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            logging.warning(f"Invalid SMILES: {smiles}")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Failed to parse SMILES: {smiles}")
            return None

        return mol

    except Exception as e:
        logging.error(f"Error parsing SMILES '{smiles}': {e}")
        return None


def compute_rdkit_descriptors(mol: Chem.Mol, config: Config) -> List[float]:
    """Compute filtered RDKit descriptors for a molecule."""
    try:
        return [func(mol) for name, func in Descriptors.descList if name not in config.useless_cols]
    except Exception as e:
        logging.error(f"Error computing RDKit descriptors: {e}")
        return [np.nan] * len(config.RD_DESC_NAMES)


def compute_graph_features(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute graph-theoretic features using NetworkX.

    Features include:
    - Global graph properties (diameter, avg path length, cycles, etc.)
    - Centrality measures (degree, betweenness, closeness, eigenvector, katz)
    - Ring analysis (aromatic vs non-aromatic)
    - Atom composition statistics
    - Bond type statistics
    """
    try:
        adj = rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)

        n = G.number_of_nodes()
        e = G.number_of_edges()

        feats: Dict[str, float] = {}

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

    except Exception as e:
        logging.error(f"Error computing graph features: {e}")
        return {}


def compute_fingerprints(mol: Chem.Mol, morgan_gen: Any) -> np.ndarray:
    """
    Compute molecular fingerprints.

    Combines:
    - Morgan fingerprints (radius=2, 128 bits)
    - MACCS keys (167 bits)
    Total: 295 bits
    """
    try:
        morgan_fp = np.array(morgan_gen.GetFingerprint(mol))
        maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        return np.concatenate([morgan_fp, maccs_fp])

    except Exception as e:
        logging.error(f"Error computing fingerprints: {e}")
        return np.zeros(295)


def extract_features(df: pd.DataFrame, label_cols: List[str], smiles_col: str, config: Config) -> pd.DataFrame:
    """
    Extract all features from a dataframe containing SMILES strings.

    Returns a dataframe with:
    - Original SMILES column
    - Embedding columns (if present)
    - RDKit descriptors
    - Graph features
    - Fingerprints
    - Label columns
    """
    logging.info("Starting feature extraction")

    # Parse SMILES into molecules
    mols = df[smiles_col].apply(safe_mol_from_smiles)

    rdkit_desc: List[List[float]] = []
    graph_desc: List[Dict[str, float]] = []
    fps: List[np.ndarray] = []

    # Create Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(radius=2, fpSize=128)

    # Get dummy graph keys once (major performance optimization)
    dummy_graph_keys = config.get_dummy_graph_keys()

    logging.info(f"Processing {len(mols)} molecules")

    for idx, mol in enumerate(mols):
        if mol is None:
            # Use NaN for invalid molecules
            rdkit_desc.append([np.nan] * len(config.RD_DESC_NAMES))
            graph_desc.append({k: np.nan for k in dummy_graph_keys})
            fps.append(np.zeros(295))
            continue

        # Compute features for valid molecules
        rdkit_desc.append(compute_rdkit_descriptors(mol, config))
        graph_desc.append(compute_graph_features(mol))
        fps.append(compute_fingerprints(mol, morgan_gen))

        # Log progress for large datasets
        if (idx + 1) % 1000 == 0:
            logging.info(f"Processed {idx + 1}/{len(mols)} molecules")

    logging.info("Creating feature dataframes")

    # Create dataframes from features
    rd_df = pd.DataFrame(rdkit_desc, columns=config.RD_DESC_NAMES)
    graph_df = pd.DataFrame(graph_desc)
    fp_df = pd.DataFrame(fps, columns=[f"FP_{i}" for i in range(295)])

    # Get embedding columns (exclude SMILES and label columns)
    embed_cols = [c for c in df.columns if c not in label_cols + [smiles_col]]

    logging.info("Concatenating all features")

    # Concatenate all features
    result_df = pd.concat(
        [
            df[[smiles_col]].reset_index(drop=True),
            df[embed_cols].reset_index(drop=True),
            rd_df,
            graph_df,
            fp_df,
            df[label_cols].reset_index(drop=True),
        ],
        axis=1
    )

    logging.info(f"Feature extraction complete. Final shape: {result_df.shape}")

    return result_df


def process_dataset(
    dataset_name: str,
    label_cols: List[str],
    config: Config
) -> None:
    """Process a single dataset through the feature extraction pipeline."""

    input_dir = config.BASE_ROOT / dataset_name
    output_dir = config.FEATURE_ROOT / dataset_name

    # Validate and create output directory
    output_dir = config._validate_path(output_dir, create=True)

    logging.info("=" * 80)
    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 80)

    # Get SMILES column name for this dataset
    smiles_col = config.smiles_mapping[dataset_name]

    for split in config.SPLITS:
        input_path = input_dir / f"{config.dataset_mapping[dataset_name]}_{split}_embed.csv"
        output_path = output_dir / f"{config.dataset_mapping[dataset_name]}_{split}_features.csv"

        if not input_path.exists():
            logging.warning(f"Missing file: {input_path}")
            continue

        try:
            logging.info(f"Processing {dataset_name} | {split}")

            # Load data
            df = pd.read_csv(input_path)
            logging.info(f"{dataset_name} | {split} | Input shape: {df.shape}")

            # Extract features
            feature_df = extract_features(df, label_cols, smiles_col, config)
            logging.info(f"{dataset_name} | {split} | Output shape: {feature_df.shape}")

            # Save features
            feature_df.to_csv(output_path, index=False)
            logging.info(f"{dataset_name} | {split} | Saved: {output_path}")

        except Exception as e:
            logging.error(f"Failed to process {dataset_name} | {split}: {e}")
            continue


def main():
    """Main execution function."""

    # Initialize configuration
    config = Config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("Starting BACE Feature Extraction Pipeline")
    logging.info(f"Base directory: {config.BASE_DIR}")
    logging.info(f"Base root: {config.BASE_ROOT}")
    logging.info(f"Feature root: {config.FEATURE_ROOT}")
    logging.info("=" * 80)

    try:
        # Process each dataset
        for dataset_name, label_cols in config.DATASET_CONFIG.items():
            try:
                process_dataset(dataset_name, label_cols, config)
            except Exception as e:
                logging.error(f"Failed to process dataset {dataset_name}: {e}")
                continue

        logging.info("=" * 80)
        logging.info("BACE feature extraction pipeline finished successfully")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
