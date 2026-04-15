# PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning

Code and data for the manuscript:

> **PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning for Molecular Property Prediction**

## Repository Structure

```
PEARL/
├── manuscript/          LaTeX source for the PEARL paper
├── data/
│   ├── raw/             Raw train/valid/test splits for BACE, BBBP, ClinTox, Flavor
│   ├── clean/           Cleaned SMILES splits
│   └── zinc250k/        ZINC-250k reference library (SMILES + properties)
├── scripts/
│   ├── data_curation/   Data cleaning and preprocessing
│   ├── base_models/     Baseline feature extraction (physicochemical + base CLM)
│   ├── smiles/          SMILES-based CLM pipelines (ChemBERTa, MolFormer)
│   │   ├── finetuning/  LoRA finetuning scripts
│   │   ├── ml/          Tree classifier training on finetuned embeddings
│   │   └── rag/         RAFE pipeline (ZINC indexing, feature extraction, modelling)
│   └── unimol/          Uni-Mol 3D pipeline
│       ├── finetuning/  LoRA finetuning with conformer inputs
│       ├── ml/          Tree classifier training on Uni-Mol embeddings
│       └── rag/         Uni-Mol RAFE pipeline
├── bash/                SLURM job scripts
│   ├── smiles/
│   └── unimol/
└── results/             Metrics (JSON/CSV) from all experiments
```

## Large Files (not in repo)

Finetuned model checkpoints, ZINC embeddings, and FAISS indices (~75GB total) are stored
externally. Set the environment variable before running any script:

```bash
export PEARL_EXTRAS="/export/cse/rmall/Raghvendra/EffiChem_Extras"
```

If you are reproducing the results from scratch, run the scripts in this order:
1. `scripts/data_curation/` — clean raw data
2. `scripts/smiles/finetuning/` or `scripts/unimol/finetuning/` — LoRA finetune
3. `scripts/smiles/rag/embed_zinc250k.py` — embed ZINC-250k
4. `scripts/smiles/rag/build_zinc_index.py` — build FAISS index
5. `scripts/smiles/rag/rag_feature_extraction.py` — extract RAFE features
6. `scripts/smiles/rag/rag_modelling_*.py` — train and evaluate

## Requirements

```bash
pip install -r requirements.txt
```

## Citation

If you use this code or data, please cite the PEARL manuscript.
