"""
Nach0 instruction fine-tuning for flavor prediction — HuggingFace Seq2SeqTrainer.
SMILES-only text input (no molecular images).

Overview
--------
Nach0 (insilicomedicine/nach0_base) is an mT5-based seq2seq model pre-trained
on diverse chemical tasks using an instruction-following format.  We fine-tune
it for 4-class flavor classification (bitter / sour / sweet / umami).

Input / output format
---------------------
  Encoder input : Nach0-tokenised instruction prompt containing the SMILES string.
                  Each SMILES token is wrapped in Nach0's <sm_…> special-token
                  scheme before the HuggingFace tokenizer processes the text,
                  matching the encoding used during Nach0's pre-training.
  Decoder target: the flavor label word  (e.g. "bitter")

The decoder is trained with teacher-forcing and cross-entropy loss; padding
positions in the target sequence are masked with -100 so they do not contribute
to the loss.

Splits
------
  train  →  data/clean/flavor_datasets/train_clean.csv   (training)
  valid  →  data/clean/flavor_datasets/valid_clean.csv   (validation / early-stop)
  test   →  data/clean/flavor_datasets/test_clean.csv    (held-out evaluation)

Usage
-----
  python nach0_flavor.py [--epochs 10] [--batch-size 16] [--lr 1e-4]

Outputs
-------
  EffiChem_Extras/nach0/flavor/best_model/ — full Nach0 weights + tokenizer (large)
  EffiChem_Extras/nach0/flavor/trainer/    — HuggingFace Trainer checkpoints (large)
  results/nach0/flavor/predictions.csv     — smiles, expected, predicted (test set)
  results/nach0/flavor/metrics.json        — accuracy, f1_macro, f1_micro, mcc
  results/nach0/flavor/finetune.log        — full training log
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Suppress noisy RDKit warnings (SMILES validation only — no image rendering)
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_ROOT   = REPO_ROOT / "data" / "clean" / "flavor_datasets"

# Small outputs (CSV predictions, metrics JSON, logs) stay inside the repo.
OUTPUT_ROOT = REPO_ROOT / "results" / "nach0" / "flavor"

# Large model files (full Nach0 weights + Trainer checkpoints) are written to
# EffiChem_Extras so they don't bloat the repo directory.
# The path can be overridden via the PEARL_EXTRAS environment variable.
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
MODEL_ROOT  = EXTRAS_ROOT / "nach0" / "flavor"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# HuggingFace model hub identifier for Nach0 base
MODEL_NAME   = "insilicomedicine/nach0_base"

# Column names in the CSV datasets
SMILES_COL   = "Standardized SMILES"
LABEL_COL    = "Canonicalized Taste"

# Only keep rows whose label is one of these four classes
VALID_LABELS = {"bitter", "sour", "sweet", "umami","undefined"}

# Maximum encoder sequence length (Nach0 / mT5-base supports up to 512)
MAX_INPUT_LEN = 512

# Maximum decoder sequence length for the label.
# "umami" tokenises to ≤3 subword tokens + 1 EOS; 8 is a safe ceiling.
MAX_LABEL_LEN = 8

# ---------------------------------------------------------------------------
# Nach0 official SMILES tokenizer
# ---------------------------------------------------------------------------
# Nach0 wraps each atomic SMILES token in a special tag: <sm_C>, <sm_N>, …
# These tags are part of Nach0's extended vocabulary and must be applied
# BEFORE passing text to the HuggingFace tokenizer so the model recognises
# the molecule as chemistry rather than natural-language text.
#
# The atom list is sorted longest-first so that two-letter symbols (e.g. "Br")
# are matched before single-letter ones (e.g. "B").
_ATOMS = sorted(
    [
        'Ag', 'Al', 'As', 'Au', 'B',  'Ba', 'Bi', 'Br', 'C',  'Ca',
        'Cd', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F',  'Fe', 'Ga', 'Gd',
        'Ge', 'H',  'Hg', 'I',  'In', 'K',  'Li', 'M',  'Mg', 'Mn',
        'Mo', 'N',  'Na', 'O',  'P',  'Pt', 'Ru', 'S',  'Sb', 'Sc',
        'Se', 'Si', 'Sn', 'V',  'W',  'Z',  'Zn', 'c',  'e',  'n',
        'o',  'p',  's',
    ],
    key=len,
    reverse=True,
)

# Regex that matches every legal SMILES token (bonds, brackets, ring-closure
# digits, and atomic symbols).  Used to split a SMILES string into its
# individual tokens before wrapping each in the <sm_…> Nach0 tag.
_SMI_REGEX = re.compile(
    r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|"
    + "|".join(_ATOMS)
    + ")"
)


def _tokenise_word(word: str) -> str:
    """
    Attempt to re-encode a single whitespace-delimited word as Nach0 SMILES tokens.

    A word is treated as a SMILES string only if:
      - the regex fully covers it (word == join of all regex hits), AND
      - it parses as a valid RDKit molecule, AND
      - it produces more than 4 tokens (avoids false-positives on short strings).

    Otherwise the word is returned unchanged (natural-language words pass through).
    """
    tokens = _SMI_REGEX.findall(word)
    if len(tokens) > 4 and word == "".join(tokens) and Chem.MolFromSmiles(word):
        return "".join(f"<sm_{t}>" for t in tokens)
    return word


def add_special_symbols(text: str) -> str:
    """
    Apply Nach0's SMILES special-token encoding to every word in *text*.

    Words that are not valid SMILES strings pass through unchanged, so this
    function is safe to call on full instruction prompts (e.g.
    "Classify the molecule: CCO Flavor:").
    """
    return " ".join(_tokenise_word(w) for w in text.split())


def clean_output_sequence(seq: str) -> str:
    """
    Strip Nach0 special tokens and EOS markers from a decoded output string.

    The Nach0 tokenizer may leave residual <sm_… > fragments or the </s> EOS
    token in the decoded text.  This removes them so downstream label matching
    works on clean natural-language text.
    """
    return (
        seq
        .replace("</s>", "")
        .replace("<sm_", "")
        .replace(" sm_", "")
        .replace(">", "")
        .strip()
    )


def build_prompt(smiles: str) -> str:
    """
    Construct the Nach0 instruction prompt for flavor classification.

    The prompt follows the instruction-tuning style used during Nach0's
    pre-training: it names the task, lists the valid output classes, and ends
    with a 'Flavor:' completion cue so the decoder knows what to generate.

    The SMILES string is embedded as plain text here; the caller must pass the
    full prompt through add_special_symbols() before tokenisation so that
    the SMILES portion receives Nach0's <sm_…> encoding.
    """
    return (
        "Possible flavor labels: bitter, sour, sweet, umami, undefined. "
        "Classify the flavor of the given molecule using one of the labels above. "
        f"Molecule: {smiles} "
        "Flavor:"
    )


def extract_label(raw: str) -> str:
    """
    Return the first valid flavor label found anywhere in *raw*, else 'unknown'.

    Used to parse model-generated text that may include filler words.
    Priority order: sweet → bitter → sour → umami  → undefined .
    """
    lower = raw.lower()
    for lbl in ("sweet", "bitter", "sour", "umami","undefined"):
        if lbl in lower:
            return lbl
    return "unknown"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlavorDataset(Dataset):
    """
    PyTorch Dataset for Nach0 instruction fine-tuning (SMILES text only).

    Each __getitem__ returns a dict with variable-length lists of token IDs:
      input_ids     : Nach0-tokenised instruction prompt (with Nach0 <sm_…> encoding)
      attention_mask: 1 for real tokens
      labels        : target label token IDs (without padding; DataCollatorForSeq2Seq
                      pads to batch max-length and replaces pad with -100)

    Dynamic padding is delegated to DataCollatorForSeq2Seq so sequences are
    padded to the longest example in each batch rather than a global maximum.
    This is more memory-efficient than padding everything to MAX_INPUT_LEN.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_input_len: int = MAX_INPUT_LEN):
        self.df            = df.reset_index(drop=True)
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row    = self.df.iloc[idx]
        smiles = row[SMILES_COL]
        label  = row[LABEL_COL]

        # --- Encode the instruction prompt ------------------------------------
        # add_special_symbols wraps the SMILES part in <sm_…> tags so Nach0
        # processes it with its chemistry-aware vocabulary.
        enc = self.tokenizer(
            add_special_symbols(build_prompt(smiles)),
            max_length  = self.max_input_len,
            truncation  = True,
            # No padding here — DataCollatorForSeq2Seq will pad to batch length
        )

        # --- Encode the target label ------------------------------------------
        # The label is a single word (e.g. "bitter").  We keep it unpadded;
        # DataCollatorForSeq2Seq pads label sequences and replaces pad tokens
        # with -100 so they are masked in the cross-entropy loss.
        lbl_enc = self.tokenizer(
            label,
            max_length  = MAX_LABEL_LEN,
            truncation  = True,
        )

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         lbl_enc["input_ids"],
        }


def load_split(split: str) -> pd.DataFrame:
    """
    Load and filter one CSV split from DATA_ROOT.

    Keeps only the SMILES and label columns, drops missing values, lower-cases
    the labels, and retains only the five valid flavor classes.
    """
    path = DATA_ROOT / f"{split}_clean.csv"
    df   = pd.read_csv(path)
    df   = df[[SMILES_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df   = df[df[LABEL_COL].isin(VALID_LABELS)].reset_index(drop=True)
    logging.info("Split %-6s: %d molecules  (%s)", split, len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_nach0(device: str) -> Tuple:
    """
    Load Nach0 base from HuggingFace Hub and move it to *device*.

    Nach0 is a T5ForConditionalGeneration model with an extended vocabulary
    for SMILES special tokens (<sm_…>).  We load both the tokenizer (which
    includes those extensions) and the full model for fine-tuning.
    """
    logging.info("Loading %s …", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model.to(device)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def make_compute_metrics(tokenizer):
    """
    Factory that returns a compute_metrics function for Seq2SeqTrainer.

    Called after each evaluation epoch with an EvalPrediction whose fields
    are numpy arrays collected across all validation batches:
      .predictions : generated token IDs  (N, seq_len)
      .label_ids   : target token IDs     (N, seq_len)

    We decode both arrays, extract flavor labels, and compute accuracy, F1
    (macro/micro), and MCC.  Rows where the model returns 'unknown' are
    excluded from metric computation.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> Dict:
        gen_ids   = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Trainer fills masked label positions with -100 before passing here;
        # replace them with pad_token_id so the tokenizer can decode normally.
        pad_id    = tokenizer.pad_token_id
        gen_ids   = np.where(gen_ids   != -100, gen_ids,   pad_id)
        label_ids = np.where(label_ids != -100, label_ids, pad_id)

        # Decode generated token IDs → raw strings → extract flavor label
        preds = [
            extract_label(
                clean_output_sequence(tokenizer.decode(g, skip_special_tokens=True))
            )
            for g in gen_ids
        ]

        # Decode reference label token IDs → ground-truth label strings
        refs = [
            tokenizer.decode(l, skip_special_tokens=True).strip().lower()
            for l in label_ids
        ]

        # Exclude samples where the model failed to output a valid label
        valid = [(r, p) for r, p in zip(refs, preds) if p != "unknown"]
        if not valid:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0, "mcc": 0.0}

        y_true, y_pred = zip(*valid)
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_true, y_pred, average="micro",  zero_division=0)), 4),
            "mcc":      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Progress bar callback
# ---------------------------------------------------------------------------
class TqdmProgressCallback(TrainerCallback):
    """
    Explicit per-step tqdm progress bar for HuggingFace Seq2SeqTrainer.

    The Trainer's built-in tqdm is suppressed in non-TTY environments (e.g.
    when running via nohup or redirecting stdout to a file).  This callback
    creates a tqdm bar with disable=False so it always renders, and updates
    it with the latest loss and accuracy after each step / evaluation.
    """

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self._pbar = tqdm(
            total        = state.max_steps,
            desc         = "Training",
            unit         = "step",
            dynamic_ncols= True,
            disable      = False,   # force bar even in non-TTY environments
        )

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self._pbar.update(1)
        # Pull the most recent loss from the trainer's log history
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self._pbar.set_postfix(
                    loss  = f"{last['loss']:.4f}",
                    epoch = f"{state.epoch:.1f}",
                )

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        metrics: Dict = None, **kwargs
    ) -> None:
        # Update the bar with validation metrics after each eval round
        if metrics:
            self._pbar.set_postfix(
                val_loss = f"{metrics.get('eval_loss', 0):.4f}",
                val_acc  = f"{metrics.get('eval_accuracy', 0):.4f}",
                epoch    = f"{state.epoch:.1f}",
            )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        self._pbar.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    # Trainer checkpoints (large) → EffiChem_Extras
    trainer_dir = MODEL_ROOT / "trainer"

    # --- Logging: console + file ---
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_ROOT / "finetune.log", mode="w"),
        ],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: %s | Args: %s", device, vars(args))

    # --- Load datasets ---
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df  = load_split("test")

    # --- Load Nach0 ---
    tokenizer, model = build_nach0(device)
    logging.info(
        "Nach0 parameters: %d total",
        sum(p.numel() for p in model.parameters()),
    )

    # --- Wrap as PyTorch Datasets ---
    train_ds = FlavorDataset(train_df, tokenizer, args.max_input_len)
    valid_ds = FlavorDataset(valid_df, tokenizer, args.max_input_len)
    test_ds  = FlavorDataset(test_df,  tokenizer, args.max_input_len)

    # DataCollatorForSeq2Seq pads input_ids and attention_mask to the longest
    # sequence in each batch, and pads labels with -100 (ignored by CE loss).
    # This is the standard HuggingFace collator for seq2seq instruction tuning.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model               = model,
        padding             = True,
        label_pad_token_id  = -100,
    )

    # --- HuggingFace Seq2SeqTrainingArguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir = str(trainer_dir),

        # Training schedule
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = args.lr,
        weight_decay                = args.weight_decay,
        # Cosine LR with 5 % linear warm-up — standard for instruction tuning
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",

        # Evaluate once per epoch; keep the checkpoint with the best
        # validation accuracy (metric returned by compute_metrics).
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        save_total_limit            = 1,

        # Use model.generate() during evaluation so the metric reflects
        # true autoregressive inference rather than teacher-forced logits.
        predict_with_generate       = True,
        generation_max_length       = MAX_LABEL_LEN,

        # Mixed precision — P100 supports fp16 but not bf16
        fp16                        = torch.cuda.is_available(),

        dataloader_num_workers      = args.num_workers,
        logging_steps               = 50,
        report_to                   = "none",   # disable W&B / TensorBoard
    )

    # --- Standard Seq2SeqTrainer with explicit progress bar callback ---
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = valid_ds,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = make_compute_metrics(tokenizer),
        callbacks       = [TqdmProgressCallback()],
    )

    # --- Instruction fine-tuning ---
    logging.info("Starting instruction fine-tuning …")
    trainer.train()

    # Save the best model weights to EffiChem_Extras (large files)
    best_dir = MODEL_ROOT / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logging.info("Best model saved → %s", best_dir)

    # --- Final test evaluation ---
    logging.info("Running predict() on test set …")
    test_out = trainer.predict(test_ds)

    # Replace -100 sentinels with pad_token_id before decoding
    gen_ids = np.where(
        test_out.predictions != -100,
        test_out.predictions,
        tokenizer.pad_token_id,
    )
    label_ids = np.where(
        test_out.label_ids != -100,
        test_out.label_ids,
        tokenizer.pad_token_id,
    )

    preds = [
        extract_label(
            clean_output_sequence(tokenizer.decode(g, skip_special_tokens=True))
        )
        for g in gen_ids
    ]
    refs = [
        tokenizer.decode(l, skip_special_tokens=True).strip().lower()
        for l in label_ids
    ]

    # Save per-sample predictions alongside the original SMILES strings
    records = [
        {"smiles": s, "expected": e, "predicted": p}
        for s, e, p in zip(test_df[SMILES_COL].tolist(), refs, preds)
    ]
    pd.DataFrame(records).to_csv(OUTPUT_ROOT / "predictions.csv", index=False)

    # Compute test metrics (exclude rows where the model returned 'unknown')
    valid = [(r, p) for r, p in zip(refs, preds) if p != "unknown"]
    y_true = [v[0] for v in valid]
    y_pred = [v[1] for v in valid]

    metrics: dict = {
        "n_total":             len(refs),
        "n_valid_predictions": len(valid),
    }
    if y_true:
        metrics.update({
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_true, y_pred, average="micro",  zero_division=0)), 4),
            "mcc":      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        })

    logging.info("Test metrics:\n%s", json.dumps(metrics, indent=2))
    with open(OUTPUT_ROOT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    logging.info("All outputs saved to %s", OUTPUT_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nach0 instruction fine-tuning — SMILES text only"
    )
    parser.add_argument("--epochs",        type=int,   default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size",    type=int,   default=16,
                        help="Per-device batch size for train and eval (default: 16)")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="Peak learning rate for AdamW with cosine schedule (default: 1e-4)")
    parser.add_argument("--weight-decay",  type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--max-input-len", type=int,   default=MAX_INPUT_LEN,
                        help=f"Max encoder token length (default: {MAX_INPUT_LEN})")
    parser.add_argument("--num-workers",   type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")
    args = parser.parse_args()
    main(args)
