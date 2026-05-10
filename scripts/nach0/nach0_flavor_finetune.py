"""
Nach0 LoRA instruction fine-tuning for flavor prediction.
SMILES-only text input (no molecular images).

Overview
--------
Nach0 (insilicomedicine/nach0_base) is an mT5-based seq2seq (T5ForConditionalGeneration)
model pre-trained on diverse chemical tasks using an instruction-following format.

This script fine-tunes Nach0 via PEFT LoRA: only the query and value projection
matrices of the T5 attention layers are updated, leaving ~99 % of parameters
frozen.  This keeps GPU memory low and training fast while still adapting the
model's representations to the flavor classification task.

Input / output format (seq2seq)
--------------------------------
  Encoder input : Nach0-tokenised instruction prompt containing the SMILES string.
                  Each SMILES token is wrapped in Nach0's <sm_…> special-token
                  scheme before the HuggingFace tokenizer processes the text,
                  matching the encoding used during Nach0's pre-training.
  Decoder target: the flavor label word  (e.g. "bitter")

The decoder is trained with teacher-forcing and cross-entropy loss; padding
positions in the target sequence are masked with -100.

Splits
------
  train  →  data/clean/flavor_datasets/train_clean.csv   (training)
  valid  →  data/clean/flavor_datasets/valid_clean.csv   (validation / checkpoint selection)
  test   →  data/clean/flavor_datasets/test_clean.csv    (held-out evaluation)

WandB Hyperparameter Sweep
--------------------------
When --sweep is passed, the script runs a Bayesian search over LoRA configuration
(rank, alpha multiplier, dropout) and optimiser hyperparameters (lr, weight_decay,
batch_size) using Weights & Biases sweeps.  Each trial trains for --sweep-epochs
epochs on train/valid, logging val_accuracy to WandB.  After all trials finish the
best configuration is retrieved via the WandB API and a full training run (--epochs)
is automatically executed with those hyperparameters.

Usage
-----
  # Direct training with default or CLI hyperparameters:
  python nach0_flavor_finetune.py [--epochs 10] [--batch-size 16] [--lr 1e-4]

  # WandB Bayesian sweep then full training with best config:
  python nach0_flavor_finetune.py --sweep --sweep-trials 20 --sweep-epochs 3

  # Override WandB project name:
  python nach0_flavor_finetune.py --sweep --wandb-project my-nach0-sweep

Outputs
-------
  EffiChem_Extras/nach0/flavor_ft/best_model/ — Nach0 LoRA adapter weights (PEFT format)
  predictions.csv   — smiles, expected, predicted   (test set)
  metrics.json      — accuracy, f1_macro, f1_micro, mcc
  training_log.csv  — epoch, train_loss, val_loss, val_accuracy, epoch_time_s
  finetune.log      — full training log with step-by-step progress
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import wandb
from rdkit import Chem, RDLogger
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm

# Suppress noisy RDKit warnings (SMILES validation only)
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_ROOT   = REPO_ROOT / "data" / "clean" / "flavor_datasets"

# Large LoRA adapter weights → EffiChem_Extras (override via PEARL_EXTRAS env var).
# Small outputs (CSV, JSON, logs) → --output-dir inside the repo.
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
MODEL_ROOT  = EXTRAS_ROOT / "nach0" / "flavor_ft"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME    = "insilicomedicine/nach0_base"
SMILES_COL    = "Standardized SMILES"
LABEL_COL     = "Canonicalized Taste"
VALID_LABELS  = {"bitter", "sour", "sweet", "umami", "undefined"}
MAX_INPUT_LEN = 512     # Nach0 / mT5-base encoder capacity
MAX_LABEL_LEN = 8       # "umami" ≤ 3 subword tokens + EOS; 8 is a safe ceiling

# ---------------------------------------------------------------------------
# WandB Bayesian sweep configuration
# ---------------------------------------------------------------------------
# Searched hyperparameter space for LoRA + optimiser.
# Metric "best_mcc" is logged at the end of every trial so WandB selects
# the trial that achieved the highest peak validation MCC.
SWEEP_CONFIG: dict = {
    "method": "bayes",
    "metric": {"name": "best_mcc", "goal": "maximize"},
    "parameters": {
        # LoRA architecture
        "lora_r": {
            "values": [4, 8, 16, 32],
        },
        # lora_alpha = lora_r * lora_alpha_mult  (common choices: 1× or 2×)
        "lora_alpha_mult": {
            "values": [1, 2, 4],
        },
        "lora_dropout": {
            "values": [0.0, 0.05, 0.1],
        },
        # Optimiser
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 5e-4,
        },
        "weight_decay": {
            "values": [0.0, 0.01, 0.05],
        },
        "batch_size": {
            "values": [8, 16, 32],
        },
    },
}

# ---------------------------------------------------------------------------
# Nach0 official SMILES tokenizer
# ---------------------------------------------------------------------------
# Nach0 extends the mT5 vocabulary with <sm_C>, <sm_N>, … special tokens for
# every atomic SMILES symbol.  add_special_symbols() wraps each SMILES token
# in this format before the HuggingFace tokenizer runs.  This is required to
# match Nach0's pre-training preprocessing — without it the model treats SMILES
# as unknown characters and performance degrades significantly.
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

_SMI_REGEX = re.compile(
    r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|"
    + "|".join(_ATOMS)
    + ")"
)


def _tokenise_word(word: str) -> str:
    """
    Wrap a valid SMILES string in Nach0's <sm_…> special-token format.

    A word is treated as SMILES only if the regex fully covers it, RDKit
    parses it as a valid molecule, and it produces more than 4 tokens
    (guards against false-positive matches on short strings like 'C' or 'N').
    All other words are returned unchanged.
    """
    tokens = _SMI_REGEX.findall(word)
    if len(tokens) > 4 and word == "".join(tokens) and Chem.MolFromSmiles(word):
        return "".join(f"<sm_{t}>" for t in tokens)
    return word


def add_special_symbols(text: str) -> str:
    """Apply Nach0 SMILES encoding to every whitespace-delimited word in *text*."""
    return " ".join(_tokenise_word(w) for w in text.split())


def clean_output_sequence(seq: str) -> str:
    """Remove Nach0 special tokens and the EOS marker from decoded output text."""
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
    Build the Nach0 instruction prompt for flavor classification.

    Format follows Nach0's pre-training style: the prompt names the task,
    lists the four valid output classes, provides the SMILES as the molecule
    descriptor, and ends with 'Flavor:' so the decoder generates a label.
    The SMILES must be passed through add_special_symbols() before tokenisation.
    """
    return (
        "Possible flavor labels: bitter, sour, sweet, umami. "
        "Classify the flavor of the given molecule using one of the labels above. "
        f"Molecule: {smiles} "
        "Flavor:"
    )


def extract_label(raw: str) -> str:
    """Return the first valid flavor label found in *raw*, else 'unknown'."""
    lower = raw.lower()
    for lbl in ("sweet", "bitter", "sour", "umami", "undefined"):
        if lbl in lower:
            return lbl
    return "unknown"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlavorDataset(Dataset):
    """
    PyTorch Dataset for Nach0 LoRA instruction fine-tuning (SMILES text only).

    Each sample:
      input_ids     : (max_input_len,) int64 — Nach0-tokenised instruction prompt
      attention_mask: (max_input_len,) int64 — 1 for real tokens, 0 for padding
      labels        : (MAX_LABEL_LEN,) int64 — target label token IDs;
                       padding positions replaced with -100 (masked in CE loss)
      smiles        : str  — original SMILES (kept for bookkeeping)
      expected      : str  — ground-truth label string (kept for bookkeeping)

    Sequences are padded to fixed lengths so that collate_fn can use torch.stack
    without dynamic padding.
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

        # --- Encode instruction prompt ----------------------------------------
        # add_special_symbols wraps the SMILES token(s) in <sm_…> tags so
        # Nach0 processes the molecule with its chemistry-aware vocabulary.
        enc = self.tokenizer(
            add_special_symbols(build_prompt(smiles)),
            max_length     = self.max_input_len,
            padding        = "max_length",   # fixed length for easy torch.stack
            truncation     = True,
            return_tensors = "pt",
        )

        # --- Encode target label ----------------------------------------------
        # Tokenise the label word (e.g. "bitter") as the decoder target.
        # Replace pad tokens with -100 so PyTorch's cross-entropy loss ignores them.
        lbl_enc = self.tokenizer(
            label,
            max_length     = MAX_LABEL_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        labels = lbl_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
            "smiles":         smiles,
            "expected":       label,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Stack tensor fields; keep string fields as Python lists.

    Tensors (input_ids, attention_mask, labels) are all the same shape due to
    fixed-length padding in __getitem__, so torch.stack works directly.
    Strings (smiles, expected) cannot be stacked and are kept as lists.
    """
    smiles   = [b.pop("smiles")   for b in batch]
    expected = [b.pop("expected") for b in batch]
    tensors  = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    tensors["smiles"]   = smiles
    tensors["expected"] = expected
    return tensors


def load_split(split: str) -> pd.DataFrame:
    """Load and filter one CSV split, keeping only SMILES + valid label rows."""
    path = DATA_ROOT / f"{split}_clean.csv"
    df   = pd.read_csv(path)
    df   = df[[SMILES_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df   = df[df[LABEL_COL].isin(VALID_LABELS)].reset_index(drop=True)
    logging.info("    %-6s: %d molecules  (%s)", split, len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_nach0_with_lora(
    device:       str,
    lora_r:       int,
    lora_alpha:   int,
    lora_dropout: float = 0.05,
) -> Tuple:
    """
    Load Nach0 and apply a PEFT LoRA adapter.

    LoRA inserts low-rank trainable matrices A and B alongside the frozen
    query (q) and value (v) projection weights of every T5 attention layer.
    At each forward pass the effective weight is  W + (B @ A) * (alpha / r).
    Only A, B matrices are updated during training — all other parameters,
    including the Nach0 encoder / decoder weights, remain frozen.

    lora_dropout applies dropout to the LoRA branch activations during training.
    Setting it to 0.0 disables dropout, which can be optimal when the dataset
    is large enough that regularisation is not needed.
    """
    logging.info("    Loading %s …", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_cfg = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,   # T5ForConditionalGeneration
        r              = lora_r,                   # low-rank dimension
        lora_alpha     = lora_alpha,               # scaling factor
        target_modules = ["q", "v"],               # T5 attention q and v projections
        lora_dropout   = lora_dropout,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    logging.info(
        "    LoRA: r=%d  alpha=%d  dropout=%.2f  target=[q, v]",
        lora_r, lora_alpha, lora_dropout,
    )
    return tokenizer, model.to(device)


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------
def train_epoch(
    loader:    DataLoader,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device:    str,
    grad_clip: float,
    epoch:     int = 0,
) -> float:
    """
    Run one full training epoch over *loader* and return mean cross-entropy loss.

    Uses fp16 mixed precision (GradScaler + autocast) for faster training.
    Gradient clipping prevents exploding gradients in the early warm-up phase.
    The LR scheduler step is called once per batch (not per epoch) to follow
    the cosine schedule with warm-up correctly.

    Per-batch loss is logged to WandB (when a run is active) for fine-grained
    monitoring of training dynamics.
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(
        loader, desc=f"  train ep{epoch:02d}", unit="batch",
        dynamic_ncols=True, leave=False, disable=False,
    )
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # fp16 forward pass: T5 encodes the SMILES prompt, teacher-forces the
        # decoder with the shifted label sequence, and returns CE loss.
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,   # T5 shifts these right internally
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()   # cosine schedule steps per batch

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / n_batches:.4f}")

        # Per-batch loss to WandB for fine-grained loss curve monitoring
        if wandb.run is not None:
            wandb.log({"batch/train_loss": loss.item()})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    loader:    DataLoader,
    model:     nn.Module,
    tokenizer,
    device:    str,
    epoch:     int = 0,
) -> Tuple[float, float]:
    """
    Evaluate on *loader* and return (mean_loss, accuracy).

    Loss is computed with teacher-forcing (same as training) so it is
    comparable to train_loss.  Accuracy is computed from greedy autoregressive
    generation — this reflects true inference-time behaviour and is the metric
    used for checkpoint selection.

    Rows where the model generates 'unknown' (no valid label found) are
    excluded from the accuracy and MCC computation.

    Returns
    -------
    (mean_loss, accuracy, mcc)  — MCC is the primary checkpoint selection metric
    as it handles class imbalance better than accuracy.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_true, all_pred    = [], []

    pbar = tqdm(
        loader, desc=f"   eval ep{epoch:02d}", unit="batch",
        dynamic_ncols=True, leave=False, disable=False,
    )
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Teacher-forced loss (same objective as training)
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
        total_loss += outputs.loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        # Greedy generation for accuracy: decoder starts from the decoder-start
        # token and generates until EOS or max_new_tokens is reached.
        gen_ids = model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            do_sample      = False,
            num_beams      = 1,
            max_new_tokens = MAX_LABEL_LEN,
        )
        raws  = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [extract_label(clean_output_sequence(r)) for r in raws]

        all_true.extend(batch["expected"])
        all_pred.extend(preds)

    # Exclude 'unknown' predictions from metrics.
    # MCC is the primary checkpoint metric — it accounts for class imbalance
    # in the flavor dataset (bitter >> umami/sour) better than accuracy.
    valid_mask = [p != "unknown" for p in all_pred]
    y_true = [t for t, v in zip(all_true, valid_mask) if v]
    y_pred = [p for p, v in zip(all_pred, valid_mask) if v]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    # MCC requires ≥ 2 unique classes in y_true; fall back to 0.0 otherwise
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0

    return total_loss / max(n_batches, 1), acc, mcc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, epoch: int, metrics: dict) -> None:
    """
    Save the LoRA adapter weights to EffiChem_Extras using PEFT's save_pretrained.

    PEFT saves only the LoRA A/B matrices (not the frozen base weights),
    so the adapter checkpoint is a few MB rather than ~GBs.
    MODEL_ROOT is defined at module level and points to EffiChem_Extras.
    """
    ckpt_dir = MODEL_ROOT / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))   # adapter_config.json + adapter weights only
    with open(ckpt_dir / "checkpoint_info.json", "w") as fh:
        json.dump({"epoch": epoch, **metrics}, fh, indent=2)
    logging.info(
        "    Checkpoint saved → %s  (epoch=%d  val_acc=%.4f  val_mcc=%.4f)",
        ckpt_dir, epoch, metrics.get("val_accuracy", 0), metrics.get("val_mcc", 0),
    )


# ---------------------------------------------------------------------------
# WandB sweep support
# ---------------------------------------------------------------------------
# Module-level dict populated before wandb.agent() runs so _sweep_trial() can
# access pre-loaded datasets and CLI args without re-loading on every trial.
# Keys: "args" (argparse.Namespace), "train_df" (DataFrame), "valid_df" (DataFrame).
_SWEEP_STATE: dict = {}


def _sweep_trial() -> None:
    """
    Single hyperparameter trial — called repeatedly by wandb.agent().

    WandB injects the trial's hyperparameter config into wandb.config before
    calling this function.  We build a fresh model for the trial, train for
    args.sweep_epochs epochs, and log val_accuracy at each epoch.  At the end
    we log best_mcc as a summary scalar — the sweep objective metric.

    GPU memory is explicitly freed at the end of each trial so subsequent
    trials start with a clean state.
    """
    with wandb.init() as run:
        cfg  = wandb.config
        args = _SWEEP_STATE["args"]

        device       = "cuda" if torch.cuda.is_available() else "cpu"
        lora_r       = cfg.lora_r
        lora_alpha   = cfg.lora_r * cfg.lora_alpha_mult
        lora_dropout = cfg.lora_dropout
        lr           = cfg.lr
        weight_decay = cfg.weight_decay
        batch_size   = cfg.batch_size

        logging.info(
            "--- Sweep trial %s | r=%d  alpha=%d  dropout=%.2f  "
            "lr=%.2e  wd=%.3f  bs=%d ---",
            run.id, lora_r, lora_alpha, lora_dropout, lr, weight_decay, batch_size,
        )

        # Build a fresh Nach0 + LoRA model for this trial's hyperparameters.
        # The tokenizer is the same for all trials but we reload it cheaply
        # since AutoTokenizer caches the vocab after the first call.
        tokenizer, model = build_nach0_with_lora(device, lora_r, lora_alpha, lora_dropout)

        # Construct DataLoaders from pre-loaded DataFrames stored in _SWEEP_STATE.
        # Avoids slow CSV reads on every trial.
        train_ds = FlavorDataset(_SWEEP_STATE["train_df"], tokenizer, args.max_input_len)
        valid_ds = FlavorDataset(_SWEEP_STATE["valid_df"], tokenizer, args.max_input_len)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

        # Optimiser and cosine LR schedule with 5 % warm-up
        optimizer    = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        total_steps  = len(train_loader) * args.sweep_epochs
        warmup_steps = max(1, int(0.05 * total_steps))
        scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler       = torch.amp.GradScaler('cuda')

        best_mcc = -1.0
        for epoch in range(1, args.sweep_epochs + 1):
            ep_start   = time.time()
            train_loss = train_epoch(
                train_loader, model, optimizer, scheduler, scaler,
                device, grad_clip=1.0, epoch=epoch,
            )
            val_loss, val_acc, val_mcc = eval_epoch( # type: ignore
                valid_loader, model, tokenizer, device, epoch=epoch
            )
            ep_elapsed = time.time() - ep_start

            logging.info(
                "  Trial ep %d/%d | train=%.4f | val=%.4f | acc=%.4f | mcc=%.4f | %.1fs",
                epoch, args.sweep_epochs, train_loss, val_loss, val_acc, val_mcc, ep_elapsed,
            )
            # Log epoch metrics; WandB Bayesian optimizer reads 'best_mcc' to rank trials.
            # MCC is the sweep objective — it accounts for class imbalance better than accuracy.
            wandb.log({
                "epoch":        epoch,
                "train_loss":   train_loss,
                "val_loss":     val_loss,
                "val_accuracy": val_acc,
                "val_mcc":      val_mcc,
            })
            best_mcc = max(best_mcc, val_mcc)

        # Log best_mcc as the sweep objective metric (matches SWEEP_CONFIG "metric" name)
        wandb.log({"best_mcc": best_mcc})
        logging.info("  Trial done | best_mcc=%.4f", best_mcc)

        # Free GPU memory before the next trial
        del model
        torch.cuda.empty_cache()


def run_sweep(args) -> dict:
    """
    Launch a WandB Bayesian sweep over LoRA + optimiser hyperparameters.

    Steps:
      1. Pre-load train/valid data once (reused across all trials via _SWEEP_STATE)
      2. Create the WandB sweep with SWEEP_CONFIG
      3. Run wandb.agent() for args.sweep_trials trials (sequential, same process)
      4. Query the WandB API to retrieve the best run's configuration
      5. Return the best config as a plain dict for use in the full training run

    Returns
    -------
    dict  Best hyperparameter configuration with keys:
          lora_r, lora_alpha_mult, lora_dropout, lr, weight_decay, batch_size
    """
    logging.info("")
    logging.info("=" * 65)
    logging.info("  [SWEEP] WandB Bayesian hyperparameter sweep")
    logging.info("  Project         : %s", args.wandb_project)
    logging.info("  Trials          : %d", args.sweep_trials)
    logging.info("  Epochs per trial: %d", args.sweep_epochs)
    logging.info("=" * 65)

    # Pre-load datasets once — shared across all trials in _SWEEP_STATE
    logging.info("")
    logging.info("  Pre-loading train/valid datasets for sweep …")
    _SWEEP_STATE["args"]     = args
    _SWEEP_STATE["train_df"] = load_split("train")
    _SWEEP_STATE["valid_df"] = load_split("valid")
    logging.info(
        "  Dataset sizes — train: %d | valid: %d",
        len(_SWEEP_STATE["train_df"]), len(_SWEEP_STATE["valid_df"]),
    )

    # Create the sweep and run all trials
    api      = wandb.Api()
    entity   = api.default_entity
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.wandb_project, entity=entity)
    logging.info("  Sweep created: https://wandb.ai/%s/%s/sweeps/%s",
                 entity, args.wandb_project, sweep_id)
    logging.info("  Starting %d trials …", args.sweep_trials)

    t0 = time.time()
    # wandb.agent runs _sweep_trial() count times sequentially in this process
    wandb.agent(sweep_id, function=_sweep_trial, count=args.sweep_trials)
    logging.info("  All trials finished in %.1fs", time.time() - t0)

    # Retrieve best configuration via WandB API
    logging.info("")
    logging.info("  Fetching best configuration from WandB API …")
    sweep_obj = api.sweep(f"{entity}/{args.wandb_project}/{sweep_id}")
    best_run  = sweep_obj.best_run()
    best_cfg  = dict(best_run.config)
    best_mcc  = best_run.summary.get("best_mcc", float("nan"))

    logging.info("  Best run     : %s", best_run.id)
    logging.info("  Best val_mcc : %.4f", best_mcc)
    logging.info("  Best config:")
    for k, v in best_cfg.items():
        logging.info("    %-20s = %s", k, v)

    return best_cfg


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
def run_test_eval(
    args:       argparse.Namespace,
    test_df:    pd.DataFrame,
    tokenizer,
    out_dir:    Path,
    batch_size: int = None,
) -> dict:
    """
    Load the best saved LoRA checkpoint, run greedy generation on the test set,
    compute accuracy/F1/MCC, save predictions.csv and metrics.json.

    Critical: the base model is loaded as a raw AutoModelForSeq2SeqLM first,
    then PeftModel.from_pretrained layers the saved adapter on top.  Never
    pass a PeftModel to PeftModel.from_pretrained — that doubly wraps the
    model and produces garbage output (all predictions become 'unknown').
    """
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir   = MODEL_ROOT / "best_model"
    batch_size = batch_size or args.batch_size

    logging.info("")
    logging.info("=" * 65)
    logging.info("  [TEST EVAL] Loading best LoRA checkpoint")
    logging.info("  Checkpoint: %s", ckpt_dir)
    logging.info("=" * 65)

    # --- Load checkpoint (raw base + adapter) --------------------------------
    t0 = time.time()
    # Load raw base model first, then layer the saved LoRA adapter on top.
    # Do NOT use build_nach0_with_lora here — that returns a PeftModel, and
    # calling PeftModel.from_pretrained on another PeftModel doubly wraps it,
    # producing garbage output and n_valid_predictions = 0.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model      = PeftModel.from_pretrained(base_model, str(ckpt_dir)).to(device)
    model.eval()
    logging.info("  Checkpoint loaded in %.1fs", time.time() - t0)

    # --- DataLoader ----------------------------------------------------------
    test_ds     = FlavorDataset(test_df, tokenizer, args.max_input_len)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    logging.info(
        "  Test set: %d molecules in %d batches (batch_size=%d)",
        len(test_df), len(test_loader), batch_size,
    )

    # --- Greedy generation ---------------------------------------------------
    records: List[dict] = []
    all_true, all_pred  = [], []
    t0 = time.time()

    for batch in tqdm(test_loader, desc="  test", unit="batch", dynamic_ncols=True, disable=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                do_sample      = False,
                num_beams      = 1,
                max_new_tokens = MAX_LABEL_LEN,
            )
        raws  = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [extract_label(clean_output_sequence(r)) for r in raws]

        for smi, exp, pred in zip(batch["smiles"], batch["expected"], preds):
            records.append({"smiles": smi, "expected": exp, "predicted": pred})
        all_true.extend(batch["expected"])
        all_pred.extend(preds)

    logging.info("  Test inference done in %.1fs", time.time() - t0)

    # --- Save predictions ----------------------------------------------------
    pred_path = out_dir / "predictions.csv"
    pd.DataFrame(records).to_csv(pred_path, index=False)
    logging.info("  Predictions saved → %s", pred_path)

    # --- Compute metrics (exclude 'unknown' predictions) ---------------------
    valid_mask = [p != "unknown" for p in all_pred]
    y_true = [t for t, v in zip(all_true, valid_mask) if v]
    y_pred = [p for p, v in zip(all_pred, valid_mask) if v]

    metrics: dict = {
        "n_total":             len(all_true),
        "n_valid_predictions": int(sum(valid_mask)),
    }
    if y_true:
        metrics.update({
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_true, y_pred, average="micro",  zero_division=0)), 4),
            "mcc":      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        })

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logging.info("  Test metrics:")
    for k, v in metrics.items():
        logging.info("    %-24s = %s", k, v)
    logging.info("  Metrics saved → %s", metrics_path)

    # Log test metrics to WandB if a run is currently active
    if wandb.run is not None:
        wandb.log({"test/" + k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace):
    script_start = time.time()

    # Prepare output directories
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Logging: console + file -------------------------------------------
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "finetune.log", mode="w"),
        ],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("")
    logging.info("=" * 65)
    logging.info("  Nach0 LoRA Flavor Fine-tuning")
    logging.info("  Mode    : %s", "WandB sweep → full train" if args.sweep else "direct training")
    logging.info("  Device  : %s", device)
    logging.info("  Outputs : %s", out_dir)
    logging.info("  Weights : %s", MODEL_ROOT)
    logging.info("  CLI args: %s", vars(args))
    logging.info("=" * 65)

    # -----------------------------------------------------------------------
    # STEP 1 / 5 — Load all three dataset splits
    # -----------------------------------------------------------------------
    t = time.time()
    logging.info("")
    logging.info("=" * 65)
    logging.info("  [1/5] Loading datasets from %s", DATA_ROOT)
    logging.info("=" * 65)
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df  = load_split("test")
    logging.info(
        "  Sizes — train: %d | valid: %d | test: %d  (loaded in %.1fs)",
        len(train_df), len(valid_df), len(test_df), time.time() - t,
    )

    # -----------------------------------------------------------------------
    # STEP 2 / 5 — Determine hyperparameters: sweep or CLI
    # -----------------------------------------------------------------------
    logging.info("")
    logging.info("=" * 65)
    logging.info("  [2/5] Hyperparameter selection")
    logging.info("=" * 65)

    if args.sweep:
        # Run WandB Bayesian sweep to discover the best LoRA + optimiser config
        best_cfg     = run_sweep(args)
        lora_r       = int(best_cfg["lora_r"])
        lora_alpha   = lora_r * int(best_cfg.get("lora_alpha_mult", 2))
        lora_dropout = float(best_cfg.get("lora_dropout", 0.05))
        lr           = float(best_cfg["lr"])
        weight_decay = float(best_cfg.get("weight_decay", 0.01))
        batch_size   = int(best_cfg.get("batch_size", args.batch_size))
        logging.info("  Source: WandB sweep best run")
    else:
        # Use values from CLI args (or their defaults)
        lora_r       = args.lora_r
        lora_alpha   = args.lora_r * 2    # standard: alpha = 2r
        lora_dropout = 0.05
        lr           = args.lr
        weight_decay = args.weight_decay
        batch_size   = args.batch_size
        logging.info("  Source: CLI arguments / defaults")

    logging.info(
        "  lora_r=%-2d  lora_alpha=%-2d  lora_dropout=%.2f  "
        "lr=%.2e  weight_decay=%.4f  batch_size=%d",
        lora_r, lora_alpha, lora_dropout, lr, weight_decay, batch_size,
    )

    # -----------------------------------------------------------------------
    # STEP 3 / 5 — Build Nach0 + LoRA model and create DataLoaders
    # -----------------------------------------------------------------------
    t = time.time()
    logging.info("")
    logging.info("=" * 65)
    logging.info("  [3/5] Building model and DataLoaders")
    logging.info("=" * 65)
    tokenizer, model = build_nach0_with_lora(device, lora_r, lora_alpha, lora_dropout)

    train_ds = FlavorDataset(train_df, tokenizer, args.max_input_len)
    valid_ds = FlavorDataset(valid_df, tokenizer, args.max_input_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Optimiser and cosine LR schedule with 5 % linear warm-up.
    # Only LoRA A/B parameters have requires_grad=True, so AdamW updates only
    # ~0.3 % of total model parameters.
    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler('cuda')

    logging.info(
        "  Steps: %d total | %d warmup | %d batches/epoch",
        total_steps, warmup_steps, len(train_loader),
    )
    logging.info("  Model + DataLoaders ready in %.1fs", time.time() - t)

    # -----------------------------------------------------------------------
    # STEP 4 / 5 — Training loop with epoch-level WandB logging
    # -----------------------------------------------------------------------
    logging.info("")
    logging.info("=" * 65)
    logging.info("  [4/5] Training for %d epochs", args.epochs)
    logging.info("=" * 65)

    # Start a WandB run for the full training (separate from any sweep runs)
    if args.wandb_project:
        wandb.init(
            project = args.wandb_project,
            name    = "full-train" if args.sweep else "train",
            config  = {
                "lora_r":        lora_r,
                "lora_alpha":    lora_alpha,
                "lora_dropout":  lora_dropout,
                "lr":            lr,
                "weight_decay":  weight_decay,
                "batch_size":    batch_size,
                "epochs":        args.epochs,
                "sweep_source":  args.sweep,
            },
            tags   = ["full-train"] if args.sweep else ["train"],
            reinit = True,
        )
        logging.info("  WandB run started: %s/%s", args.wandb_project, wandb.run.name)

    best_val_mcc = -1.0
    log_rows: List[dict] = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        logging.info("")
        logging.info("  -- Epoch %d/%d --", epoch, args.epochs)

        # Training pass
        train_loss = train_epoch(
            train_loader, model, optimizer, scheduler, scaler,
            device, args.grad_clip, epoch=epoch,
        )
        logging.info("  train_loss = %.4f", train_loss)

        # Validation pass — returns (loss, accuracy, mcc)
        val_loss, val_acc, val_mcc = eval_epoch(
            valid_loader, model, tokenizer, device, epoch=epoch
        )
        ep_elapsed = time.time() - ep_start

        logging.info(
            "  val_loss = %.4f | val_acc = %.4f | val_mcc = %.4f | epoch_time = %.1fs",
            val_loss, val_acc, val_mcc, ep_elapsed,
        )
        log_rows.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_mcc":      round(val_mcc, 6),
            "epoch_time_s": round(ep_elapsed, 1),
        })

        # Log epoch metrics to WandB
        if wandb.run is not None:
            wandb.log({
                "epoch":        epoch,
                "train_loss":   train_loss,
                "val_loss":     val_loss,
                "val_accuracy": val_acc,
                "val_mcc":      val_mcc,
            })

        # Checkpoint on best MCC — MCC is the primary metric as it handles the
        # class imbalance in the flavor dataset (bitter >> umami/sour) correctly.
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            save_checkpoint(model, epoch, {
                "val_loss": val_loss, "val_accuracy": val_acc, "val_mcc": val_mcc,
            })
            logging.info("  *** New best val_mcc=%.4f — checkpoint saved ***", best_val_mcc)

    total_train_time = time.time() - train_start
    logging.info("")
    logging.info(
        "  Training complete | best_val_mcc=%.4f | total_time=%.1fs",
        best_val_mcc, total_train_time,
    )
    if wandb.run is not None:
        wandb.log({"best_val_mcc": best_val_mcc, "total_train_time_s": total_train_time})

    log_path = out_dir / "training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    logging.info("  Training log saved → %s", log_path)

    # -----------------------------------------------------------------------
    # STEP 5 / 5 — Test evaluation from best checkpoint
    # -----------------------------------------------------------------------
    metrics = run_test_eval(args, test_df, tokenizer, out_dir, batch_size=batch_size)

    # Finish the WandB run for this full training
    if wandb.run is not None:
        wandb.finish()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - script_start
    logging.info("")
    logging.info("=" * 65)
    logging.info("  DONE  —  total elapsed: %.1fs  (%.1f min)", total_elapsed, total_elapsed / 60)
    logging.info("  Small outputs  : %s", out_dir)
    logging.info("  Model weights  : %s", MODEL_ROOT)
    logging.info("  Best val MCC   : %.4f  (checkpoint selection criterion)", best_val_mcc)
    logging.info("  Test MCC       : %.4f", metrics.get("mcc", float("nan")))
    logging.info("  Test accuracy  : %.4f", metrics.get("accuracy", float("nan")))
    logging.info("  Test F1 macro  : %.4f", metrics.get("f1_macro", float("nan")))
    logging.info("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nach0 LoRA instruction fine-tuning — SMILES text only"
    )

    # --- Training hyperparameters (used in direct mode; overridden by sweep) ---
    parser.add_argument("--epochs",        type=int,   default=10,
                        help="Training epochs for the final model (default: 10)")
    parser.add_argument("--batch-size",    type=int,   default=16,
                        help="Per-device batch size (default: 16)")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="AdamW peak learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay",  type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--grad-clip",     type=float, default=1.0,
                        help="Gradient norm clipping threshold (default: 1.0)")
    parser.add_argument("--lora-r",        type=int,   default=16,
                        help="LoRA rank — higher = more trainable params (default: 16)")
    parser.add_argument("--max-input-len", type=int,   default=MAX_INPUT_LEN,
                        help=f"Max encoder token length (default: {MAX_INPUT_LEN})")
    parser.add_argument("--num-workers",   type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--output-dir",    type=str,
                        default=str(REPO_ROOT / "results" / "nach0" / "flavor_ft"),
                        help="Directory for predictions, metrics, and log files")

    # --- WandB sweep options ---
    parser.add_argument("--sweep",         action="store_true",
                        help="Run WandB Bayesian sweep then train with best config")
    parser.add_argument("--sweep-trials",  type=int,   default=20,
                        help="Number of sweep trials (default: 20)")
    parser.add_argument("--sweep-epochs",  type=int,   default=3,
                        help="Training epochs per sweep trial (default: 3)")
    parser.add_argument("--wandb-project", type=str,   default="nach0-flavor-ft",
                        help="WandB project name (default: nach0-flavor-ft)")

    args = parser.parse_args()
    main(args)
