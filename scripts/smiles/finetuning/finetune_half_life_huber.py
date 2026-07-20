# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: effichem
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Hyperparameter tuning using WandB -- HALF_LIFE | HUBER Loss (regression)

# %%
#Importing Libraries
import wandb
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Repo-relative paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from safe_model_loading import load_seq_cls_model_safe, load_tokenizer_safe  # noqa: E402
DATA_DIR = REPO_ROOT / "data" / "clean" / "half_life_datasets"
RESULTS_DIR = REPO_ROOT / "results" / "lora_finetuning" / "half_life"
# NEW artifacts go to PEARL_EXTRAS_V2 (the original PEARL_EXTRAS default was
# found read-only from this host during Phase 3 -- see editor_response_suggestions.md).
PEARL_EXTRAS = Path(os.getenv("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
TRAINING_DIR = PEARL_EXTRAS / "lora_training" / "half_life"  # scratch per-trial checkpoints, not repo-tracked
os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(str(TRAINING_DIR), exist_ok=True)

LABEL_COL = "Half_Life_Hours"
TARGET_TRANSFORM = 'log1p'  # None or "log1p"

from dotenv import load_dotenv
load_dotenv()
wandb_api_key=os.getenv("WANDB_API_KEY")

#Set up WANDB Key
if wandb_api_key is not None:
    print(f"API Key loaded successfully! Key snippet: {wandb_api_key[:5]}...")
else:
    print("API Key not found. Please check your .env file and environment setup.")

# WandB's own local run cache defaults to CWD ("wandb/" under the repo root) --
# redirect it to EffiChem_Extras_v2 too, same rationale as TRAINING_DIR/PEARL_EXTRAS above.
os.environ.setdefault("WANDB_DIR", str(PEARL_EXTRAS / "wandb_logs"))
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

wandb.login(key=wandb_api_key)


# %%
def apply_target_transform(y):
    return np.log1p(y) if TARGET_TRANSFORM == "log1p" else y


def invert_target_transform(y):
    return np.expm1(y) if TARGET_TRANSFORM == "log1p" else y


# %%
# Load train and valid datasets
def data_load():
    train_df = pd.read_csv(str(DATA_DIR / 'train_clean.csv'))
    val_df = pd.read_csv(str(DATA_DIR / 'valid_clean.csv'))
    return train_df, val_df


# %%
#Process the data into tensor format
def data_prep(data_process, tokenizer):

    smiles_list = data_process['Standardized SMILES'].tolist()
    tokenized = tokenizer(smiles_list)

    dataset = Dataset.from_dict(tokenized)
    labels = apply_target_transform(data_process[LABEL_COL].astype(float).values).tolist()

    dataset = dataset.add_column("labels", labels)

    return dataset

# %%
def lora_config(r,lora_alpha,dropout):

    lora_config = LoraConfig(
        task_type="SEQ_CLS",  # HF regression head reuses the SEQ_CLS classifier class with num_labels=1
        r=r,
        lora_alpha=lora_alpha,
        target_modules='all-linear',
        lora_dropout=dropout
    )

    return lora_config


# %% [markdown]
# ### HUBER Loss Function (regression)

# %%
#Create custom Huber loss trainer (robust to outliers vs. plain MSE)
class HuberLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs.get("logits").squeeze(-1)

        loss = F.huber_loss(preds, labels.float(), delta=1.0)
        return (loss, outputs) if return_outputs else loss

# %%
#Evaluation for validation and test sets
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    preds_orig = invert_target_transform(preds)
    labels_orig = invert_target_transform(labels)

    return {
        "eval_r2": float(r2_score(labels_orig, preds_orig)),
        "MAE": float(mean_absolute_error(labels_orig, preds_orig)),
        "RMSE": float(np.sqrt(mean_squared_error(labels_orig, preds_orig))),
        "Spearman": float(spearmanr(labels_orig, preds_orig).correlation),
    }

# %%
import re

#Perform a sweep using WANDb
sweep_config = {
"name": "HALF_LIFE_HUBER_Hyperparameter_Tuning",
"method": "bayes",
"metric": {
    "goal": "maximize",
    "name": "eval/r2"},
"parameters": {"lr": {
        "distribution": "uniform",
        "min": 1e-5,
        "max": 2e-3},
    "r": {"values": [4,8,16,32,64]},
    "lora_alpha": {"values": [8,16,32,64,128]},
    "dropout": {"values": [0.0,0.1,0.2] },

    "optimizer": {"value": ["adamw"]}}
}

model_list= ["DeepChem/ChemBERTa-77M-MLM",
             "DeepChem/ChemBERTa-77M-MTR",
             "ibm/MoLFormer-XL-both-10pct"]

best_configs={}

def safe_model_name(name1):
        return re.sub(r"[^a-zA-Z0-9]", "__", name1)

for model_name in model_list:

    sweep_id = wandb.sweep(sweep_config, project="huggingface")

    model_id_clean = safe_model_name(model_name)
    print(f"Running sweep for model: {model_id_clean}")

    def run_training():
        print(f"Running training for model: {model_id_clean}")
        # Initialize W&B with sweep
        run = wandb.init(project="huggingface")
        config = run.config

        print(f"Model ID cleaned: {model_id_clean}")
        run_id = wandb.run.id

        # Define unique output folders
        save_dir = str(TRAINING_DIR / "HUBER_Loss" / model_id_clean / run_id)
        logging_dir = str(TRAINING_DIR / "HUBER_Loss" / model_id_clean / run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Load tokenizer and model
        tokenizer = load_tokenizer_safe(model_name,trust_remote_code=True)
        model = load_seq_cls_model_safe(
            model_name,
            num_labels=1,
            problem_type="regression",
            trust_remote_code=True
        )

        # Load and preprocess data
        train_data, val_data = data_load()
        training_data = data_prep(train_data, tokenizer)
        validation_data = data_prep(val_data, tokenizer)

        # Apply LoRA
        peft_config = lora_config(config.r, config.lora_alpha, config.dropout)
        lora_model = get_peft_model(model, peft_config)
        lora_model.print_trainable_parameters()

        # Define training args
        training_args = TrainingArguments(
            output_dir=save_dir,
            eval_strategy="epoch",
            learning_rate=config.lr,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            num_train_epochs=20,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=logging_dir,
            logging_strategy="steps",
            logging_steps=500,
            report_to="wandb",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_r2"
        )

        trainer = HuberLossTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=training_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        print(f"Training completed for model: {model_name}")
        del trainer
        torch.cuda.empty_cache()

        wandb.finish()

    #Run a wandb agent
    wandb.agent(sweep_id, function=run_training, count=20, project="huggingface")

    #Perform a sweep
    api = wandb.Api()
    sweep = api.sweep(f"huggingface/{sweep_id}")

    #Find the run with the best R2
    runs_with_r2 = [run for run in sweep.runs if 'eval/r2' in run.summary_metrics]

    def get_best_epoch_r2(run):
        history = run.scan_history(keys=["eval/r2"])
        val = max((row["eval/r2"] for row in history if row.get("eval/r2") is not None),default=float("-inf"))
        print(f"Best eval/r2 for {run.id} was: {val}")
        return val

    if runs_with_r2:
        best_run = max(runs_with_r2, key=get_best_epoch_r2)
    else:
        raise ValueError("No runs found with 'eval/r2' metric.")

    best_hyperparameters = best_run.config
    print(f"Best hyperparameters: {best_hyperparameters}")
    print("Completed sweep for model: ",model_id_clean)

    best_configs[model_id_clean] = best_run

    del api, sweep, sweep_id, runs_with_r2, model_id_clean
    gc.collect()

print(best_configs)

# %%

# Map your folder names to the base HuggingFace model names
MODEL_NAME_MAP = {
    "DeepChem__ChemBERTa__77M__MLM": "DeepChem/ChemBERTa-77M-MLM",
    "DeepChem__ChemBERTa__77M__MTR": "DeepChem/ChemBERTa-77M-MTR",
    "ibm__MoLFormer__XL__both__10pct": "ibm/MoLFormer-XL-both-10pct",
}

#Load the test data
test_data=pd.read_csv(str(DATA_DIR / 'test_clean.csv'))

#Path to model root
models_root_dir = str(TRAINING_DIR / "HUBER_Loss") + "/"
output_dir = str(RESULTS_DIR / "test_results_huber") + "/"
os.makedirs(output_dir, exist_ok=True)

#Essential args for evaluation
eval_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=64,
    report_to="none",
    disable_tqdm=True,
)

all_test_results = []
for key in MODEL_NAME_MAP.keys():

    model_folder = models_root_dir+key
    print("Model folder: ",model_folder)

    best_run = best_configs[key]
    checkpoint_path = model_folder+"/"+best_run.id
    print("Best PEFT folder: ",checkpoint_path)

    hf_model_name = MODEL_NAME_MAP[key]
    print(f"Using base model: {hf_model_name}")

    # Load tokenizer and base model for the model type
    tokenizer = load_tokenizer_safe(hf_model_name, trust_remote_code=True)
    base_model = load_seq_cls_model_safe(
        hf_model_name,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=True
    )

    # Create the SMILES dataset in ingestible format
    smiles_test = test_data['Standardized SMILES'].tolist()
    test_tokenized = tokenizer(smiles_test)
    test_dataset = Dataset.from_dict(test_tokenized)
    test_labels = apply_target_transform(test_data[LABEL_COL].astype(float).values).tolist()
    test_dataset = test_dataset.add_column("labels", test_labels)

    # Load the adapter checkpoint
    adapter_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    adapter_model.eval()

    # Evaluation
    trainer = Trainer(
        model=adapter_model,
        args=eval_args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"\n Evaluating {checkpoint_path}")

    test_results = trainer.evaluate()
    test_results["best_checkpoint"]=checkpoint_path
    all_test_results.append(test_results)

    # Save Base Model with LoRA weights
    final_adapter_model = adapter_model.merge_and_unload()
    save_path = str(PEARL_EXTRAS / "huber_loss_HALF_LIFE" / (key+"_LoRA_Finetuned")) + "/"
    final_adapter_model.save_pretrained(save_path)

    import shutil
    shutil.rmtree(model_folder, ignore_errors=True)

all_test_results_df = pd.DataFrame(all_test_results)
# Column names come straight from trainer.evaluate()'s own dict keys (eval_r2,
# eval_MAE, eval_RMSE, eval_Spearman, eval_loss, eval_runtime, eval_samples_per_second,
# eval_steps_per_second, epoch, best_checkpoint) -- do NOT hardcode a renamed
# column list here, HF's automatic eval_ prefixing means the real column count
# and order don't match a fixed short list (this previously crashed with a
# pandas "Length mismatch" error; a hardcoded list that happened to match the
# count would silently mislabel columns instead).
for col in all_test_results_df.columns:
    all_test_results_df[col] = all_test_results_df[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
print(all_test_results_df)

all_test_results_df.to_csv(output_dir+"/huber_loss_test_results.csv",header="infer",index=False)

import shutil
shutil.rmtree(str(TRAINING_DIR / "HUBER_Loss"), ignore_errors=True)

# %%
