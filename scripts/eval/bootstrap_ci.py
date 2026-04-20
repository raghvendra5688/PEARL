"""
bootstrap_ci.py — Bootstrap confidence intervals for PEARL classification metrics.

Computes 95% bootstrap CIs (and SE) for MCC, AUC, F1-macro, and Accuracy using
stratified resampling.  Can be used in two ways:

  1. Standalone CLI — pass a CSV with y_true, y_pred, y_prob columns:
         python scripts/eval/bootstrap_ci.py \
             --csv path/to/predictions.csv \
             --label y_true --pred y_pred --prob "prob_0,prob_1" \
             --n-bootstraps 10000 --alpha 0.05

  2. Importable function — call `bootstrap_metrics()` from other PEARL scripts:
         from scripts.eval.bootstrap_ci import bootstrap_metrics
         ci = bootstrap_metrics(y_true, y_pred, y_prob)

CSV format expected by the CLI:
  - y_true : integer class labels  (0/1 for binary; 0..C-1 for multiclass)
  - y_pred : predicted class labels (same encoding)
  - y_prob : probability columns — one per class, comma-separated in --prob
             e.g. for binary: "prob_0,prob_1"; for 5-class: "p0,p1,p2,p3,p4"
             For AUC computation only y_prob of the positive class is needed
             for binary tasks; all columns are used for multi-class OvR AUC.

Output: a dict (or printed table) with keys:
  metric -> {"mean", "se", "ci_lower", "ci_upper", "observed"}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ── core bootstrap engine ──────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> dict:
    """Compute MCC, AUC, F1-macro, and Accuracy for one sample."""
    acc  = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if n_classes == 2:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            auc = float("nan")
    else:
        try:
            auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc = float("nan")

    return {"MCC": mcc, "AUC": auc, "F1_macro": f1, "Accuracy": acc}


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true        : 1-D integer array of true class labels.
    y_pred        : 1-D integer array of predicted class labels.
    y_prob        : 2-D float array of class probabilities, shape (n, n_classes).
                    For binary tasks pass shape (n, 2) or (n, 1) — the latter is
                    treated as P(class=1).
    n_bootstraps  : Number of bootstrap resamples (default 10 000).
    alpha         : Significance level; CIs are (alpha/2, 1-alpha/2) percentiles.
    random_state  : Seed for reproducibility.

    Returns
    -------
    dict mapping metric name -> {
        "observed" : point estimate on original sample,
        "mean"     : mean of bootstrap distribution,
        "se"       : standard error of bootstrap distribution,
        "ci_lower" : lower percentile CI,
        "ci_upper" : upper percentile CI,
    }
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])

    n_classes = y_prob.shape[1]
    n_samples = len(y_true)
    classes   = np.unique(y_true)

    observed = _compute_metrics(y_true, y_pred, y_prob, n_classes)

    rng = np.random.default_rng(random_state)
    boot_scores = {k: [] for k in observed}

    for _ in range(n_bootstraps):
        # Stratified resample: sample with replacement within each class
        idx = np.concatenate([
            rng.choice(np.where(y_true == c)[0],
                       size=np.sum(y_true == c), replace=True)
            for c in classes
        ])
        bt, bp, bpr = y_true[idx], y_pred[idx], y_prob[idx]
        # Skip degenerate samples where a class is absent after resample
        if len(np.unique(bt)) < len(classes):
            continue
        m = _compute_metrics(bt, bp, bpr, n_classes)
        for k, v in m.items():
            if not np.isnan(v):
                boot_scores[k].append(v)

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    results = {}
    for k, vals in boot_scores.items():
        arr = np.array(vals)
        results[k] = {
            "observed" : round(observed[k], 4),
            "mean"     : round(float(np.mean(arr)), 4),
            "se"       : round(float(np.std(arr, ddof=1)), 4),
            "ci_lower" : round(float(np.percentile(arr, lo)), 4),
            "ci_upper" : round(float(np.percentile(arr, hi)), 4),
            "n_valid_bootstraps": len(arr),
        }
    return results


def pairwise_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_prob_b: np.ndarray,
    metric: str = "MCC",
    n_bootstraps: int = 10_000,
    random_state: int = 42,
) -> dict:
    """
    Paired bootstrap test for the difference in a metric between two models.

    Tests H0: metric(A) == metric(B) using the bootstrap distribution of
    (metric_A - metric_B).  Returns the two-sided p-value and 95% CI on the
    difference.

    Parameters
    ----------
    y_true           : shared true labels for both models.
    y_pred_a/b       : predicted labels for model A and B.
    y_prob_a/b       : predicted probabilities for model A and B.
    metric           : one of "MCC", "AUC", "F1_macro", "Accuracy".
    n_bootstraps     : number of resamples.

    Returns
    -------
    dict with keys:
        "delta_observed", "ci_lower", "ci_upper", "p_value", "se"
    """
    y_true   = np.asarray(y_true).ravel()
    y_pred_a = np.asarray(y_pred_a).ravel()
    y_pred_b = np.asarray(y_pred_b).ravel()
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    if y_prob_a.ndim == 1:
        y_prob_a = np.column_stack([1 - y_prob_a, y_prob_a])
    if y_prob_b.ndim == 1:
        y_prob_b = np.column_stack([1 - y_prob_b, y_prob_b])

    n_classes = y_prob_a.shape[1]
    classes   = np.unique(y_true)

    def _metric(yt, yp, ypr):
        return _compute_metrics(yt, yp, ypr, n_classes)[metric]

    obs_a = _metric(y_true, y_pred_a, y_prob_a)
    obs_b = _metric(y_true, y_pred_b, y_prob_b)
    delta_obs = obs_a - obs_b

    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(n_bootstraps):
        idx = np.concatenate([
            rng.choice(np.where(y_true == c)[0],
                       size=np.sum(y_true == c), replace=True)
            for c in classes
        ])
        yt = y_true[idx]
        if len(np.unique(yt)) < len(classes):
            continue
        da = _metric(yt, y_pred_a[idx], y_prob_a[idx])
        db = _metric(yt, y_pred_b[idx], y_prob_b[idx])
        if not (np.isnan(da) or np.isnan(db)):
            deltas.append(da - db)

    deltas = np.array(deltas)
    # Two-sided p-value: fraction of bootstrap deltas as or more extreme than 0
    # under the shifted null (shift distribution to have mean 0)
    shifted = deltas - np.mean(deltas)
    p_value = float(np.mean(np.abs(shifted) >= np.abs(delta_obs)))

    return {
        "metric"         : metric,
        "model_a_observed": round(obs_a, 4),
        "model_b_observed": round(obs_b, 4),
        "delta_observed" : round(delta_obs, 4),
        "ci_lower"       : round(float(np.percentile(deltas, 2.5)), 4),
        "ci_upper"       : round(float(np.percentile(deltas, 97.5)), 4),
        "se"             : round(float(np.std(deltas, ddof=1)), 4),
        "p_value"        : round(p_value, 4),
        "significant_95" : p_value < 0.05,
        "n_valid_bootstraps": len(deltas),
    }


# ── convenience: run CIs for all PEARL datasets from saved metrics CSVs ────────

def run_all_datasets(
    results_root: Path,
    n_bootstraps: int = 10_000,
    out_json: Optional[Path] = None,
) -> dict:
    """
    Run bootstrap CIs for all four PEARL datasets using the aggregated result CSVs.

    NOTE: Because the aggregated CSVs contain only summary metrics (not raw
    predictions), this function estimates CI width using the normal approximation:
        SE ≈ metric_value * (1 - metric_value) / sqrt(n_test)
    with dataset-specific test-set sizes.  For exact bootstrap CIs, call
    `bootstrap_metrics()` with raw predictions from your evaluation pipeline.

    Parameters
    ----------
    results_root : path to PEARL/results/
    out_json     : optional path to write the CI table as JSON.

    Returns
    -------
    dict: dataset -> list of dicts with metric CIs per configuration.
    """
    TEST_SIZES = {"bace": 152, "bbbp": 204, "clintox": 143, "flavor": 1503}
    datasets = ["bace", "bbbp", "clintox", "flavor"]
    all_ci = {}

    for ds in datasets:
        csv_path = results_root / "rag" / "aggregated" / f"{ds}_rag_results.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping {ds}")
            continue

        df = pd.read_csv(csv_path)
        n  = TEST_SIZES[ds]
        rows = []
        for _, row in df.iterrows():
            r = {"dataset": ds, "embedding_model": row["embedding_model"],
                 "ml_model": row.get("ml_model", "---")}
            for metric in ["MCC", "AUC", "F1_macro", "Accuracy"]:
                col = metric if metric in row else metric.replace("_", "")
                if col not in row:
                    continue
                v  = float(row[col])
                # Normal approximation SE for bounded [0,1] metrics
                se = float(np.sqrt(max(v * (1 - v), 1e-6) / n))
                r[f"{metric}_observed"] = round(v, 4)
                r[f"{metric}_se"]       = round(se, 4)
                r[f"{metric}_ci_lower"] = round(max(0.0, v - 1.96 * se), 4)
                r[f"{metric}_ci_upper"] = round(min(1.0, v + 1.96 * se), 4)
            rows.append(r)
        all_ci[ds] = rows
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}  (n_test = {n})")
        print(f"{'='*60}")
        _print_ci_table(rows)

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(all_ci, f, indent=2)
        print(f"\n[INFO] CIs written to {out_json}")

    return all_ci


def _print_ci_table(rows: list):
    header = f"{'Model':<45} {'MCC':>6}  {'95% CI':>16}  {'SE':>6}  {'AUC':>6}  {'95% CI':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        model = f"{r.get('embedding_model','')}/{r.get('ml_model','')}"
        mcc   = r.get("MCC_observed", float("nan"))
        mcc_l = r.get("MCC_ci_lower", float("nan"))
        mcc_u = r.get("MCC_ci_upper", float("nan"))
        mcc_s = r.get("MCC_se", float("nan"))
        auc   = r.get("AUC_observed", float("nan"))
        auc_l = r.get("AUC_ci_lower", float("nan"))
        auc_u = r.get("AUC_ci_upper", float("nan"))
        print(
            f"{model:<45} {mcc:>6.3f}  [{mcc_l:.3f}, {mcc_u:.3f}]  "
            f"{mcc_s:>6.4f}  {auc:>6.3f}  [{auc_l:.3f}, {auc_u:.3f}]"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for PEARL metrics.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- single-model CIs from a CSV of raw predictions ----
    ci = sub.add_parser(
        "ci",
        help="Bootstrap CIs for one model from a raw-predictions CSV.",
    )
    ci.add_argument("--csv", required=True, help="Path to predictions CSV.")
    ci.add_argument("--label", default="y_true",
                    help="Column name for true labels (default: y_true).")
    ci.add_argument("--pred", default="y_pred",
                    help="Column name for predicted labels (default: y_pred).")
    ci.add_argument(
        "--prob", default=None,
        help=(
            "Comma-separated probability column names, one per class.\n"
            "E.g. for binary: 'prob_0,prob_1'  or  'prob_1' (positive class only).\n"
            "If omitted, AUC cannot be computed."
        ),
    )
    ci.add_argument("--n-bootstraps", type=int, default=10_000)
    ci.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level for CIs (default 0.05 → 95%% CI).")
    ci.add_argument("--out", default=None, help="Path to write JSON results.")

    # ---- pairwise test ----
    pw = sub.add_parser(
        "pairwise",
        help="Paired bootstrap test: model A vs model B.",
    )
    pw.add_argument("--csv-a", required=True, help="CSV for model A.")
    pw.add_argument("--csv-b", required=True, help="CSV for model B.")
    pw.add_argument("--label", default="y_true")
    pw.add_argument("--pred", default="y_pred")
    pw.add_argument("--prob", default=None,
                    help="Comma-separated prob columns (same for both CSVs).")
    pw.add_argument("--metric", default="MCC",
                    choices=["MCC", "AUC", "F1_macro", "Accuracy"])
    pw.add_argument("--n-bootstraps", type=int, default=10_000)
    pw.add_argument("--out", default=None)

    # ---- approximate CIs over all datasets from aggregated CSVs ----
    agg = sub.add_parser(
        "all-datasets",
        help=(
            "Approximate 95%% CIs for all PEARL datasets using the aggregated\n"
            "results CSVs (normal approximation; requires only summary metrics)."
        ),
    )
    agg.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
        help="Path to PEARL/results/ directory.",
    )
    agg.add_argument("--out", default=None, help="Path to write JSON.")

    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode == "all-datasets":
        run_all_datasets(
            results_root=Path(args.results_root),
            out_json=Path(args.out) if args.out else None,
        )
        return

    def _load_pred_csv(path, label_col, pred_col, prob_cols):
        df = pd.read_csv(path)
        y_true = df[label_col].values
        y_pred = df[pred_col].values
        if prob_cols:
            cols  = [c.strip() for c in prob_cols.split(",")]
            y_prob = df[cols].values
            if y_prob.shape[1] == 1:
                y_prob = np.column_stack([1 - y_prob[:, 0], y_prob[:, 0]])
        else:
            n_cls  = len(np.unique(y_true))
            y_prob = np.eye(n_cls)[y_pred]   # hard probs — AUC will be meaningless
            print("[WARN] --prob not supplied; AUC results are unreliable.")
        return y_true, y_pred, y_prob

    if args.mode == "ci":
        y_true, y_pred, y_prob = _load_pred_csv(
            args.csv, args.label, args.pred, args.prob
        )
        results = bootstrap_metrics(
            y_true, y_pred, y_prob,
            n_bootstraps=args.n_bootstraps,
            alpha=args.alpha,
        )
        print(f"\nBootstrap CIs  (n={len(y_true)}, {args.n_bootstraps} resamples, "
              f"alpha={args.alpha})")
        print(f"{'Metric':<12} {'Observed':>9} {'Mean':>9} {'SE':>7}  "
              f"{'CI Lower':>9}  {'CI Upper':>9}  {'n_valid':>8}")
        print("-" * 72)
        for m, v in results.items():
            print(f"{m:<12} {v['observed']:>9.4f} {v['mean']:>9.4f} "
                  f"{v['se']:>7.4f}  {v['ci_lower']:>9.4f}  "
                  f"{v['ci_upper']:>9.4f}  {v['n_valid_bootstraps']:>8d}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n[INFO] Results written to {args.out}")

    elif args.mode == "pairwise":
        y_true_a, y_pred_a, y_prob_a = _load_pred_csv(
            args.csv_a, args.label, args.pred, args.prob
        )
        y_true_b, y_pred_b, y_prob_b = _load_pred_csv(
            args.csv_b, args.label, args.pred, args.prob
        )
        if not np.array_equal(y_true_a, y_true_b):
            sys.exit("[ERROR] y_true arrays differ between the two CSVs.")

        result = pairwise_bootstrap_test(
            y_true_a, y_pred_a, y_prob_a, y_pred_b, y_prob_b,
            metric=args.metric,
            n_bootstraps=args.n_bootstraps,
        )
        print(f"\nPaired bootstrap test — metric: {args.metric}")
        print(f"  Model A observed : {result['model_a_observed']:.4f}")
        print(f"  Model B observed : {result['model_b_observed']:.4f}")
        print(f"  Delta (A − B)    : {result['delta_observed']:.4f}")
        print(f"  95% CI on delta  : [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  SE               : {result['se']:.4f}")
        print(f"  p-value (two-sided): {result['p_value']:.4f}")
        print(f"  Significant (α=0.05): {result['significant_95']}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[INFO] Results written to {args.out}")


if __name__ == "__main__":
    main()
