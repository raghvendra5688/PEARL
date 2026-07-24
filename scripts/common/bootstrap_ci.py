"""
Bootstrapped confidence intervals, matching the exact protocol used throughout
PEARL_paper.tex: "bootstrapped 95% confidence intervals (10,000 stratified
resamples, normal-approximation)" (Section "Evaluation Metrics"). The paper's
own comparisons treat non-overlapping CIs as the bar for a real difference
between two methods -- e.g. it reports that BACE's Uni-Mol vs. MolFormer gap
(MCC 0.623 vs 0.577) does NOT reach significance on the n=152 test set.

Any new baseline (PC-only, Chemprop, GCN, ...) must report CIs computed the
same way, or "X beats Y" claims are not on the same footing as the paper's own
statistically-qualified claims and are not a fair comparison.

Method (matches the paper's stated protocol):
- Resample the TEST SET predictions (not retrain the model) with replacement,
  n_resamples times.
- Classification: resampling is STRATIFIED by class label, so each resample
  preserves the original class proportions (appropriate for the class-imbalanced
  binary/multiclass tasks throughout PEARL).
- Regression: plain (non-stratified) resampling of (y_true, y_pred) pairs.
- CI = point_estimate +/- 1.96 * SE, where point_estimate is the metric computed
  on the FULL test set (not the bootstrap mean) and SE is the bootstrap
  distribution's standard deviation (normal-approximation, not percentile).
"""

from typing import Callable, Dict, Optional

import numpy as np

N_RESAMPLES = 10_000
Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    stratified: bool = True,
    n_resamples: int = N_RESAMPLES,
    seed: int = 42,
) -> Dict[str, float]:
    """Returns {point, ci_lo, ci_hi, se} for metric_fn(y_true, y_pred) under
    10,000 (default) stratified (classification) or plain (regression) bootstrap
    resamples of the test set, with a normal-approximation 95% CI.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y_true, y_pred))

    if stratified:
        classes, class_indices = np.unique(y_true), None
        class_indices = {c: np.where(y_true == c)[0] for c in classes}

    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        if stratified:
            idx = np.concatenate([
                rng.choice(class_indices[c], size=len(class_indices[c]), replace=True)
                for c in classes
            ])
        else:
            idx = rng.choice(n, size=n, replace=True)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            scores[i] = np.nan

    se = float(np.nanstd(scores))
    return {
        "point": round(point, 4),
        "ci_lo": round(point - Z_95 * se, 4),
        "ci_hi": round(point + Z_95 * se, 4),
        "se": round(se, 4),
    }


def ci_overlap(ci_a: Dict[str, float], ci_b: Dict[str, float]) -> bool:
    """True if two CIs overlap -- matches the paper's own bar: non-overlapping
    CIs are treated as evidence of a statistically meaningful difference."""
    return not (ci_a["ci_hi"] < ci_b["ci_lo"] or ci_b["ci_hi"] < ci_a["ci_lo"])


def se_overlap(ci_a: Dict[str, float], ci_b: Dict[str, float]) -> bool:
    """True if two mean +/- 1 SE bands overlap.

    Narrower than ci_overlap() (1 SE vs. the paper's 1.96 SE / 95% CI band),
    per the updated evaluation convention adopted after Phase 7: comparisons
    going forward judge "is method A really better than method B" against a
    mean +/- 1 SE band rather than the full 95% CI, tolerating less overlap
    before declaring a real difference. Both bands are computed from the same
    bootstrap_ci() output -- only the comparison width changes, not how SE
    itself is estimated.
    """
    lo_a, hi_a = ci_a["point"] - ci_a["se"], ci_a["point"] + ci_a["se"]
    lo_b, hi_b = ci_b["point"] - ci_b["se"], ci_b["point"] + ci_b["se"]
    return not (hi_a < lo_b or hi_b < lo_a)
