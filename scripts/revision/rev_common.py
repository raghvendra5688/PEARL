"""Shared harness for the PEARL revision experiments.

Protocol replicates the manuscript's ML stage (pc_only_modelling.py): Optuna TPE, 20 trials,
5-fold CV on the TRAIN split only (MCC for classification, -RMSE for regression), balanced
sample weights for classification, then one fit on train and one evaluation on test.
Differences (deliberate): the seed is a parameter (controls Optuna sampler, CV shuffle and
model RNG) and per-molecule test predictions are stored for paired bootstrapping.
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "revision"
CLEAN = REPO / "data" / "clean"
PCF = REPO / "data" / "pc_only_features"
EX1 = Path(os.environ.get("EXTRAS_V1", "/export/qcai-omics/Raghvendra/EffiChem_Extras"))
EX2 = Path(os.environ.get("EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))

DS = {
    "bace":      dict(label="Class", task="binary"),
    "bbbp":      dict(label="p_np", task="binary"),
    "herg":      dict(label="hERG_Inhib", task="binary"),
    "dili":      dict(label="DILI_Label", task="binary"),
    "caco2":     dict(label="Caco2_LogPapp", task="regression", tt=None),
    "half_life": dict(label="Half_Life_Hours", task="regression", tt="log1p"),
    "flavor":    dict(label="Canonicalized Taste", task="multiclass"),
}
SMI = "Standardized SMILES"
CLFS = ["XGBoost", "LightGBM", "CatBoost"]


def load_clean(ds):
    """train/valid/test dataframes (SMILES, label) exactly as the PC-only pipeline reads them."""
    out = {}
    for sp in ["train", "valid", "test"]:
        df = pd.read_csv(CLEAN / f"{ds}_datasets" / f"{sp}_clean.csv")
        df = df.dropna(subset=[SMI, DS[ds]["label"]]).reset_index(drop=True)
        out[sp] = df
    return out


def load_pc(ds):
    """473-d (467 for DILI/Caco-2) PC feature matrices, row-aligned with load_clean()."""
    out = {}
    for sp in ["train", "valid", "test"]:
        f = pd.read_csv(PCF / ds / f"{sp}_pc_features.csv")
        f = f.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        f = f.fillna(f.median()).clip(-1e6, 1e6)
        out[sp] = f
    return out


def encode_labels(ds, dfs):
    """returns y arrays (raw scale), and class list for multiclass."""
    lab = DS[ds]["label"]
    if DS[ds]["task"] == "multiclass":
        classes = sorted(dfs["train"][lab].unique())
        m = {c: i for i, c in enumerate(classes)}
        return {sp: dfs[sp][lab].map(m).values.astype(int) for sp in dfs}, classes
    if DS[ds]["task"] == "binary":
        return {sp: dfs[sp][lab].astype(int).values for sp in dfs}, [0, 1]
    return {sp: dfs[sp][lab].astype(float).values for sp in dfs}, None


def _tt(y, tt):
    return np.log1p(y) if tt == "log1p" else y


def _itt(y, tt):
    return np.expm1(y) if tt == "log1p" else y


def _dev():
    # XGBoost runs on GPU (hist, cuda) to keep the HPO budget tractable; LightGBM/CatBoost stay on CPU
    n = int(os.environ.get("REV_NGPU", "2"))   # number of GPUs visible; 0 -> CPU XGBoost
    return "cpu" if n == 0 else f"cuda:{os.getpid() % n}"


def make_model(clf, task, ncls, params, seed, nj):
    import xgboost as xgb, lightgbm as lgb, catboost as cb
    if task == "regression":
        if clf == "XGBoost":
            return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, tree_method="hist", device=_dev(), n_jobs=nj, **params)
        if clf == "LightGBM":
            return lgb.LGBMRegressor(objective="regression", random_state=seed, n_jobs=nj, verbosity=-1, **params)
        return cb.CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=0, thread_count=nj, **params)
    mc = task == "multiclass"
    if clf == "XGBoost":
        return xgb.XGBClassifier(objective="multi:softprob" if mc else "binary:logistic",
                                 eval_metric="mlogloss" if mc else "logloss", random_state=seed,
                                 tree_method="hist", device=_dev(), n_jobs=nj, **({"num_class": ncls} if mc else {}), **params)
    if clf == "LightGBM":
        return lgb.LGBMClassifier(objective="multiclass" if mc else "binary", class_weight="balanced",
                                  random_state=seed, n_jobs=nj, verbosity=-1, **({"num_class": ncls} if mc else {}), **params)
    return cb.CatBoostClassifier(loss_function="MultiClass" if mc else "Logloss", auto_class_weights="Balanced",
                                 random_seed=seed, verbose=0, thread_count=nj, **params)


def tune(clf, Xtr, ytr, sw, task, ncls, seed, trials, nj):
    import optuna
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import matthews_corrcoef, mean_squared_error
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        p = dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                 learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                 n_estimators=trial.suggest_int("n_estimators", 100, 600))
        m = make_model(clf, task, ncls, p, seed, nj)
        if task == "regression":
            cv = KFold(5, shuffle=True, random_state=seed); sp = cv.split(Xtr)
        else:
            cv = StratifiedKFold(5, shuffle=True, random_state=seed); sp = cv.split(Xtr, ytr)
        sc = []
        for a, b in sp:
            m.fit(Xtr[a], ytr[a], sample_weight=sw[a]); pr = m.predict(Xtr[b])
            sc.append(-np.sqrt(mean_squared_error(ytr[b], pr)) if task == "regression" else matthews_corrcoef(ytr[b], pr))
        return float(np.mean(sc))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=trials)
    return st.best_params, float(st.best_value)   # best_value = 5-fold CV score on the training split (selection criterion that never sees test data)


def metrics(task, y, pred, proba):
    from sklearn.metrics import matthews_corrcoef, roc_auc_score, r2_score, mean_absolute_error, f1_score, accuracy_score
    from scipy.stats import spearmanr
    if task == "regression":
        return dict(Spearman=float(spearmanr(y, pred).correlation), R2=float(r2_score(y, pred)), MAE=float(mean_absolute_error(y, pred)))
    m = dict(MCC=float(matthews_corrcoef(y, pred)), Accuracy=float(accuracy_score(y, pred)), F1_macro=float(f1_score(y, pred, average="macro")))
    try:
        m["AUC"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro")) if task == "multiclass" else float(roc_auc_score(y, proba))
    except Exception:
        m["AUC"] = float("nan")
    return m


def primary(task):
    return "Spearman" if task == "regression" else "MCC"


def run_cell(out_prefix, ds, Xtr, ytr_raw, Xte, yte_raw, clf, seed, trials=20, nj=4, meta=None):
    """Tune on train-CV, fit, predict test. Writes <out_prefix>.json and .npz. Idempotent."""
    out_prefix = Path(out_prefix)
    if out_prefix.with_suffix(".json").exists():
        return json.load(open(out_prefix.with_suffix(".json")))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    task = DS[ds]["task"]; tt = DS[ds].get("tt")
    ncls = int(len(np.unique(ytr_raw))) if task == "multiclass" else 2
    ytr = _tt(ytr_raw, tt) if task == "regression" else ytr_raw
    sw = np.ones(len(ytr), np.float32)
    if task != "regression":
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", ytr)
    t0 = time.time()
    Xtr = np.asarray(Xtr, np.float32); Xte = np.asarray(Xte, np.float32)
    bp, cv_best = tune(clf, Xtr, ytr, sw, task, ncls, seed, trials, nj)
    m = make_model(clf, task, ncls, bp, seed, nj)
    m.fit(Xtr, ytr, sample_weight=sw)
    if task == "regression":
        pred = _itt(m.predict(Xte), tt); proba = None
    else:
        pred = m.predict(Xte); proba = m.predict_proba(Xte); proba = proba if task == "multiclass" else proba[:, 1]
    res = dict(dataset=ds, clf=clf, seed=seed, n_train=int(len(ytr)), n_test=int(len(yte_raw)), n_feat=int(Xtr.shape[1]),
               best_params=bp, cv_best=round(cv_best, 4), seconds=round(time.time() - t0, 1), **(meta or {}), **metrics(task, yte_raw, pred, proba))
    np.savez_compressed(out_prefix.with_suffix(".npz"), y=yte_raw, pred=pred, proba=proba if proba is not None else np.zeros(1))
    json.dump(res, open(out_prefix.with_suffix(".json"), "w"), indent=1)
    return res


def pool_run(fn, jobs, workers):
    """run fn(job) over jobs in a process pool (spawn); returns results list."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        return list(p.imap_unordered(fn, jobs, chunksize=1))
