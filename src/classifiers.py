
import warnings

import numpy as np
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                              balanced_accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 42, 123, 456, 1337]

# Published / fallback RF config (manuscript Table 2). Used directly when
# grid search is disabled, and as the default/tie-break point of the grid.
RF_N_EST = 150
RF_DEPTH = 12
RF_MAX_FEATURES = "sqrt"
RF_CRITERION = "gini"

# LR isolation classifier config (manuscript Table 2: L2, C = 1.0).
LR_C = 1.0

N_BOOTSTRAP = 2_000
SHAPIRO_ALPHA = 0.05

# Grid search configuration.
RF_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [8, 12, 16],
}
LR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
}
GRID_VAL_FRACTION = 0.2

warnings.filterwarnings("ignore")


def _preprocess(X_tr, X_te, y_tr):
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    for j in range(X_tr.shape[1]):
        nan_tr = ~np.isfinite(X_tr[:, j])
        if nan_tr.any():
            for lbl in np.unique(y_tr):
                fill = np.nanmedian(X_tr[y_tr == lbl, j])
                if not np.isfinite(fill):
                    fill = 0.
                X_tr[nan_tr & (y_tr == lbl), j] = fill
        nan_te = ~np.isfinite(X_te[:, j])
        if nan_te.any():
            fill = np.nanmedian(X_tr[:, j])
            X_te[nan_te, j] = fill if np.isfinite(fill) else 0.
    for j in range(X_tr.shape[1]):
        fin = X_tr[:, j][np.isfinite(X_tr[:, j])]
        if len(fin) >= 3 and np.min(fin) >= 0:
            try:
                n_sw = min(len(fin), 5000)
                _, p_sw = shapiro(fin[:n_sw])
                if p_sw < SHAPIRO_ALPHA:
                    X_tr[:, j] = np.log1p(X_tr[:, j])
                    X_te[:, j] = np.log1p(np.clip(X_te[:, j], 0, None))
            except Exception:
                pass
    mu = np.mean(X_tr, axis=0)
    sig = np.std(X_tr, axis=0)
    sig = np.where(sig < 1e-10, 1., sig)
    return (X_tr - mu) / sig, (X_te - mu) / sig


def _fold(df_tr, df_te, cols, fold_seed=None):
    X_tr = df_tr[cols].values.astype(float)
    y_tr = df_tr["label"].values
    X_te = df_te[cols].values.astype(float)
    y_te = df_te["label"].values
    X_tr, X_te = _preprocess(X_tr, X_te, y_tr)
    seed = fold_seed if fold_seed is not None else SEEDS[0]
    rng = np.random.default_rng(seed)
    n = min(np.sum(y_tr == 0), np.sum(y_tr == 1))
    i0 = rng.choice(np.where(y_tr == 0)[0], n, replace=False)
    i1 = rng.choice(np.where(y_tr == 1)[0], n, replace=False)
    idx = np.sort(np.concatenate([i0, i1]))
    return X_tr[idx], y_tr[idx], X_te, y_te


def _rf(seed, n_estimators=RF_N_EST, max_depth=RF_DEPTH):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=RF_MAX_FEATURES, criterion=RF_CRITERION,
        class_weight="balanced", random_state=seed, n_jobs=-1)


def _lr(seed, C=LR_C):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", max_iter=1000,
                                    class_weight="balanced", solver="lbfgs",
                                    random_state=seed)),
    ])


def _eer(y_true, y_score):
    try:
        fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1. - tpr
        diff = fpr - fnr
        sc = np.where(np.diff(np.sign(diff)))[0]
        if len(sc) == 0:
            i = np.argmin(np.abs(diff))
            return float((fpr[i] + fnr[i]) / 2), float(thr[i])
        i = sc[0]
        d0, d1 = diff[i], diff[i + 1]
        if abs(d1 - d0) < 1e-12:
            return float((fpr[i] + fnr[i]) / 2), float(thr[i])
        a = d0 / (d0 - d1)
        return float(fpr[i] + a * (fpr[i + 1] - fpr[i])), float(thr[i] + a * (thr[i + 1] - thr[i]))
    except Exception:
        return np.nan, 0.5


def rf_grid_search(X_tr, y_tr, seed, param_grid=None, val_fraction=GRID_VAL_FRACTION):
    """Grid search over RF hyperparameters on a stratified validation split
    carved out of the training fold, scored by validation AUC. Matches the
    manuscript's stated protocol ("Grid search, stratified validation split").
    Falls back to the published Table 2 config (RF_N_EST/RF_DEPTH) if the
    split is infeasible (e.g. too few samples per class)."""
    param_grid = param_grid or RF_PARAM_GRID
    try:
        X_g, X_v, y_g, y_v = train_test_split(
            X_tr, y_tr, test_size=val_fraction, random_state=seed, stratify=y_tr)
    except ValueError:
        return {"n_estimators": RF_N_EST, "max_depth": RF_DEPTH}

    best_params = {"n_estimators": RF_N_EST, "max_depth": RF_DEPTH}
    best_auc = -np.inf
    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            clf = _rf(seed, n_estimators=n_est, max_depth=depth)
            clf.fit(X_g, y_g)
            yp = clf.predict_proba(X_v)[:, 1]
            try:
                auc = roc_auc_score(y_v, yp)
            except Exception:
                continue
            if auc > best_auc:
                best_auc = auc
                best_params = {"n_estimators": n_est, "max_depth": depth}
    return best_params


def lr_grid_search(X_tr, y_tr, seed, param_grid=None, val_fraction=GRID_VAL_FRACTION):
    """Grid search over LR's C on a stratified validation split, scored by
    validation AUC. Falls back to the published C=1.0 if the split fails."""
    param_grid = param_grid or LR_PARAM_GRID
    try:
        X_g, X_v, y_g, y_v = train_test_split(
            X_tr, y_tr, test_size=val_fraction, random_state=seed, stratify=y_tr)
    except ValueError:
        return {"C": LR_C}

    best_params = {"C": LR_C}
    best_auc = -np.inf
    for C in param_grid["C"]:
        clf = _lr(seed, C=C)
        clf.fit(X_g, y_g)
        yp = clf.predict_proba(X_v)[:, 1]
        try:
            auc = roc_auc_score(y_v, yp)
        except Exception:
            continue
        if auc > best_auc:
            best_auc = auc
            best_params = {"C": C}
    return best_params


def _multiseed(X_tr, y_tr, X_te, y_te, lr_only=False, grid_search=True):
    """Multi-seed train/eval. When grid_search=True, hyperparameters are
    selected once (on the first seed's stratified validation split, per the
    manuscript's protocol) and then reused across all seeds for stability,
    rather than re-searching per seed."""
    aucs, eers, baccs, f1s = [], [], [], []

    params = None
    if grid_search:
        if lr_only:
            params = lr_grid_search(X_tr, y_tr, SEEDS[0])
        else:
            params = rf_grid_search(X_tr, y_tr, SEEDS[0])

    for s in SEEDS:
        if lr_only:
            clf = _lr(s, **params) if params else _lr(s)
        else:
            clf = _rf(s, **params) if params else _rf(s)
        clf.fit(X_tr, y_tr)
        yp = (clf.predict_proba(X_te)[:, 1]
              if hasattr(clf, "predict_proba") else clf.decision_function(X_te))
        yb = (yp >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(y_te, yp))
            eers.append(_eer(y_te, yp)[0])
            baccs.append(balanced_accuracy_score(y_te, yb))
            f1s.append(f1_score(y_te, yb, zero_division=0))
        except Exception:
            pass
    if not aucs:
        return {"auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "eer": np.nan, "balanced_acc": np.nan, "f1": np.nan}
    arr = np.array(aucs)
    rng = np.random.default_rng(SEEDS[0])
    boot = rng.choice(arr, (N_BOOTSTRAP, len(arr)), replace=True).mean(axis=1)
    return {"auc": float(np.mean(arr)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "eer": float(np.nanmean(eers)),
            "balanced_acc": float(np.nanmean(baccs)),
            "f1": float(np.nanmean(f1s))}
