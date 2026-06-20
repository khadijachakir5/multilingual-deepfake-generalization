from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .classifiers import SEEDS, _fold, _rf, _multiseed, _eer
from .common import _stable_seed, _fmt, cprint, get_logger
from .stats_geometry import (_cohen, _hellinger, _srh_fake, _srh_classwise,
                              _bootstrap_eta2_ratio, _interaction_test)

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = get_logger("pipeline")

MIN_SAMPLES_KW = 20
ALPHA = 0.05


def sq1(df, families, tables_dir, languages, out_name="SQ1_representation_space.csv"):
    out_file = Path(tables_dir) / out_name
    if out_file.exists():
        logger.info(f"  SQ1 deja calcule -> {out_file.name}")
        return pd.read_csv(out_file)

    logger.info("SQ1: Acoustic Space Geometry [dual-space]")
    lb = df["label"].values
    la = df["language"].values
    ge = (df["generator"].values if "generator" in df.columns else lb.copy())
    rng = np.random.default_rng(SEEDS[0])
    recs = []

    for fam, cols in families.items():
        X = df[cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.)

        if X.shape[1] == 1:
            sc_global = X[:, 0]
            pc1_var_gl = 1.0
        else:
            scaler_gl = StandardScaler().fit(X)
            pca_gl = PCA(n_components=1, random_state=SEEDS[0])
            pca_gl.fit(scaler_gl.transform(X))
            sc_global = pca_gl.transform(scaler_gl.transform(X)).ravel()
            pc1_var_gl = float(pca_gl.explained_variance_ratio_[0])

        X_fake = X[lb == 1]
        if X_fake.shape[0] < 10 or X.shape[1] == 1:
            sc_fake_proj = sc_global.copy()
            pc1_var_fk = pc1_var_gl
        else:
            scaler_fk = StandardScaler().fit(X_fake)
            pca_fk = PCA(n_components=1, random_state=SEEDS[0])
            pca_fk.fit(scaler_fk.transform(X_fake))
            sc_fake_proj = pca_fk.transform(scaler_fk.transform(X)).ravel()
            pc1_var_fk = float(pca_fk.explained_variance_ratio_[0])

        sc_fake = sc_fake_proj[lb == 1]
        la_fake = la[lb == 1]
        ge_fake = ge[lb == 1]

        hbl = {}
        for lang in languages:
            mf = (lb == 1) & (la == lang)
            mr = (lb == 0) & (la == lang)
            if mf.sum() >= 5 and mr.sum() >= 5:
                hbl[lang] = _hellinger(sc_global[mf], sc_global[mr])
        hv = np.array(list(hbl.values()))
        hm = float(np.mean(hv)) if len(hv) > 0 else np.nan
        hc = (float(np.std(hv) / (hm + 1e-10)) if hm > 1e-5 else np.nan)

        d = _cohen(sc_global[lb == 1], sc_global[lb == 0])

        sr = _srh_fake(sc_fake, la_fake, ge_fake)

        it = _interaction_test(sc_fake, la_fake, ge_fake)

        ci_lo, ci_hi = _bootstrap_eta2_ratio(sc_fake, la_fake, ge_fake, rng)

        cw = _srh_classwise(sc_global, la, lb)

        row = {
            "family": fam, "n_features": len(cols),
            "projection_H_d": "PC1_global", "projection_eta2": "PC1_fake",
            "pc1_var_global": round(pc1_var_gl, 4),
            "pc1_var_fake": round(pc1_var_fk, 4),
            "H_mean": round(hm, 4),
            "H_cv": round(hc, 4) if not np.isnan(hc) else np.nan,
            "cohens_d": round(d, 4),
            "eta2_lang": round(sr["eta2_lang"], 4),
            "eta2_gen": round(sr["eta2_gen"], 4),
            "ratio_gen_lang": (round(sr["ratio"], 2) if np.isfinite(sr["ratio"]) else np.nan),
            "ratio_ci_lo": (round(ci_lo, 2) if not np.isnan(ci_lo) else np.nan),
            "ratio_ci_hi": (round(ci_hi, 2) if not np.isnan(ci_hi) else np.nan),
            "eta2_interaction": (round(it["eta2_interaction"], 4)
                                  if not np.isnan(it.get("eta2_interaction", np.nan)) else np.nan),
            "p_interaction": (round(it["p_interaction"], 4)
                               if not np.isnan(it.get("p_interaction", np.nan)) else np.nan),
            "eta2_fake": round(cw.get("eta2_fake", np.nan), 4),
            "eta2_real": round(cw.get("eta2_real", np.nan), 4),
            "ratio_fr": (round(cw.get("ratio_fr", np.nan), 2)
                         if not np.isnan(cw.get("ratio_fr", np.nan)) else np.nan),
            "profile": cw.get("profile", "N/A"),
            "cross_corpus_note": (
                "PC1_fake estimated independently per corpus; cross-corpus "
                "eta2 comparisons reflect intrinsic local geometry, not a "
                "shared latent axis."),
        }
        row.update({f"H_{l}": round(v, 4) for l, v in hbl.items()})
        recs.append(row)
        cprint(f"  {fam:8s}: H={_fmt(hm)} [global] d={_fmt(d,2)} "
               f"eta2_gen={_fmt(sr['eta2_gen'])} eta2_lang={_fmt(sr['eta2_lang'])} "
               f"ratio={_fmt(sr['ratio'], 1)}x "
               f"CI=[{_fmt(ci_lo, 1)},{_fmt(ci_hi, 1)}] [fake] "
               f"p_int={_fmt(it.get('p_interaction', np.nan))}")

    dout = pd.DataFrame(recs).sort_values("H_mean", ascending=False)
    dout.to_csv(out_file, index=False)
    logger.info(f"  SQ1 -> {out_file.name}")
    return dout


def sq2(df, families, tables_dir, languages, out_name="SQ2_invariance.csv"):
    out_file = Path(tables_dir) / out_name
    if out_file.exists():
        logger.info(f"  SQ2 deja calcule -> {out_file.name}")
        return pd.read_csv(out_file)

    logger.info("SQ2: Discriminative Invariance [PC1_global]")
    lb = df["label"].values
    la = df["language"].values
    recs = []

    for fam, cols in families.items():
        X = df[cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.)
        if X.shape[1] == 1:
            sc = X[:, 0]
            pc1_var = 1.0
        else:
            scaler = StandardScaler().fit(X)
            pca = PCA(n_components=1, random_state=SEEDS[0])
            pca.fit(scaler.transform(X))
            sc = pca.transform(scaler.transform(X)).ravel()
            pc1_var = float(pca.explained_variance_ratio_[0])

        a = sc[lb == 1][np.isfinite(sc[lb == 1])]
        b = sc[lb == 0][np.isfinite(sc[lb == 0])]
        if len(a) >= 2 and len(b) >= 2:
            stat, pmw = mannwhitneyu(a, b, alternative="two-sided")
            rb = abs(1 - 2 * stat / (len(a) * len(b)))
        else:
            pmw, rb = np.nan, np.nan

        def kw(cv):
            grps = {l: sc[(lb == cv) & (la == l)] for l in languages}
            grps = {l: v[np.isfinite(v)] for l, v in grps.items()
                    if v[np.isfinite(v)].shape[0] >= MIN_SAMPLES_KW}
            if len(grps) < 2:
                return np.nan, 0
            try:
                _, p = kruskal(*grps.values())
                return float(p), len(grps)
            except ValueError:
                return np.nan, 0

        pkf, nkf = kw(1)
        pkr, nkr = kw(0)
        recs.append({"family": fam, "projection": "PC1_global",
                      "pc1_variance_ratio": round(pc1_var, 4),
                      "rb": round(rb, 4) if not np.isnan(rb) else np.nan,
                      "p_mw_raw": pmw, "p_kwf_raw": pkf, "p_kwr_raw": pkr,
                      "n_groups_kwf": nkf, "n_groups_kwr": nkr})
        cprint(f"  {fam:8s}: r_b={_fmt(rb)}  p_MW={_fmt(pmw)}  p_KWf={_fmt(pkf)}")

    dout = pd.DataFrame(recs)
    for raw, corr in [("p_mw_raw", "p_mw_corrected"),
                       ("p_kwf_raw", "p_kwf_corrected"),
                       ("p_kwr_raw", "p_kwr_corrected")]:
        _, c, _, _ = multipletests(dout[raw].fillna(1.).tolist(), alpha=ALPHA, method="fdr_bh")
        dout[corr] = c

    dout["discriminative"] = dout["p_mw_corrected"] < ALPHA
    dout["stable_fake"] = (dout["p_kwf_corrected"] >= ALPHA) & (dout["n_groups_kwf"] >= 2)
    dout["stable_real"] = (dout["p_kwr_corrected"] >= ALPHA) & (dout["n_groups_kwr"] >= 2)
    dout["language_independent"] = (
        dout["discriminative"] & dout["stable_fake"] & dout["stable_real"]
    )

    logger.info(f"  Discriminatif  : {dout['discriminative'].sum()}/{len(dout)}")
    logger.info(f"  Stable fake    : {dout['stable_fake'].sum()}/{len(dout)}")
    logger.info(f"  Stable real    : {dout['stable_real'].sum()}/{len(dout)}")
    logger.info(f"  Lang-invariant : {dout['language_independent'].sum()}/{len(dout)} "
                f"(discriminatif and stable_fake and stable_real)")
    dout.to_csv(out_file, index=False)
    logger.info(f"  SQ2 -> {out_file.name}")
    return dout


def sq3_lolo(df, families, tables_dir, ckpt_dir, languages,
             out_name="SQ3_lolo.csv", ckpt_name="lolo.pkl", run_lr=True):
    out_file = Path(tables_dir) / out_name
    cp = Path(ckpt_dir) / ckpt_name

    records = []
    done = set()
    if cp.exists():
        ex = pd.read_pickle(cp)
        records = ex.to_dict("records")
        done = {(r["held_out"], r["family"]) for r in records}

    all_keys = {(l, f) for l in languages for f in families}
    todo_keys = all_keys - done
    if not todo_keys:
        logger.info(f"  SQ3-LOLO deja complet -> {out_file.name}")
        dout = pd.DataFrame(records)
        dout.to_csv(out_file, index=False)
        return dout

    logger.info("SQ3-LOLO")
    splits = {}
    for lang in languages:
        if all((lang, f) in done for f in families):
            continue
        dtr = df[df["language"] != lang].copy()
        dte = df[df["language"] == lang].copy()
        if len(dte) < 10 or len(np.unique(dte["label"].values)) < 2:
            logger.warning(f"  SQ3-LOLO skip {lang} : donnees insuffisantes")
            continue
        splits[lang] = (dtr, dte)

    for lang, (dtr, dte) in splits.items():
        fs = _stable_seed(lang)
        for fam, cols in families.items():
            if (lang, fam) in done:
                continue
            Xtr, ytr, Xte, yte = _fold(dtr, dte, cols, fs)
            rf = _multiseed(Xtr, ytr, Xte, yte)
            row = {
                "protocol": "LOLO", "held_out": lang, "family": fam,
                "auc_rf": rf["auc"], "ci_lo_rf": rf["ci_lo"], "ci_hi_rf": rf["ci_hi"],
                "eer_rf": rf["eer"], "balanced_acc_rf": rf["balanced_acc"], "f1_rf": rf["f1"],
                "n_test": len(yte),
            }
            if run_lr:
                lr = _multiseed(Xtr, ytr, Xte, yte, lr_only=True)
                row.update({
                    "auc_lr": lr["auc"], "eer_lr": lr["eer"],
                    "balanced_acc_lr": lr["balanced_acc"], "f1_lr": lr["f1"],
                })
            records.append(row)
            done.add((lang, fam))
            cprint(f"  LOLO {lang} {fam}: AUC={rf['auc']:.4f} "
                   f"[{rf['ci_lo']:.3f},{rf['ci_hi']:.3f}] EER={rf['eer']:.3f}")
            tmp = pd.DataFrame(records)
            tmp.to_pickle(cp)
            tmp.to_csv(out_file, index=False)

    dout = pd.DataFrame(records)
    dout.to_pickle(cp)
    dout.to_csv(out_file, index=False)
    logger.info(f"  SQ3-LOLO -> {out_file.name}")
    return dout


def sq3_logo(df, families, tables_dir, ckpt_dir,
             out_name="SQ3_logo.csv", ckpt_name="logo.pkl"):
    if "generator" not in df.columns:
        return None
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = df["generator"].where(
            df["label"] == 0, df["generator"].str.split("/").str[-1])

    models = sorted(df.loc[df["label"] == 1, "model_name"].dropna().unique().tolist())
    models = [m for m in models if m != "real"]
    if len(models) < 2:
        return None

    out_file = Path(tables_dir) / out_name
    cp = Path(ckpt_dir) / ckpt_name

    records = []
    done = set()
    if cp.exists():
        ex = pd.read_pickle(cp)
        records = ex.to_dict("records")
        done = {(r["held_out"], r["family"]) for r in records}

    all_keys = {(m, f) for m in models for f in families}
    todo_keys = all_keys - done
    if not todo_keys:
        logger.info(f"  SQ3-LOGO deja complet -> {out_file.name}")
        dout = pd.DataFrame(records)
        dout.to_csv(out_file, index=False)
        return dout

    logger.info(f"SQ3-LOGO ({len(models)} modeles)")
    dreal = df[df["label"] == 0].copy()
    models_todo = sorted({m for (m, _) in todo_keys})

    for model in tqdm(models_todo, desc="LOGO"):
        dtef = df[(df["label"] == 1) & (df["model_name"] == model)].copy()
        dtrf = df[(df["label"] == 1) & (df["model_name"] != model)].copy()
        if len(dtef) < 10:
            continue
        gs = _stable_seed(model)
        nte = len(dtef)
        rte = (dreal.sample(n=min(nte, len(dreal)), random_state=gs)
               if len(dreal) >= nte else dreal)
        rtr = dreal.drop(rte.index) if len(dreal) > nte else dreal
        dtr = pd.concat([dtrf, rtr]).reset_index(drop=True)
        dte = pd.concat([dtef, rte]).reset_index(drop=True)
        if len(np.unique(dte["label"].values)) < 2:
            continue

        for fam, cols in families.items():
            if (model, fam) in done:
                continue
            Xtr, ytr, Xte, yte = _fold(dtr, dte, cols, gs)
            rf = _multiseed(Xtr, ytr, Xte, yte)
            records.append({
                "protocol": "LOGO", "held_out": model, "family": fam,
                "auc_rf": rf["auc"], "ci_lo_rf": rf["ci_lo"], "ci_hi_rf": rf["ci_hi"],
                "eer_rf": rf["eer"], "balanced_acc_rf": rf["balanced_acc"], "f1_rf": rf["f1"],
                "n_test": len(yte),
            })
            done.add((model, fam))
            tmp = pd.DataFrame(records)
            tmp.to_pickle(cp)
            tmp.to_csv(out_file, index=False)

    dout = pd.DataFrame(records)
    dout.to_pickle(cp)
    dout.to_csv(out_file, index=False)
    logger.info(f"  SQ3-LOGO -> {out_file.name}")
    return dout


def compute_delta(dlolo, dlogo, tables_dir, out_name="SQ3_delta.csv"):
    tables_dir = Path(tables_dir)
    lm = (dlolo[dlolo["protocol"] == "LOLO"]
          .groupby("family")
          .agg(auc_lolo=("auc_rf", "mean"),
               ci_lo_lolo=("ci_lo_rf", "mean"),
               ci_hi_lolo=("ci_hi_rf", "mean"))
          .reset_index())
    if dlogo is None or len(dlogo) == 0:
        d = lm.copy()
        d["auc_logo"] = np.nan
        d["delta"] = np.nan
        d["wilcoxon_p"] = np.nan
        d.to_csv(tables_dir / out_name, index=False)
        return d
    lom = (dlogo.groupby("family")
           .agg(auc_logo=("auc_rf", "mean"),
                ci_lo_logo=("ci_lo_rf", "mean"),
                ci_hi_logo=("ci_hi_rf", "mean"))
           .reset_index())
    d = lm.merge(lom, on="family", how="outer")
    d["delta"] = (d["auc_logo"] - d["auc_lolo"]).round(4)
    wp = {}
    for fam in d["family"]:
        la = dlolo[dlolo["family"] == fam]["auc_rf"].dropna().values
        lo = dlogo[dlogo["family"] == fam]["auc_rf"].dropna().values
        n = min(len(la), len(lo))
        if n >= 5:
            try:
                _, p = wilcoxon(la[:n], lo[:n], alternative="two-sided")
                wp[fam] = float(p)
            except Exception:
                wp[fam] = np.nan
        else:
            wp[fam] = np.nan
    d["wilcoxon_p"] = d["family"].map(wp)
    d = d.sort_values("delta")
    d.to_csv(tables_dir / out_name, index=False)
    logger.info(f"  Delta -> {out_name}")
    return d


def sq4(df, families, tables_dir, ckpt_dir, held_out_values, protocol="lolo",
        out_name=None, ckpt_name=None, model_name_col="model_name"):
    if not SHAP_AVAILABLE:
        logger.warning("shap non disponible - SQ4 ignore")
        return None

    out_name = out_name or f"SQ4_shap_{protocol}.csv"
    ckpt_name = ckpt_name or f"shap_{protocol}_ckpt.pkl"
    out_file = Path(tables_dir) / out_name
    cp = Path(ckpt_dir) / ckpt_name

    all_cols = [c for cs in families.values() for c in cs]

    records = []
    done = set()
    if cp.exists():
        ex = pd.read_pickle(cp)
        records = ex.to_dict("records")
        done = set(ex["held_out"].unique())

    todo = [h for h in sorted(held_out_values) if h not in done]
    if not todo:
        logger.info(f"  SQ4-{protocol} deja complet -> {out_file.name}")
        dout = pd.DataFrame(records)
        if len(dout) > 0:
            dout.to_csv(out_file, index=False)
        return dout if len(dout) > 0 else None

    logger.info(f"SQ4: SHAP ({protocol.upper()})")

    for held in tqdm(todo, desc=f"SHAP {protocol}"):
        if protocol == "lolo":
            dtr = df[df["language"] != held].copy()
            dte = df[df["language"] == held].copy()
            fs = _stable_seed(held)
        else:
            dp = df[df["label"] == 0].copy()
            dtef = df[(df["label"] == 1) & (df[model_name_col] == held)].copy()
            dtrf = df[(df["label"] == 1) & (df[model_name_col] != held)].copy()
            if len(dtef) < 5:
                continue
            fs = _stable_seed(held)
            nte = len(dtef)
            rte = (dp.sample(n=min(nte, len(dp)), random_state=fs)
                   if len(dp) >= nte else dp)
            dte = pd.concat([dtef, rte]).reset_index(drop=True)
            dtr = pd.concat([dtrf, dp.drop(rte.index)]).reset_index(drop=True)
        if len(dte) < 10 or len(np.unique(dte["label"].values)) < 2:
            continue

        Xtr, ytr, Xte, yte = _fold(dtr, dte, all_cols, fs)
        clf = _rf(SEEDS[0])
        clf.fit(Xtr, ytr)
        yp = clf.predict_proba(Xte)[:, 1]
        ev, thr = _eer(yte, yp)
        yb = (yp >= thr).astype(int)
        oc = np.where((yte == 1) & (yb == 1), "TP",
             np.where((yte == 0) & (yb == 0), "TN",
             np.where((yte == 1) & (yb == 0), "FN", "FP")))

        try:
            expl = shap_lib.TreeExplainer(clf)
            rng = np.random.default_rng(SEEDS[0])
            ns = min(300, len(Xte))
            ids = rng.choice(len(Xte), ns, replace=False)
            sv = expl.shap_values(Xte[ids])
            if isinstance(sv, list):
                sv = np.abs(sv[1])
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                sv = np.abs(sv[:, :, 1])
            else:
                sv = np.abs(sv)
        except Exception as e:
            logger.warning(f"SHAP {held}: {e}")
            continue

        ocs = oc[ids]
        n_added = 0
        for st in ["TP", "TN", "FP", "FN"]:
            mk = ocs == st
            if mk.sum() == 0:
                continue
            msv = sv[mk].mean(axis=0)
            row = {"held_out": held, "protocol": protocol, "strata": st,
                   "n": int(mk.sum()),
                   "auc": float(roc_auc_score(yte, yp)),
                   "eer": float(ev)}
            off = 0
            for fam, cols in families.items():
                row[f"phi_{fam}"] = float(msv[off:off + len(cols)].mean())
                off += len(cols)
            records.append(row)
            n_added += 1

        done.add(held)
        cprint(f"  SHAP {protocol} {held}: AUC={roc_auc_score(yte, yp):.4f} "
               f"EER={ev:.3f} strates={n_added}")
        tmp = pd.DataFrame(records)
        tmp.to_pickle(cp)
        tmp.to_csv(out_file, index=False)

    if not records:
        logger.warning(f"  SQ4-{protocol} : aucun resultat")
        return None

    dout = pd.DataFrame(records)
    dout.to_pickle(cp)
    dout.to_csv(out_file, index=False)

    phi = [c for c in dout.columns if c.startswith("phi_")]
    drecs = []
    for ho in dout["held_out"].unique():
        sub = dout[dout["held_out"] == ho]
        for err, ctrl in [("FP", "TN"), ("FN", "TP")]:
            er = sub[sub["strata"] == err]
            cr = sub[sub["strata"] == ctrl]
            if len(er) == 0 or len(cr) == 0:
                continue
            rec = {"held_out": ho, "error_type": err,
                   "n_error": int(er["n"].values[0]),
                   "n_correct": int(cr["n"].values[0])}
            for c in phi:
                rec[f"delta_{c.replace('phi_', '')}"] = float(er[c].values[0]) - float(cr[c].values[0])
            drecs.append(rec)
    if drecs:
        dd = pd.DataFrame(drecs)
        dcols = [c for c in dd.columns if c.startswith("delta_")]
        dd.to_csv(Path(tables_dir) / f"SQ4_delta_shap_{protocol}.csv", index=False)
        logger.info(dd.groupby("error_type")[dcols].mean().round(4).to_string())

    logger.info(f"  SQ4-{protocol} -> {out_file.name}")
    return dout
