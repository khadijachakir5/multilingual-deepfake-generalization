
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import get_logger, cprint, _seed, _feat_cols
from src.features import (FAMILY_DIMS, FAMILY_PREFIXES, EXCLUDED_FEATS,
                           families_from_columns, load_and_extract)
from src.sq_pipeline import sq1, sq2, sq3_lolo, sq3_logo, compute_delta, sq4

logger = get_logger("pipeline_mlaad")

MLAAD_ROOT = "/content/drive/MyDrive/DOCTORAT/Mlaad/mlaad_v5/fake"
MAILABS_ROOT = "/content/drive/MyDrive/DOCTORAT/mailbs"
OUTPUT_DIR = "/content/drive/MyDrive/results_paper1_v5"
SKIP_EXTRACT = False
MAX_REAL_PER_LANG = 2000
MIN_GEN_FILES = 50
SAVE_INTERVAL = 100

MLAAD_LANGUAGES = ["de", "en", "es", "fr", "it", "pl", "ru", "uk"]
MAILABS_MAP = {
    "de_DE": "de", "en_US": "en", "en_UK": "en",
    "es_ES": "es", "fr_FR": "fr", "it_IT": "it",
    "pl_PL": "pl", "ru_RU": "ru", "uk_UK": "uk",
}

META_COLS_SET = {"filepath", "language", "label", "generator",
                  "model_name", "duration"}


def scan_datasets():
    items = []

    mailabs = Path(MAILABS_ROOT)
    logger.info("Scan M-AILABS...")
    real_by_lang = defaultdict(list)
    if mailabs.exists():
        for lang_dir in sorted(mailabs.iterdir()):
            if not lang_dir.is_dir():
                continue
            code = MAILABS_MAP.get(lang_dir.name)
            if not code:
                continue
            for wf in lang_dir.rglob("*.wav"):
                real_by_lang[code].append(str(wf))
        rng = np.random.default_rng(42)
        for lang, fps in real_by_lang.items():
            sel = (rng.choice(fps, MAX_REAL_PER_LANG, replace=False).tolist()
                   if len(fps) > MAX_REAL_PER_LANG else fps)
            for fp in sel:
                items.append({"filepath": fp, "language": lang,
                               "label": 0, "generator": "real",
                               "model_name": "real"})
        n_real = sum(1 for it in items if it["label"] == 0)
        logger.info(f"  M-AILABS : {n_real:,} real (max {MAX_REAL_PER_LANG}/langue)")
    else:
        logger.warning(f"  M-AILABS non trouve : {MAILABS_ROOT}")

    mlaad = Path(MLAAD_ROOT)
    logger.info("Scan MLAAD v5...")
    n_fake = 0
    n_skip = 0
    collision_check = {}

    for lang in MLAAD_LANGUAGES:
        lang_dir = mlaad / lang
        if not lang_dir.exists():
            logger.warning(f"  Langue absente : {lang}")
            continue
        for gen_dir in sorted(lang_dir.iterdir()):
            if not gen_dir.is_dir():
                continue
            gen_id = f"{lang}/{gen_dir.name}"
            meta_path = gen_dir / "meta.csv"
            if meta_path.exists():
                try:
                    meta = pd.read_csv(meta_path, sep="|", on_bad_lines="skip")
                    if "path" not in meta.columns:
                        continue
                    if len(meta) < MIN_GEN_FILES:
                        n_skip += len(meta)
                        continue
                    gen_col = next((c for c in
                                     ["model_name", "model", "generator"]
                                     if c in meta.columns), None)
                    model_name = (str(meta[gen_col].iloc[0]).strip()
                                   if gen_col else gen_dir.name)
                    key = (lang, model_name)
                    if key in collision_check:
                        logger.debug(
                            f"  Collision model_name '{model_name}' ({lang}) : "
                            f"{collision_check[key]} vs {gen_dir.name}")
                    collision_check[key] = gen_dir.name
                    for _, row in meta.iterrows():
                        p = Path(str(row["path"]))
                        fp = str(p) if p.is_absolute() else str(gen_dir / p)
                        if not os.path.exists(fp):
                            fp = str(gen_dir / p.name)
                        items.append({
                            "filepath": fp,
                            "language": lang,
                            "label": 1,
                            "generator": gen_id,
                            "model_name": model_name,
                        })
                        n_fake += 1
                except Exception as e:
                    logger.debug(f"  meta.csv {meta_path}: {e}")
            else:
                wavs = (list(gen_dir.glob("*.wav")) +
                        list(gen_dir.glob("*.flac")))
                if len(wavs) < MIN_GEN_FILES:
                    n_skip += len(wavs)
                    continue
                for wf in wavs:
                    items.append({
                        "filepath": str(wf),
                        "language": lang,
                        "label": 1,
                        "generator": gen_id,
                        "model_name": gen_dir.name,
                    })
                    n_fake += 1

    n_real = sum(1 for it in items if it["label"] == 0)
    n_gens = len(set(it["generator"] for it in items if it["label"] == 1))
    logger.info(f"  MLAAD v5 : {n_fake:,} fake | {n_gens} gen_id uniques | "
                f"{n_skip} ignores (<{MIN_GEN_FILES})")
    logger.info(f"  Total    : {len(items):,} (real={n_real:,}, fake={n_fake:,})")

    fake_items = [it for it in items if it["label"] == 1]
    by_lang = defaultdict(set)
    for it in fake_items:
        by_lang[it["language"]].add(it["generator"])
    logger.info("  Fake par langue :")
    for lang in MLAAD_LANGUAGES:
        n = sum(1 for it in fake_items if it["language"] == lang)
        logger.info(f"    {lang} : {len(by_lang[lang]):>3} generateurs  {n:>7,} utt")
    return items


def extract_features(audio_items, output_dir):
    feat_file = Path(output_dir) / "features_mlaad.pkl"
    ckpt_file = Path(output_dir) / "checkpoints" / "extraction_full.pkl"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    if feat_file.exists():
        df = pd.read_pickle(str(feat_file))
        logger.info(f"Features chargees : {feat_file} ({len(df):,} rows)")
        return df

    processed = {}
    if ckpt_file.exists():
        with open(ckpt_file, "rb") as f:
            processed = pickle.load(f)
        logger.info(f"  Reprise checkpoint : {len(processed):,} traites")

    meta_map = {it["filepath"]: it for it in audio_items}
    fps = [it["filepath"] for it in audio_items]
    errors = []

    logger.info(f"  Restants : {len(fps) - len(processed):,} / {len(fps):,}")

    for fp in tqdm(fps, desc="Extraction (155 dims)",
                    initial=len(processed), total=len(fps)):
        if fp in processed:
            continue
        meta = meta_map[fp]
        try:
            feats = load_and_extract(fp)
            feats["filepath"] = fp
            feats["language"] = meta["language"]
            feats["label"] = meta["label"]
            feats["generator"] = meta["generator"]
            feats["model_name"] = meta.get("model_name", "")
            processed[fp] = feats
        except Exception as e:
            errors.append({"file": fp, "error": str(e)})

        if len(processed) % SAVE_INTERVAL == 0:
            with open(ckpt_file, "wb") as f:
                pickle.dump(processed, f)

    with open(ckpt_file, "wb") as f:
        pickle.dump(processed, f)

    if errors:
        pd.DataFrame(errors).to_csv(Path(output_dir) / "extraction_errors.csv", index=False)
        logger.warning(f"  {len(errors)} erreurs")

    df = pd.DataFrame([v for v in processed.values()
                        if "label" in v and v["label"] in [0, 1]])
    df.to_pickle(str(feat_file))
    logger.info(f"  {len(df):,} utterances -> {feat_file}")

    all_cols = [c for c in df.columns
                 if c not in META_COLS_SET and c not in set(EXCLUDED_FEATS)
                 and df[c].dtype in (np.float32, np.float64, float)]
    total = 0
    for fam, fn in FAMILY_PREFIXES.items():
        cols = [c for c in all_cols if fn(c)]
        exp = FAMILY_DIMS.get(fam, "?")
        total += len(cols)
        logger.info(f"  {fam:8s}: d={len(cols)} (attendu={exp})")
    logger.info(f"  Total : {total}/155")
    return df


def run():
    _seed(42)
    tables_dir = Path(OUTPUT_DIR) / "tables"
    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cprint(f"MLAAD : {MLAAD_ROOT}")
    cprint(f"M-AILABS: {MAILABS_ROOT}")
    cprint(f"Output : {OUTPUT_DIR}")

    feat_file = Path(OUTPUT_DIR) / "features_mlaad.pkl"
    if SKIP_EXTRACT and feat_file.exists():
        logger.info(f"Chargement : {feat_file}")
        df = pd.read_pickle(str(feat_file))
    else:
        audio_items = scan_datasets()
        df = extract_features(audio_items, OUTPUT_DIR)

    cprint(f"Dataset : {len(df):,} fake={int((df['label']==1).sum()):,} "
           f"real={int((df['label']==0).sum()):,} "
           f"modeles={df[df['label']==1]['model_name'].nunique()}")

    all_cols = _feat_cols(df, META_COLS_SET)
    families = families_from_columns(all_cols)
    total = sum(len(c) for c in families.values())
    cprint(f"Familles : {list(families.keys())} | Dims : {total}/155")

    dsq1 = sq1(df, families, tables_dir, MLAAD_LANGUAGES,
               out_name="SQ1_representation_space.csv")
    cprint("\n-- SQ1 resume --")
    cprint(dsq1[["family", "H_mean", "cohens_d", "eta2_lang", "eta2_gen",
                  "ratio_gen_lang", "ratio_ci_lo", "ratio_ci_hi",
                  "p_interaction", "profile"]].to_string(index=False))

    dsq2 = sq2(df, families, tables_dir, MLAAD_LANGUAGES,
               out_name="SQ2_invariance.csv")
    cprint("\n-- SQ2 resume --")
    cprint(dsq2[["family", "rb", "discriminative", "language_independent",
                  "p_mw_corrected", "p_kwf_corrected", "p_kwr_corrected"]]
           .to_string(index=False))

    dlolo = sq3_lolo(df, families, tables_dir, ckpt_dir, MLAAD_LANGUAGES,
                      out_name="SQ3_lolo_mlaad.csv", ckpt_name="lolo_mlaad.pkl",
                      run_lr=True)
    dlogo = sq3_logo(df, families, tables_dir, ckpt_dir,
                      out_name="SQ3_logo_mlaad.csv", ckpt_name="logo_mlaad.pkl")

    if dlolo is not None and len(dlolo) > 0:
        compute_delta(dlolo, dlogo, tables_dir, out_name="SQ3_delta.csv")

    sq4(df, families, tables_dir, ckpt_dir, MLAAD_LANGUAGES, protocol="lolo",
        out_name="SQ4_shap_lolo.csv", ckpt_name="shap_lolo_ckpt.pkl")
    if dlogo is not None and len(dlogo) > 0:
        models = sorted(df.loc[df["label"] == 1, "model_name"].dropna().unique().tolist())
        models = [m for m in models if m != "real"]
        sq4(df, families, tables_dir, ckpt_dir, models, protocol="logo",
            out_name="SQ4_shap_logo.csv", ckpt_name="shap_logo_ckpt.pkl",
            model_name_col="model_name")

    cprint("Termine")
    for fp in sorted(tables_dir.glob("*.csv")):
        cprint(f"  {fp.name}")


if __name__ == "__main__":
    run()
