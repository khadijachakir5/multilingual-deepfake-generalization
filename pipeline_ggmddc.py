"""
GGMDDC pipeline -- SQ1 through SQ4.
"""
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import get_logger, cprint, _seed, _feat_cols
from src.features import (FAMILY_DIMS, FAMILY_PREFIXES, EXCLUDED_FEATS,
                           families_from_columns, load_and_extract)
from src.sq_pipeline import sq1, sq2, sq3_lolo, sq4

logger = get_logger("pipeline_ggmddc")

GGMDDC_ROOT = "/content/gdrive/MyDrive/GGMDDC_Dataset/GGMDDC"
OUTPUT_DIR = "/content/gdrive/MyDrive/results_ggmddc_v5"
SKIP_EXTRACT = False
MAX_PER_LANG = 2000
MIN_FILES = 10
SAVE_INTERVAL = 100

GGMDDC_LANGUAGES = [
    "arabic", "english", "french", "hindi", "mandarin",
    "portuguese", "russian", "sanskrit", "spanish", "vietnamese",
]
GGMDDC_SPLITS = ["Training", "Testing", "Validation"]

META_COLS_SET = {"filepath", "language", "label", "generator", "split", "duration"}

# canonical mapping for orthographic variants found in GGMDDC filenames
_LANG_PREFIX_MAP = {
    "sanskarit": "sanskrit",
    "vitnamese": "vietnamese",
    "portugese": "portuguese",
    "mandarin_chinese": "mandarin",
}


def _infer_language(filepath):
    fp = Path(filepath)
    for part in fp.parts[::-1][1:]:
        token = part.lower()
        token = _LANG_PREFIX_MAP.get(token, token)
        if token in GGMDDC_LANGUAGES:
            return token
    stem = fp.stem.lower()
    for lang in GGMDDC_LANGUAGES:
        if stem.startswith(lang):
            return lang
    token = stem.split("_")[0]
    token = _LANG_PREFIX_MAP.get(token, token)
    if token in GGMDDC_LANGUAGES:
        return token
    return "unknown"


def scan_ggmddc():
    root = Path(GGMDDC_ROOT)
    items = []
    logger.info(f"Scan GGMDDC : {root}")

    for split in GGMDDC_SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            logger.warning(f"  Split absent : {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cname = class_dir.name.lower()
            if "real" in cname:
                label = 0
                gen = "real"
            elif "deepfake" in cname or "fake" in cname:
                label = 1
                gen = "hifigan"
            else:
                continue

            subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
            if subdirs:
                for lang_dir in sorted(subdirs):
                    lang = lang_dir.name.lower()
                    lang = _LANG_PREFIX_MAP.get(lang, lang)
                    if lang not in GGMDDC_LANGUAGES:
                        lang = _infer_language(str(lang_dir))
                    for wf in list(lang_dir.rglob("*.wav")) + list(lang_dir.rglob("*.flac")):
                        items.append({"filepath": str(wf), "language": lang,
                                       "label": label, "generator": gen, "split": split})
            else:
                for wf in list(class_dir.glob("*.wav")) + list(class_dir.glob("*.flac")):
                    lang = _infer_language(str(wf))
                    items.append({"filepath": str(wf), "language": lang,
                                   "label": label, "generator": gen, "split": split})

    n_real = sum(1 for it in items if it["label"] == 0)
    n_fake = sum(1 for it in items if it["label"] == 1)
    langs = sorted(set(it["language"] for it in items))
    logger.info(f"  Total : {len(items):,} (real={n_real:,}, fake={n_fake:,})")
    logger.info(f"  Langues detectees : {langs}")

    unknown = [it for it in items if it["language"] == "unknown"]
    if unknown:
        logger.warning(f"  {len(unknown)} fichiers sans langue, ex: {unknown[0]['filepath']}")

    df_tmp = pd.DataFrame(items)
    if len(df_tmp) > 0:
        tab = df_tmp.groupby(["language", "split", "label"]).size().unstack(fill_value=0)
        logger.info("\n" + tab.to_string())

    return items


def _subsample_items(items, max_per_lang=MAX_PER_LANG, min_files=MIN_FILES, seed=42):
    if not items:
        return items
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[(it["split"], it["language"], it["label"])].append(it)

    selected = []
    for key, vals in groups.items():
        if len(vals) < min_files:
            logger.warning(f"  Groupe ignore (<{min_files} fichiers) : {key} -> {len(vals)}")
            continue
        vals = vals.copy()
        rng.shuffle(vals)
        selected.extend(vals[:max_per_lang] if max_per_lang else vals)

    logger.info(f"  Apres sous-echantillonnage : {len(selected):,}/{len(items):,} fichiers")
    return selected


def extract_features(audio_items, output_dir):
    feat_file = Path(output_dir) / "features_ggmddc.pkl"
    ckpt_file = Path(output_dir) / "checkpoints" / "extraction_ggmddc.pkl"
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

    logger.info(f"  Restants : {len(fps) - len(processed):,}/{len(fps):,}")

    for fp in tqdm(fps, desc="Extraction GGMDDC (155 dims)",
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
            feats["split"] = meta["split"]
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

    tab = df.groupby(["language", "label"]).size().unstack(fill_value=0)
    tab.columns = ["real", "fake"]
    logger.info("\n" + tab.to_string())
    return df


def run():
    _seed(42)
    tables_dir = Path(OUTPUT_DIR) / "tables"
    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cprint(f"GGMDDC : {GGMDDC_ROOT}")
    cprint(f"Output : {OUTPUT_DIR}")

    feat_file = Path(OUTPUT_DIR) / "features_ggmddc.pkl"
    if SKIP_EXTRACT and feat_file.exists():
        logger.info(f"Chargement : {feat_file}")
        df = pd.read_pickle(str(feat_file))
    else:
        audio_items = scan_ggmddc()
        if not audio_items:
            logger.error("Aucun fichier trouve - verifier GGMDDC_ROOT")
            return
        audio_items = _subsample_items(audio_items)
        if not audio_items:
            logger.error("Aucun fichier retenu apres filtrage")
            return
        df = extract_features(audio_items, OUTPUT_DIR)

    lang_counts = df.groupby("language")["label"].nunique()
    valid_langs = sorted(lang_counts[lang_counts == 2].index.tolist())
    valid_langs = [l for l in valid_langs if l != "unknown"]

    cprint(f"Dataset : {len(df):,} fake={int((df['label']==1).sum()):,} "
           f"real={int((df['label']==0).sum()):,}")
    cprint(f"Langues valides ({len(valid_langs)}) : {valid_langs}")

    df_valid = df[df["language"].isin(valid_langs)].copy()
    all_cols = _feat_cols(df_valid, META_COLS_SET)
    families = families_from_columns(all_cols)
    total = sum(len(c) for c in families.values())
    cprint(f"Familles : {list(families.keys())} | Dims : {total}/155")

    dsq1 = sq1(df_valid, families, tables_dir, valid_langs,
               out_name="SQ1_representation_space_ggmddc.csv")
    cprint("\n-- SQ1 resume --")
    cprint(dsq1[["family", "H_mean", "cohens_d", "eta2_lang", "eta2_gen",
                  "ratio_gen_lang", "ratio_ci_lo", "ratio_ci_hi",
                  "p_interaction", "profile"]].to_string(index=False))

    dsq2 = sq2(df_valid, families, tables_dir, valid_langs,
               out_name="SQ2_invariance_ggmddc.csv")
    cprint("\n-- SQ2 resume --")
    cprint(dsq2[["family", "rb", "discriminative", "language_independent",
                  "p_mw_corrected", "p_kwf_corrected", "p_kwr_corrected"]]
           .to_string(index=False))

    # GGMDDC has a single generator (HiFi-GAN) -> LOLO only, no LOGO.
    sq3_lolo(df_valid, families, tables_dir, ckpt_dir, valid_langs,
             out_name="SQ3_lolo_ggmddc.csv", ckpt_name="lolo_ggmddc.pkl",
             run_lr=True)
    sq4(df_valid, families, tables_dir, ckpt_dir, valid_langs, protocol="lolo",
        out_name="SQ4_shap_lolo_ggmddc.csv", ckpt_name="shap_lolo_ggmddc_ckpt.pkl")

    cprint("Termine")
    for fp in sorted(tables_dir.glob("*ggmddc*.csv")):
        cprint(f"  {fp.name}")


if __name__ == "__main__":
    run()
