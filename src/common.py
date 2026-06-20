"""
Shared logging, seeding, and small formatting utilities.

These are byte-identical helpers that were duplicated verbatim across
pipeline_mlaad.py and pipeline_ggmddc.py. Factored out here so both
corpus-specific pipelines import the exact same implementation.
"""
import hashlib
import logging
import os
import random
import sys

import numpy as np

from .features import EXCLUDED_FEATS


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            handlers=[_FlushHandler(sys.stdout)],
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger(name)


logger = get_logger("pipeline")


def cprint(msg):
    print(msg, flush=True)


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def _stable_seed(text):
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def _fmt(x, d=3):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{d}f}"


def _feat_cols(df, meta_cols_set):
    """meta_cols_set differs slightly per corpus (e.g. 'split' vs 'model_name'),
    so it is passed in explicitly rather than hardcoded here."""
    excl = set(EXCLUDED_FEATS)
    return [c for c in df.columns
            if c not in meta_cols_set and c not in excl
            and df[c].dtype in (np.float32, np.float64, float)]
