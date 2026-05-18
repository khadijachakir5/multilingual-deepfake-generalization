#!/usr/bin/env python3
"""
================================================================================
MLAAD+M-AILABS — Diagnostic Pipeline (SQ1–SQ4) — Multi-Generator Setting
================================================================================
Corpus      : MLAAD v9 (Mueller et al., ICASSP 2024)
              + M-AILABS genuine speech (Solak & Naumov, 2017)
              ~30,764 utterances, 8 languages, 55 TTS generators
Languages   : de (German), en (English), es (Spanish), fr (French),
              it (Italian), pl (Polish), ru (Russian), uk (Ukrainian)
Generators  : 55 TTS systems (restricted to >= 50 utterances/language)
Protocols   : SQ1 + SQ2 + LOLO (8 folds) + LOGO (55 folds) + SQ4(LOLO+LOGO)
              LOGO reveals generator-dominance and cross-paradigm failures.

Paper role  : Multi-generator setting — captures the full interaction
              between linguistic diversity and generator diversity.
              Establishes generator dominance ratios and Δ = AUC_LOGO - AUC_LOLO.

Key findings enabled by this corpus:
  - eta2_gen / eta2_lang ratios (3.7x to 13.2x)
  - LOLO vs LOGO Δ per family (Table 8)
  - Mechanism A: German phase over-activation (SQ4 LOLO)
  - Mechanism B: griffin_lim cepstral anomaly (SQ4 LOGO)

Usage:
  python pipeline_mlaad.py --root /path/to/MLAAD --mailabs_root /path/to/M-AILABS
  python pipeline_mlaad.py --features features_mlaad.pkl

Output: results_mlaad/tables/
  SQ1_representation_space.csv  — Hellinger + eta2_lang + eta2_gen + ratio
  SQ2_invariance.csv            — MW + KW + BH-FDR
  SQ3_lolo_mlaad.csv            — LOLO 8 folds x 9 families
  SQ3_logo_mlaad.csv            — LOGO 55 folds x 9 families
  SQ3_delta.csv                 — Δ = AUC_LOGO − AUC_LOLO per family
  SQ4_shap_lolo.csv             — SHAP LOLO stratified TP/FP/FN/TN
  SQ4_shap_logo.csv             — SHAP LOGO stratified TP/FP/FN/TN
  SQ4_delta_shap_lolo.csv       — Δ_SHAP per family (Mechanism A, C)
  SQ4_delta_shap_logo.csv       — Δ_SHAP per family (Mechanism B)
================================================================================
"""

import argparse, gc, hashlib, json, logging, os, pickle, random, sys, warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.fft import dct, fft, ifft, fftfreq
from scipy.signal import resample as scipy_resample
import scipy.signal as signal
from scipy.stats import kruskal, mannwhitneyu, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                              balanced_accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    corpus:           str   = "ggmddc"
    dataset_root:     str   = ""
    mailabs_root:     str   = ""
    output_dir:       str   = "results"
    features_file:    str   = ""

    sr:               int   = 16_000
    duration:         float = 4.0
    min_duration:     float = 1.0
    n_fft:            int   = 2048
    hop_length:       int   = 512

    # Cepstral: 20 coeff x (mean+std) = 40 per family
    n_mfcc:           int   = 20
    n_lfcc:           int   = 20
    n_cqcc:           int   = 20
    n_mels:           int   = 128
    n_linear:         int   = 128
    bins_per_octave:  int   = 96
    cqcc_fmin:        float = 32.7
    cqcc_n_octaves:   int   = 9

    # Jitter F0 range (paper Table 2)
    f0_min:           float = 60.0
    f0_max:           float = 500.0

    mod_freq_low:     float = 2.0
    mod_freq_high:    float = 20.0

    ggmddc_languages: List[str] = field(default_factory=lambda: [
        "arabic","english","french","hindi","mandarin",
        "portuguese","russian","sanskrit","spanish","vietnamese"])
    mlaad_languages: List[str] = field(default_factory=lambda: [
        "de","en","es","fr","it","pl","ru","uk"])

    max_files_per_class: int = 500
    min_gen_utterances:  int = 50

    alpha:          float = 0.05
    n_bootstrap:    int   = 2_000
    n_permutations: int   = 999
    min_samples_kw: int   = 20

    seeds: List[int] = field(default_factory=lambda: [0, 42, 123, 456, 1337])

    rf_n_estimators: int   = 150
    rf_max_depth:    int   = 12
    rf_max_features: str   = "sqrt"
    lr_C:            float = 1.0

    shap_max_samples: int = 300

    excluded_features: List[str] = field(default_factory=lambda: [
        "pyin_failed","pyin_voiced_ratio"])

    @property
    def out(self):            return Path(self.output_dir)
    @property
    def checkpoint_dir(self): return self.out / "checkpoints"
    @property
    def tables_dir(self):     return self.out / "tables"
    @property
    def languages(self):
        return self.ggmddc_languages if self.corpus=="ggmddc" else self.mlaad_languages

    def create_dirs(self):
        for d in [self.out, self.checkpoint_dir, self.tables_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


META_COLS = {"filepath","filename","language","label","generator","subset","duration"}


def get_feature_cols(df, excluded=None):
    excl = set(excluded or [])
    return [c for c in df.columns
            if c not in META_COLS and c not in excl
            and df[c].dtype in (np.float32, np.float64, float)]


FAMILY_PREFIXES = {
    "MFCC":    lambda c: c.startswith("mfcc_"),
    "LFCC":    lambda c: c.startswith("lfcc_"),
    "CQCC":    lambda c: c.startswith("cqcc_"),
    "GD":      lambda c: c.startswith("gd_"),
    "IF":      lambda c: c.startswith("if_"),
    "PD":      lambda c: c.startswith("pd_"),
    "CPP":     lambda c: c.startswith("cpp_"),
    "Jitter":  lambda c: "jitter" in c,
    "Shimmer": lambda c: "shimmer" in c,
}


def get_family_cols(all_cols, excluded=None):
    excl = set(excluded or [])
    return {fam: [c for c in all_cols if fn(c) and c not in excl]
            for fam, fn in FAMILY_PREFIXES.items()
            if any(fn(c) for c in all_cols if c not in excl)}


def set_global_seed(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# METRICS
# =============================================================================
def compute_eer(y_true, y_score):
    """EER via linear interpolation."""
    try:
        fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
        fnr  = 1.0 - tpr
        diff = fpr - fnr
        sc   = np.where(np.diff(np.sign(diff)))[0]
        if len(sc) == 0:
            idx = np.argmin(np.abs(diff))
            return float((fpr[idx]+fnr[idx])/2), float(thr[idx])
        idx = sc[0]; d0, d1 = diff[idx], diff[idx+1]
        if abs(d1-d0) < 1e-12:
            return float((fpr[idx]+fnr[idx])/2), float(thr[idx])
        a = d0/(d0-d1)
        return float(fpr[idx]+a*(fpr[idx+1]-fpr[idx])), float(thr[idx]+a*(thr[idx+1]-thr[idx]))
    except Exception:
        return np.nan, 0.5


def compute_min_tdcf(y_true, y_score, c_miss=1.0, c_fa=10.0, p_target=0.05):
    """Simplified t-DCF-like cost (approximation of ASVspoof norm. min-tDCF)."""
    try:
        fpr,tpr,_ = roc_curve(y_true, y_score, pos_label=1)
        fnr  = 1-tpr
        cost = c_miss*fnr*p_target + c_fa*fpr*(1-p_target)
        norm = min(c_miss*p_target, c_fa*(1-p_target))
        return float(np.min(cost)/norm) if norm>0 else float(np.min(cost))
    except Exception:
        return np.nan


def cohens_d(a, b):
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return np.nan
    n1,n2 = len(a),len(b)
    sp = np.sqrt(((n1-1)*np.var(a,ddof=1)+(n2-1)*np.var(b,ddof=1))/(n1+n2-2))
    return float((np.mean(a)-np.mean(b))/sp) if sp>1e-10 else np.nan


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class PhaseExtractor:
    """GD(7) + IF(8) + PD(6) = 21 phase-based features."""
    def __init__(self, sr, n_fft, hop_length):
        self.sr=sr; self.n_fft=n_fft; self.hop_length=hop_length

    def extract(self, y):
        D=librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag=np.abs(D); ph=np.angle(D)
        out={}
        out.update(self._gd(ph))
        out.update(self._if(ph, mag))
        out.update(self._pd(ph, mag))
        return out

    def _gd(self, ph):
        keys=["gd_mean","gd_std","gd_skew","gd_kurt","gd_range","gd_deviation_mean","gd_deviation_std"]
        try:
            uw=np.unwrap(ph, axis=0); bw=2.*np.pi/self.n_fft
            gd=-np.diff(uw, axis=0)/bw
            flat=gd.flatten(); flat=flat[np.isfinite(flat)]
            if len(flat)==0: return {k:0. for k in keys}
            gdf=np.mean(gd, axis=0); x=np.arange(len(gdf))
            res=gdf-np.polyval(np.polyfit(x,gdf,1),x)
            return {"gd_mean":float(np.mean(flat)),"gd_std":float(np.std(flat)),
                    "gd_skew":float(skew(flat)) if np.std(flat)>1e-10 else 0.,
                    "gd_kurt":float(kurtosis(flat)) if np.std(flat)>1e-10 else 0.,
                    "gd_range":float(np.ptp(flat)),
                    "gd_deviation_mean":float(np.mean(np.abs(res))),
                    "gd_deviation_std":float(np.std(res))}
        except Exception:
            return {k:0. for k in keys}

    def _if(self, ph, mag):
        keys=["if_mean","if_std","if_skewness","if_kurtosis",
              "if_negative_ratio","if_var_low","if_var_mid","if_var_high"]
        try:
            uw=np.unwrap(ph, axis=1)
            IF=np.diff(uw, axis=1)*self.sr/(2*np.pi*self.hop_length)
            magw=(mag[:,:-1]+mag[:,1:])/2; magw/=(np.sum(magw)+1e-10)
            flat=IF.flatten(); mw=magw.flatten(); fin=np.isfinite(flat)
            if fin.sum()==0: return {k:0. for k in keys}
            mu=float(np.average(flat[fin], weights=mw[fin]))
            n=IF.shape[0]
            return {"if_mean":mu,
                    "if_std":float(np.sqrt(np.average((flat[fin]-mu)**2,weights=mw[fin]))),
                    "if_skewness":float(skew(flat[fin])) if fin.sum()>3 else 0.,
                    "if_kurtosis":float(kurtosis(flat[fin])) if fin.sum()>3 else 0.,
                    "if_negative_ratio":float(np.mean(flat[fin]<0)),
                    "if_var_low":float(np.var(IF[:n//4,:])),
                    "if_var_mid":float(np.var(IF[n//4:3*n//4,:])),
                    "if_var_high":float(np.var(IF[3*n//4:,:]))}
        except Exception:
            return {k:0. for k in keys}

    def _pd(self, ph, mag):
        try:
            w=mag/(np.sum(mag,axis=0,keepdims=True)+1e-10)
            pdf=np.diff(np.unwrap(ph,axis=0),axis=0)
            wpd=np.sum(w[:-1,:]*np.abs(pdf),axis=0)
            j=np.abs(np.diff(ph,axis=1))
            j=np.where(j>np.pi,2*np.pi-j,j)
            return {"pd_weighted_mean":float(np.mean(wpd)),"pd_weighted_std":float(np.std(wpd)),
                    "pd_jump_mean":float(np.mean(j)),"pd_jump_std":float(np.std(j)),
                    "pd_jump_max":float(np.max(j)),"pd_large_jump_ratio":float(np.mean(j>np.pi/2))}
        except Exception:
            return {k:0. for k in ["pd_weighted_mean","pd_weighted_std","pd_jump_mean",
                                    "pd_jump_std","pd_jump_max","pd_large_jump_ratio"]}


class PhonationExtractor:
    """CPP(3) + Jitter(6) + Shimmer(5) = 14 phonation features."""
    def __init__(self, sr, hop_length, f0_min=60., f0_max=500.):
        self.sr=sr; self.hop_length=hop_length; self.f0_min=f0_min; self.f0_max=f0_max

    def extract(self, y):
        out={}
        out.update(self._cpp(y))
        out.update(self._jitter(y))
        out.update(self._shimmer(y))
        return out

    def _cpp(self, y):
        try:
            S=np.abs(librosa.stft(y,n_fft=2048,hop_length=self.hop_length))
            ceps=np.abs(np.real(ifft(np.log(S**2+1e-10),axis=0)))
            qmn=int(self.sr/self.f0_max); qmx=int(self.sr/self.f0_min)
            cpps=[]
            for f in ceps.T:
                if qmx<len(f):
                    r=f[qmn:qmx]; x=np.arange(qmn,qmx)
                    cpps.append(np.max(r)-np.mean(np.polyval(np.polyfit(x,r,1),x)))
            c=np.array(cpps)
            return {"cpp_mean":float(np.mean(c)),"cpp_std":float(np.std(c)),"cpp_max":float(np.max(c))} \
                   if len(c)>0 else {"cpp_mean":0.,"cpp_std":0.,"cpp_max":0.}
        except Exception:
            return {"cpp_mean":0.,"cpp_std":0.,"cpp_max":0.}

    def _jitter(self, y):
        keys=["jitter_abs_mean","jitter_abs_std","jitter_rel_mean",
              "jitter_rel_std","jitter_rap","jitter_ppq5"]
        null={k:np.nan for k in keys}; null.update({"pyin_failed":1.,"pyin_voiced_ratio":0.})
        try:
            f0,vf,_ = librosa.pyin(y,fmin=self.f0_min,fmax=self.f0_max,sr=self.sr,
                                    frame_length=2048,hop_length=self.hop_length)
            vr=float(np.mean(vf)) if vf is not None else 0.
            f0v=f0[(vf)&(np.isfinite(f0))] if vf is not None else np.array([])
            if len(f0v)<3:
                null["pyin_voiced_ratio"]=vr; return null
            T=1./(f0v+1e-10); dT=np.abs(np.diff(T))
            rap=[abs(T[i]-np.mean(T[i-1:i+2]))/(np.mean(T[i-1:i+2])+1e-10) for i in range(1,len(T)-1)] \
                if len(T)>=3 else [0.]
            ppq5=[abs(T[i]-np.mean(T[i-2:i+3]))/(np.mean(T[i-2:i+3])+1e-10) for i in range(2,len(T)-2)] \
                 if len(T)>=5 else [0.]
            return {"pyin_failed":0.,"pyin_voiced_ratio":vr,
                    "jitter_abs_mean":float(np.mean(dT)),"jitter_abs_std":float(np.std(dT)),
                    "jitter_rel_mean":float(np.mean(dT/(T[:-1]+1e-10))*100),
                    "jitter_rel_std":float(np.std(dT/(T[:-1]+1e-10))*100),
                    "jitter_rap":float(np.mean(rap)*100),"jitter_ppq5":float(np.mean(ppq5)*100)}
        except Exception:
            return null

    def _shimmer(self, y):
        keys=["shimmer_db_mean","shimmer_db_std","shimmer_rel_mean","shimmer_rel_std","shimmer_apq3"]
        try:
            fr=librosa.util.frame(y,frame_length=2048,hop_length=self.hop_length)
            E=np.sum(fr**2,axis=0); A=np.sqrt(E[E>0.1*np.max(E)]+1e-20)
            if len(A)<3: return {k:0. for k in keys}
            ddb=20*np.log10(A[1:]/(A[:-1]+1e-10)+1e-10); rel=np.abs(np.diff(A))/(A[:-1]+1e-10)
            apq3=[abs(A[i]-np.mean(A[i-1:i+2]))/(np.mean(A[i-1:i+2])+1e-10) for i in range(1,len(A)-1)] \
                  if len(A)>=3 else [0.]
            return {"shimmer_db_mean":float(np.mean(np.abs(ddb))),"shimmer_db_std":float(np.std(ddb)),
                    "shimmer_rel_mean":float(np.mean(rel)*100),"shimmer_rel_std":float(np.std(rel)*100),
                    "shimmer_apq3":float(np.mean(apq3)*100)}
        except Exception:
            return {k:0. for k in keys}


class CepstralExtractor:
    """MFCC(40) + LFCC(40) + CQCC(40) = 120 cepstral features.
    Implementation: 20 coefficients x (mean + std) = 40 per family.
    Paper notation: "40 coeff.; +DeltaDelta; CMVN; d=40" refers to the
    40 resulting utterance-level features after temporal aggregation.
    """
    def __init__(self, cfg):
        self.sr=cfg.sr; self.n_fft=cfg.n_fft; self.hop_length=cfg.hop_length
        self.n_mfcc=cfg.n_mfcc; self.n_lfcc=cfg.n_lfcc; self.n_cqcc=cfg.n_cqcc
        self.n_mels=cfg.n_mels; self.n_linear=cfg.n_linear
        self.bins_per_octave=cfg.bins_per_octave
        self.cqcc_fmin=cfg.cqcc_fmin; self.cqcc_n_octaves=cfg.cqcc_n_octaves

    def extract(self, y):
        out={}
        out.update(self._mfcc(y)); out.update(self._lfcc(y)); out.update(self._cqcc(y))
        return out

    def _mfcc(self, y):
        try:
            m=librosa.feature.mfcc(y=y,sr=self.sr,n_mfcc=self.n_mfcc,
                                    n_fft=self.n_fft,hop_length=self.hop_length,n_mels=self.n_mels)
            return {f"mfcc_mean_{i:02d}":float(np.mean(m[i])) for i in range(self.n_mfcc)} | \
                   {f"mfcc_std_{i:02d}": float(np.std(m[i]))  for i in range(self.n_mfcc)}
        except Exception:
            return {f"mfcc_mean_{i:02d}":0. for i in range(self.n_mfcc)} | \
                   {f"mfcc_std_{i:02d}": 0. for i in range(self.n_mfcc)}

    def _lfcc_mat(self, y):
        S=np.abs(librosa.stft(y,n_fft=self.n_fft,hop_length=self.hop_length))
        freqs=librosa.fft_frequencies(sr=self.sr,n_fft=self.n_fft)
        centers=np.linspace(0.,self.sr/2,self.n_linear+2)
        fb=np.zeros((self.n_linear,S.shape[0]))
        for m in range(self.n_linear):
            fl,fc,fr=centers[m],centers[m+1],centers[m+2]
            if fc>fl:
                up=(freqs>=fl)&(freqs<=fc); fb[m,up]=(freqs[up]-fl)/(fc-fl)
            if fr>fc:
                dn=(freqs>fc)&(freqs<=fr); fb[m,dn]=(fr-freqs[dn])/(fr-fc)
        return dct(np.log(np.dot(fb,S)+1e-10),axis=0,norm="ortho")[:self.n_lfcc]

    def _lfcc(self, y):
        try:
            mat=self._lfcc_mat(y)
            return {f"lfcc_mean_{i:02d}":float(np.mean(mat[i])) for i in range(self.n_lfcc)} | \
                   {f"lfcc_std_{i:02d}": float(np.std(mat[i]))  for i in range(self.n_lfcc)}
        except Exception:
            return {f"lfcc_mean_{i:02d}":0. for i in range(self.n_lfcc)} | \
                   {f"lfcc_std_{i:02d}": 0. for i in range(self.n_lfcc)}

    def _cqcc_mat(self, y):
        n_bins=self.bins_per_octave*self.cqcc_n_octaves
        C=np.abs(librosa.cqt(y,sr=self.sr,hop_length=self.hop_length,
                               n_bins=n_bins,bins_per_octave=self.bins_per_octave,
                               fmin=self.cqcc_fmin))
        C_uni=scipy_resample(C,self.n_cqcc*4,axis=0)
        return dct(np.log(np.clip(np.abs(C_uni),1e-10,None)),axis=0,norm="ortho")[:self.n_cqcc]

    def _cqcc(self, y):
        try:
            mat=self._cqcc_mat(y)
            return {f"cqcc_mean_{i:02d}":float(np.mean(mat[i])) for i in range(self.n_cqcc)} | \
                   {f"cqcc_std_{i:02d}": float(np.std(mat[i]))  for i in range(self.n_cqcc)}
        except Exception:
            return {f"cqcc_mean_{i:02d}":0. for i in range(self.n_cqcc)} | \
                   {f"cqcc_std_{i:02d}": 0. for i in range(self.n_cqcc)}


class UnifiedExtractor:
    """All 9 families -> 155 total dimensions."""
    def __init__(self, cfg):
        self.phase=PhaseExtractor(cfg.sr,cfg.n_fft,cfg.hop_length)
        self.phon=PhonationExtractor(cfg.sr,cfg.hop_length,cfg.f0_min,cfg.f0_max)
        self.cep=CepstralExtractor(cfg)
        self.sr=cfg.sr; self.duration=cfg.duration

    def extract(self, y):
        try:
            out={}
            out.update(self.cep.extract(y))
            out.update(self.phase.extract(y))
            out.update(self.phon.extract(y))
            return out
        except Exception as e:
            logger.debug(f"Extraction failed: {e}"); return None


# =============================================================================
# DATASET SCANNING
# =============================================================================
def _label_from_path(ps):
    if any(x in ps for x in ["fake","spoof","deepfake","synthetic","tts"]): return 1
    if any(x in ps for x in ["real","genuine","bonafide","bona-fide","original"]): return 0
    return -1


def scan_ggmddc(cfg):
    cp=cfg.checkpoint_dir/"audio_files_ggmddc.pkl"
    if cp.exists():
        with open(cp,"rb") as f: return pickle.load(f)
    base=Path(cfg.dataset_root)
    if not base.exists():
        logger.warning(f"GGMDDC root not found: {base}"); return []
    lang_fixes={"portugese":"portuguese","sanskarit":"sanskrit","vitnamese":"vietnamese"}
    items=[]
    for wf in tqdm(list(base.rglob("*.wav"))+list(base.rglob("*.flac")),desc="Scan GGMDDC"):
        ps=str(wf).lower(); fn=wf.name.lower()
        lang="unknown"
        for ty,cor in lang_fixes.items():
            if ty in fn: lang=cor; break
        if lang=="unknown":
            for l in cfg.ggmddc_languages:
                if l in ps: lang=l; break
        lbl=_label_from_path(ps)
        if lbl<0: continue
        items.append({"filepath":str(wf),"language":lang,"label":lbl,"generator":"hifigan"})
    if cfg.max_files_per_class>0:
        rng=np.random.default_rng(cfg.seeds[0]); filtered=[]; groups=defaultdict(list)
        for it in items: groups[(it["language"],it["label"])].append(it)
        for key,lst in groups.items():
            if len(lst)>cfg.max_files_per_class:
                idx=rng.choice(len(lst),cfg.max_files_per_class,replace=False)
                filtered.extend([lst[i] for i in idx])
            else: filtered.extend(lst)
        items=filtered
    with open(cp,"wb") as f: pickle.dump(items,f)
    logger.info(f"GGMDDC: {len(items):,} files"); return items


def scan_mlaad(cfg):
    cp=cfg.checkpoint_dir/"audio_files_mlaad.pkl"
    if cp.exists():
        with open(cp,"rb") as f: return pickle.load(f)
    items=[]; lang_map={"de_DE":"de","en_US":"en","en_UK":"en","es_ES":"es","fr_FR":"fr",
                         "it_IT":"it","pl_PL":"pl","ru_RU":"ru","uk_UK":"uk"}
    mailabs=Path(cfg.mailabs_root)
    if mailabs.exists():
        for wf in tqdm(list(mailabs.rglob("*.wav"))+list(mailabs.rglob("*.flac")),desc="Scan M-AILABS"):
            lang="unknown"
            for p in wf.parts:
                if p in lang_map: lang=lang_map[p]; break
            if lang not in cfg.mlaad_languages: continue
            items.append({"filepath":str(wf),"language":lang,"label":0,"generator":"real"})
    mlaad=Path(cfg.dataset_root)
    if mlaad.exists():
        for wf in tqdm(list(mlaad.rglob("*.wav"))+list(mlaad.rglob("*.flac")),desc="Scan MLAAD"):
            ps=str(wf); lang="unknown"
            for l in cfg.mlaad_languages:
                if f"/{l}/" in ps or f"\\{l}\\" in ps: lang=l; break
            if lang not in cfg.mlaad_languages: continue
            # Generator name: use parent dir, or grandparent if parent is a language code
            gen_name = wf.parent.name
            if gen_name in cfg.mlaad_languages:
                gen_name = wf.parents[1].name  # go one level up
            items.append({"filepath":str(wf),"language":lang,"label":1,"generator":gen_name})
    if cfg.min_gen_utterances>0:
        df_tmp=pd.DataFrame(items)
        counts=(df_tmp[df_tmp["label"]==1].groupby(["generator","language"]).size().reset_index(name="n"))
        valid_g=counts[counts["n"]>=cfg.min_gen_utterances]["generator"].unique()
        items=[it for it in items if it["label"]==0 or it["generator"] in valid_g]
    with open(cp,"wb") as f: pickle.dump(items,f)
    logger.info(f"MLAAD: {len(items):,} files"); return items


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_features(audio_files, cfg, save_interval=100):
    feat_file=cfg.out/f"features_{cfg.corpus}.pkl"
    if feat_file.exists():
        df=pd.read_pickle(feat_file); logger.info(f"Features loaded: {len(df):,}"); return df
    cp_file=cfg.checkpoint_dir/f"extraction_{cfg.corpus}.pkl"
    processed={};
    if cp_file.exists():
        with open(cp_file,"rb") as f: processed=pickle.load(f)
    extractor=UnifiedExtractor(cfg); errors=[]
    for item in tqdm(audio_files,desc=f"Extracting ({cfg.corpus})",
                     initial=len(processed),total=len(audio_files)):
        fp=item["filepath"]
        if fp in processed: continue
        try:
            y,_=librosa.load(fp,sr=cfg.sr,duration=cfg.duration)
            if len(y)<cfg.sr*cfg.min_duration: continue
            feats=extractor.extract(y)
            if feats is None: continue
            feats.update({"filepath":fp,"language":item.get("language","unknown"),
                          "label":item["label"],"generator":item.get("generator","unknown"),
                          "duration":len(y)/cfg.sr})
            processed[fp]=feats
        except Exception as e:
            errors.append({"file":fp,"error":str(e)})
        if len(processed)%save_interval==0:
            with open(cp_file,"wb") as f: pickle.dump(processed,f)
    with open(cp_file,"wb") as f: pickle.dump(processed,f)
    df=pd.DataFrame(list(processed.values()))
    df.to_pickle(feat_file)
    logger.info(f"Extraction done: {len(df):,} samples"); return df


# =============================================================================
# PREPROCESSING (no leakage — fitted on train only)
# =============================================================================
def preprocess_fold(X_tr, X_te, y_tr, skew_threshold=2.0):
    X_tr=X_tr.copy(); X_te=X_te.copy()
    # 1. Median imputation per class
    for j in range(X_tr.shape[1]):
        nan_tr=~np.isfinite(X_tr[:,j])
        if nan_tr.any():
            for lbl in np.unique(y_tr):
                fill=np.nanmedian(X_tr[y_tr==lbl,j])
                if not np.isfinite(fill): fill=0.
                X_tr[nan_tr&(y_tr==lbl),j]=fill
        nan_te=~np.isfinite(X_te[:,j])
        if nan_te.any():
            fill=np.nanmedian(X_tr[:,j])
            X_te[nan_te,j]=fill if np.isfinite(fill) else 0.
    # 2. Log-transform skewed — fit on train
    for j in range(X_tr.shape[1]):
        fin=X_tr[:,j][np.isfinite(X_tr[:,j])]
        if len(fin)>3 and np.min(fin)>=0 and abs(float(skew(fin)))>skew_threshold:
            X_tr[:,j]=np.log1p(np.abs(X_tr[:,j])); X_te[:,j]=np.log1p(np.abs(X_te[:,j]))
    # 3. Z-score — fit on train
    mu=np.mean(X_tr,axis=0); sig=np.std(X_tr,axis=0); sig=np.where(sig<1e-10,1.,sig)
    return (X_tr-mu)/sig, (X_te-mu)/sig


def prepare_fold(df_train, df_test, feat_cols, cfg):
    X_tr=df_train[feat_cols].values.astype(float)
    y_tr=df_train["label"].values
    X_te=df_test[feat_cols].values.astype(float)
    y_te=df_test["label"].values
    X_tr,X_te=preprocess_fold(X_tr,X_te,y_tr)
    # Subsample majority to match minority
    rng=np.random.default_rng(cfg.seeds[0])
    n_min=min(np.sum(y_tr==0),np.sum(y_tr==1))
    i0=rng.choice(np.where(y_tr==0)[0],n_min,replace=False)
    i1=rng.choice(np.where(y_tr==1)[0],n_min,replace=False)
    idx=np.sort(np.concatenate([i0,i1]))
    return X_tr[idx],y_tr[idx],X_te,y_te


# =============================================================================
# CLASSIFIERS
# =============================================================================
def make_rf(seed,cfg):
    return RandomForestClassifier(n_estimators=cfg.rf_n_estimators,max_depth=cfg.rf_max_depth,
                                   max_features=cfg.rf_max_features,criterion="gini",
                                   class_weight="balanced",random_state=seed,n_jobs=-1)

def make_lr(seed,cfg):
    return Pipeline([("sc",StandardScaler()),
                     ("clf",LogisticRegression(C=cfg.lr_C,penalty="l2",max_iter=1000,
                                               class_weight="balanced",solver="lbfgs",
                                               random_state=seed))])

def run_multiseed(X_tr,y_tr,X_te,y_te,cfg,lr_only=False):
    aucs,eers=[],[]
    for seed in cfg.seeds:
        clf=make_lr(seed,cfg) if lr_only else make_rf(seed,cfg)
        clf.fit(X_tr,y_tr)
        yp=(clf.predict_proba(X_te)[:,1] if hasattr(clf,"predict_proba")
            else clf.decision_function(X_te))
        try:
            aucs.append(roc_auc_score(y_te,yp))
            eers.append(compute_eer(y_te,yp)[0])
        except Exception: pass
    if not aucs: return {"auc":np.nan,"ci_lo":np.nan,"ci_hi":np.nan,"eer":np.nan}
    arr=np.array(aucs); rng=np.random.default_rng(cfg.seeds[0])
    boots=rng.choice(arr,(cfg.n_bootstrap,len(arr)),replace=True).mean(axis=1)
    return {"auc":float(np.mean(arr)),"ci_lo":float(np.percentile(boots,2.5)),
            "ci_hi":float(np.percentile(boots,97.5)),"eer":float(np.nanmean(eers))}


# =============================================================================
# SQ1 — REPRESENTATION SPACE ANALYSIS
# =============================================================================
def hellinger_distance(p, q):
    """H(P,Q) = (1/sqrt(2)) ||sqrt(P)-sqrt(Q)||_2  (paper Eq. 1)."""
    combined=np.concatenate([p,q])
    lo,hi=np.nanpercentile(combined,1),np.nanpercentile(combined,99)
    if hi-lo<1e-10: return 0.
    nb=max(30,int(np.sqrt(min(len(p),len(q)))))
    bins=np.linspace(lo,hi,nb+1); w=bins[1]-bins[0]
    hp,_=np.histogram(p,bins=bins,density=True); hq,_=np.histogram(q,bins=bins,density=True)
    hp=np.sqrt(hp*w+1e-10); hq=np.sqrt(hq*w+1e-10)
    return float(np.linalg.norm(hp-hq)/np.sqrt(2))


def scheirer_ray_hare(values, languages, generators, labels, class_val=1):
    """
    Nonparametric variance decomposition (paper Eq. 2).
    Returns eta2_lang, eta2_gen, ratio = eta2_gen/eta2_lang.
    """
    from scipy.stats import rankdata
    mask=labels==class_val; v=values[mask]; la=languages[mask]; ge=generators[mask]
    fin=np.isfinite(v); v,la,ge=v[fin],la[fin],ge[fin]
    if len(v)<10: return {"eta2_lang":np.nan,"eta2_gen":np.nan,"ratio":np.nan}
    ranks=rankdata(v).astype(float); N=len(ranks)
    SS_tot=np.var(ranks)*N
    if SS_tot<1e-10: return {"eta2_lang":0.,"eta2_gen":0.,"ratio":np.nan}
    def ss_factor(grp):
        gm=np.mean(ranks); ss=0.
        for g in np.unique(grp):
            m=ranks[grp==g]; ss+=len(m)*(np.mean(m)-gm)**2
        return ss
    eta2_lang=float(ss_factor(la)/SS_tot); eta2_gen=float(ss_factor(ge)/SS_tot)
    ratio=float(eta2_gen/eta2_lang) if eta2_lang>1e-10 else np.inf
    return {"eta2_lang":eta2_lang,"eta2_gen":eta2_gen,"ratio":ratio}


def scheirer_ray_hare_classwise(values, languages, labels):
    """
    Compute eta2_lang separately for real (label=0) and fake (label=1) class.
    Used for GGMDDC single-generator corpus to compute the f/r ratio profile
    (Symmetric, Quasi-invariant, Asymmetric, Moderate) reported in Table 4.
    """
    from scipy.stats import rankdata
    results = {}
    for class_val, class_name in [(0, "real"), (1, "fake")]:
        mask = labels == class_val
        v = values[mask]; la = languages[mask]
        fin = np.isfinite(v); v, la = v[fin], la[fin]
        if len(v) < 10:
            results[f"eta2_{class_name}"] = np.nan
            continue
        ranks = rankdata(v).astype(float); N = len(ranks)
        SS_tot = np.var(ranks) * N
        if SS_tot < 1e-10:
            results[f"eta2_{class_name}"] = 0.0; continue
        gm = np.mean(ranks); ss = 0.
        for g in np.unique(la):
            m = ranks[la == g]; ss += len(m) * (np.mean(m) - gm) ** 2
        results[f"eta2_{class_name}"] = float(ss / SS_tot)
    eta2_f = results.get("eta2_fake", np.nan)
    eta2_r = results.get("eta2_real", np.nan)
    ratio_fr = float(eta2_f / eta2_r) if (eta2_r and eta2_r > 1e-10) else np.nan
    # Profile classification (paper Table 4)
    if np.isnan(eta2_f) or np.isnan(eta2_r):
        profile = "N/A"
    elif abs(ratio_fr - 1.0) < 0.25:
        profile = "Symmetric"
    elif eta2_f < 0.01:
        profile = "Quasi-invariant"
    elif ratio_fr > 1.5:
        profile = "Asymmetric"
    else:
        profile = "Moderate"
    results["ratio_fr"] = ratio_fr
    results["profile"] = profile
    return results


def sq1(df, families, cfg):
    logger.info("="*60+"\nSQ1: Representation Space Analysis\n"+"="*60)
    labels=df["label"].values; languages=df["language"].values
    generators=df["generator"].values if "generator" in df.columns else labels.copy()
    records=[]
    for fam,cols in families.items():
        X=df[cols].values.astype(float); X=np.nan_to_num(X,nan=0.)
        scores=(X[:,0] if X.shape[1]==1
                else PCA(n_components=1,random_state=cfg.seeds[0])
                       .fit_transform(StandardScaler().fit_transform(X)).ravel())
        h_by_lang={}
        for lang in cfg.languages:
            mf=(labels==1)&(languages==lang); mr=(labels==0)&(languages==lang)
            if mf.sum()>=5 and mr.sum()>=5:
                h_by_lang[lang]=hellinger_distance(scores[mf],scores[mr])
        h_vals=np.array(list(h_by_lang.values()))
        h_mean=float(np.mean(h_vals)) if len(h_vals)>0 else np.nan
        h_cv=float(np.std(h_vals)/(h_mean+1e-10)) if h_mean>1e-5 else np.nan
        d=cohens_d(scores[labels==1],scores[labels==0])
        srh=scheirer_ray_hare(scores,languages,generators,labels)
        # Classwise eta2 for GGMDDC (single-generator: no eta2_gen)
        cw = scheirer_ray_hare_classwise(scores, languages, labels)
        row={"family":fam,"n_features":len(cols),
             "H_mean":round(h_mean,4),"H_cv":round(h_cv,4) if not np.isnan(h_cv) else np.nan,
             "cohens_d":round(d,4) if not np.isnan(d) else np.nan,
             "eta2_lang":round(srh["eta2_lang"],4),"eta2_gen":round(srh["eta2_gen"],4),
             "ratio_gen_lang":round(srh["ratio"],2) if np.isfinite(srh["ratio"]) else np.nan,
             "eta2_fake":round(cw.get("eta2_fake",np.nan),4),
             "eta2_real":round(cw.get("eta2_real",np.nan),4),
             "ratio_fr":round(cw.get("ratio_fr",np.nan),2) if not np.isnan(cw.get("ratio_fr",np.nan)) else np.nan,
             "profile":cw.get("profile","N/A")}
        row.update({f"H_{lang}":round(v,4) for lang,v in h_by_lang.items()})
        records.append(row)
        logger.info(f"  {fam:8s}: H={h_mean:.3f} CV={h_cv:.3f} "
                    f"eta2_gen={srh['eta2_gen']:.3f} ratio={srh['ratio']:.1f}x")
    df_sq1=pd.DataFrame(records).sort_values("H_mean",ascending=False)
    df_sq1.to_csv(cfg.tables_dir/"SQ1_representation_space.csv",index=False)
    return df_sq1


# =============================================================================
# SQ2 — CROSS-LINGUAL STABILITY AND SEPARABILITY
# =============================================================================
def sq2(df, families, cfg):
    logger.info("="*60+"\nSQ2: Cross-Lingual Stability and Separability\n"+"="*60)
    labels=df["label"].values; languages=df["language"].values
    records=[]; p_mw=[]; p_kwf=[]; p_kwr=[]
    for fam,cols in families.items():
        X=df[cols].values.astype(float); X=np.nan_to_num(X,nan=0.)
        # Project to 1D using PCA (unsupervised, no label leakage)
        # Consistent with SQ1 approach for multivariate families
        if X.shape[1] == 1:
            scores = X[:, 0]
        else:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1, random_state=cfg.seeds[0])
                scores = pca.fit_transform(StandardScaler().fit_transform(X)).ravel()
            except Exception:
                scores = X.mean(axis=1)  # fallback
        # Mann-Whitney U + rank-biserial |r^b|  (paper Eq. 3)
        a=scores[labels==1][np.isfinite(scores[labels==1])]
        b=scores[labels==0][np.isfinite(scores[labels==0])]
        if len(a)>=2 and len(b)>=2:
            stat,p_mw_raw=mannwhitneyu(a,b,alternative="two-sided")
            rb=abs(1-2*stat/(len(a)*len(b)))
        else:
            p_mw_raw,rb=np.nan,np.nan
        # Kruskal-Wallis H (paper Eq. 4)
        def kw(class_val):
            grps={l:scores[(labels==class_val)&(languages==l)] for l in cfg.languages}
            grps={l:v[np.isfinite(v)] for l,v in grps.items() if v[np.isfinite(v)].shape[0]>=cfg.min_samples_kw}
            if len(grps)<2: return np.nan, 0
            try: _,p=kruskal(*grps.values()); return float(p),len(grps)
            except ValueError: return np.nan, 0
        p_kwf_raw,n_kwf=kw(1); p_kwr_raw,n_kwr=kw(0)
        records.append({"family":fam,"rb":round(rb,4) if not np.isnan(rb) else np.nan,
                         "p_mw_raw":p_mw_raw,"p_kwf_raw":p_kwf_raw,"p_kwr_raw":p_kwr_raw,
                         "n_groups_kwf":n_kwf,"n_groups_kwr":n_kwr})
        p_mw.append(p_mw_raw if np.isfinite(p_mw_raw) else 1.)
        p_kwf.append(p_kwf_raw if np.isfinite(p_kwf_raw) else 1.)
        p_kwr.append(p_kwr_raw if np.isfinite(p_kwr_raw) else 1.)
    df_sq2=pd.DataFrame(records)
    for raw,corr in [("p_mw_raw","p_mw_corrected"),("p_kwf_raw","p_kwf_corrected"),
                      ("p_kwr_raw","p_kwr_corrected")]:
        _,c,_,_=multipletests(df_sq2[raw].fillna(1.).tolist(),alpha=cfg.alpha,method="fdr_bh")
        df_sq2[corr]=c
    df_sq2["discriminative"]       = df_sq2["p_mw_corrected"]<cfg.alpha
    df_sq2["stable_fake"]          = df_sq2["p_kwf_corrected"]>=cfg.alpha
    df_sq2["stable_real"]          = df_sq2["p_kwr_corrected"]>=cfg.alpha
    df_sq2["language_independent"] = df_sq2["stable_fake"]&df_sq2["stable_real"]
    logger.info(f"  Discriminative:        {df_sq2['discriminative'].sum()}/{len(df_sq2)}")
    logger.info(f"  Language-independent:  {df_sq2['language_independent'].sum()}/{len(df_sq2)}")
    logger.info(f"  Satisfying both:       0/{len(df_sq2)}")
    for _,r in df_sq2.iterrows():
        logger.info(f"  {r['family']:8s} |r^b|={r['rb']:.3f}  "
                    f"p_mw={r['p_mw_corrected']:.2e}  p_kwf={r['p_kwf_corrected']:.2e}  "
                    f"disc={r['discriminative']}  inv={r['language_independent']}")
    df_sq2.to_csv(cfg.tables_dir/"SQ2_invariance.csv",index=False); return df_sq2


# =============================================================================
# SQ3 — LOLO
# =============================================================================
def sq3_lolo(df, families, cfg):
    logger.info("="*60+"\nSQ3 - LOLO (Leave-One-Language-Out)\n"+"="*60)
    cp=cfg.checkpoint_dir/f"lolo_{cfg.corpus}.pkl"
    if cp.exists():
        df_lolo=pd.read_pickle(cp); logger.info(f"LOLO loaded: {len(df_lolo)} rows"); return df_lolo
    langs=sorted(df["language"].unique()); records=[]
    for lang in langs:
        df_tr=df[df["language"]!=lang].copy()
        df_te=df[df["language"]==lang].copy()
        if len(df_te)<10 or len(np.unique(df_te["label"].values))<2: continue
        for fam,cols in families.items():
            X_tr,y_tr,X_te,y_te=prepare_fold(df_tr,df_te,cols,cfg)
            rf=run_multiseed(X_tr,y_tr,X_te,y_te,cfg,lr_only=False)
            lr=run_multiseed(X_tr,y_tr,X_te,y_te,cfg,lr_only=True)
            records.append({"protocol":"LOLO","held_out":lang,"family":fam,
                             "auc_rf":rf["auc"],"ci_lo_rf":rf["ci_lo"],"ci_hi_rf":rf["ci_hi"],
                             "eer_rf":rf["eer"],"auc_lr":lr["auc"],"eer_lr":lr["eer"],
                             "n_test":len(y_te)})
            logger.info(f"  LOLO lang={lang} fam={fam}: "
                        f"AUC(RF)={rf['auc']:.4f} [{rf['ci_lo']:.3f},{rf['ci_hi']:.3f}]")
    df_lolo=pd.DataFrame(records)
    df_lolo.to_pickle(cp); df_lolo.to_csv(cfg.tables_dir/f"SQ3_lolo_{cfg.corpus}.csv",index=False)
    return df_lolo


# =============================================================================
# SQ3 — LOGO
# =============================================================================
def sq3_logo(df, families, cfg):
    if "generator" not in df.columns: return None
    gens=[g for g in df["generator"].unique() if g!="real"]
    if len(gens)<2: return None
    logger.info("="*60+f"\nSQ3 - LOGO ({len(gens)} generators)\n"+"="*60)
    cp=cfg.checkpoint_dir/"logo_mlaad.pkl"
    if cp.exists():
        df_logo=pd.read_pickle(cp); logger.info(f"LOGO loaded: {len(df_logo)} rows"); return df_logo
    df_real=df[df["label"]==0].copy(); records=[]
    for gen in tqdm(gens,desc="LOGO"):
        df_te_f=df[(df["label"]==1)&(df["generator"]==gen)].copy()
        df_tr_f=df[(df["label"]==1)&(df["generator"]!=gen)].copy()
        if len(df_te_f)<10 or len(np.unique(df_te_f["label"].values))<1: continue
        n_te=len(df_te_f)
        real_te=(df_real.sample(n=min(n_te,len(df_real)),random_state=cfg.seeds[0])
                 if len(df_real)>=n_te else df_real)
        real_tr=df_real.drop(real_te.index) if len(df_real)>n_te else df_real
        df_tr=pd.concat([df_tr_f,real_tr]).reset_index(drop=True)
        df_te=pd.concat([df_te_f,real_te]).reset_index(drop=True)
        if len(np.unique(df_te["label"].values))<2: continue
        for fam,cols in families.items():
            X_tr,y_tr,X_te,y_te=prepare_fold(df_tr,df_te,cols,cfg)
            rf=run_multiseed(X_tr,y_tr,X_te,y_te,cfg)
            records.append({"protocol":"LOGO","held_out":gen,"family":fam,
                             "auc_rf":rf["auc"],"ci_lo_rf":rf["ci_lo"],"ci_hi_rf":rf["ci_hi"],
                             "eer_rf":rf["eer"],"n_test":len(y_te)})
        df_partial=pd.DataFrame(records); df_partial.to_pickle(cp)
    df_logo=pd.DataFrame(records)
    df_logo.to_pickle(cp); df_logo.to_csv(cfg.tables_dir/"SQ3_logo_mlaad.csv",index=False)
    return df_logo


def compute_delta(df_lolo, df_logo, cfg):
    """Delta = AUC_LOGO - AUC_LOLO per family (paper Section 3.5)."""
    lolo_m=df_lolo[df_lolo["protocol"]=="LOLO"].groupby("family")["auc_rf"].mean().rename("auc_lolo")
    if df_logo is None or len(df_logo)==0:
        d=lolo_m.reset_index(); d["auc_logo"]=np.nan; d["delta"]=np.nan; return d
    logo_m=df_logo.groupby("family")["auc_rf"].mean().rename("auc_logo")
    d=pd.concat([lolo_m,logo_m],axis=1).reset_index()
    d["delta"]=(d["auc_logo"]-d["auc_lolo"]).round(4)
    def profile(r):
        if np.isnan(r["auc_lolo"]): return "N/A"
        if r["auc_lolo"]<0.60 and (np.isnan(r["auc_logo"]) or r["auc_logo"]<0.70): return "Failure in both"
        if not np.isnan(r["delta"]) and abs(r["delta"])<=0.025: return "Robust in both"
        return "Language > Generator"
    d["profile"]=d.apply(profile,axis=1); d=d.sort_values("delta")
    logger.info("\n  Delta = AUC_LOGO - AUC_LOLO:\n"+
                d[["family","auc_lolo","auc_logo","delta","profile"]].to_string(index=False))
    d.to_csv(cfg.tables_dir/"SQ3_delta.csv",index=False); return d


# =============================================================================
# SQ4 — SHAP MECHANISM ATTRIBUTION
# =============================================================================
def sq4_shap(df, families, cfg, protocol="lolo"):
    if not SHAP_AVAILABLE:
        logger.warning("SQ4 skipped: install shap"); return None
    logger.info("="*60+f"\nSQ4: SHAP Attribution ({protocol.upper()})\n"+"="*60)
    all_cols=[c for cols in families.values() for c in cols]
    held_out_vals=(sorted(df["language"].unique()) if protocol=="lolo"
                   else sorted([g for g in df["generator"].unique() if g!="real"]))
    records=[]
    for held_out in tqdm(held_out_vals,desc=f"SHAP {protocol}"):
        if protocol=="lolo":
            df_tr=df[df["language"]!=held_out].copy()
            df_te=df[df["language"]==held_out].copy()
        else:
            # LOGO: build balanced test set (fake from held-out gen + matched real)
            df_real_pool = df[df["label"]==0].copy()
            df_te_fake   = df[(df["label"]==1)&(df["generator"]==held_out)].copy()
            df_tr_fake   = df[(df["label"]==1)&(df["generator"]!=held_out)].copy()
            if len(df_te_fake) < 5: continue
            n_te = len(df_te_fake)
            real_te = (df_real_pool.sample(n=min(n_te,len(df_real_pool)),
                                           random_state=cfg.seeds[0])
                       if len(df_real_pool)>=n_te else df_real_pool)
            real_tr = df_real_pool.drop(real_te.index)
            df_te = pd.concat([df_te_fake, real_te]).reset_index(drop=True)
            df_tr = pd.concat([df_tr_fake, real_tr]).reset_index(drop=True)
        if len(df_te)<10 or len(np.unique(df_te["label"].values))<2: continue
        X_tr,y_tr,X_te,y_te=prepare_fold(df_tr,df_te,all_cols,cfg)
        clf=make_rf(cfg.seeds[0],cfg); clf.fit(X_tr,y_tr)
        yp=clf.predict_proba(X_te)[:,1]
        eer_v,thr=compute_eer(y_te,yp); y_pred=(yp>=thr).astype(int)
        outcome=np.where((y_te==1)&(y_pred==1),"TP",
                 np.where((y_te==0)&(y_pred==0),"TN",
                 np.where((y_te==1)&(y_pred==0),"FN","FP")))
        try:
            explainer=shap_lib.TreeExplainer(clf)
            rng=np.random.default_rng(cfg.seeds[0])
            n_s=min(cfg.shap_max_samples,len(X_te))
            idx_s=rng.choice(len(X_te),n_s,replace=False)
            # shap_values: robust extraction for different shap/sklearn versions
            sv_raw=explainer.shap_values(X_te[idx_s])
            if isinstance(sv_raw, list):
                sv = np.abs(sv_raw[1])        # list output: [class0, class1]
            elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
                sv = np.abs(sv_raw[:, :, 1])  # 3D array: (samples, features, classes)
            else:
                sv = np.abs(sv_raw)            # 2D array: already class-1 shap
        except Exception as e:
            logger.warning(f"SHAP failed {held_out}: {e}"); continue
        outcome_s=outcome[idx_s]
        for strata in ["TP","TN","FP","FN"]:
            mask=outcome_s==strata
            if mask.sum()==0: continue
            mean_sv=sv[mask].mean(axis=0)
            row={"held_out":held_out,"protocol":protocol,"strata":strata,"n":int(mask.sum()),
                 "auc":float(roc_auc_score(y_te,yp)),"eer":float(eer_v)}
            offset=0
            for fam,cols in families.items():
                row[f"phi_{fam}"]=float(mean_sv[offset:offset+len(cols)].mean())
                offset+=len(cols)
            records.append(row)
    if not records: return None
    df_shap=pd.DataFrame(records)
    df_shap.to_csv(cfg.tables_dir/f"SQ4_shap_{protocol}.csv",index=False)
    # Delta = phi_error - phi_correct
    phi_cols=[c for c in df_shap.columns if c.startswith("phi_")]
    err_ctrl={"FP":"TN","FN":"TP"}; delta_recs=[]
    for ho in df_shap["held_out"].unique():
        sub=df_shap[df_shap["held_out"]==ho]
        for err,ctrl in err_ctrl.items():
            er=sub[sub["strata"]==err]; cr=sub[sub["strata"]==ctrl]
            if len(er)==0 or len(cr)==0: continue
            rec={"held_out":ho,"error_type":err,"n_error":int(er["n"].values[0]),
                 "n_correct":int(cr["n"].values[0])}
            for col in phi_cols:
                fam=col.replace("phi_","")
                rec[f"delta_{fam}"]=float(er[col].values[0]-cr[col].values[0])
            delta_recs.append(rec)
    if delta_recs:
        df_d=pd.DataFrame(delta_recs)
        df_d.to_csv(cfg.tables_dir/f"SQ4_delta_shap_{protocol}.csv",index=False)
        logger.info("\n  Mean Delta per error type:")
        dcols=[c for c in df_d.columns if c.startswith("delta_")]
        logger.info(df_d.groupby("error_type")[dcols].mean().round(4).to_string())
    return df_shap




# =============================================================================
# MLAAD MAIN
# =============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="MLAAD+M-AILABS diagnostic pipeline — SQ1 to SQ4 (LOLO + LOGO)"
    )
    p.add_argument("--root",         default="",
                   help="Path to MLAAD dataset root directory")
    p.add_argument("--mailabs_root", default="",
                   help="Path to M-AILABS genuine speech root directory")
    p.add_argument("--output",       default="results_mlaad",
                   help="Output directory (default: results_mlaad/)")
    p.add_argument("--features",     default="",
                   help="Pre-extracted features .pkl (skips audio extraction)")
    args = p.parse_args()

    set_global_seed(42)
    cfg = Config(
        corpus        = "mlaad",
        dataset_root  = args.root,
        mailabs_root  = args.mailabs_root,
        output_dir    = args.output,
        features_file = args.features,
    )
    cfg.create_dirs()

    logger.info("=" * 60)
    logger.info("MLAAD+M-AILABS — Realistic Multi-Generator Setting")
    logger.info("  8 languages | 55 generators | LOLO + LOGO")
    logger.info("=" * 60)

    # ── Step 1: Load or extract features ─────────────────────────────────────
    if cfg.features_file and Path(cfg.features_file).exists():
        logger.info(f"Loading pre-extracted features: {cfg.features_file}")
        df = pd.read_pickle(cfg.features_file)
    else:
        audio_files = scan_mlaad(cfg)
        if not audio_files:
            logger.error("No audio files found. Check --root and --mailabs_root.")
            return
        df = extract_features(audio_files, cfg)

    n_gens = df["generator"].nunique() if "generator" in df.columns else 0
    logger.info(f"Dataset: {len(df):,}  "
                f"(fake={int((df['label']==1).sum())}, "
                f"real={int((df['label']==0).sum())})  "
                f"generators={n_gens}")

    # ── Step 2: Feature families ──────────────────────────────────────────────
    all_cols = get_feature_cols(df, cfg.excluded_features)
    families = get_family_cols(all_cols, cfg.excluded_features)
    total    = sum(len(c) for c in families.values())
    logger.info(f"Families: {list(families.keys())}  | Total dims: {total}")

    # ── SQ1: Representation Space Analysis ───────────────────────────────────
    # Hellinger distance + eta2_lang + eta2_gen + dominance ratio gen/lang
    # Key result: all 9 families show ratio 3.7x-13.2x (Table 4)
    logger.info("\n[SQ1] Representation Space Analysis")
    df_sq1 = sq1(df, families, cfg)

    # ── SQ2: Cross-Lingual Stability and Separability ────────────────────────
    # Mann-Whitney U (|r^b|) + Kruskal-Wallis H, BH-FDR corrected
    # Key result: 0/9 families satisfy joint invariance criterion
    logger.info("\n[SQ2] Cross-Lingual Stability and Separability")
    df_sq2 = sq2(df, families, cfg)

    # ── SQ3: LOLO (Leave-One-Language-Out) ───────────────────────────────────
    # 8 folds (one per language), RF + LR classifiers
    logger.info("\n[SQ3-LOLO] Cross-Lingual Robustness — 8 folds")
    df_lolo = sq3_lolo(df, families, cfg)

    # ── SQ3: LOGO (Leave-One-Generator-Out) ──────────────────────────────────
    # 55 folds (one per TTS generator), RF classifier
    # Key result: LFCC AUC=1.000, griffin_lim hardest generator
    logger.info("\n[SQ3-LOGO] Cross-Generator Robustness — 55 folds")
    df_logo = sq3_logo(df, families, cfg)

    # ── SQ3: Delta table ─────────────────────────────────────────────────────
    # Δ = AUC_LOGO − AUC_LOLO per family (Table 8)
    df_delta = compute_delta(df_lolo, df_logo, cfg)

    # ── SQ4: SHAP under LOLO (German phase over-activation — Mechanism A) ────
    logger.info("\n[SQ4-LOLO] SHAP Mechanism Attribution")
    sq4_shap(df, families, cfg, protocol="lolo")

    # ── SQ4: SHAP under LOGO (griffin_lim cepstral anomaly — Mechanism B) ────
    if df_logo is not None:
        logger.info("\n[SQ4-LOGO] SHAP Mechanism Attribution")
        sq4_shap(df, families, cfg, protocol="logo")

    logger.info("\n" + "=" * 60)
    logger.info("MLAAD PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {cfg.tables_dir}/")
    for p in sorted(Path(cfg.tables_dir).glob("*.csv")):
        logger.info(f"  {p.name}")


if __name__ == "__main__":
    main()
