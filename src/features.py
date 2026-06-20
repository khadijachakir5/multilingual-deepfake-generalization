import numpy as np
import librosa
from scipy.fft import dct, ifft
from scipy.signal import resample as scipy_resample
from scipy.stats import skew, kurtosis

SR = 16_000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
DURATION = 4.0
MIN_DUR = 1.0

N_MFCC = 20
N_MELS = 40

N_LFCC = 20
N_LINEAR = 40

N_CQCC = 20
BINS_PER_OCTAVE = 96
N_OCTAVES = 9
CQCC_FMIN = 32.7

F0_MIN = 60.0
F0_MAX = 500.0

FAMILY_DIMS = {
    "MFCC": 40, "LFCC": 40, "CQCC": 40,
    "GD": 7, "IF": 8, "PD": 6,
    "CPP": 3, "Jitter": 6, "Shimmer": 5,
}

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

EXCLUDED_FEATS = ["pyin_failed", "pyin_voiced_ratio"]


def families_from_columns(all_cols):
    excl = set(EXCLUDED_FEATS)
    return {fam: [c for c in all_cols if fn(c) and c not in excl]
            for fam, fn in FAMILY_PREFIXES.items()
            if any(fn(c) for c in all_cols if c not in excl)}


def extract_mfcc(y):
    try:
        m = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        return ({f"mfcc_mean_{i:02d}": float(np.mean(m[i])) for i in range(N_MFCC)} |
                {f"mfcc_std_{i:02d}":  float(np.std(m[i]))  for i in range(N_MFCC)})
    except Exception:
        return ({f"mfcc_mean_{i:02d}": 0. for i in range(N_MFCC)} |
                {f"mfcc_std_{i:02d}":  0. for i in range(N_MFCC)})


def extract_lfcc(y):
    try:
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 win_length=WIN_LENGTH))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        centers = np.linspace(0., SR / 2, N_LINEAR + 2)
        fb = np.zeros((N_LINEAR, S.shape[0]))
        for m in range(N_LINEAR):
            fl, fc, fr = centers[m], centers[m + 1], centers[m + 2]
            if fc > fl:
                up = (freqs >= fl) & (freqs <= fc)
                fb[m, up] = (freqs[up] - fl) / (fc - fl)
            if fr > fc:
                dn = (freqs > fc) & (freqs <= fr)
                fb[m, dn] = (fr - freqs[dn]) / (fr - fc)
        log_spec = np.log(np.dot(fb, S) + 1e-10)
        mat = dct(log_spec, axis=0, norm="ortho")[:N_LFCC]
        return ({f"lfcc_mean_{i:02d}": float(np.mean(mat[i])) for i in range(N_LFCC)} |
                {f"lfcc_std_{i:02d}":  float(np.std(mat[i]))  for i in range(N_LFCC)})
    except Exception:
        return ({f"lfcc_mean_{i:02d}": 0. for i in range(N_LFCC)} |
                {f"lfcc_std_{i:02d}":  0. for i in range(N_LFCC)})


def extract_cqcc(y):
    try:
        requested = BINS_PER_OCTAVE * N_OCTAVES
        max_bins = int(np.floor(
            BINS_PER_OCTAVE * np.log2((SR / 2) / CQCC_FMIN)))
        n_bins = max(1, min(requested, max_bins))
        C = np.abs(librosa.cqt(
            y, sr=SR, hop_length=HOP_LENGTH,
            n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE,
            fmin=CQCC_FMIN))
        C_uni = scipy_resample(C, N_CQCC * 4, axis=0)
        log_C = np.log(np.clip(np.abs(C_uni), 1e-10, None))
        mat = dct(log_C, axis=0, norm="ortho")[:N_CQCC]
        return ({f"cqcc_mean_{i:02d}": float(np.mean(mat[i])) for i in range(N_CQCC)} |
                {f"cqcc_std_{i:02d}":  float(np.std(mat[i]))  for i in range(N_CQCC)})
    except Exception:
        return ({f"cqcc_mean_{i:02d}": 0. for i in range(N_CQCC)} |
                {f"cqcc_std_{i:02d}":  0. for i in range(N_CQCC)})


def extract_phase(y):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mag = np.abs(D)
    ph = np.angle(D)
    out = {}

    try:
        uw = np.unwrap(ph, axis=0)
        bw = 2. * np.pi / N_FFT
        gd = -np.diff(uw, axis=0) / bw
        fl = gd.flatten()
        fl = fl[np.isfinite(fl)]
        gdf = np.mean(gd, axis=0)
        x = np.arange(len(gdf))
        res = gdf - np.polyval(np.polyfit(x, gdf, 1), x)
        out.update({
            "gd_mean": float(np.mean(fl)), "gd_std": float(np.std(fl)),
            "gd_skew": float(skew(fl)) if np.std(fl) > 1e-10 else 0.,
            "gd_kurt": float(kurtosis(fl)) if np.std(fl) > 1e-10 else 0.,
            "gd_range": float(np.ptp(fl)),
            "gd_deviation_mean": float(np.mean(np.abs(res))),
            "gd_deviation_std": float(np.std(res)),
        })
    except Exception:
        out.update({k: 0. for k in ["gd_mean", "gd_std", "gd_skew", "gd_kurt",
                                     "gd_range", "gd_deviation_mean", "gd_deviation_std"]})

    try:
        uw2 = np.unwrap(ph, axis=1)
        IF_ = np.diff(uw2, axis=1) * SR / (2 * np.pi * HOP_LENGTH)
        magw = (mag[:, :-1] + mag[:, 1:]) / 2
        magw /= (np.sum(magw) + 1e-10)
        fl = IF_.flatten()
        mwf = magw.flatten()
        fin = np.isfinite(fl)
        mu = float(np.average(fl[fin], weights=mwf[fin]))
        n = IF_.shape[0]
        out.update({
            "if_mean": mu,
            "if_std": float(np.sqrt(np.average((fl[fin] - mu) ** 2, weights=mwf[fin]))),
            "if_skewness": float(skew(fl[fin])) if fin.sum() > 3 else 0.,
            "if_kurtosis": float(kurtosis(fl[fin])) if fin.sum() > 3 else 0.,
            "if_negative_ratio": float(np.mean(fl[fin] < 0)),
            "if_var_low": float(np.var(IF_[:n // 4, :])),
            "if_var_mid": float(np.var(IF_[n // 4:3 * n // 4, :])),
            "if_var_high": float(np.var(IF_[3 * n // 4:, :])),
        })
    except Exception:
        out.update({k: 0. for k in ["if_mean", "if_std", "if_skewness", "if_kurtosis",
                                     "if_negative_ratio", "if_var_low", "if_var_mid", "if_var_high"]})

    try:
        w = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
        pdf = np.diff(np.unwrap(ph, axis=0), axis=0)
        wpd = np.sum(w[:-1, :] * np.abs(pdf), axis=0)
        j = np.abs(np.diff(ph, axis=1))
        j = np.where(j > np.pi, 2 * np.pi - j, j)
        out.update({
            "pd_weighted_mean": float(np.mean(wpd)),
            "pd_weighted_std": float(np.std(wpd)),
            "pd_jump_mean": float(np.mean(j)),
            "pd_jump_std": float(np.std(j)),
            "pd_jump_max": float(np.max(j)),
            "pd_large_jump_ratio": float(np.mean(j > np.pi / 2)),
        })
    except Exception:
        out.update({k: 0. for k in ["pd_weighted_mean", "pd_weighted_std",
                                     "pd_jump_mean", "pd_jump_std",
                                     "pd_jump_max", "pd_large_jump_ratio"]})
    return out


def extract_phonation(y):
    out = {}

    try:
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        ceps = np.abs(np.real(ifft(np.log(S ** 2 + 1e-10), axis=0)))
        qmn = int(SR / F0_MAX)
        qmx = int(SR / F0_MIN)
        cpps = [np.max(f[qmn:qmx]) -
                np.mean(np.polyval(np.polyfit(np.arange(qmn, qmx), f[qmn:qmx], 1),
                                    np.arange(qmn, qmx)))
                for f in ceps.T if qmx < len(f)]
        c = np.array(cpps)
        if len(c) > 0:
            out.update({"cpp_mean": float(np.mean(c)), "cpp_std": float(np.std(c)),
                         "cpp_max": float(np.max(c))})
        else:
            out.update({"cpp_mean": 0., "cpp_std": 0., "cpp_max": 0.})
    except Exception:
        out.update({"cpp_mean": 0., "cpp_std": 0., "cpp_max": 0.})

    jnull = {k: np.nan for k in ["jitter_abs_mean", "jitter_abs_std",
                                  "jitter_rel_mean", "jitter_rel_std",
                                  "jitter_rap", "jitter_ppq5"]}
    jnull.update({"pyin_failed": 1., "pyin_voiced_ratio": 0.})
    try:
        f0, vf, _ = librosa.pyin(
            y, fmin=F0_MIN, fmax=F0_MAX, sr=SR,
            frame_length=N_FFT, hop_length=HOP_LENGTH)
        vr = float(np.mean(vf)) if vf is not None else 0.
        f0v = f0[(vf) & (np.isfinite(f0))] if vf is not None else np.array([])
        if len(f0v) < 3:
            jnull["pyin_voiced_ratio"] = vr
            out.update(jnull)
        else:
            T = 1. / (f0v + 1e-10)
            dT = np.abs(np.diff(T))
            rap = [abs(T[i] - np.mean(T[i - 1:i + 2])) / (np.mean(T[i - 1:i + 2]) + 1e-10)
                   for i in range(1, len(T) - 1)] if len(T) >= 3 else [0.]
            ppq5 = [abs(T[i] - np.mean(T[i - 2:i + 3])) / (np.mean(T[i - 2:i + 3]) + 1e-10)
                    for i in range(2, len(T) - 2)] if len(T) >= 5 else [0.]
            out.update({
                "pyin_failed": 0., "pyin_voiced_ratio": vr,
                "jitter_abs_mean": float(np.mean(dT)),
                "jitter_abs_std": float(np.std(dT)),
                "jitter_rel_mean": float(np.mean(dT / (T[:-1] + 1e-10)) * 100),
                "jitter_rel_std": float(np.std(dT / (T[:-1] + 1e-10)) * 100),
                "jitter_rap": float(np.mean(rap) * 100),
                "jitter_ppq5": float(np.mean(ppq5) * 100),
            })
    except Exception:
        out.update(jnull)

    skeys = ["shimmer_db_mean", "shimmer_db_std", "shimmer_rel_mean",
             "shimmer_rel_std", "shimmer_apq3"]
    try:
        fr = librosa.util.frame(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        E = np.sum(fr ** 2, axis=0)
        A = np.sqrt(E[E > 0.1 * np.max(E)] + 1e-20)
        if len(A) < 3:
            out.update({k: 0. for k in skeys})
        else:
            ddb = 20 * np.log10(A[1:] / (A[:-1] + 1e-10) + 1e-10)
            rel = np.abs(np.diff(A)) / (A[:-1] + 1e-10)
            apq3 = [abs(A[i] - np.mean(A[i - 1:i + 2])) / (np.mean(A[i - 1:i + 2]) + 1e-10)
                    for i in range(1, len(A) - 1)] if len(A) >= 3 else [0.]
            out.update({
                "shimmer_db_mean": float(np.mean(np.abs(ddb))),
                "shimmer_db_std": float(np.std(ddb)),
                "shimmer_rel_mean": float(np.mean(rel) * 100),
                "shimmer_rel_std": float(np.std(rel) * 100),
                "shimmer_apq3": float(np.mean(apq3) * 100),
            })
    except Exception:
        out.update({k: 0. for k in skeys})
    return out


def extract_all(y):
    feats = {}
    feats.update(extract_mfcc(y))
    feats.update(extract_lfcc(y))
    feats.update(extract_cqcc(y))
    feats.update(extract_phase(y))
    feats.update(extract_phonation(y))
    return feats


def load_and_extract(filepath):
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak
    if len(y) < SR * MIN_DUR:
        raise ValueError(f"duration<{MIN_DUR}s")
    return extract_all(y)
