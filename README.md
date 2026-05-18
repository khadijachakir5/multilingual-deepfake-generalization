# Understanding Generalisation in Multilingual Deepfake Speech Detection through Acoustic Space Geometry

**Chakir, K., Gahi, Y. & El-Khatib, K.**  
*EURASIP Journal on Audio, Speech, and Music Processing*

---

## Overview

This repository provides the complete reproducible implementation for the diagnostic study described in the paper. The analysis is conducted on two complementary multilingual corpora and structured around four sequential research questions (SQ1–SQ4).

**Two independent scripts are provided — one per corpus:**

| Script | Corpus | Protocols |
|--------|--------|-----------|
| `pipeline_ggmddc.py` | GGMDDC (controlled, 1 generator) | SQ1 + SQ2 + LOLO + SQ4(LOLO) |
| `pipeline_mlaad.py` | MLAAD + M-AILABS (55 generators) | SQ1 + SQ2 + LOLO + LOGO + SQ4(LOLO+LOGO) |

---

## Diagnostic Stages

| Stage | Method | Description |
|-------|--------|-------------|
| **SQ1** | Hellinger distance + Scheirer–Ray–Hare η² | Characterises the global organisation of the acoustic representation space |
| **SQ2** | Mann–Whitney U + Kruskal–Wallis H (BH-FDR) | Evaluates statistical separability and cross-lingual distributional stability |
| **SQ3** | LOLO + LOGO leave-one-out protocols | Measures operational generalisation under unseen languages and generators |
| **SQ4** | SHAP (TreeExplainer, stratified TP/FP/FN/TN) | Identifies the acoustic drivers of each error class |

---

## Descriptor Families — 155 Total Dimensions

| Dimension | Family | d | Acoustic Role |
|-----------|--------|---|---------------|
| Spectral/Cepstral | MFCC | 40 | Vocal tract envelope (mel) |
| Spectral/Cepstral | LFCC | 40 | Linear-scale spectral envelope |
| Spectral/Cepstral | CQCC | 40 | Multi-resolution cepstral |
| Phase-Based | GD | 7 | Phase variation across frequencies |
| Phase-Based | IF | 8 | Instantaneous frequency |
| Phase-Based | PD | 6 | Inter-frame phase distortion |
| Phonation | CPP | 3 | Glottal periodicity |
| Phonation | Jitter | 6 | Pitch micro-variations F0 ∈ [60, 500] Hz |
| Phonation | Shimmer | 5 | Amplitude micro-variations |

> **Implementation note.** MFCC, LFCC, and CQCC are computed as 20 coefficients × (temporal mean + standard deviation) = 40 utterance-level features per family, after CMVN normalisation.

---

## Corpora

### GGMDDC
- **Reference:** Purohit et al. (2024), *APSIPA ASC*
- **Structure:** 80,000 utterances — 40,000 genuine + 40,000 synthetic
- **Languages (10):** Arabic, English, French, Hindi, Mandarin, Portuguese, Russian, Sanskrit, Spanish, Vietnamese
- **Generator:** HiFi-GAN (single generator — controlled setting)
- **Genuine speech:** VoxLingua107
- **Download:** [TODO — insert URL/DOI]

### MLAAD v9 + M-AILABS
- **Reference:** Müller et al. (2024), *ICASSP*
- **Structure:** ~30,764 utterances (restricted to 8 languages common to both datasets)
- **Languages (8):** German (DE), English (EN), Spanish (ES), French (FR), Italian (IT), Polish (PL), Russian (RU), Ukrainian (UK)
- **Generators:** 55 TTS systems (generators with < 50 utterances/language excluded)
- **Genuine speech:** M-AILABS — Solak & Naumov (2017)
  - Download: https://github.com/i-celeste-aurora/m-ailabs-dataset
- **MLAAD download:** [TODO — insert URL/DOI]

---

## Installation

```bash
git clone https://github.com/[TODO]/deepfake-acoustic-geometry
cd deepfake-acoustic-geometry
pip install -r requirements.txt
```

For SQ4 SHAP mechanism attribution (optional but recommended):
```bash
pip install shap>=0.44.0
```

---

## Usage

### Script 1 — GGMDDC (controlled baseline)

```bash
python pipeline_ggmddc.py --root /path/to/GGMDDC
```

**With pre-extracted features (skips audio extraction):**
```bash
python pipeline_ggmddc.py --features /path/to/features_ggmddc.pkl
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Path to GGMDDC dataset root | `` |
| `--output` | Output directory | `results_ggmddc/` |
| `--features` | Pre-extracted features `.pkl` | `` |

**Expected dataset structure:**
```
GGMDDC/
├── arabic/
│   ├── real/    *.wav
│   └── fake/    *.wav
├── english/
│   ├── real/
│   └── fake/
...
```

---

### Script 2 — MLAAD + M-AILABS (multi-generator)

```bash
python pipeline_mlaad.py \
    --root         /path/to/MLAAD \
    --mailabs_root /path/to/M-AILABS
```

**With pre-extracted features:**
```bash
python pipeline_mlaad.py --features /path/to/features_mlaad.pkl
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Path to MLAAD dataset root | `` |
| `--mailabs_root` | Path to M-AILABS genuine speech | `` |
| `--output` | Output directory | `results_mlaad/` |
| `--features` | Pre-extracted features `.pkl` | `` |

**Expected dataset structure:**
```
MLAAD/
├── de/
│   ├── tacotron2-DDC/    *.wav
│   ├── vits/             *.wav
│   └── ...
├── en/
│   └── ...

M-AILABS/
├── de_DE/  *.wav
├── en_US/  *.wav
...
```

---

## Output Files

### GGMDDC — `results_ggmddc/tables/`

| File | Content | Paper table |
|------|---------|-------------|
| `SQ1_representation_space.csv` | Hellinger H̄, CV, Cohen's d, η²_lang, η²_fake, η²_real, profile | Table 3 |
| `SQ2_invariance.csv` | Mann–Whitney \|r^b\|, Kruskal–Wallis p-values (BH-corrected) | Table 5 |
| `SQ3_lolo_ggmddc.csv` | LOLO AUC/EER per family × language fold (RF + LR) | Table 6 |
| `SQ4_shap_lolo.csv` | Mean \|SHAP\| per family × prediction outcome | Table 9 |
| `SQ4_delta_shap_lolo.csv` | Δ_SHAP = φ_error − φ_correct per family | Table 9 |

### MLAAD — `results_mlaad/tables/`

| File | Content | Paper table |
|------|---------|-------------|
| `SQ1_representation_space.csv` | Hellinger H̄, CV, d, η²_lang, η²_gen, ratio gen/lang | Tables 3–4 |
| `SQ2_invariance.csv` | Mann–Whitney \|r^b\|, Kruskal–Wallis p-values (BH-corrected) | Table 5 |
| `SQ3_lolo_mlaad.csv` | LOLO AUC/EER per family × language fold | Table 6 |
| `SQ3_logo_mlaad.csv` | LOGO AUC/EER per family × generator fold (55 generators) | Table 7 |
| `SQ3_delta.csv` | Δ = AUC_LOGO − AUC_LOLO per family + generalisation profile | Table 8 |
| `SQ4_shap_lolo.csv` | SHAP LOLO stratified by TP/FP/FN/TN | Table 9 |
| `SQ4_shap_logo.csv` | SHAP LOGO stratified by TP/FP/FN/TN | Table 9 |
| `SQ4_delta_shap_lolo.csv` | Δ_SHAP LOLO — Mechanisms A (German) and C (Mandarin) | Table 9 |
| `SQ4_delta_shap_logo.csv` | Δ_SHAP LOGO — Mechanism B (griffin_lim) | Table 9 |

---

## Experimental Configuration

| Component | Setting |
|-----------|---------|
| Sampling rate | 16 kHz |
| Signal duration | 4.0 s (max) |
| Primary classifier | Random Forest — 150 trees, max depth 12, max_features=sqrt |
| Isolation classifier | Logistic Regression — L2, C=1.0 |
| Class weighting | balanced (all models) |
| Random seeds | {0, 42, 123, 456, 1337} |
| Bootstrap CI | 95%, B = 2,000 |
| Statistical correction | Benjamini–Hochberg FDR (α = 0.05) |
| SHAP sample size | ≤ 300 utterances/fold (stratified) |
| Min. generator utterances | 50 per language (MLAAD) |

---

## Key Implementation Details

### Preprocessing — no data leakage
All preprocessing steps (NaN imputation, log-transform, z-score normalisation) are fitted exclusively on the training fold and applied to the test fold without any information from the test set.

```
Train fold only:
  1. Median imputation (per class)
  2. Log-transform (skewed features, threshold = 2.0)
  3. Z-score normalisation (μ=0, σ=1)
  → Transform applied to test fold using train statistics
```

### LOLO vs LOGO protocols

| Protocol | Applied to | Folds | Purpose |
|----------|-----------|-------|---------|
| LOLO (Leave-One-Language-Out) | Both corpora | 10 (GGMDDC) / 8 (MLAAD) | Language generalisation |
| LOGO (Leave-One-Generator-Out) | MLAAD only | 55 | Generator generalisation |

### Δ = AUC_LOGO − AUC_LOLO
A positive Δ indicates that generator-related variability is better absorbed than language-related variability. Values close to zero reflect comparable robustness in both dimensions.

### SQ2 projection
Mann–Whitney and Kruskal–Wallis tests are applied to the first principal component (PCA, unsupervised) of each descriptor family to obtain a scalar representation without label leakage.

### SHAP compatibility
SHAP values are extracted robustly across different versions of the `shap` and `scikit-learn` libraries:
```python
if isinstance(sv_raw, list):
    sv = np.abs(sv_raw[1])        # list: [class0, class1]
elif sv_raw.ndim == 3:
    sv = np.abs(sv_raw[:, :, 1])  # 3D: (samples, features, classes)
else:
    sv = np.abs(sv_raw)           # 2D: class-1 SHAP values
```

### t-DCF
The tandem detection cost function reported is a simplified approximation of the ASVspoof normalised min-tDCF and should be interpreted accordingly.

---

## Checkpointing

Both scripts checkpoint intermediate results automatically. If execution is interrupted (e.g. on Colab or a cluster), simply rerun the same command — the pipeline resumes from the last completed step.

Checkpoint files are saved in `{output}/checkpoints/`:
- `audio_files_{corpus}.pkl` — scanned file list
- `extraction_{corpus}.pkl` — partially extracted features
- `lolo_{corpus}.pkl` — partial LOLO results
- `logo_mlaad.pkl` — partial LOGO results

---

## Reproducing Paper Tables

| Paper table | Script | Output file |
|-------------|--------|-------------|
| Table 3 (Hellinger) | Both | `SQ1_representation_space.csv` |
| Table 4 (η² decomposition) | Both | `SQ1_representation_space.csv` |
| Table 5 (Invariance) | Both | `SQ2_invariance.csv` |
| Table 6 (LOLO AUC) | Both | `SQ3_lolo_{corpus}.csv` |
| Table 7 (LOGO AUC) | `pipeline_mlaad.py` | `SQ3_logo_mlaad.csv` |
| Table 8 (Δ table) | `pipeline_mlaad.py` | `SQ3_delta.csv` |
| Table 9 (SHAP) | Both | `SQ4_delta_shap_{protocol}.csv` |

---

## Citation

```bibtex
@article{chakir2025generalisation,
  title   = {Understanding Generalisation in Multilingual Deepfake Speech
             Detection through Acoustic Space Geometry},
  author  = {Chakir, Khadija and Gahi, Youssef and El-Khatib, Khalil},
  journal = {EURASIP Journal on Audio, Speech, and Music Processing},
  year    = {2025},
  doi     = {[TODO]}
}
```

---

## Repository Structure

```
deepfake-acoustic-geometry/
│
├── pipeline_ggmddc.py     # GGMDDC pipeline — SQ1 + SQ2 + LOLO + SQ4
├── pipeline_mlaad.py      # MLAAD pipeline  — SQ1 + SQ2 + LOLO + LOGO + SQ4
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## License

[TODO — specify license, e.g. MIT or Apache 2.0]

---

## Contact

**Khadija Chakir** — khadija.chakir5@uit.ac.ma  
**Youssef Gahi** — youssef.gahi@uit.ac.ma  
Laboratoire Sciences de l'Ingénieur, École Nationale des Sciences Appliquées,  
Université Ibn Tofail, Kénitra, Morocco.

**Khalil El-Khatib** — Khalil.El-Khatib@ontariotechu.ca  
Institute for Cyber Security and Resilient Systems,  
Faculty of Business and Information Technology,  
Ontario Tech University, Oshawa, ON, Canada.
