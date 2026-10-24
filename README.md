# ROGII Wellbore Geology Prediction

**A Computational Intelligence Pipeline for True Vertical Thickness Prediction in Horizontal Wellbores**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-4.7-02A28F">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Kaggle public RMSE" src="https://img.shields.io/badge/Kaggle%20Public-15.077%20ft%20RMSE-20BEFF?logo=kaggle&logoColor=white">
  <img alt="Kaggle private RMSE" src="https://img.shields.io/badge/Kaggle%20Private-13.717%20ft%20RMSE-20BEFF?logo=kaggle&logoColor=white">
</p>

> CIS 6005 — Computational Intelligence · WRIT1 Deep Learning Plus AI Mini Project
> Cardiff Metropolitan University
>
> Competition: [ROGII - Wellbore Geology Prediction on Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

---

## The problem

While geosteering a horizontal well, a geologist needs to know the **True Vertical
Thickness (TVT)** — the well's stratigraphic position within the target formation —
in order to steer the bit. Traditionally this is done by manually matching the
lateral's Gamma Ray (GR) signature against a vertical reference well ("typewell").

In this competition the TVT for the **toe-end of each lateral is masked**. Given the
well's trajectory, its partial GR log, and its paired typewell, the task is to
predict TVT for every masked row. Scoring is RMSE in feet.

The dataset is real and messy: 773 training wells, ~43% GR sensor dropout, and
wells at wildly different absolute depths.

---

## Results

| Model | CI paradigm | Mean CV RMSE (ft) | Std dev | Fold scores |
| --- | --- | --- | --- | --- |
| Ridge Regression (L2) | Statistical baseline | 15.57 | 0.68 | 15.12, 15.36, 14.79, 15.84, 16.74 |
| **LightGBM Ensemble** | Gradient boosting | **15.12** ← best mean | 0.67 | 14.48, 15.12, 14.38, 15.39, 16.21 |
| PyTorch MLP | Neural network | 15.30 | **0.61** ← most stable | 14.82, 15.45, 14.48, 15.47, 16.26 |

*5-fold `GroupKFold`, all 773 wells (3,783,989 rows), identical folds across all three
models. Source: [`outputs/results_full_comparison.json`](outputs/results_full_comparison.json).*

**Kaggle scores (LightGBM final submission):**

| Leaderboard | RMSE (ft) |
| --- | --- |
| Public (while the competition was open) | 15.077 (public rank 4,859) |
| Private (hidden test data, final score) | 13.717 |

The public score is within **0.3%** of the local CV estimate of 15.12 ft. That
agreement between an independently-scored hidden test set and the local estimate is
the strongest evidence in the project that the model generalises rather than being
overfit to the validation scheme. The private score (13.717 ft) is lower still, which
points the same way.

The honest reading of this table: all three models converge into a narrow
15.1–15.6 ft band despite being structurally very different learning algorithms.
LightGBM beats Ridge by only 2.9%. **Feature engineering, not model architecture,
is doing most of the work here.**

> ℹ️ The MLP is not seeded (`torch.manual_seed` is never called), so its numbers
> shift slightly between runs. The written report quotes 15.36 ± 0.59 ft from an
> earlier run; the table above matches the committed JSON.
> Planned fix: set a fixed seed and re-run `evaluate_models.py`.

---

## How it works

### PS anchoring — the core design decision

The models **never predict absolute TVT.** A naive approach would regress absolute
TVT against absolute MD/Z, which implicitly assumes every well sits in a flat,
identically-datumed formation — the EDA showed that assumption is false across 773
wells.

Instead, `features.py` locates each well's **Prediction-Start (PS) point** — the last
row where `TVT_input` is known — and broadcasts that row's state onto every
subsequent row as a per-well reference frame. The model predicts only
`dTVT = TVT − ps_TVT`, the *offset* from that anchor, so one trained model
generalises across wells at completely different absolute depths.

At inference, `ps_TVT` is added back to recover absolute TVT. **This reversal step is
essential to every number the pipeline reports.**

### The five features

| Feature | Definition | Importance (Gain) |
| --- | --- | --- |
| `inclination_proxy` | `delta_Z / delta_MD` — normalised local dip-rate | **2.04 × 10⁸** |
| `GR` | Gamma Ray (~43% dropout, mean-imputed per well) | 7.48 × 10⁷ |
| `delta_Z` | Elevation change since the PS anchor | 5.16 × 10⁷ |
| `delta_MD` | Measured depth travelled since the PS anchor | 3.46 × 10⁷ |
| `dist_XY` | Horizontal step-out from the PS anchor | 2.84 × 10⁷ |

`inclination_proxy` dominates — and it was *not* the feature initially hypothesised.
It is physically coherent in hindsight: two wells with identical `delta_Z` but very
different `delta_MD` are in completely different structural situations, and the ratio
is what distinguishes them.

### Avoiding data leakage

A single well contributes thousands of highly correlated, closely-spaced rows. A
random row split would let the model interpolate between near-duplicate neighbours
and report an artificially low error.

The fix is **5-fold `GroupKFold` grouped by `well_id`** — every row of a given well
sits entirely inside either the training or the validation fold. The same split
object is reused across all three models, which is what makes the comparison fair.

### One shared feature module

`features.py` is the single source of truth for the pipeline, imported by the
notebooks, all four scripts, and the Streamlit app. Because the transformation is
never reimplemented per-consumer, inference-time features cannot silently drift from
training-time features (*training-serving skew*).

The one deliberate exception is `kaggle/kaggle_submission_notebook.py`, which
duplicates the logic because Kaggle code competitions run in an isolated environment
with no access to this repository. Its header documents exactly where it diverges.

---

## Project structure

```
ROGII/
├── src/
│   ├── config.py                       Paths, feature list, model hyperparameters
│   ├── data_loader.py                  CSV I/O, lateral ↔ typewell pairing, integrity checks
│   ├── features.py                     PS anchoring and trajectory deltas
│   ├── train_model.py                  Train the production LightGBM
│   ├── evaluate_models.py              Ridge vs LightGBM vs MLP on shared folds
│   ├── error_analysis.py               Per-well error breakdown
│   └── generate_submission.py          Test-set inference → submission.csv
├── app/
│   └── app.py                          Streamlit demonstration app
├── notebooks/
│   ├── 01_EDA.ipynb                    Exploratory data analysis
│   ├── 02_Baseline_Model.ipynb         Ridge regression
│   ├── 03_Ensemble_Model.ipynb         LightGBM
│   └── 04_Deep_Learning_Model.ipynb    PyTorch MLP
├── kaggle/
│   └── kaggle_submission_notebook.py   Self-contained competition script
├── models/
│   ├── lgbm_model.joblib               Served by the Streamlit app
│   └── lgbm_model_full.joblib          Retrained after CV; used for error analysis
├── outputs/
│   ├── submission.csv                  14,151 predicted rows
│   ├── results_full_comparison.json    Per-fold RMSE for all three models
│   └── per_well_error_analysis.csv     Per-well RMSE across all 773 wells
├── data/                               Competition data — not committed, see Setup
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

The dataset (~1.2 GB) is gitignored. Download it from the Kaggle competition and
place it so the structure is:

```
data/
├── train/    {well_id}__horizontal_well.csv + {well_id}__typewell.csv   (773 pairs)
├── test/     {well_id}__horizontal_well.csv + {well_id}__typewell.csv
└── sample_submission.csv
```

---

## Usage

Every script resolves its own paths from `config.py`, so it can be run from anywhere
in the repository.

Train the production LightGBM model on all 773 wells:

```bash
python src/train_model.py
```

Reproduce the full three-model comparison (takes several minutes):

```bash
python src/evaluate_models.py
```

Score the model per-well and test what drives error:

```bash
python src/error_analysis.py
```

Generate a Kaggle submission from the test split:

```bash
python src/generate_submission.py
```

Launch the interactive demo:

```bash
streamlit run app/app.py
```

The app takes a lateral CSV and its paired typewell CSV, locates the PS boundary,
predicts TVT for every masked row, and plots known TVT merging into predicted TVT.
The smooth, continuous transition at the PS boundary is a quick sanity check that
the anchoring is working correctly.

---

## What the error analysis shows

Scoring each of the 773 wells independently gives a mean per-well RMSE of
**10.92 ft** (std 6.66, range 1.41–66.54).

| Tested relationship | Correlation |
| --- | --- |
| Error vs post-PS dip variability | r = 0.20 (weak-to-moderate) |
| Error vs mean typewell offset distance | r = 0.07 (negligible) |

This is a mixed, honest result. The dip-variability correlation gives *some* support
to the idea that erratically-steered wells are harder to predict, but several of the
ten worst wells (RMSE 33–67 ft) show no unusual dip variability. **A meaningful share
of the largest errors remains unexplained by the current five features.**

---

## Limitations

- **No uncertainty quantification.** The model outputs a single deterministic value
  per row, with no confidence interval. This is the single biggest missed
  opportunity: it is precisely the capability where neural approaches have a
  demonstrated edge in this domain, and the MLP here does not exploit it.
- **No live, adaptive correlation.** Inference is static and one-shot. The system
  cannot re-correlate against the typewell as new lateral data streams in, so it
  cannot react to an unexpected formation change mid-lateral.
- **No explicit GR-to-GR correlation.** The EDA showed lateral and typewell GR curves
  share matching peaks and troughs on a common TVT axis — strong evidence the signal
  is usable — but the current feature set does not yet exploit it directly.
- **Untested cross-field generalisation.** All evaluation comes from one competition
  dataset and field distribution. Performance on a different geological basin is
  untested and plausibly weaker.

### Directions for improvement

1. Add a Mixture Density Network output head so the app reports a confidence range
   per prediction rather than a single number.
2. Add a sliding-window normalised cross-correlation or lightweight DTW pass between
   the lateral's known GR and the typewell's GR, converting the system from a static
   extrapolator into an adaptive correlator.
3. Engineer features that capture geological discontinuity directly — rate-of-change
   of GR, or a formation-boundary-crossing count derived from the typewell.
4. Seed the MLP for reproducible results, and add unit tests for `features.py`
   (for example, a check that `ps_TVT` is added back correctly at inference).

---

## Tech stack

**LightGBM** (production model) · **PyTorch** (deep-learning comparator) ·
**scikit-learn** (Ridge baseline, `GroupKFold`, metrics) · **pandas** / **NumPy** ·
**Streamlit** (demo app) · **matplotlib** / **seaborn**

Exact versions are pinned in [`requirements.txt`](requirements.txt).

---

## Notes

- `data/` and `venv/` are gitignored for size. `models/*.joblib` are tracked
  deliberately — they are under 1 MB each and let the Streamlit demo run
  immediately after cloning.
- Two models are kept by design: `train_model.py` produces `lgbm_model.joblib`
  (served by the app), and `evaluate_models.py` produces `lgbm_model_full.joblib`
  after cross-validation (used for the feature-importance and error analysis).

## License

No license yet. Add a `LICENSE` file (for example MIT) if you want others to reuse the code.
