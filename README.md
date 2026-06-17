# WiDS Wildfire Prediction Challenge 2026

> Predict the probability that an active wildfire reaches within 5km of an evacuation zone at **12h, 24h, 48h, and 72h** horizons, using only the first 5 hours of fire perimeter data.

**Competition:** [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26) · Kaggle  
**Final public leaderboard score: 0.96373**

---

## Results

| Metric | Value |
|---|---|
| Public leaderboard score | **0.96373** |
| CV hybrid score (mean) | **0.973** |
| CV hybrid score (std) | ±0.008 |
| CV folds | 5-fold stratified |
| Competition | WiDS Global Datathon 2026 |

### Per-fold breakdown

| Fold | C-index | Brier | Hybrid |
|---|---|---|---|
| 1 | 0.959 | 0.012 | 0.979 |
| 2 | 0.954 | 0.003 | 0.984 |
| 3 | 0.937 | 0.019 | 0.968 |
| 4 | 0.934 | 0.010 | 0.973 |
| 5 | 0.940 | 0.028 | 0.963 |
| **Mean** | **0.945** | **0.014** | **0.973** |

---

## Problem statement

Given a snapshot of a wildfire event — its size, growth rate, speed, direction, and distance from the nearest structure — output four **calibrated, monotonically non-decreasing** probabilities:

| Output | Meaning |
|---|---|
| `prob_12h` | P(wildfire within 5km of evacuation zone within 12 hours) |
| `prob_24h` | P(wildfire within 5km of evacuation zone within 24 hours) |
| `prob_48h` | P(wildfire within 5km of evacuation zone within 48 hours) |
| `prob_72h` | P(wildfire within 5km of evacuation zone within 72 hours) |

The evaluation metric is a **hybrid score** combining the C-index and Brier loss:

```
Hybrid = 0.3 × C-index + 0.7 × (1 − weighted Brier)

Weighted Brier = 0.1×BS(12h) + 0.2×BS(24h) + 0.3×BS(48h) + 0.4×BS(72h)
```

---

## Approach

### Why survival analysis?

Standard classification treats each horizon (12h, 24h, etc.) as an independent binary problem and ignores the time-to-event structure. A wildfire impact is fundamentally a **time-to-event process** — the question is not just *if* it hits but *when*. Random Survival Forest directly models this hazard function, capturing censored observations (fires that hadn't hit yet at observation time) that classification would discard or mishandle.

### Model: Random Survival Forest

Algorithm: `RandomSurvivalForest` from `scikit-survival`

| Parameter | Value | Reasoning |
|---|---|---|
| `n_estimators` | 500 | Enough trees for stable survival function estimates |
| `max_depth` | 8 | Controls overfitting on a small dataset (221 training rows) |
| `min_samples_leaf` | 3 | Prevents leaf-level memorisation |
| `max_features` | 0.6 | Feature subsampling for diversity across trees |
| `n_jobs` | -1 | Parallelised across all cores |

### Synthetic data augmentation

The training set has only 221 observations. To stabilise the survival function at the 72h boundary, **10 synthetic observations** are added at `t = 73h` (just beyond the prediction horizon). This prevents the survival curve from collapsing to zero at the edge of observed time, which would produce unreliable 72h probability estimates.

```
Original train size:  221
Extended train size:  231  (+10 synthetic rows at 73h)
```

### Feature engineering

Domain-specific kinematic and threat features derived from raw fire perimeter data:

| Feature | Description |
|---|---|
| `closing_ratio` | Closing speed relative to current distance |
| `est_time_to_hit` | Distance / clipped closing speed (capped at 300h) |
| `log_dist_min` | Log-transformed minimum distance to structure |
| `kinematic_hit_{h}h` | Binary flag: estimated impact within h hours |
| `threat_score` | Fire alignment × centroid speed |
| `aligned_closing` | Alignment × positive closing speed |
| `size_distance_ratio` | Log fire area / distance |
| `data_quality_score` | Temporal coverage × resolution flag |
| `is_afternoon` | Flag for 12:00–20:00 window (peak fire behaviour) |

Weak features (permutation importance < 0.005 by cross-validated RF) are dropped, reducing 52 raw columns to **23 selected features**.

### Prediction pipeline

```
Raw features (52)
    ↓ engineer_features()
Engineered features (23)
    ↓ RandomSurvivalForest.fit()
Survival function S(t) per test row
    ↓ interpolate at t = 12, 24, 48, 72
Raw probabilities: 1 − S(t)
    ↓ monotonicity enforcement (np.maximum propagation)
    ↓ clip to [0.01, 0.99]
Final output: prob_12h, prob_24h, prob_48h, prob_72h
```

**Monotonicity enforcement** ensures `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h` — a physical constraint the scoring system requires (a fire cannot be less likely to hit at 72h than at 12h).

---

## Repository structure

```
Wildfire_Prediction/
├── survival_model.py        ← Final model (generates submission_rsf.csv)
├── survival_model_ensemble.py  ← 3-seed RSF ensemble variant
├── feature_selection.py     ← Cross-validated feature importance analysis
├── final_model.py           ← Earlier 5-model sklearn ensemble (baseline)
├── tuning.py                ← HGB hyperparameter search (GridSearchCV)
├── metaData.csv             ← Feature descriptions from competition
├── submission_rsf.csv       ← Best submission (public score: 0.96373)
├── time_distribution.png    ← EDA: event time distribution
├── top_feature_boxplot.png  ← EDA: top feature importances by class
└── .gitignore
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
scikit-survival>=0.21
lifelines>=0.27
```

---

## Running

```bash
# Feature importance analysis (optional)
python feature_selection.py

# Train final model, run 5-fold CV, generate submission
python survival_model.py
```

Output: `submission_rsf.csv` with columns `event_id, prob_12h, prob_24h, prob_48h, prob_72h`

Expected terminal output:
```
Original train size: 221
Extended train size: 231 (added 10 synthetic rows at 73h)

=== RSF CV (with synthetic 73h observations) ===
  Fold 1 | C-index: 0.959 | Brier: 0.012 | Hybrid: 0.979
  Fold 2 | C-index: 0.954 | Brier: 0.003 | Hybrid: 0.984
  Fold 3 | C-index: 0.937 | Brier: 0.019 | Hybrid: 0.968
  Fold 4 | C-index: 0.934 | Brier: 0.010 | Hybrid: 0.973
  Fold 5 | C-index: 0.940 | Brier: 0.028 | Hybrid: 0.963

Mean Hybrid Score: 0.973
Std:               0.008
```

---

## Key design decisions

**Survival analysis over classification** — The dataset has only 221 training rows and includes censored observations (fires still active at observation time). Treating each horizon as a separate binary classification problem would discard censoring information and produce poorly calibrated probabilities. RSF handles censored data natively and outputs a full survival curve, which is then queried at each horizon.

**Synthetic boundary observations** — Without observations near t = 72h, the RSF survival function has no data to anchor against at the prediction boundary, causing the curve to extrapolate unreliably. Adding 10 synthetic non-events at t = 73h gives the model a soft anchor without introducing real-label leakage.

**Monotonicity via post-hoc enforcement** — Rather than constraining the model during training (which complicates the survival function interpolation), monotonicity is enforced post-prediction using a simple forward `np.maximum` pass. This is both more flexible and more interpretable.

**Probability clipping to [0.01, 0.99]** — Hard 0 or 1 predictions are penalised heavily by the Brier score. Clipping to a narrow range ensures the model expresses appropriate uncertainty even for the most confident predictions.

---

## Technologies

`Python` · `scikit-survival` · `scikit-learn` · `pandas` · `numpy` · `lifelines` · Random Survival Forest · Survival analysis · Feature engineering · Cross-validation · Brier score · C-index
