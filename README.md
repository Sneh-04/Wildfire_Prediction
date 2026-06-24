# 🔥 Wildfire Spread Prediction — WiDS Global Datathon 2026

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![lifelines](https://img.shields.io/badge/lifelines-survival%20analysis-red)
![Competition](https://img.shields.io/badge/Kaggle-WiDS%20Datathon%202026-20BEFF?logo=kaggle&logoColor=white)
![Score](https://img.shields.io/badge/Public%20Leaderboard-0.96373-brightgreen)

> Survival analysis + ensemble ML to predict whether a wildfire will reach a critical infrastructure point within 12, 24, 48, and 72 hours — submitted as part of the **Women in Data Science (WiDS) Global Datathon 2026**.

---

## 📌 Problem Statement

Given real wildfire event data with fire kinematics, spatial proximity metrics, and temporal attributes, predict the **probability of a wildfire hitting a critical infrastructure point** at four time horizons: **12h, 24h, 48h, and 72h**.

The evaluation metric is a **hybrid score** combining:
- **C-index** (Harrell's concordance index, from survival analysis)
- **Weighted Brier Score** across all four time horizons

`Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`

---

## 🏆 Results

| Metric | Score |
|---|---|
| Public Leaderboard Score | **0.96373** |
| CV Mean Hybrid Score | **~0.973 ± 0.008** |
| CV Strategy | Repeated Stratified K-Fold (5 splits × 3 repeats) |

---

## 🧠 Approach

### 1. Feature Engineering
Crafted domain-relevant features from raw fire kinematics:

- **Distance features** — closing ratio, estimated time-to-hit, log-transformed distance
- **Growth features** — size-distance ratio, log radial growth, size-growth interaction
- **Threat scores** — alignment × speed composites, cross-track threat
- **Temporal features** — afternoon flag (peak burn hours), data quality score
- **Weak feature pruning** — dropped 28 low-importance features identified through permutation analysis

### 2. Survival Analysis Integration (Cox PH)
- Fitted a **CoxPHFitter** (penalizer=0.1) on training data
- Used the predicted **partial hazard (cox_risk)** as an additional feature for all downstream classifiers
- This bridges survival-analysis-style reasoning with classical ML

### 3. Ensemble of 5 Classifiers
Trained and blended predictions across four time-horizon labels using a **weighted ensemble**:

| Model | Weight |
|---|---|
| HistGradientBoosting (calibrated, isotonic) | 0.35 |
| Random Forest | 0.25 |
| Extra Trees | 0.20 |
| Gradient Boosting | 0.15 |
| Logistic Regression (scaled) | 0.05 |

### 4. Monotonicity Enforcement
Applied `np.maximum` across horizons to ensure:
`P(hit_12h) ≤ P(hit_24h) ≤ P(hit_48h) ≤ P(hit_72h)`

---

## 📁 Repository Structure

```
Wildfire_Prediction/
│
├── explore.py              # EDA — distributions, correlations, missing values
├── feature_selection.py    # Permutation importance & weak feature identification
├── model.py                # Baseline model experiments
├── tuning.py               # Hyperparameter tuning
├── final_model.py          # ✅ Full pipeline — feature engineering, Cox, ensemble, CV
├── submission.py           # Submission generation script
├── brier.py                # Brier score utility
├── c.index.py              # C-index evaluation utility
├── summary.py              # Results summary
├── lib.py                  # Shared utilities
│
├── train.csv               # Training data
├── test.csv                # Test data
├── metaData.csv            # Feature metadata
├── sample_submission.csv   # Kaggle submission format
├── submission_final.csv    # ✅ Best submission (score: 0.96373)
│
├── time_distribution.png   # EDA: time-to-hit distribution
└── top_feature_boxplot.png # EDA: top feature distributions by event
```

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **ML** — scikit-learn (HistGradientBoosting, RandomForest, ExtraTrees, GradientBoosting, LogisticRegression)
- **Survival Analysis** — `lifelines` (CoxPHFitter, concordance_index)
- **Evaluation** — Custom hybrid metric (C-index + Weighted Brier Score)

---

## ⚡ Quick Start

```bash
git clone https://github.com/Sneh-04/Wildfire_Prediction.git
cd Wildfire_Prediction
pip install pandas numpy scikit-learn lifelines
python final_model.py
```

> Requires `train.csv` and `test.csv` from the [WiDS Datathon 2026 Kaggle competition](https://www.kaggle.com/competitions/widsdatathon2026).

---

## 📊 EDA Highlights

| Visual | Insight |
|---|---|
| `time_distribution.png` | Most wildfire hits occur within 24–48h; long tails indicate difficult edge cases |
| `top_feature_boxplot.png` | `dist_min_ci_0_5h` and `alignment_abs` are the strongest separators between hit/no-hit |

---

## 🔑 Key Takeaways

- Survival analysis (Cox PH) meaningfully improved ranking ability on censored fire data
- Weak feature pruning (28 features dropped) reduced noise without hurting the C-index
- Probability calibration via isotonic regression on HGB improved Brier scores noticeably
- Monotonicity constraint was essential to meet competition submission requirements

---

## 👩‍💻 Author

**Sneha** — [GitHub: Sneh-04](https://github.com/Sneh-04)

*Submitted to the WiDS Global Datathon 2026 on Kaggle*
