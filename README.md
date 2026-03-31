# WiDS Wildfire Prediction Challenge

## Problem
Predict the probability that a wildfire reaches within 5km of an evacuation zone within 12h, 24h, 48h, and 72h using fire perimeter data from the first 5 hours.

## Approach
- Survival analysis framing with censored data
- Horizon-specific binary classification (4 targets)
- 5-model weighted ensemble (HGB + RF + ET + GB + LR)
- Permutation-based feature selection (52 → 23 features)
- Probability calibration + monotonicity correction
- Repeated 15-fold cross validation

## Results
- CV Hybrid Score: 0.965
- Metric: 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)

## Files
- `final_model.py` — main pipeline, ensemble, generates submission
- `feature_selection.py` — feature importance analysis
- `tuning.py` — hyperparameter search
- `submission_final.csv` — final predictions

## How to Run
pip install pandas numpy scikit-learn lifelines
python final_model.py
