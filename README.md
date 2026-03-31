# WiDS Wildfire Prediction Challenge

## Problem
Predict whether a wildfire will reach critical infrastructure or populated areas within 12-72 hours. This is a survival analysis and classification challenge using spatiotemporal wildfire characteristics to enable timely evacuation and resource allocation.

## Approach
The solution uses a feature engineering pipeline followed by an ensemble of tree-based and linear models with Cox Proportional Hazards risk scores as meta-features. The workflow consists of:

1. **Feature Engineering**: Distance metrics (closing speed, kinematic predictions), growth features (area expansion rates), threat scores (alignment and speed interactions), and temporal indicators
2. **Feature Selection**: Permutation importance analysis to identify weak features, reducing dimensionality from 50+ to 24 key predictors
3. **Cox Model Integration**: Survival analysis using Cox PH to generate risk scores as additional ensemble input
4. **Multi-Horizon Targets**: Separate models trained for 12h, 24h, 48h, and 72h horizons with weighted predictions

## Model
**Ensemble with 5 base learners:**
- HistGradientBoostingClassifier (35% weight) - Fast gradient boosting with calibration
- RandomForestClassifier (25% weight) - Robust non-linear patterns
- ExtraTreesClassifier (20% weight) - Reduced overfitting variance
- GradientBoostingClassifier (15% weight) - Flexible hyperparameter tuning
- LogisticRegression (5% weight) - Linear decision boundary

**Calibration & Stacking:**
- Isotonic calibration on HGB for probability calibration
- Repeated stratified 5-fold CV (5×3 repeats) for robust evaluation
- Weighted averaging of base learner predictions

## Results
- **CV Hybrid Score (weighted Brier + C-index)**: 0.965
- **C-Index**: 0.941 (concordance on duration ordering)
- **Brier Score**: 0.007 (calibration quality)
- **Multi-horizon coverage**: 12h, 24h, 48h, 72h predictions

## Files
- **final_model.py** - Main training pipeline with feature engineering, Cox integration, cross-validation, and ensemble
- **feature_selection.py** - Importance analysis using permutation importance on RandomForest (identifies weak features to drop)
- **tuning.py** - Hyperparameter optimization and model selection utilities
- **submission_final.csv** - Final predictions on test set (format: event_id, hit_within_12h, hit_within_24h, etc.)
- **metaData.csv** - Feature metadata and data dictionary
- **train.csv** - Training dataset (221 samples)
- **test.csv** - Test dataset (95 samples)

## How to Run

**Setup:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn lifelines
```

**Run Feature Selection:**
```bash
python feature_selection.py
```
Generates importance rankings and identifies weak features for removal.

**Train Final Model:**
```bash
python final_model.py
```
Trains ensemble with repeated CV, generates test predictions in `submission_final.csv`.

**Tune Hyperparameters:**
```bash
python tuning.py
```
Runs grid/random search on model hyperparameters.

## Key Insights
- **Distance features dominate**: `dist_min_ci_0_5h` and `log_dist_min` are top predictors
- **Weak features removed**: 29 features with <0.5% importance dropped (kinematic_hit_*, closing_ratio, etc.)
- **Cox risk score helps**: Survival analysis risk incorporates duration information missed by pure classifiers
- **Ensemble stability**: 15 CV folds show consistent 0.96+ score across all repeats
