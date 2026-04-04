import pandas as pd
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier,
                               ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sksurv.ensemble import RandomSurvivalForest
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from lifelines.utils import concordance_index
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def engineer_features(df, add_exp1_features=False):
    df = df.copy()
    df['closing_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['est_time_to_hit'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0.1))
    df['est_time_to_hit'] = df['est_time_to_hit'].clip(upper=300)
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    for h in [12, 24, 48, 72]:
        df[f'kinematic_hit_{h}h'] = (df['est_time_to_hit'] <= h).astype(int)
    df['size_growth_interaction'] = df['log1p_area_first'] * df['area_growth_rate_ha_per_h']
    df['log_radial_growth'] = np.log1p(df['radial_growth_m'])
    df['size_distance_ratio'] = df['log1p_area_first'] / (df['dist_min_ci_0_5h'] + 1)
    df['threat_score'] = df['alignment_abs'] * df['centroid_speed_m_per_h']
    df['aligned_closing'] = df['alignment_abs'] * df['closing_speed_m_per_h'].clip(lower=0)
    df['direct_threat'] = df['along_track_speed'] * df['alignment_abs']
    df['data_quality_score'] = df['dt_first_last_0_5h'] * (1 - df['low_temporal_resolution_0_5h'])
    df['dist_x_alignment'] = df['dist_min_ci_0_5h'] * (1 - df['alignment_abs'])
    df['threat_per_distance'] = df['threat_score'] / (df['dist_min_ci_0_5h'] + 1)
    df['is_afternoon'] = ((df['event_start_hour'] >= 12) &
                          (df['event_start_hour'] <= 20)).astype(int)
    df['is_summer'] = df['event_start_month'].isin([6, 7, 8, 9]).astype(int)
    # Square of distance — penalizes very far fires more
    df['dist_squared'] = df['dist_min_ci_0_5h'] ** 2 / 1e10

# Growth × alignment interaction
    df['growth_alignment'] = df['area_growth_rate_ha_per_h'] * df['alignment_abs']

    # Distance zones — is fire already dangerously close?
    df['is_very_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)
    df['is_close'] = (df['dist_min_ci_0_5h'] < 15000).astype(int)

    # Rank of distance among all fires — relative threat
    df['dist_rank'] = df['dist_min_ci_0_5h'].rank(pct=True)

    # Fire size × alignment — big fire pointing at zone
    df['size_x_alignment'] = df['log1p_area_first'] * df['alignment_abs']

    # Distance × data quality — confident measurement of close fire
    df['confident_close'] = df['data_quality_score'] / (df['dist_min_ci_0_5h'] + 1)

    # Square root of distance — different scale sensitivity
    df['sqrt_dist'] = np.sqrt(df['dist_min_ci_0_5h'])

    # Log transform of key ratios
    df['log_closing_ratio'] = np.log1p(
        df['closing_speed_m_per_h'].clip(lower=0) / 
        (df['dist_min_ci_0_5h'] + 1)
    )

    # Fire perimeter density
    df['perimeter_density'] = (
        df['num_perimeters_0_5h'] / 
        (df['dt_first_last_0_5h'] + 0.1)
    )

    # Distance × bearing alignment
    df['dist_bearing_score'] = (
        df['spread_bearing_cos'] * 
        df['alignment_abs'] /
        (df['log_dist_min'] + 1)
    )

    # EXPERIMENT 1: Add 3 new features BEFORE WEAK_FEATURES drop
    if add_exp1_features:
        df['bearing_consistency'] = df['spread_bearing_cos'] * df['alignment_abs'] * df['num_perimeters_0_5h']
        df['dist_urgency'] = np.exp(-df['dist_min_ci_0_5h'] / 10000)
        df['quality_threat'] = df['threat_score'] * df['data_quality_score']

    WEAK_FEATURES = [
        'area_growth_abs_0_5h', 'radial_growth_m', 'centroid_speed_m_per_h',
        'size_growth_interaction', 'area_growth_rel_0_5h', 'spread_bearing_sin',
        'log1p_growth', 'cross_track_component', 'direct_threat',
        'radial_growth_rate_m_per_h', 'relative_growth_0_5h', 'log_area_ratio_0_5h',
        'dist_fit_r2_0_5h', 'along_track_speed', 'is_summer',
        'closing_speed_abs_m_per_h', 'dist_std_ci_0_5h', 'dist_change_ci_0_5h',
        'dist_accel_m_per_h2', 'closing_speed_m_per_h', 'closing_ratio',
        'aligned_closing', 'dist_slope_ci_0_5h', 'projected_advance_m',
        'est_time_to_hit', 'kinematic_hit_12h', 'kinematic_hit_24h',
        'kinematic_hit_48h', 'kinematic_hit_72h'
    ]
    df = df.drop(columns=[c for c in WEAK_FEATURES if c in df.columns])
    return df

# ── Level 1 base models ──────────────────────────────────────────
def get_base_models():
    return [
        ('HGB', CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.03,
                max_depth=5, min_samples_leaf=3,
                random_state=42),
            cv=3, method='isotonic')),
        ('RF', RandomForestClassifier(
            n_estimators=500, max_depth=6,
            min_samples_leaf=4, max_features='sqrt',
            random_state=42)),
        ('ET', ExtraTreesClassifier(
            n_estimators=500, max_depth=6,
            min_samples_leaf=4, max_features='sqrt',
            random_state=42)),
        ('GB', GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03,
            max_depth=3, min_samples_leaf=5,
            random_state=42)),
        ('LR', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=0.1, max_iter=1000, random_state=42))
        ])),
        ('XGB', XGBClassifier(
            n_estimators=500, learning_rate=0.03,
            max_depth=3, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )),
        ('LGB', LGBMClassifier(
            n_estimators=500, learning_rate=0.03,
            max_depth=4, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        ))
    ]

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

train = engineer_features(train)
test  = engineer_features(test)

HORIZONS = [12, 24, 48, 72]
WEIGHTS  = [0.1, 0.2, 0.3, 0.4]

for h in HORIZONS:
    train[f'hit_within_{h}h'] = (
        (train['event'] == 1) &
        (train['time_to_hit_hours'] <= h)
    ).astype(int)

FEATURES = [c for c in train.columns
            if c not in ['event_id', 'event', 'time_to_hit_hours']
            and 'hit_within' not in c]

X_train = train[FEATURES]
X_test  = test[FEATURES]

# ── Level 2 meta learner ─────────────────────────────────────────
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# ── Stacking CV ──────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

hybrid_scores = []
prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

print("=== Stacking CV Evaluation ===")

for fold, (tr_idx, val_idx) in enumerate(outer_cv.split(X_train, train['event'])):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]

    weighted_brier = 0
    fold_risk_scores = np.zeros(len(val_idx))

    for h, w, col in zip(HORIZONS, WEIGHTS, prob_cols):
        y_tr  = train[f'hit_within_{h}h'].iloc[tr_idx]
        y_val = train[f'hit_within_{h}h'].iloc[val_idx]

        # ── Level 1: generate OOF meta-features ──────────────────
        meta_train = np.zeros((len(tr_idx), len(get_base_models())))
        meta_val   = np.zeros((len(val_idx), len(get_base_models())))

        for i, (name, model) in enumerate(get_base_models()):
            # OOF predictions for meta training
            oof = np.zeros(len(tr_idx))
            for in_tr, in_val in inner_cv.split(X_tr, y_tr):
                clone_models = get_base_models()
                clone_models[i][1].fit(X_tr.iloc[in_tr], y_tr.iloc[in_tr])
                oof[in_val] = clone_models[i][1].predict_proba(
                    X_tr.iloc[in_val])[:, 1]
            meta_train[:, i] = oof

            # Full retrain on tr for val predictions
            model.fit(X_tr, y_tr)
            meta_val[:, i] = model.predict_proba(X_val)[:, 1]

        # ── Level 2: meta learner ─────────────────────────────────
        meta_model.fit(meta_train, y_tr)
        fold_preds = meta_model.predict_proba(meta_val)[:, 1]

        bs = np.mean((fold_preds - y_val.values) ** 2)
        weighted_brier += w * bs

        if h == 72:
            fold_risk_scores = fold_preds

    y_val_time  = train['time_to_hit_hours'].iloc[val_idx].values
    y_val_event = train['event'].iloc[val_idx].values
    ci = concordance_index(y_val_time, -fold_risk_scores, y_val_event)

    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)
    hybrid_scores.append(hybrid)
    print(f"  Fold {fold+1:2d} | C-index: {ci:.3f} | "
          f"Brier: {weighted_brier:.3f} | Hybrid: {hybrid:.3f}")

print(f"\nMean Hybrid Score: {np.mean(hybrid_scores):.3f}")
print(f"Std Hybrid Score:  {np.std(hybrid_scores):.3f}")

# ── Generate final submission ────────────────────────────────────
print("\n=== Generating Stacked Submission ===")

final_preds = {}

for h, col in zip(HORIZONS, prob_cols):
    y_tr = train[f'hit_within_{h}h']

    # Level 1: train all base models on full data
    meta_train = np.zeros((len(X_train), len(get_base_models())))
    meta_test  = np.zeros((len(X_test),  len(get_base_models())))

    for i, (name, model) in enumerate(get_base_models()):
        # OOF for meta training
        oof = np.zeros(len(X_train))
        for in_tr, in_val in inner_cv.split(X_train, y_tr):
            clone = get_base_models()[i][1]
            clone.fit(X_train.iloc[in_tr], y_tr.iloc[in_tr])
            oof[in_val] = clone.predict_proba(X_train.iloc[in_val])[:, 1]
        meta_train[:, i] = oof

        # Full retrain for test
        model.fit(X_train, y_tr)
        meta_test[:, i] = model.predict_proba(X_test)[:, 1]

    meta_model.fit(meta_train, y_tr)
    final_preds[col] = meta_model.predict_proba(meta_test)[:, 1]

pred_array = np.column_stack([final_preds[c] for c in prob_cols])
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)

submission = pd.DataFrame({'event_id': test['event_id']})
for i, col in enumerate(prob_cols):
    submission[col] = pred_array[:, i]

submission.to_csv("submission_stacked.csv", index=False)
print(submission.head(10))
print("\nShape:", submission.shape)
print("\nProbability ranges:")
for col in prob_cols:
    print(f"  {col}: min={submission[col].min():.3f}, "
          f"max={submission[col].max():.3f}")
print("\n✓ submission_stacked.csv saved!")