import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def engineer_features(df):
    df = df.copy()
    
    # === DISTANCE FEATURES ===
    df['closing_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['est_time_to_hit'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0.1))
    df['est_time_to_hit'] = df['est_time_to_hit'].clip(upper=300)
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    for h in [12, 24, 48, 72]:
        df[f'kinematic_hit_{h}h'] = (df['est_time_to_hit'] <= h).astype(int)
    
    # === GROWTH FEATURES ===
    df['size_growth_interaction'] = df['log1p_area_first'] * df['area_growth_rate_ha_per_h']
    df['log_radial_growth'] = np.log1p(df['radial_growth_m'])
    df['size_distance_ratio'] = df['log1p_area_first'] / (df['dist_min_ci_0_5h'] + 1)
    
    # === THREAT FEATURES ===
    df['threat_score'] = df['alignment_abs'] * df['centroid_speed_m_per_h']
    df['aligned_closing'] = df['alignment_abs'] * df['closing_speed_m_per_h'].clip(lower=0)
    df['direct_threat'] = df['along_track_speed'] * df['alignment_abs']
    
    # === DATA QUALITY ===
    df['data_quality_score'] = df['dt_first_last_0_5h'] * (1 - df['low_temporal_resolution_0_5h'])
    
    # === DISTANCE x THREAT ===
    df['dist_x_alignment'] = df['dist_min_ci_0_5h'] * (1 - df['alignment_abs'])
    df['threat_per_distance'] = df['threat_score'] / (df['dist_min_ci_0_5h'] + 1)
    
    # === TIME FEATURES ===
    df['is_afternoon'] = ((df['event_start_hour'] >= 12) & 
                          (df['event_start_hour'] <= 20)).astype(int)
    df['is_summer'] = df['event_start_month'].isin([6, 7, 8, 9]).astype(int)
    
    # Drop weak features identified by importance analysis
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

def add_cox_feature(train_df, test_df, features):
    # Train Cox on training data
    cox_cols = features + ['time_to_hit_hours', 'event']
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df[cox_cols], 
            duration_col='time_to_hit_hours',
            event_col='event')
    
    # Predict risk score for both train and test
    train_df['cox_risk'] = cph.predict_partial_hazard(train_df[cox_cols])
    test_df['cox_risk']  = cph.predict_partial_hazard(test_df[features])
    
    return train_df, test_df

train = engineer_features(train)
test = engineer_features(test)

FEATURES = [c for c in train.columns 
            if c not in ['event_id', 'event', 'time_to_hit_hours']]

train, test = add_cox_feature(train, test, FEATURES)

# Add cox_risk to features
FEATURES = [c for c in train.columns
            if c not in ['event_id', 'event', 'time_to_hit_hours']]

print("Features after Cox addition:", len(FEATURES))
print("Train shape:", train[FEATURES].shape)
# ── Horizon labels ──────────────────────────────────────────────
HORIZONS = [12, 24, 48, 72]
WEIGHTS =  [0.1, 0.2, 0.3, 0.4]

for h in HORIZONS:
    train[f'hit_within_{h}h'] = (
        (train['event'] == 1) &
        (train['time_to_hit_hours'] <= h)
    ).astype(int)

X_train = train[FEATURES]
X_test  = test[FEATURES]

# ── Five models to ensemble ──────────────────────────────────────
def get_models():
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
        ]))
    ]

# ── Model weights based on known performance ──────────────────────
MODEL_WEIGHTS = {
    'HGB': 0.35,
    'RF':  0.25,
    'ET':  0.20,
    'GB':  0.15,
    'LR':  0.05
}

def get_weighted_pred(models, X):
    pred = np.zeros(len(X))
    for name, model in models:
        pred += MODEL_WEIGHTS[name] * model.predict_proba(X)[:, 1]
    return pred

# ── Repeated CV - 5 folds × 3 repeats = 15 evaluations ─────────────
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
hybrid_scores = []

print("\n=== Repeated CV Evaluation (5x3) ===")
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, train['event'])):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]

    weighted_brier = 0
    fold_risk_scores = np.zeros(len(val_idx))

    for h, w in zip(HORIZONS, WEIGHTS):
        y_tr  = train[f'hit_within_{h}h'].iloc[tr_idx]
        y_val = train[f'hit_within_{h}h'].iloc[val_idx]

        trained_models = []
        for name, model in get_models():
            model.fit(X_tr, y_tr)
            trained_models.append((name, model))
        fold_preds = get_weighted_pred(trained_models, X_val)

        bs = np.mean((fold_preds - y_val.values) ** 2)
        weighted_brier += w * bs

        if h == 72:
            fold_risk_scores = fold_preds

    y_val_time  = train['time_to_hit_hours'].iloc[val_idx].values
    y_val_event = train['event'].iloc[val_idx].values
    ci = concordance_index(y_val_time, -fold_risk_scores, y_val_event)

    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)
    hybrid_scores.append(hybrid)
    print(f"  Fold {fold+1:2d} | C-index: {ci:.3f} | Brier: {weighted_brier:.3f} | Hybrid: {hybrid:.3f}")

print(f"\nMean Hybrid Score: {np.mean(hybrid_scores):.3f}")
print(f"Std Hybrid Score:  {np.std(hybrid_scores):.3f}")
# ── Generate final submission ─────────────────────────────────────
print("\n=== Generating Final Submission ===")

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
final_preds = {col: np.zeros(len(X_test)) for col in prob_cols}

for h, col in zip(HORIZONS, prob_cols):
    y_tr = train[f'hit_within_{h}h']
    
    trained_models = []
    for name, model in get_models():
        model.fit(X_train, y_tr)
        trained_models.append((name, model))
    final_preds[col] = get_weighted_pred(trained_models, X_test)

# Stack → monotonicity → clip
pred_array = np.column_stack([final_preds[c] for c in prob_cols])
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)

# Build and save
submission = pd.DataFrame({'event_id': test['event_id']})
for i, col in enumerate(prob_cols):
    submission[col] = pred_array[:, i]

submission.to_csv("submission_final.csv", index=False)
print(submission.head(10))
print("\nShape:", submission.shape)
print("\nProbability ranges:")
for col in prob_cols:
    print(f"  {col}: min={submission[col].min():.3f}, max={submission[col].max():.3f}")
print("\n✓ submission_final.csv saved!")