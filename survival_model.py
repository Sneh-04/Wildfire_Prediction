import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

def engineer_features(df):
    df = df.copy()
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    df['size_distance_ratio'] = df['log1p_area_first'] / (df['dist_min_ci_0_5h'] + 1)
    df['dist_x_alignment'] = df['dist_min_ci_0_5h'] * (1 - df['alignment_abs'])
    df['threat_score'] = df['alignment_abs'] * df['centroid_speed_m_per_h']
    df['threat_per_distance'] = df['threat_score'] / (df['dist_min_ci_0_5h'] + 1)
    df['data_quality_score'] = df['dt_first_last_0_5h'] * (1 - df['low_temporal_resolution_0_5h'])
    df['is_afternoon'] = ((df['event_start_hour'] >= 12) &
                          (df['event_start_hour'] <= 20)).astype(int)
    df['log_closing_ratio'] = np.log1p(
        df['closing_speed_m_per_h'].clip(lower=0) /
        (df['dist_min_ci_0_5h'] + 1))
    df['perimeter_density'] = (
        df['num_perimeters_0_5h'] / (df['dt_first_last_0_5h'] + 0.1))
    df['dist_bearing_score'] = (
        df['spread_bearing_cos'] * df['alignment_abs'] /
        (df['log_dist_min'] + 1))
    df['dist_rank'] = df['dist_min_ci_0_5h'].rank(pct=True)
    df['dist_squared'] = df['dist_min_ci_0_5h'] ** 2 / 1e10
    df['growth_alignment'] = df['area_growth_rate_ha_per_h'] * df['alignment_abs']
    return df

train = engineer_features(train)
test  = engineer_features(test)

FEATURES = [
    'log_dist_min', 'dist_min_ci_0_5h', 'size_distance_ratio',
    'dist_x_alignment', 'alignment_abs', 'num_perimeters_0_5h',
    'data_quality_score', 'dt_first_last_0_5h', 'threat_score',
    'threat_per_distance', 'log1p_area_first', 'area_first_ha',
    'is_afternoon', 'event_start_month', 'event_start_hour',
    'log_closing_ratio', 'perimeter_density', 'dist_bearing_score',
    'dist_rank', 'low_temporal_resolution_0_5h', 'alignment_cos',
    'spread_bearing_cos', 'spread_bearing_deg',
    'dist_squared', 'growth_alignment'
]

# Add synthetic censored observations at 73h to extend time axis
# This ensures RSF's unique_times_ covers the 72h horizon without extrapolation
print(f"Original train size: {len(train)}")
synthetic = pd.DataFrame([train[FEATURES].median(numeric_only=True)] * 10)
synthetic['event'] = 0  # censored
synthetic['time_to_hit_hours'] = 73.0  # just beyond 72h
train = pd.concat([train, synthetic], ignore_index=True)
print(f"Extended train size: {len(train)} (added 10 synthetic rows at 73h)")

X_train = train[FEATURES].values
X_test  = test[FEATURES].values

y_surv = np.array(
    [(bool(e), t) for e, t in
     zip(train['event'], train['time_to_hit_hours'])],
    dtype=[('event', bool), ('time', float)]
)

HORIZONS = [12, 24, 48, 72]
WEIGHTS  = [0.1, 0.2, 0.3, 0.4]

# ── CV ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("=== RSF CV (with synthetic 73h observations) ===")
hybrid_scores = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, train['event'])):
    X_tr  = X_train[tr_idx]
    X_val = X_train[val_idx]
    y_tr  = y_surv[tr_idx]

    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.6,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rsf.fit(X_tr, y_tr)

    t = rsf.unique_times_
    surv_funcs = rsf.predict_survival_function(X_val)

    weighted_brier = 0
    for h, w in zip(HORIZONS, WEIGHTS):
        probs = []
        for fn in surv_funcs:
            s = fn(t)
            if h <= t.min():
                prob = 0.01
            elif h >= t.max():
                prob = float(1 - s[-1])
            else:
                idx = np.searchsorted(t, h)
                t0, t1 = t[idx-1], t[idx]
                s0, s1 = s[idx-1], s[idx]
                s_h = s0 + (s1-s0)*(h-t0)/(t1-t0)
                prob = float(1 - s_h)
            probs.append(max(prob, 0.01))

        probs = np.array(probs)
        y_val_label = (
            (train['event'].iloc[val_idx].values == 1) &
            (train['time_to_hit_hours'].iloc[val_idx].values <= h)
        ).astype(int)
        bs = np.mean((probs - y_val_label) ** 2)
        weighted_brier += w * bs

    risk_scores = rsf.predict(X_val)
    ci = concordance_index(
        train['time_to_hit_hours'].iloc[val_idx].values,
        -risk_scores,
        train['event'].iloc[val_idx].values
    )
    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)
    hybrid_scores.append(hybrid)
    print(f"  Fold {fold+1} | C-index: {ci:.3f} | "
          f"Brier: {weighted_brier:.3f} | Hybrid: {hybrid:.3f}")

print(f"\nMean Hybrid Score: {np.mean(hybrid_scores):.3f}")
print(f"Std:               {np.std(hybrid_scores):.3f}")

# ── Quick submission to check flat count ──────────────────────
print("\n=== Quick Submission (t.max() check) ===")
rsf_final = RandomSurvivalForest(
    n_estimators=500, min_samples_leaf=3,
    max_features=0.6, max_depth=8,
    random_state=42, n_jobs=-1
)
rsf_final.fit(X_train, y_surv)

print(f"Unique times max: {rsf_final.unique_times_.max():.1f}h")
print(f"Target horizons:  {HORIZONS}")

t = rsf_final.unique_times_
surv_funcs = rsf_final.predict_survival_function(X_test)

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
preds = {}

for h, col in zip(HORIZONS, prob_cols):
    probs = []
    for fn in surv_funcs:
        s = fn(t)
        if h <= t.min():
            prob = 0.01
        elif h >= t.max():
            prob = float(1 - s[-1])
        else:
            idx = np.searchsorted(t, h)
            t0, t1 = t[idx-1], t[idx]
            s0, s1 = s[idx-1], s[idx]
            s_h = s0 + (s1-s0)*(h-t0)/(t1-t0)
            prob = float(1 - s_h)
        probs.append(max(prob, 0.01))
    preds[col] = np.array(probs)

pred_array = np.column_stack([preds[c] for c in prob_cols])
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.97)

submission = pd.DataFrame({'event_id': test['event_id']})
for i, col in enumerate(prob_cols):
    submission[col] = pred_array[:, i]

flat = (submission['prob_72h'] == submission['prob_12h']).sum()
print(f"\nFlat predictions: {flat}")
print(submission[prob_cols].describe().round(3))