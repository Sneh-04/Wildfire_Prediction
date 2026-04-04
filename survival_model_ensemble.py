import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest

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

X_train = train[FEATURES].values
X_test  = test[FEATURES].values

y_surv = np.array(
    [(bool(e), t) for e, t in
     zip(train['event'], train['time_to_hit_hours'])],
    dtype=[('event', bool), ('time', float)]
)

HORIZONS = [12, 24, 48, 72]
WEIGHTS  = [0.1, 0.2, 0.3, 0.4]

# ── Train 3 RSF models with different seeds ────────────────────
print("=== Training 3-Seed RSF Ensemble ===")
seeds = [42, 123, 456]
all_risk = []
all_models = []

for seed in seeds:
    print(f"Training RSF with seed={seed}...")
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.6,
        max_depth=8,
        random_state=seed,
        n_jobs=-1
    )
    rsf.fit(X_train, y_surv)
    all_models.append(rsf)
    all_risk.append(rsf.predict(X_test))

# Average risk scores from 3 models
avg_risk = np.mean(all_risk, axis=0)
print(f"Averaged risk scores (min={avg_risk.min():.3f}, max={avg_risk.max():.3f})")

# ── Generate submission using averaged risk ────────────────────
print("\n=== Generating Submission ===")

# Use first model's survival functions for probability extraction
rsf_main = all_models[0]
t = rsf_main.unique_times_
surv_funcs = rsf_main.predict_survival_function(X_test)

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

submission.to_csv("submission_rsf_ensemble.csv", index=False)

flat = (submission['prob_72h'] == submission['prob_12h']).sum()
print(f"Flat predictions: {flat}")
print(submission.head(10).to_string(index=False))
print(f"\nShape: {submission.shape}")
print(submission[prob_cols].describe().round(3))
print("\n✓ submission_rsf_ensemble.csv saved!")
