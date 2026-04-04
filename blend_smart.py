import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold

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
    'spread_bearing_cos', 'spread_bearing_deg'
]

X_train = train[FEATURES].values
X_test  = test[FEATURES].values

y_surv = np.array(
    [(bool(e), t) for e, t in
     zip(train['event'], train['time_to_hit_hours'])],
    dtype=[('event', bool), ('time', float)]
)

# Train RSF on full data
rsf = RandomSurvivalForest(
    n_estimators=500, min_samples_leaf=3,
    max_features=0.6, max_depth=8,
    random_state=42, n_jobs=-1
)
rsf.fit(X_train, y_surv)

# Get RSF risk scores for test (higher = more dangerous)
rsf_risk = rsf.predict(X_test)

# Normalize risk to 0-1
rsf_risk_norm = (rsf_risk - rsf_risk.min()) / (rsf_risk.max() - rsf_risk.min())

# Load stacking submission
stack = pd.read_csv("submission_stacked.csv")
prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Blend: use RSF risk to adjust stacking probabilities
# RSF tells us WHO is dangerous, stacking tells us WHEN
blended = stack.copy()
for col in prob_cols:
    stack_prob = stack[col].values
    # Pull high-risk fires higher, low-risk fires lower
    blended[col] = 0.7 * stack_prob + 0.3 * rsf_risk_norm

# Monotonicity + clip
pred_array = blended[prob_cols].values
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)

for i, col in enumerate(prob_cols):
    blended[col] = pred_array[:, i]

blended.to_csv("submission_blend_smart.csv", index=False)

# Quality checks
flat = (blended['prob_72h'] == blended['prob_12h']).sum()
print(f"Flat predictions: {flat}")
print(blended[prob_cols].describe().round(3))
print("\nSample:")
print(blended.head(10).to_string(index=False))
print("\n✓ submission_blend_smart.csv saved!")