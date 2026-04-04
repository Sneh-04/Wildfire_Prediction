import pandas as pd
import numpy as np
from sksurv.ensemble import ExtraSurvivalTrees

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

y_surv = np.array(
    [(bool(e), t) for e, t in
     zip(train['event'], train['time_to_hit_hours'])],
    dtype=[('event', bool), ('time', float)]
)

# Train ExtraSurvivalTrees on full data
rsf = ExtraSurvivalTrees(
    n_estimators=500, min_samples_leaf=3,
    max_features=0.6, max_depth=8,
    random_state=42, n_jobs=-1
)
rsf.fit(train[FEATURES].values, y_surv)

# Get RSF risk scores — higher = more dangerous
rsf_risk = rsf.predict(test[FEATURES].values)

# Normalize to 0-1
rsf_rank = pd.Series(rsf_risk).rank(pct=True).values

# Load stacking
stack = pd.read_csv("submission_stacked.csv")
prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Key idea: rank-based isotonic adjustment
# Use RSF risk rank to reorder stacking probabilities
result = stack.copy()
for col in prob_cols:
    stack_vals = stack[col].values
    
    # Sort stacking values by RSF risk rank
    # High RSF risk → keep high stacking prob
    # Low RSF risk → pull toward lower end
    rsf_order = np.argsort(np.argsort(rsf_risk))  # rank of each fire
    stack_order = np.argsort(np.argsort(stack_vals))  # rank in stacking
    
    # Average the two rankings
    combined_rank = 0.4 * rsf_order + 0.6 * stack_order
    
    # Map combined rank back to stacking probability values
    sorted_stack = np.sort(stack_vals)
    new_rank_idx = np.argsort(combined_rank).argsort()
    result[col] = sorted_stack[new_rank_idx]

# Monotonicity + clip
pred_array = result[prob_cols].values
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)
for i, col in enumerate(prob_cols):
    result[col] = pred_array[:, i]

result.to_csv("submission_rank_blend.csv", index=False)

flat = (result['prob_72h'] == result['prob_12h']).sum()
print(f"Flat: {flat}")
print(result[prob_cols].describe().round(3))
print(result.head(10).to_string(index=False))
print("\n✓ submission_rank_blend.csv saved!")