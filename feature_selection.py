import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("train.csv")

def engineer_features(df):
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
    return df

train = engineer_features(train)

FEATURES = [c for c in train.columns
            if c not in ['event_id', 'event', 'time_to_hit_hours']]

train['hit_within_12h'] = (
    (train['event'] == 1) &
    (train['time_to_hit_hours'] <= 12)
).astype(int)

X = train[FEATURES]
y = train['hit_within_12h']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importance_matrix = np.zeros((5, len(FEATURES)))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=4, random_state=42
    )
    rf.fit(X_tr, y_tr)
    importance_matrix[fold] = rf.feature_importances_

mean_importance = importance_matrix.mean(axis=0)
std_importance  = importance_matrix.std(axis=0)

importance_df = pd.DataFrame({
    'feature':    FEATURES,
    'importance': mean_importance,
    'std':        std_importance
}).sort_values('importance', ascending=False)

print("=== Top 20 Features ===")
print(importance_df.head(20).to_string(index=False))

print("\n=== Bottom 15 Features ===")
print(importance_df.tail(15).to_string(index=False))

weak = importance_df[importance_df['importance'] < 0.005]
print(f"\n=== Weak features to DROP ({len(weak)}): ===")
print(weak['feature'].tolist())