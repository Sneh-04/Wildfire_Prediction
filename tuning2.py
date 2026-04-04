import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

# ── Tune Random Forest ───────────────────────────────────────────
print("=== Tuning Random Forest ===")
rf_grid = {
    'n_estimators':    [300, 500, 800],
    'max_depth':       [4, 6, 8],
    'min_samples_leaf':[2, 4, 6],
    'max_features':    ['sqrt', 0.5]
}
rf_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_grid, cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=0
)
rf_search.fit(X, y)
print("Best RF params:", rf_search.best_params_)
print(f"Best RF AUC:    {rf_search.best_score_:.4f}")

# ── Tune Extra Trees ─────────────────────────────────────────────
print("\n=== Tuning Extra Trees ===")
et_grid = {
    'n_estimators':    [300, 500, 800],
    'max_depth':       [4, 6, 8],
    'min_samples_leaf':[2, 4, 6],
    'max_features':    ['sqrt', 0.5]
}
et_search = GridSearchCV(
    ExtraTreesClassifier(random_state=42),
    et_grid, cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=0
)
et_search.fit(X, y)
print("Best ET params:", et_search.best_params_)
print(f"Best ET AUC:    {et_search.best_score_:.4f}")