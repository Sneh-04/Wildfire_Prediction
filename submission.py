import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def engineer_features(df):
    df = df.copy()
    df['closing_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['est_time_to_hit'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0.1))
    df['est_time_to_hit'] = df['est_time_to_hit'].clip(upper=300)
    df['size_growth_interaction'] = df['log1p_area_first'] * df['area_growth_rate_ha_per_h']
    df['threat_score'] = df['alignment_abs'] * df['centroid_speed_m_per_h']
    return df

train = engineer_features(train)
test = engineer_features(test)

FEATURES = [c for c in train.columns
            if c not in ['event_id', 'event', 'time_to_hit_hours']]
HORIZONS = [12, 24, 48, 72]

for h in HORIZONS:
    train[f'hit_within_{h}h'] = (
        (train['event'] == 1) &
        (train['time_to_hit_hours'] <= h)
    ).astype(int)

X_train = train[FEATURES]
X_test = test[FEATURES]
preds = {}

for h in HORIZONS:
    y_train = train[f'hit_within_{h}h']
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        max_depth=4, min_samples_leaf=5, random_state=42
    )
    model.fit(X_train, y_train)
    preds[f'prob_{h}h'] = model.predict_proba(X_test)[:, 1]

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Step 1: Stack predictions
pred_array = np.column_stack([preds[c] for c in prob_cols])

# Step 2: Monotonicity correction
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])

# Step 3: Clip
for i in range(len(prob_cols)):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)

# Step 4: Build submission
submission = pd.DataFrame({'event_id': test['event_id']})
for i, col in enumerate(prob_cols):
    submission[col] = pred_array[:, i]

# Step 5: Save
submission.to_csv("submission.csv", index=False)

print(submission.head(10))
print("\nShape:", submission.shape)
print("\nProbability ranges:")
for col in prob_cols:
    print(f"  {col}: min={submission[col].min():.3f}, max={submission[col].max():.3f}")
print("\n✓ Submission saved!")