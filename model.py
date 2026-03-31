from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Define features (exclude target and id columns)
FEATURES = [col for col in train.columns if col not in ['event_id', 'time_to_hit_hours', 'event']]

X = train[FEATURES]
y_event = train['event']
y_time = train['time_to_hit_hours']

# Baseline Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation AUC
scores = cross_val_score(rf, X, y_event, cv=5, scoring='roc_auc')
print("=== Random Forest - 5-fold CV AUC ===")
print(f"Scores: {scores.round(3)}")
print(f"Mean AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
# For each time horizon, create a binary target
# Label = 1 if fire HIT and arrived WITHIN that horizon
for h in [12, 24, 48, 72]:
    col = f'hit_within_{h}h'
    train[col] = ((train['event'] == 1) & 
                  (train['time_to_hit_hours'] <= h)).astype(int)

print("=== Horizon label counts ===")
for h in [12, 24, 48, 72]:
    col = f'hit_within_{h}h'
    count = train[col].sum()
    print(f"  {col}: {count} positives out of {len(train)} ({count/len(train):.1%})")