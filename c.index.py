from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Define features (exclude target and id columns)
FEATURES = [col for col in train.columns if col not in ['event_id', 'time_to_hit_hours', 'event']]

# C-index needs: predicted risk score, actual time, event flag
# Higher predicted probability = higher risk = should have LOWER time_to_hit

X = train[FEATURES].values
y_time = train['time_to_hit_hours'].values
y_event = train['event'].values

rf = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cindex_scores = []

for train_idx, val_idx in cv.split(X, y_event):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr_event = y_event[train_idx]
    y_val_time = y_time[val_idx]
    y_val_event = y_event[val_idx]
    
    rf.fit(X_tr, y_tr_event)
    
    # Predicted probability of hitting (risk score)
    risk_scores = rf.predict_proba(X_val)[:, 1]
    
    # C-index: does higher risk = shorter time to hit?
    ci = concordance_index(y_val_time, -risk_scores, y_val_event)
    cindex_scores.append(ci)

print("=== C-index (5-fold CV) ===")
print(f"Scores: {np.array(cindex_scores).round(3)}")
print(f"Mean C-index: {np.mean(cindex_scores):.3f}")