from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

# Weights per horizon (given by competition)
HORIZONS = [12, 24, 48, 72]
WEIGHTS =  [0.1, 0.2, 0.3, 0.4]   # more weight on longer horizons

def brier_score_horizon(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hybrid_scores = []
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Define features (exclude target and id columns)
FEATURES = [col for col in train.columns if col not in ['event_id', 'time_to_hit_hours', 'event']]

# Create hit_within_{h}h columns
for h in HORIZONS:
    train[f'hit_within_{h}h'] = ((train['time_to_hit_hours'] <= h) & (train['event'] == 1)).astype(int)
for train_idx, val_idx in cv.split(train[FEATURES], train['event']):
    X_tr = train[FEATURES].iloc[train_idx]
    X_val = train[FEATURES].iloc[val_idx]
    
    weighted_brier = 0
    risk_scores_fold = None
    
    for h, w in zip(HORIZONS, WEIGHTS):
        y_tr = train[f'hit_within_{h}h'].iloc[train_idx]
        y_val = train[f'hit_within_{h}h'].iloc[val_idx]
        
        model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05,
    max_depth=4, min_samples_leaf=5, random_state=42
)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        
        bs = brier_score_horizon(y_val.values, prob)
        weighted_brier += w * bs
        
        if h == 72:
            risk_scores_fold = prob
    
    # C-index using 72h risk score
    y_val_time = train['time_to_hit_hours'].iloc[val_idx].values
    y_val_event = train['event'].iloc[val_idx].values
    ci = concordance_index(y_val_time, -risk_scores_fold, y_val_event)
    
    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)
    hybrid_scores.append(hybrid)
    print(f"  C-index: {ci:.3f} | Weighted Brier: {weighted_brier:.3f} | Hybrid: {hybrid:.3f}")

print(f"\nMean Hybrid Score: {np.mean(hybrid_scores):.3f}")