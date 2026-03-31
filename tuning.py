import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

train = pd.read_csv("train.csv")

# Define features (excluding id and target columns)
FEATURES = [c for c in train.columns 
            if c not in ['event_id', 'event', 
                        'time_to_hit_hours',
                        'hit_within_12h']]

# We tune on 72h horizon (most important)
# 12h is harder - only 49 positives out of 221
train['hit_within_12h'] = (
    (train['event'] == 1) &
    (train['time_to_hit_hours'] <= 12)
).astype(int)

X = train[FEATURES]
y = train['hit_within_12h']

# Parameter grid to search
param_grid = {
    'max_iter':        [300, 500, 800],
    'learning_rate':   [0.01, 0.03, 0.05],
    'max_depth':       [3, 4, 5],
    'min_samples_leaf':[3, 5, 8]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

search.fit(X, y)

print("\n=== Best Parameters ===")
print(search.best_params_)
print(f"\nBest AUC: {search.best_score_:.4f}")