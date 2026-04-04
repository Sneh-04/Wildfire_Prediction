import pandas as pd
import numpy as np

rsf   = pd.read_csv("submission_rsf_clean.csv")
stack = pd.read_csv("submission_stacked.csv")

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Use RSF for 12h/24h/48h (good interpolation)
# Use stacking for 72h (avoids extrapolation problem)
hybrid = rsf.copy()
hybrid['prob_72h'] = 0.5 * rsf['prob_72h'] + 0.5 * stack['prob_72h']
hybrid['prob_48h'] = 0.7 * rsf['prob_48h'] + 0.3 * stack['prob_48h']

# Monotonicity + clip
pred_array = hybrid[prob_cols].values
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.97)
for i, col in enumerate(prob_cols):
    hybrid[col] = pred_array[:, i]

hybrid.to_csv("submission_hybrid.csv", index=False)

flat = (hybrid['prob_72h'] == hybrid['prob_12h']).sum()
print(f"Flat: {flat}")
print(hybrid[prob_cols].describe().round(3))
print(hybrid.head(10).to_string(index=False))