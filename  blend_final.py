import pandas as pd
import numpy as np

# Load both submissions
rsf   = pd.read_csv("submission_rsf.csv")
stack = pd.read_csv("submission_stacked.csv")

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Blend 60% RSF + 40% stacking
blended = rsf.copy()
for col in prob_cols:
    blended[col] = 0.6 * rsf[col] + 0.4 * stack[col]

# Check flat predictions
flat = (blended['prob_72h'] == blended['prob_12h']).sum()
print(f"Flat predictions after blend: {flat}")

# Monotonicity check
pred_array = blended[prob_cols].values
for i in range(1, 4):
    pred_array[:, i] = np.maximum(pred_array[:, i], pred_array[:, i-1])
for i in range(4):
    pred_array[:, i] = np.clip(pred_array[:, i], 0.01, 0.99)

for i, col in enumerate(prob_cols):
    blended[col] = pred_array[:, i]

blended.to_csv("submission_blend.csv", index=False)
print(blended.head(10))
print("\nProbability ranges:")
for col in prob_cols:
    print(f"  {col}: min={blended[col].min():.3f}, "
          f"max={blended[col].max():.3f}")
print("\n✓ submission_blend.csv saved!")