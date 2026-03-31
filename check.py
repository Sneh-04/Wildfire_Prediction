import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Missing values
print("=== Missing values ===")
print(train.isnull().sum()[train.isnull().sum() > 0])

# Distribution of time_to_hit_hours split by event
plt.figure(figsize=(8, 4))
for event_val, label, color in [(1, 'Hit (event=1)', 'red'), (0, 'Censored (event=0)', 'blue')]:
    subset = train[train['event'] == event_val]['time_to_hit_hours']
    plt.hist(subset, bins=20, alpha=0.6, label=label, color=color)

plt.xlabel('time_to_hit_hours')
plt.ylabel('Count')
plt.title('Distribution of time_to_hit_hours by event')
plt.legend()
plt.tight_layout()
plt.savefig('time_distribution.png')
plt.show()
print("Plot saved.")
# Correlation of features with 'event'
features = [c for c in train.columns if c not in ['event_id', 'event', 'time_to_hit_hours']]

correlations = train[features + ['event']].corr()['event'].drop('event')
top_corr = correlations.abs().sort_values(ascending=False).head(10)

print("=== Top 10 features correlated with event ===")
print(top_corr)
# Box plot: top correlated feature vs event
top_feature = top_corr.index[0]
print(f"Top feature: {top_feature}")

plt.figure(figsize=(6, 4))
train.boxplot(column=top_feature, by='event')
plt.title(f'{top_feature} by event')
plt.suptitle('')
plt.tight_layout()
plt.savefig('top_feature_boxplot.png')
plt.show()