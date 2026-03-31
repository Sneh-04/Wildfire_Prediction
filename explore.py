import pandas as pd
import numpy as np

# Load all files - adjust paths if needed
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Task 2.1
print("=== Target columns ===")
print(train[['event', 'time_to_hit_hours']].describe())
print("\nEvent value counts:", train['event'].value_counts().to_dict())

print("\n=== Train columns ===")
print(train.columns.tolist())

print("\n=== Metadata categories ===")
print(meta.groupby('category')['column'].apply(list))