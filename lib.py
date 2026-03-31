import pandas as pd
import numpy as np
import matplotlib
import sklearn
import lifelines

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("lifelines:", lifelines.__version__)

# Load all 4 files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
meta = pd.read_csv("metaData.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Basic shape check
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("Meta shape:", meta.shape)
print("Sample submission shape:", sample_sub.shape)