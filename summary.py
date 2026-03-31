from lifelines import CoxPHFitter
import pandas as pd

# Load data
train = pd.read_csv("train.csv")

# Define features
FEATURES = [col for col in train.columns if col not in ['event_id', 'time_to_hit_hours', 'event']]

cox_train = train[FEATURES + ['time_to_hit_hours', 'event']].copy()

cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_train, duration_col='time_to_hit_hours', event_col='event')

cph.print_summary()