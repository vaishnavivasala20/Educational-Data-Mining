import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
DATASET_PATH = 'Students Performance Dataset.csv'
df = pd.read_csv(DATASET_PATH)

# Define nuanced At_Risk target
# At_Risk = 1 if (Grade is D or F) OR (Attendance (%) < 75) OR (Total_Score < 50)
df['At_Risk'] = (
    (df['Grade'].isin(['D', 'F'])) |
    (df['Attendance (%)'] < 75) |
    (df['Total_Score'] < 50)
).astype(int)

# Features for training (do NOT use Grade)
features = ['Attendance (%)', 'Total_Score', 'Projects_Score', 'Study_Hours_per_Week']
X = df[features]
y = df['At_Risk']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('student_risk_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

# Compute and save feature importances
importances = model.feature_importances_
feature_names = features
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df.to_csv('feature_importances_v2.csv', index=False)
