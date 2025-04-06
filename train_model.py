import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Encode target labels (if needed)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the mapping if needed for reverse lookup
crop_dict = dict(zip(range(len(le.classes_)), le.classes_))

# Scaling
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

X_minmax = minmax_scaler.fit_transform(X)
X_scaled = standard_scaler.fit_transform(X_minmax)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scalers
with open("minmaxscaler.pkl", "wb") as f:
    pickle.dump(minmax_scaler, f)

with open("standscaler.pkl", "wb") as f:
    pickle.dump(standard_scaler, f)

# Optionally save label encoder (to convert predictions back to crop names)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and scalers saved successfully.")
