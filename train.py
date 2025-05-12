import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, accuracy_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("PCOS_dataset.csv")

# Select features and target
selected_features = [
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)", 
    "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)", 
    "RBS(mg/dl)", "TSH (mIU/L)"
]
target = "PCOS (Y/N)"

# Coerce non-numeric values to NaN and drop rows with NaN
df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Prepare features and target
X = df[selected_features].values
y = df[target].astype(int).values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,  # Number of boosting rounds
    learning_rate=0.05,  # Step size shrinkage
    max_depth=4,  # Maximum depth of a tree
    objective='binary:logistic',  # Binary classification task
    eval_metric='logloss',  # Evaluation metric
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(xgb_model, "xgb_pcos_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predict on test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]  # For probability

