import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
import torch.optim as optim

# Load dataset
df = pd.read_csv("PCOS_dataset.csv")

# Select features and target
selected_features = [
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)", 
    "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)", 
    "RBS(mg/dl)", "TSH (mIU/L)"
]
target = "PCOS (Y/N)"

# Convert columns to numeric and drop NaNs
df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

X = df[selected_features].values
y = df[target].astype(int).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TabNet setup with correct optimizer
model = TabNetClassifier(
    n_d=8,                # Dimensionality of the decision step
    n_a=8,                # Dimensionality of the attention step
    n_steps=3,            # Fewer steps
    gamma=1.3,            # Regularization coefficient
    lambda_sparse=0.0001, # Regularization for sparsity
    optimizer_fn=optim.Adam,  # Pass the optimizer function directly
    mask_type="entmax",   # Masking method
    verbose=0
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Print in terminal
print("Confusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.2f}")
print(f"AUC Score: {auc:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add accuracy & AUC as text below plot
plt.figtext(0.5, -0.1, f"Accuracy: {accuracy:.2f}     AUC Score: {auc:.2f}", wrap=True, horizontalalignment='center', fontsize=10)
plt.tight_layout()
plt.show()
