# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv("PCOS_dataset.csv")  # Replace with your actual file name

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in "Marraige Status (Yrs)" with the mean
df["Marraige Status (Yrs)"].fillna(df["Marraige Status (Yrs)"].mean(), inplace=True)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Verify cleaned column names
print("\nUpdated Column Names:", df.columns.tolist())

# Drop non-symptom-based columns
df.drop(columns=["Sl. No", "Patient File No.", "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)", 
                 "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)", "PRL(ng/mL)", 
                 "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)", "Follicle No. (L)", 
                 "Follicle No. (R)", "Avg. F size (L) (mm)", "Avg. F size (R) (mm)", 
                 "Endometrium (mm)"], axis=1, inplace=True)

# Compute correlation matrix
print("\nFeature Correlations with PCOS:")
target_corr = df.corr()['PCOS (Y/N)'].abs().sort_values(ascending=False)[1:]
print(target_corr)

# Selected features based on correlation analysis
selected_features = [
    "hair growth(Y/N)", "Skin darkening (Y/N)", "Weight gain(Y/N)", "Cycle(R/I)", 
    "Fast food (Y/N)", "Pimples(Y/N)", "Weight (Kg)", "BMI", "Waist(inch)", "Age(yrs)", 
    "Hair loss(Y/N)", "Hip(inch)", "Cycle length(days)"
]

# Define input (X) and target (y)
X = df[selected_features]
y = df["PCOS (Y/N)"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"\nDataset Split: Train Shape {X_train.shape}, Test Shape {X_test.shape}")

# Train and evaluate multiple models
models = [
    ('Naive Bayes', GaussianNB()),
    ('KNN', KNeighborsClassifier(n_neighbors=8)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('DecisionTree', DecisionTreeClassifier()),
    ('SVM', SVC(kernel='linear'))
]

print("\nTraining Models and Evaluating Performance:")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name} Model:")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Select the best-performing model (Random Forest) for deployment
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Save the trained model
with open("pcos_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Model training complete. Model saved as 'pcos_model.pkl'")
print(X.columns.tolist())  # This should exactly match API input

