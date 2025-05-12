import joblib
import numpy as np

# Load the trained model and scaler
xgb_model = joblib.load("xgb_pcos_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the function to get risk level
def get_risk_level(prob):
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# Function to predict PCOS risk based on user input
def predict_pcos(input_data):
    # Preprocess input data (Scale it)
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

    # Predict the probability of PCOS (positive class)
    prob = xgb_model.predict_proba(input_data_scaled)[:, 1][0]
    
    # Get the risk level based on the probability
    risk_level = get_risk_level(prob)
    
    # Predict whether the person has PCOS (0 or 1)
    pcos_diagnosis = "Yes" if prob >= 0.5 else "No"

    # Output the results
    print(f"\nPCOS Risk Prediction:")
    print(f"➡ Probability of PCOS: {prob:.2f}")
    print(f"⚠ Risk Level: {risk_level}")
    print(f"✅ PCOS Diagnosis: {pcos_diagnosis}")

# Main function to get input and predict
if __name__ == "__main__":
    # Input values from the user (make sure the order matches the training data)
    print("Enter the values for the following parameters:")
    
    FSH = float(input("FSH(mIU/mL): "))
    LH = float(input("LH(mIU/mL): "))
    FSH_LH = float(input("FSH/LH: "))
    AMH = float(input("AMH(ng/mL): "))
    PRL = float(input("PRL(ng/mL): "))
    Vit_D3 = float(input("Vit D3 (ng/mL): "))
    PRG = float(input("PRG(ng/mL): "))
    RBS = float(input("RBS(mg/dl): "))
    TSH = float(input("TSH (mIU/L): "))

    # Create the input data array (order matters, should match training data)
    input_data = [FSH, LH, FSH_LH, AMH, PRL, Vit_D3, PRG, RBS, TSH]
    
    # Predict the PCOS risk
    predict_pcos(input_data)
