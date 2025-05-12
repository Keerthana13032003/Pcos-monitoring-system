import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


xgb_model = joblib.load("xgb_pcos_model.pkl") 
scaler = joblib.load("scaler.pkl")  
cnn_model = load_model("bestmodel.h5")  


selected_features = [
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)",
    "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)",
    "RBS(mg/dl)", "TSH (mIU/L)"
]


def preprocess_hormonal_data(hormonal_data):
    df = pd.DataFrame([hormonal_data], columns=selected_features)
    scaled = scaler.transform(df)
    risk = xgb_model.predict_proba(scaled)[0][1]
    return risk

def preprocess_ultrasound_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_arr = tf.keras.utils.img_to_array(img)
    print(f"Image shape after loading: {img_arr.shape}")

    img_arr = preprocess_input(img_arr)
    print(f"Image shape after preprocessing: {img_arr.shape}")

    input_arr = np.expand_dims(img_arr, axis=0)
    ultrasound_pred = cnn_model.predict(input_arr)[0][0]

    print(f"Ultrasound Model Raw Prediction: {ultrasound_pred}")
    return ultrasound_pred

# Combined decision logic
def predict_pcos(hormonal_data, image_path):
    hormonal_risk = preprocess_hormonal_data(hormonal_data)
    print(f"Hormonal Risk Score: {hormonal_risk:.2f}")

    ultrasound_pred = preprocess_ultrasound_image(image_path)

    # 🔁 Inversion based on class label (PCOS = class 0)
    # So, lower probability → PCOS
    ultrasound_threshold = 0.5
    hormonal_threshold = 0.5

    # Interpret the prediction from CNN
    ultrasound_diagnosis = "Yes" if ultrasound_pred < ultrasound_threshold else "No"
    print(f"Ultrasound Diagnosis: {ultrasound_diagnosis} (raw prob: {ultrasound_pred:.2f})")

    # Final diagnosis logic
    if ultrasound_pred >= ultrasound_threshold and hormonal_risk < hormonal_threshold:
        combined_risk = (hormonal_risk + (1 - ultrasound_pred)) / 2
        diagnosis = "No"
    elif ultrasound_pred < 0.1 and hormonal_risk > 0.6:
        combined_risk = (hormonal_risk + (1 - ultrasound_pred)) / 2
        diagnosis = "Yes"
    else:
        # Mixed case, weight hormonal risk more
        combined_risk = (0.7 * hormonal_risk + 0.3 * (1 - ultrasound_pred))
        diagnosis = "Yes" if combined_risk > 0.5 else "No"

    # Risk level
    if combined_risk < 0.5:
        risk_level = "Low"
    elif combined_risk < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return combined_risk, diagnosis, risk_level

# =================== TEST SAMPLE ======================

hormonal_data_input = {
    "FSH(mIU/mL)": 5.05,
    "LH(mIU/mL)": 2.01,
    "FSH/LH": 2.51,
    "AMH(ng/mL)": 0.9,
    "PRL(ng/mL)": 14.85,
    "Vit D3 (ng/mL)": 32.8,
    "PRG(ng/mL)": 0.24,
    "RBS(mg/dl)": 92,
    "TSH (mIU/L)": 1.27
}

ultrasound_image_path = r"F:\ULTRASOUND\arch.jpg"  # Change this to test other images

# Run prediction
combined_risk, final_diagnosis, risk_level = predict_pcos(hormonal_data_input, ultrasound_image_path)

# Output
print(f"\n Combined Risk Score: {combined_risk:.2f}")
print(f" Final Diagnosis (PCOS): {final_diagnosis}")
print(f" Risk Level: {risk_level}")
