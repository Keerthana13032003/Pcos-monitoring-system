from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
import uvicorn
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import joblib
from tensorflow.keras.applications.resnet50 import preprocess_input

# After loading the image


import tensorflow as tf

# Load models
model = pickle.load(open("pcos_model.pkl", "rb"))  # Symptom-based model
xgb_model = joblib.load("xgb_pcos_model.pkl")  # Hormonal-based model
scaler = joblib.load("scaler.pkl")  # Hormonal data scaler
cnn_model = load_model("bestmodel.h5")  # CNN for ultrasound

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ===================== Integrate Preprocessing Logic =====================
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
    img_arr = preprocess_input(img_arr)
    input_arr = np.expand_dims(img_arr, axis=0)
    ultrasound_pred = cnn_model.predict(input_arr)[0][0]
    return ultrasound_pred

def predict_pcos_combined(hormonal_data, image_path):
    hormonal_risk = preprocess_hormonal_data(hormonal_data)
    ultrasound_pred = preprocess_ultrasound_image(image_path)

    ultrasound_threshold = 0.5
    hormonal_threshold = 0.5

    ultrasound_diagnosis = "Yes" if ultrasound_pred < ultrasound_threshold else "No"

    if ultrasound_pred >= ultrasound_threshold and hormonal_risk < hormonal_threshold:
        combined_risk = (hormonal_risk + (1 - ultrasound_pred)) / 2
        diagnosis = "No"
    elif ultrasound_pred < 0.1 and hormonal_risk > 0.6:
        combined_risk = (hormonal_risk + (1 - ultrasound_pred)) / 2
        diagnosis = "Yes"
    else:
        combined_risk = (0.7 * hormonal_risk + 0.3 * (1 - ultrasound_pred))
        diagnosis = "Yes" if combined_risk > 0.5 else "No"

    if combined_risk < 0.5:
        risk_level = "Low"
    elif combined_risk < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return combined_risk, diagnosis, risk_level


# ===================== FastAPI Routes =====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/general-test", response_class=HTMLResponse)
async def general_test(request: Request):
    return templates.TemplateResponse("symptomtest.html", {"request": request})


@app.get("/professional-test", response_class=HTMLResponse)
async def professional_test(request: Request):
    return templates.TemplateResponse("ultra.html", {"request": request})


@app.post("/predict_symptom", response_class=HTMLResponse)
async def predict_symptom(
    request: Request,
    hair_growth: int = Form(0),
    skin_darkening: int = Form(0),
    weight_gain: int = Form(0),
    pimples: int = Form(0),
    hair_loss: int = Form(0),
    cycle: str = Form(...),
    fast_food: str = Form(...),
    weight_Kg: float = Form(...),
    bmi: float = Form(...),
    age_yrs: int = Form(...),
    waist_inch: float = Form(...),
    hip_inch: float = Form(...),
    cycle_length_days: int = Form(...)
):
    cycle_val = 1 if cycle == "R" else 0
    fast_food_val = 1 if fast_food == "Y" else 0

    features = [
        hair_growth,
        skin_darkening,
        weight_gain,
        cycle_val,
        fast_food_val,
        pimples,
        weight_Kg,
        bmi,
        waist_inch,
        age_yrs,
        hair_loss,
        hip_inch,
        cycle_length_days
    ]

    columns = [
        "hair growth(Y/N)", "Skin darkening (Y/N)", "Weight gain(Y/N)", "Cycle(R/I)",
        "Fast food (Y/N)", "Pimples(Y/N)", "Weight (Kg)", "BMI", "Waist(inch)",
        "Age(yrs)", "Hair loss(Y/N)", "Hip(inch)", "Cycle length(days)"
    ]

    input_df = pd.DataFrame([features], columns=columns)
    prediction = model.predict(input_df)[0]
    result_text = "PCOS" if prediction == 1 else "No PCOS"

    return templates.TemplateResponse("symptomtest.html", {
        "request": request,
        "prediction": result_text
    })


@app.post("/predict_pcos", response_class=HTMLResponse)
async def predict_pcos(
    request: Request,
    FSH: float = Form(...),
    LH: float = Form(...),
    FSH_LH: float = Form(...),
    AMH: float = Form(...),
    PRL: float = Form(...),
    VitD: float = Form(...),
    PRG: float = Form(...),
    RBS: float = Form(...),
    TSH: float = Form(...),
    ultrasound_image: UploadFile = Form(...)
):
    # Hormonal prediction
    hormonal_data = {
        "FSH(mIU/mL)": FSH,
        "LH(mIU/mL)": LH,
        "FSH/LH": FSH_LH,
        "AMH(ng/mL)": AMH,
        "PRL(ng/mL)": PRL,
        "Vit D3 (ng/mL)": VitD,
        "PRG(ng/mL)": PRG,
        "RBS(mg/dl)": RBS,
        "TSH (mIU/L)": TSH
    }

    # Predict using combined logic
    contents = await ultrasound_image.read()
    img_path = f"static/{ultrasound_image.filename}"
    with open(img_path, "wb") as f:
        f.write(contents)

    combined_risk, final_diagnosis, risk_level = predict_pcos_combined(hormonal_data, img_path)

    os.remove(img_path)  # Clean up after prediction

    return templates.TemplateResponse("ultra.html", {
        "request": request,
        "result": {
            "diagnosis": final_diagnosis,
            "risk_level": f"PCOS Risk Level: {risk_level}",
            "combined_risk": f"Combined Risk Score: {combined_risk:.2f}"
        }
    })


#if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=5000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use $PORT from environment or default to 5000 locally
    uvicorn.run("app:app", host="0.0.0.0", port=port)
