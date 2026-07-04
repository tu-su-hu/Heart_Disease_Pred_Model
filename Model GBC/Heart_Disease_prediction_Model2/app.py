import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# Check if model exists
# -------------------------
if not os.path.exists("models/heart_model.pkl"):
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

# Load Model
model = joblib.load("models/heart_model.pkl")

st.title("❤️ Heart Disease Prediction System")
st.markdown("### Enter Patient Details")

col1, col2 = st.columns(2)

# -------------------------
# LEFT COLUMN
# -------------------------

with col1:

    age = st.slider(
    "Age",
    min_value=1,
    max_value=120,
    value=45,
    step=1
)

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    cp_dict = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp = cp_dict[cp_option]

    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        80,
        220,
        120
    )

    chol = st.slider(
        "Serum Cholesterol (mg/dL)",
        100,
        500,
        220
    )

    fbs_option = st.selectbox(
        "Fasting Blood Sugar",
        [
            "≤ 120 mg/dL",
            "> 120 mg/dL"
        ]
    )

    fbs = 0 if fbs_option == "≤ 120 mg/dL" else 1

    restecg_option = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    restecg_dict = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    restecg = restecg_dict[restecg_option]

# -------------------------
# RIGHT COLUMN
# -------------------------

with col2:

    thalach = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exang_option = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    exang = 0 if exang_option == "No" else 1

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        6.0,
        1.0
    )

    slope_option = st.selectbox(
        "ST Segment Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    slope_dict = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    slope = slope_dict[slope_option]

    ca = st.selectbox(
        "Number of Major Vessels (0-4)",
        [0, 1, 2, 3, 4]
    )

    thal_option = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect",
            "Unknown"
        ]
    )

    thal_dict = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2,
        "Unknown": 3
    }

    thal = thal_dict[thal_option]

# Convert Gender
sex = 1 if sex == "Male" else 0

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Heart Disease"):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.subheader("Prediction Probability")

    st.progress(float(probability[1]))

    st.write(f"Healthy Probability: {probability[0] * 100:.2f}%")
    st.write(f"Heart Disease Probability: {probability[1] * 100:.2f}%")
    # -------------------------
    # -------------------------
    # Clinical Report
    # -------------------------

    st.markdown("---")
    st.subheader("🩺 Doctor's Clinical Report")

    risk_factors = []
    normal_factors = []

    # Age
    if age >= 60:
        risk_factors.append(f"Age ({age} years) is associated with an increased cardiovascular risk.")
    else:
        normal_factors.append(f"Age ({age} years) is not considered a major risk factor.")

    # Blood Pressure
    if trestbps >= 140:
        risk_factors.append(f"Resting blood pressure ({trestbps} mm Hg) is elevated.")
    else:
        normal_factors.append(f"Resting blood pressure ({trestbps} mm Hg) is within the normal range.")

    # Cholesterol
    if chol >= 240:
        risk_factors.append(f"Cholesterol level ({chol} mg/dL) is high.")
    else:
        normal_factors.append(f"Cholesterol level ({chol} mg/dL) is within the desirable range.")

    # Blood Sugar
    if fbs == 1:
        risk_factors.append("Fasting blood sugar is above 120 mg/dL.")
    else:
        normal_factors.append("Fasting blood sugar is within the normal range.")

    # Exercise Angina
    if exang == 1:
        risk_factors.append("Exercise-induced angina is present.")
    else:
        normal_factors.append("No exercise-induced angina was reported.")

    # Heart Rate
    if thalach < 120:
        risk_factors.append(f"Maximum heart rate ({thalach} bpm) is relatively low.")
    else:
        normal_factors.append(f"Maximum heart rate ({thalach} bpm) is acceptable.")

    # Oldpeak
    if oldpeak > 2:
        risk_factors.append(f"Oldpeak ({oldpeak}) is elevated.")
    else:
        normal_factors.append("Oldpeak is within the expected range.")

    # Major Vessels
    if ca >= 2:
        risk_factors.append(f"{ca} major vessels were identified, increasing cardiovascular risk.")
    else:
        normal_factors.append("Major vessel count does not indicate elevated risk.")

    st.markdown("### 🔍 Factors Increasing Risk")

    if risk_factors:
        for item in risk_factors:
            st.write("•", item)
    else:
        st.success("No major risk factors were identified from the provided information.")

    st.markdown("### ✅ Positive Findings")

    for item in normal_factors:
        st.write("•", item)

    st.markdown("---")
    st.subheader("📋 Clinical Summary")

    if prediction == 1:
        st.error(f"""
The model predicts a **high likelihood of heart disease** with a probability of **{probability[1]*100:.2f}%**.

The prediction is influenced by the patient's overall profile, including factors such as age, blood pressure, cholesterol, fasting blood sugar, exercise-induced angina, and other clinical measurements.

This result is a **screening prediction only** and should be confirmed through a medical evaluation and appropriate diagnostic tests.
""")
    else:
        st.success(f"""
The model predicts a **low likelihood of heart disease** with a probability of **{probability[0]*100:.2f}%**.

Most of the patient's clinical measurements fall within acceptable ranges, resulting in a lower predicted risk.

This result is a **screening prediction only** and should not replace professional medical advice.
""")

    st.markdown("---")
    st.subheader("📋 Patient Summary")
    st.dataframe(patient)