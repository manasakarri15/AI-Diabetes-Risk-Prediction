
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = xgb.XGBClassifier()
model.load_model("diabetes_xgboost_model.json")


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 AI-Based Diabetes Risk Prediction")
st.write(
    "Enter patient information to estimate diabetes risk "
    "using an XGBoost machine learning model."
)

st.warning(
    "This application provides risk estimation for educational "
    "and research purposes only. It is not a medical diagnosis."
)


# --------------------------------------------------
# Patient Input
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=30.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])


    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # Risk level
    if probability < 0.30:
        risk_level = "Low Risk"
    elif probability < 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Risk Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Risk Level",
            risk_level
        )


    if risk_level == "High Risk":
        st.error("⚠️ High predicted risk")

    elif risk_level == "Medium Risk":
        st.warning("⚠️ Medium predicted risk")

    else:
        st.success("✅ Low predicted risk")


    # --------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------

    st.subheader("🤖 Why did the AI make this prediction?")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    patient_shap = shap_values[0]

    explanation = shap.Explanation(
        values=patient_shap,
        base_values=explainer.expected_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        explanation,
        max_display=8,
        show=False
    )

    st.pyplot(fig)

    st.info(
        "SHAP explains how each patient feature contributed "
        "to the model's prediction."
    )
