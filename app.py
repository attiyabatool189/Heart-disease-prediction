"""
Heart Disease Prediction - Streamlit App (Assignment 2, Task 2)
----------------------------------------------------------------
Loads the Support Vector Classifier pipeline trained in
Heart_Disease_Prediction.ipynb (heart_disease_model.pkl) and predicts
whether heart disease is present from exactly four inputs:
age, sex, cp (chest-pain type) and thalach (maximum heart rate).

The saved pipeline already contains the scaling / one-hot encoding fitted
on the training data, so we only have to build a DataFrame with the same
four columns in the same order.

EDUCATIONAL USE ONLY - not a medical diagnosis tool.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"
FEATURES = ["age", "sex", "cp", "thalach"]          # same order as in the notebook

# Labels shown to the user -> codes used in the UCI dataset / training data
SEX_OPTIONS = {"Male": 1, "Female": 0}
CP_OPTIONS = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as err:                              # missing file / version mismatch
    st.error(f"Could not load the trained model: {err}")
    st.stop()

if list(model.feature_names_in_) != FEATURES:
    st.error("The saved model does not use the expected four input features.")
    st.stop()

st.title("❤️ Heart Disease Prediction")
st.markdown(
    """
    This app uses a **Support Vector Classifier (SVC)** trained on the UCI
    **Heart Disease (Cleveland)** dataset. Enter the four patient details below
    and the model predicts whether heart disease is **present** or **absent**.
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1,
                          help="Training data covers ages 29-77.")
    sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()))
with col2:
    cp_label = st.selectbox("Chest pain type (cp)", list(CP_OPTIONS.keys()), index=3)
    thalach = st.number_input("Maximum heart rate achieved (bpm)", min_value=40, max_value=250,
                              value=150, step=1, help="Training data covers 71-202 bpm.")

if st.button("Predict", type="primary", use_container_width=True):
    # Input validation (outside the range seen in training the result is unreliable)
    if not (1 <= age <= 120) or not (40 <= thalach <= 250):
        st.error("Please enter a valid age (1-120) and maximum heart rate (40-250 bpm).")
    else:
        user_input = pd.DataFrame({
            "age": [int(age)],
            "sex": [SEX_OPTIONS[sex_label]],
            "cp": [CP_OPTIONS[cp_label]],
            "thalach": [int(thalach)],
        })[FEATURES]

        prediction = int(model.predict(user_input)[0])

        st.divider()
        if prediction == 1:
            st.error("### ⚠️ Heart disease PRESENT (predicted class 1)")
        else:
            st.success("### ✅ NO heart disease (predicted class 0)")

        if not (29 <= age <= 77) or not (71 <= thalach <= 202):
            st.warning("Some values are outside the range of the training data (age 29-77, "
                       "max heart rate 71-202), so this prediction is less reliable.")

        with st.expander("See the four inputs sent to the model"):
            st.dataframe(user_input, hide_index=True)

st.divider()
st.caption(
    "⚠️ Educational use only. This is a university assignment model trained on 303 patients "
    "using just four attributes. It is NOT a medical diagnosis tool and cannot confirm or rule "
    "out heart disease. Data: UCI Heart Disease (Cleveland) - R. Detrano, M.D., Ph.D., "
    "Cleveland Clinic Foundation."
)
