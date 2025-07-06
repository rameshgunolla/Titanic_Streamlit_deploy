import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Base dir
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'

# Load model + preprocessor
try:
    model = joblib.load(MODEL_DIR / 'model.joblib')
    preprocessor = joblib.load(MODEL_DIR / 'preprocessor.joblib')
    print("Model and preprocessor loaded.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🚢 Titanic Survival Prediction")

# Input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=20.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [0 if sex == 'male' else 1],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'FamilySize': [sibsp + parch + 1]
    })

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    st.write(f"🎉 Prediction: **{'Survived' if prediction == 1 else 'Did NOT survive'}**")
