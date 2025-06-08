import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('model/random_forest_model.pkl')
feature_cols = joblib.load('model/feature_columns.pkl')

# App Title
st.title("Mental Health Prediction in Tech Industry")

# User Inputs
age = st.slider("Age", 18, 70, 25)
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
self_employed = st.selectbox("Self-employed", ['Yes', 'No'])
family_history = st.selectbox("Family history of mental illness", ['Yes', 'No'])
remote_work = st.selectbox("Remote work", ['Yes', 'No'])
work_interfere = st.selectbox("Work interfere with mental health", ['Never', 'Rarely', 'Sometimes', 'Often'])

# Extra multi-choice fields
benefits = st.selectbox("Mental health benefits", ['Yes', 'No', 'Unknown'])
care_options = st.selectbox("Care options provided", ['Yes', 'No', 'Unknown'])
anonymity = st.selectbox("Anonymity assured", ['Yes', 'No', 'Unknown'])
coworkers = st.selectbox("Discuss with coworkers", ['Yes', 'No', 'Unknown'])

# Preprocess Input
input_dict = {
    'age': age,
    'gender': 0 if gender == 'Male' else 1 if gender == 'Female' else 2,
    'self_employed': 1 if self_employed == 'Yes' else 0,
    'family_history': 1 if family_history == 'Yes' else 0,
    'remote_work': 1 if remote_work == 'Yes' else 0,
    'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}[work_interfere],
    'benefits_Yes': 1 if benefits == 'Yes' else 0,
    'benefits_Unknown': 1 if benefits == 'Unknown' else 0,
    'care_options_Yes': 1 if care_options == 'Yes' else 0,
    'care_options_Unknown': 1 if care_options == 'Unknown' else 0,
    'anonymity_Yes': 1 if anonymity == 'Yes' else 0,
    'anonymity_Unknown': 1 if anonymity == 'Unknown' else 0,
    'coworkers_Yes': 1 if coworkers == 'Yes' else 0,
    'coworkers_Unknown': 1 if coworkers == 'Unknown' else 0,
}

# Fill missing one-hot features with 0
for col in feature_cols:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[feature_cols]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("Needs Treatment" if prediction == 1 else "No Treatment Required")
