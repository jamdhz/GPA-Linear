import streamlit as st
import numpy as np
import joblib

st.title("GPA Prediction Model")

st.header("Input Parameters")

spec = st.selectbox("Specialization (Spec)", ["CS", "MIS", "ACC", "BA"])
gender = st.selectbox("Gender", ["F", "M"])
st_hr = st.slider("Study Hours (St_Hr)", 1, 8, 1)
school_av = st.slider("School Average (School_AV)", 0, 100, 50)
branch = st.selectbox("Branch", ["P", "A"])
level = st.slider("Level", 1, 6, 1)
age = st.slider("Age", 18, 25, 18)

def preprocess_input(spec, gender, branch):
    spec_map = {"CS": 0, "MIS": 1, "ACC": 2, "BA": 3}
    gender_map = {"F": 0, "M": 1}
    branch_map = {"P": 0, "A": 1}
    return spec_map[spec], gender_map[gender], branch_map[branch]

def predict_gpa(model_file, spec, gender, st_hr, school_av, branch, level, age):
    spec, gender, branch = preprocess_input(spec, gender, branch)
    input_data = np.array([[spec, gender, st_hr, school_av, branch, level, age]])
    model = joblib.load(model_file)

    if model_file == 'gpa_model_2.pkl':
        poly = joblib.load('poly_transform.pkl')
        scaler = joblib.load('scaler.pkl')
        input_data = poly.transform(input_data)
        input_data = scaler.transform(input_data)
    return model.predict(input_data)[0]

def predict_gpa_best(model_file, spec, gender, st_hr, school_av, branch, level, age):
    spec, gender, branch = preprocess_input(spec, gender, branch)
    input_data = np.array([[spec, gender, st_hr, school_av, branch, level, age]])
    model = joblib.load(model_file)
    return model.predict(input_data)[0]

def predict_gpa_selected(model_file, branch, school_av, gender, st_hr):
    branch_map = {"P": 0, "A": 1}
    gender_map = {"F": 0, "M": 1}
    branch = branch_map[branch]
    gender = gender_map[gender]
    input_data = np.array([[branch, school_av, gender, st_hr]])
    scaler = joblib.load('scaler_selected.pkl')
    model = joblib.load(model_file)
    input_data = scaler.transform(input_data)
    return model.predict(input_data)[0]

def convert_to_4_scale(gpa):
    return (gpa / 100) * 4

model_option = st.radio(
    "Choose Model",
    (
        "Model 1: Linear Regression (All Features)",
        "Model 2: Ridge Regression + Poly Features",
        "Model 3: XGBoost (Best Model)",
        "Model 4: Linear Regression (Selected Features)"
    )
)

if st.button("Predict GPA"):
    if model_option == "Model 1: Linear Regression (All Features)":
        gpa = predict_gpa('gpa_model_1.pkl', spec, gender, st_hr, school_av, branch, level, age)
    elif model_option == "Model 2: Ridge Regression + Poly Features":
        gpa = predict_gpa('gpa_model_2.pkl', spec, gender, st_hr, school_av, branch, level, age)
    elif model_option == "Model 3: XGBoost (Best Model)":
        gpa = predict_gpa_best('best_model.pkl', spec, gender, st_hr, school_av, branch, level, age)
    elif model_option == "Model 4: Linear Regression (Selected Features)":
        gpa = predict_gpa_selected('model_selected.pkl', branch, school_av, gender, st_hr)
    else:
        gpa = None

    if gpa is not None:
        gpa_4_scale = convert_to_4_scale(gpa)
        st.success(f"Predicted GPA: {gpa:.2f} (Raw), {gpa_4_scale:.2f} (0-4 Scale)")