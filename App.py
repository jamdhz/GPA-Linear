import streamlit as st
import numpy as np
import joblib

st.title("GPA Prediction Model")

# Input fields
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


def predict_gpa(spec, gender, st_hr, school_av, branch, level, age):

    spec, gender, branch = preprocess_input(spec, gender, branch)

    with open('gpa_model_1.pkl', 'rb') as file:
        model = joblib.load(file)

    input_data = np.array([[spec, gender, st_hr, school_av, branch, level, age]])

    gpa_prediction = model.predict(input_data)

    return gpa_prediction[0]


def convert_to_4_scale(gpa):
    return (gpa / 100) * 4

if st.button("Predict GPA"):
    gpa = predict_gpa(spec, gender, st_hr, school_av, branch, level, age)
    st.success(f"Predicted GPA: {gpa}")
    gpa_4_scale = convert_to_4_scale(gpa)
    st.success(f"Predicted GPA: {gpa:.2f} (Raw), {gpa_4_scale:.2f} (0-4 Scale)")