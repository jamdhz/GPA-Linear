import streamlit as st
import numpy as np

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

def predict_gpa(spec, gender, st_hr, school_av, branch, level, age):
    # Nanti pake model
    print("A")

if st.button("Predict GPA"):
    gpa = predict_gpa(spec, gender, st_hr, school_av, branch, level, age)
    st.success(f"Predicted GPA: {gpa}")