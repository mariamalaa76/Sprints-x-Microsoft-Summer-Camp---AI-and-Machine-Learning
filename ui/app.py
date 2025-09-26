import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("E:/Heart_Disease_Project/models/final_model.pkl")

st.title("❤️ Heart Disease Prediction App")

st.sidebar.header("🧑‍⚕️ Input Patient Data")

# -----------------------------
# Sidebar Inputs
# -----------------------------
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_blood_pressure = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
cholestoral = st.sidebar.number_input("Cholestoral", 100, 600, 200)
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
rest_ecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
Max_heart_rate = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exercise_induced_angina = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
vessels_colored_by_flourosopy = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
thalassemia = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

# -----------------------------
# Create Input DataFrame with EXACT training column names
# -----------------------------
input_data = pd.DataFrame([[
    age, sex, chest_pain_type, resting_blood_pressure, cholestoral,
    fasting_blood_sugar, rest_ecg, Max_heart_rate, exercise_induced_angina,
    oldpeak, slope, vessels_colored_by_flourosopy, thalassemia
]], columns=[
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholestoral",
    "fasting_blood_sugar",
    "rest_ecg",
    "Max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "vessels_colored_by_flourosopy",
    "thalassemia"
])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")

# -----------------------------
# Data Exploration Section
# -----------------------------
st.subheader("📊 Dataset Overview")

try:
    df = pd.read_csv("E:\Heart_Disease_Project\data\HeartDiseaseTrain-Test.csv")
    st.write(df.head())

    st.subheader("🔍 Cholestoral vs Heart Disease")
    fig, ax = plt.subplots()
    sns.boxplot(x="target", y="cholestoral", data=df, ax=ax)
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("Dataset not found. Please check the path in app.py")
