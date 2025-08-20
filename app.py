import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Load dataset
df = pd.read_csv("HR Dataset.csv")

# Load trained Random Forest model
with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("📊 Employee Attrition Prediction App")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Prediction"])

# -------------------------------
# EDA Page
# -------------------------------
if page == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Attrition Count
    fig1 = px.histogram(df, x="Attrition", color="Attrition", title="Attrition Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Department vs Attrition
    fig2 = px.histogram(df, x="Department", color="Attrition", barmode="group", title="Department vs Attrition")
    st.plotly_chart(fig2, use_container_width=True)

    # Job Role vs Attrition
    fig3 = px.histogram(df, x="JobRole", color="Attrition", barmode="group", title="Job Role vs Attrition")
    st.plotly_chart(fig3, use_container_width=True)

    # Age vs Attrition
    fig4 = px.box(df, x="Attrition", y="Age", color="Attrition", title="Age vs Attrition")
    st.plotly_chart(fig4, use_container_width=True)

    # Monthly Income vs Attrition
    fig5 = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition", title="Monthly Income vs Attrition")
    st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Prediction":
    st.subheader("🔮 Predict Employee Attrition")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=60, value=30)
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        JobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        OverTime = st.selectbox("OverTime", ["Yes", "No"])

    with col2:
        DistanceFromHome = st.number_input("Distance From Home (km)", min_value=1, max_value=30, value=5)
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
        YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        WorkLifeBalance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        "Age": [Age],
        "MonthlyIncome": [MonthlyIncome],
        "JobSatisfaction": [JobSatisfaction],
        "OverTime": [1 if OverTime == "Yes" else 0],
        "DistanceFromHome": [DistanceFromHome],
        "TotalWorkingYears": [TotalWorkingYears],
        "YearsAtCompany": [YearsAtCompany],
        "WorkLifeBalance": [WorkLifeBalance]
    })

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ The employee is likely to LEAVE (Yes Attrition).")
        else:
            st.success("✅ The employee is NOT likely to leave (No Attrition).")
