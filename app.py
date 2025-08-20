import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("HR Dataset.csv")
    return df

# Load model
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
rf_model = load_model()

# Sidebar
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["EDA", "Predict Attrition"])

# ================== EDA ================== #
if options == "EDA":
    st.title("📊 Employee Attrition - EDA")

    st.subheader("Attrition Distribution")
    fig = px.histogram(df, x="Attrition", color="Attrition", title="Attrition Distribution")
    st.plotly_chart(fig)

    st.subheader("Department-wise Attrition")
    fig = px.histogram(df, x="Department", color="Attrition", barmode="group", title="Attrition by Department")
    st.plotly_chart(fig)

    st.subheader("Age vs Attrition")
    fig = px.box(df, x="Attrition", y="Age", color="Attrition", title="Age vs Attrition")
    st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig)

# ================== Prediction ================== #
if options == "Predict Attrition":
    st.title("🔮 Employee Attrition Prediction")

    st.write("Fill in the employee details:")

    # Collect features from user
    features = {}
    for col in df.drop(["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1).columns:
        if df[col].dtype == "object":
            features[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            features[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    # Convert to dataframe
    input_df = pd.DataFrame([features])

    # Encode categorical vars same way as training
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    if st.button("Predict"):
        prediction = rf_model.predict(input_df)[0]
        if prediction == 1:
            st.error(⚠️ Employee is likely to Attrite")
        else:
            st.success("✅ Employee is NOT likely to Attrite")
