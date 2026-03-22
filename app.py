import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    if not os.path.exists('models/xgb_model.pkl'):
        return None, None, None
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, scaler, feature_cols

st.title("💳 Credit Risk Prediction System")
st.markdown("Enter customer details below to predict the likelihood of credit card default using our advanced XGBoost Machine Learning model.")

model, scaler, feature_cols = load_models()

if not model:
    st.error("Error: Models not found! Please ensure 'src/train.py' has been run to generate the model files in the 'models/' directory.")
    st.stop()

# Sidebar for Input Features
st.sidebar.header("📋 Customer Information")
st.sidebar.markdown("Please provide the borrower's details:")

# Create input fields
with st.sidebar.form("input_form"):
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["M", "F", "XNA"])
    owns_car = st.selectbox("Owns a Car?", ["Y", "N"])
    owns_house = st.selectbox("Owns a House?", ["Y", "N"])
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    net_yearly_income = st.number_input("Net Yearly Income ($)", min_value=0, value=50000, step=1000)
    no_of_days_employed = st.number_input("Days Employed", min_value=0, value=1500, step=100)
    
    # Common Occupation Types from dataset
    occ_types = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                 'High skill tech staff', 'Accountants', 'Medicine staff', 
                 'Security staff', 'Cooking staff', 'Cleaning staff', 
                 'Private service staff', 'Low-skill Laborers', 'Secretaries', 
                 'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff', 'Unknown']
    occupation_type = st.selectbox("Occupation Type", occ_types)
    
    total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=2)
    migrant_worker = st.selectbox("Migrant Worker?", ["No", "Yes"])
    migrant_worker_val = 1.0 if migrant_worker == "Yes" else 0.0
    
    yearly_debt_payments = st.number_input("Yearly Debt Payments ($)", min_value=0, value=10000, step=500)
    credit_limit = st.number_input("Credit Limit ($)", min_value=0, value=20000, step=1000)
    credit_limit_used = st.slider("Credit Limit Used (%)", min_value=0, max_value=100, value=30)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
    
    prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=50, value=0)
    default_in_last_6months = st.number_input("Defaults in Last 6 Months", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict Credit Risk 🚀")

# Main Panel Layout
col1, col2 = st.columns([1, 1])

if submitted:
    # Prepare input dictionary
    input_data = {
        'age': age,
        'gender': gender,
        'owns_car': owns_car,
        'owns_house': owns_house,
        'no_of_children': float(no_of_children),
        'net_yearly_income': float(net_yearly_income),
        'no_of_days_employed': float(no_of_days_employed),
        'occupation_type': occupation_type,
        'total_family_members': float(total_family_members),
        'migrant_worker': migrant_worker_val,
        'yearly_debt_payments': float(yearly_debt_payments),
        'credit_limit': float(credit_limit),
        'credit_limit_used(%)': int(credit_limit_used),
        'credit_score': float(credit_score),
        'prev_defaults': int(prev_defaults),
        'default_in_last_6months': int(default_in_last_6months)
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply get_dummies matching the training pipeline
    input_encoded = pd.get_dummies(input_df)
    
    # Reindex to ensure all expected feature columns are present (fill missing ones with 0)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
    
    # Scale Features
    input_scaled = scaler.transform(input_encoded)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # Display Results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    if prediction == 1:
        st.error("🚨 **High Risk of Default Detected!**")
        st.markdown(f"The model predicts the client **WILL DEFAULT** on their credit card with a probability of **{prediction_proba * 100:.2f}%**.")
        st.image("https://img.icons8.com/color/96/000000/cancel--v1.png", width=64)
    else:
        st.success("✅ **Low Risk / Safe Client**")
        st.markdown(f"The model predicts the client **WILL NOT DEFAULT** on their credit card. (Default Probability: **{prediction_proba * 100:.2f}%**)")
        st.image("https://img.icons8.com/color/96/000000/ok--v1.png", width=64)
        
    st.markdown("### Risk Analysis Metrics")
    c1, c2, c3 = st.columns(3)
    
    risk_level = "High" if prediction == 1 else ("Medium" if prediction_proba > 0.3 else "Low")
    risk_color = "red" if risk_level == "High" else ("orange" if risk_level == "Medium" else "green")
    
    c1.markdown(f"<div class='metric-card'><h4 style='color:#7f8c8d'>Risk Level</h4><h2 style='color:{risk_color}'>{risk_level}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h4 style='color:#7f8c8d'>Default Probability</h4><h2>{prediction_proba * 100:.1f}%</h2></div>", unsafe_allow_html=True)
    
    health = "Poor" if credit_score < 580 else ("Fair" if credit_score < 670 else ("Good" if credit_score < 740 else "Excellent"))
    c3.markdown(f"<div class='metric-card'><h4 style='color:#7f8c8d'>Credit Health</h4><h2>{health}</h2></div>", unsafe_allow_html=True)

else:
    st.info("👈 Please fill out the customer details in the sidebar to generate a prediction.")
    
    st.markdown("### How it Works")
    st.markdown("""
    1. **Data Input**: The sidebar allows entry of demographic and financial indicators.
    2. **Processing**: The data is encoded and scaled to match the trained schema dynamically.
    3. **Evaluation**: An **XGBoost Classifier** rigorously evaluates the multi-dimensional risk vector.
    4. **Output**: An immediate classification (Default / No Default) along with the exact probability.
    """)
