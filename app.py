import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CUSTOM CSS FOR HIGH-DEF UI -----------------
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background-color: #f8fafc;
    }

    /* Hide default Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #64748b;
        margin-bottom: 40px;
    }

    /* Input grouping cards */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
    }

    /* Custom Button Styling */
    .stButton>button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-top: 20px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.15) !important;
        background-color: #1e293b !important;
    }

    /* Result Cards */
    .result-card-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    .status-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .prob-text {
        font-size: 1.2rem;
        color: #475569;
        font-weight: 500;
    }

    /* Metric Containers */
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 5px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_models():
    if not os.path.exists('models/xgb_model.pkl'):
        return None, None, None
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, scaler, feature_cols

model, scaler, feature_cols = load_models()

# ----------------- UI HEADER -----------------
st.markdown("<div class='main-header'>Credit Risk Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A high-definition enterprise tool for behavioral credit risk analysis and default probability assessment.</div>", unsafe_allow_html=True)

if not model:
    st.error("Model artifacts not found. Please verify deployment configuration.")
    st.stop()

# ----------------- MAIN FORM DASHBOARD -----------------
with st.form("risk_assessment_form"):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Demographics**")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["M", "F", "XNA"])
        no_of_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
        total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=2)
        migrant_worker = st.selectbox("Migrant Worker Status", ["No", "Yes"])
        migrant_worker_val = 1.0 if migrant_worker == "Yes" else 0.0

    with col2:
        st.markdown("**Financial & Employment Factors**")
        net_yearly_income = st.number_input("Net Yearly Income (USD)", min_value=0, value=50000, step=1000)
        yearly_debt_payments = st.number_input("Yearly Debt Payments (USD)", min_value=0, value=10000, step=500)
        no_of_days_employed = st.number_input("Total Days Employed", min_value=0, value=1500, step=100)
        
        occ_types = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                     'High skill tech staff', 'Accountants', 'Medicine staff', 
                     'Security staff', 'Cooking staff', 'Cleaning staff', 
                     'Private service staff', 'Low-skill Laborers', 'Secretaries', 
                     'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff', 'Unknown']
        occupation_type = st.selectbox("Occupation Classification", occ_types)
        
        c21, c22 = st.columns(2)
        owns_car = c21.selectbox("Owns Automobile", ["Y", "N"])
        owns_house = c22.selectbox("Owns Property", ["Y", "N"])

    with col3:
        st.markdown("**Credit History Metrics**")
        credit_limit = st.number_input("Available Credit Limit (USD)", min_value=0, value=20000, step=1000)
        credit_limit_used = st.slider("Credit Limit Utilization (%)", min_value=0, max_value=100, value=30)
        credit_score = st.slider("Federated Credit Score", min_value=300, max_value=900, value=700)
        
        prev_defaults = st.number_input("Historical Default Events", min_value=0, max_value=50, value=0)
        default_in_last_6months = st.number_input("Defaults (Trailing 6 Months)", min_value=0, max_value=10, value=0)

    # Empty space for alignment
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit Button
    submitted = st.form_submit_button("Execute Risk Subroutine")

# ----------------- PREDICTION LOGIC -----------------
if submitted:
    st.markdown("<br><hr style='border-color: #e2e8f0;'><br>", unsafe_allow_html=True)
    
    # Structure input mapping identically to training vector
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
    
    # Vectorization & Encoding
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
    
    # Statistical Scaling
    input_scaled = scaler.transform(input_encoded)
    
    # Inference
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # ----------------- RESULTS DASHBOARD -----------------
    st.markdown("<div class='main-header' style='font-size: 1.5rem; margin-bottom: 20px;'>Risk Analysis Output</div>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card-danger">
            <div class="status-title" style="color: #b91c1c;">HIGH RISK PROFILE DETECTED</div>
            <div class="prob-text">The analytical subroutine indicates a severe statistical trajectory toward credit card default.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "Critical"
        risk_color = "#b91c1c"
    else:
        st.markdown(f"""
        <div class="result-card-safe">
            <div class="status-title" style="color: #15803d;">LOW RISK PROFILE VERIFIED</div>
            <div class="prob-text">The behavioral and financial footprint aligns with statistically secure repayment models.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "Secure" if prediction_proba < 0.25 else "Moderate"
        risk_color = "#15803d" if risk_level == "Secure" else "#b45309"

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics row
    r1, r2, r3, r4 = st.columns(4)
    
    r1.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Calculated Risk Index</div>
        <div class="metric-value" style="color: {risk_color}">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)
    
    r2.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Default Probability</div>
        <div class="metric-value">{prediction_proba * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    health = "Subprime" if credit_score < 580 else ("Prime" if credit_score >= 670 else "Near Prime")
    r3.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Credit Tier</div>
        <div class="metric-value">{health}</div>
    </div>
    """, unsafe_allow_html=True)

    dti = (yearly_debt_payments / net_yearly_income) * 100 if net_yearly_income > 0 else 0
    r4.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">DTI Ratio</div>
        <div class="metric-value">{dti:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
