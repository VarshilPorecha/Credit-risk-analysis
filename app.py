import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CLEAN MODERN CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }
    
    .stApp {
        background-color: #f4f7f6;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Cards */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    h3 {
        color: #334155 !important;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Results */
    .result-safe {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .result-danger {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .status-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .safe-text { color: #166534; }
    .danger-text { color: #991b1b; }

    /* Metrics */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-num {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
st.markdown("<div class='main-header'>Credit Risk Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict the likelihood of credit card default based on customer profiles.</div>", unsafe_allow_html=True)

if not model:
    st.error("Error: Models not found in the 'models/' directory.")
    st.stop()

# ----------------- MAIN FORM DASHBOARD -----------------
with st.form("risk_assessment_form"):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3>Personal Details</h3>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["M", "F", "XNA"])
        no_of_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
        total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=2)
        migrant_worker = st.selectbox("Migrant Worker", ["No", "Yes"])
        migrant_worker_val = 1.0 if migrant_worker == "Yes" else 0.0

    with col2:
        st.markdown("<h3>Employment & Financial</h3>", unsafe_allow_html=True)
        net_yearly_income = st.number_input("Net Yearly Income ($)", min_value=0, value=50000, step=1000)
        yearly_debt_payments = st.number_input("Yearly Debt Payments ($)", min_value=0, value=10000, step=500)
        no_of_days_employed = st.number_input("Days Employed", min_value=0, value=1500, step=100)
        
        occ_types = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                     'High skill tech staff', 'Accountants', 'Medicine staff', 
                     'Security staff', 'Cooking staff', 'Cleaning staff', 
                     'Private service staff', 'Low-skill Laborers', 'Secretaries', 
                     'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff', 'Unknown']
        occupation_type = st.selectbox("Occupation", occ_types)
        
        ca, cb = st.columns(2)
        owns_car = ca.selectbox("Owns Car", ["Y", "N"])
        owns_house = cb.selectbox("Owns House", ["Y", "N"])

    with col3:
        st.markdown("<h3>Credit History</h3>", unsafe_allow_html=True)
        credit_limit = st.number_input("Credit Limit ($)", min_value=0, value=20000, step=1000)
        credit_limit_used = st.slider("Credit Limit Used (%)", min_value=0, max_value=100, value=30)
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
        
        prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=50, value=0)
        default_in_last_6months = st.number_input("Defaults in Last 6 Months", min_value=0, max_value=10, value=0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Using the native Streamlit primary button to completely avoid CSS visibility bugs
    submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

# ----------------- PREDICTION LOGIC -----------------
if submitted:
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
    
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-danger">
            <div class="status-text danger-text">High Risk Detected</div>
            <div style="color: #475569; font-size: 1.1rem;">This customer has a high statistical probability of defaulting on their credit card.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "High"
        risk_color = "#ef4444"
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="status-text safe-text">Low Risk / Safe</div>
            <div style="color: #475569; font-size: 1.1rem;">This customer has a low statistical probability of defaulting.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "Low" if prediction_proba < 0.25 else "Medium"
        risk_color = "#22c55e" if risk_level == "Low" else "#eab308"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Customer Financial Health</h3>", unsafe_allow_html=True)
    
    # Financial Health Metrics
    s1, s2, s3 = st.columns(3)
    
    s1.markdown(f"""
    <div class="metric-card">
        <div class="metric-num" style="color: {risk_color}">{risk_level}</div>
        <div class="metric-label">Model Risk Level</div>
    </div>
    """, unsafe_allow_html=True)
    
    health = "Poor" if credit_score < 580 else ("Good" if credit_score >= 670 else "Fair")
    s2.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{health}</div>
        <div class="metric-label">Credit Bureau Status</div>
    </div>
    """, unsafe_allow_html=True)

    dti = (yearly_debt_payments / net_yearly_income) * 100 if net_yearly_income > 0 else 0
    s3.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{dti:.1f}%</div>
        <div class="metric-label">Debt-to-Income (DTI)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Basel IRB Risk Exposure (Quantitative)</h3>", unsafe_allow_html=True)
    
    # Calculate Basel quantities
    current_balance = float(credit_limit) * (float(credit_limit_used) / 100.0)
    undrawn_amount = float(credit_limit) - current_balance
    ccf = 0.75 # 75% credit conversion factor on undrawn amount
    ead = current_balance + (ccf * undrawn_amount)
    
    lgd = 0.75 # 75% loss given default assumption for unsecured credit cards
    pd_val = prediction_proba
    expected_loss = pd_val * lgd * ead
    
    r1, r2, r3, r4 = st.columns(4)

    r1.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{pd_val * 100:.1f}%</div>
        <div class="metric-label">PD (Probability of Default)</div>
    </div>
    """, unsafe_allow_html=True)

    r2.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">75.0%</div>
        <div class="metric-label">LGD (Assumed)</div>
    </div>
    """, unsafe_allow_html=True)

    r3.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">${ead:,.0f}</div>
        <div class="metric-label">EAD (Exposure at Default)</div>
    </div>
    """, unsafe_allow_html=True)

    r4.markdown(f"""
    <div class="metric-card" style="border: 2px solid #ef4444;">
        <div class="metric-num" style="color: #ef4444;">${expected_loss:,.0f}</div>
        <div class="metric-label">Expected Loss (EL)</div>
    </div>
    """, unsafe_allow_html=True)
