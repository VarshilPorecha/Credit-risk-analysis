import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Fintech Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- ULTRA HD DARK GLASSMORPHISM CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    /* True Dark Background */
    html, body, .stApp {
        background-color: #050505 !important;
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0;
    }

    /* Hide Streamlit Junk */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Title Animations */
    @keyframes glow {
        0% { text-shadow: 0 0 10px #06b6d4; }
        50% { text-shadow: 0 0 20px #3b82f6, 0 0 30px #06b6d4; }
        100% { text-shadow: 0 0 10px #06b6d4; }
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
        animation: glow 3s infinite alternate;
    }
    
    .sub-header {
        font-size: 1.1rem;
        font-weight: 300;
        color: #94a3b8;
        margin-bottom: 50px;
        text-align: center;
    }

    /* Glassmorphism Inputs & Form */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Text Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* Typography inside Form */
    .stMarkdown p {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
    }
    
    h3 {
        color: #e2e8f0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-weight: 500;
    }

    /* Neon Cyber Button */
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        margin-top: 30px !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.8) !important;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%) !important;
    }

    /* Result Dashboard */
    .dashboard-container {
        margin-top: 40px;
    }

    .result-glass-safe {
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.15);
        animation: slideUp 0.6s ease-out;
    }
    
    .result-glass-danger {
        background: rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
        animation: slideUp 0.6s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-text {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .status-safe { color: #34d399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.5); }
    .status-danger { color: #f87171; text-shadow: 0 0 20px rgba(248, 113, 113, 0.6); }

    .prob-subtext {
        font-size: 1.2rem;
        color: #cbd5e1;
        font-weight: 300;
        margin-top: 10px;
    }

    /* Statistics Grid */
    .stat-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.3);
        background: rgba(15, 23, 42, 0.8);
    }
    
    .stat-num {
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }
    
    .stat-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 1px;
        font-weight: 600;
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
st.markdown("<div class='main-header'>QUANTUM RISK ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A unified machine-learning dashboard for algorithmic credit scoring.</div>", unsafe_allow_html=True)

if not model:
    st.error("SYSTEM ERROR: Analytical payload (XGBoost .pkl) not detected in sector /models/.")
    st.stop()

# ----------------- MAIN FORM DASHBOARD -----------------
with st.form("risk_assessment_form"):
    
    # 3-Column Glass Grid
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("<h3>Identity Core</h3>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Biological Sex", ["M", "F", "XNA"])
        no_of_children = st.number_input("Dependents (Children)", min_value=0, max_value=20, value=0)
        total_family_members = st.number_input("Household Size", min_value=1, max_value=20, value=2)
        migrant_worker = st.selectbox("Migratory Status", ["No", "Yes"])
        migrant_worker_val = 1.0 if migrant_worker == "Yes" else 0.0

    with col_b:
        st.markdown("<h3>Capital & Assets</h3>", unsafe_allow_html=True)
        net_yearly_income = st.number_input("Net Income (Annual)", min_value=0, value=50000, step=1000)
        yearly_debt_payments = st.number_input("Debt Obligations (Annual)", min_value=0, value=10000, step=500)
        no_of_days_employed = st.number_input("Employment Tenure (Days)", min_value=0, value=1500, step=100)
        
        occ_types = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                     'High skill tech staff', 'Accountants', 'Medicine staff', 
                     'Security staff', 'Cooking staff', 'Cleaning staff', 
                     'Private service staff', 'Low-skill Laborers', 'Secretaries', 
                     'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff', 'Unknown']
        occupation_type = st.selectbox("Industry Classification", occ_types)
        
        ca, cb = st.columns(2)
        owns_car = ca.selectbox("Auto Asset", ["Y", "N"])
        owns_house = cb.selectbox("Real Estate", ["Y", "N"])

    with col_c:
        st.markdown("<h3>Credit Meta-Data</h3>", unsafe_allow_html=True)
        credit_limit = st.number_input("Approved Credit Volume", min_value=0, value=20000, step=1000)
        credit_limit_used = st.slider("Credit Line Trajectory (%)", min_value=0, max_value=100, value=30)
        credit_score = st.slider("Federated TransUnion Score", min_value=300, max_value=900, value=700)
        
        prev_defaults = st.number_input("Prior Liquidations / Defaults", min_value=0, max_value=50, value=0)
        default_in_last_6months = st.number_input("Defaults (T-6 Months)", min_value=0, max_value=10, value=0)
    
    # Submit Button
    submitted = st.form_submit_button("Synthesize Prediction Model")

# ----------------- PREDICTION LOGIC -----------------
if submitted:
    
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
    st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-glass-danger">
            <div class="status-text status-danger">DEFAULT PROBABILITY CRITICAL</div>
            <div class="prob-subtext">The algorithm projects an irreversible statistical deviation indicating financial collapse.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "CRITICAL"
        risk_color = "#f87171"
    else:
        st.markdown(f"""
        <div class="result-glass-safe">
            <div class="status-text status-safe">PROGNOSIS: SOLVENT</div>
            <div class="prob-subtext">The algorithmic matrix confirms the target's financial markers are securely within institutional risk parameters.</div>
        </div>
        """, unsafe_allow_html=True)
        risk_level = "STABLE" if prediction_proba < 0.25 else "OBSERVE"
        risk_color = "#34d399" if risk_level == "STABLE" else "#fbbf24"

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics Grid
    s1, s2, s3, s4 = st.columns(4)
    
    s1.markdown(f"""
    <div class="stat-box">
        <div class="stat-num" style="color: {risk_color}">{risk_level}</div>
        <div class="stat-label">Vector Status</div>
    </div>
    """, unsafe_allow_html=True)
    
    s2.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{prediction_proba * 100:.1f}%</div>
        <div class="stat-label">Failure Confidence</div>
    </div>
    """, unsafe_allow_html=True)
    
    health = "Omega" if credit_score < 580 else ("Alpha" if credit_score >= 670 else "Beta")
    s3.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{health}</div>
        <div class="stat-label">Credit Sub-Tier</div>
    </div>
    """, unsafe_allow_html=True)

    dti = (yearly_debt_payments / net_yearly_income) * 100 if net_yearly_income > 0 else 0
    s4.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{dti:.1f}%</div>
        <div class="stat-label">Debt Kinetic Ratio</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
