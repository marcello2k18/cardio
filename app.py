import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="CardioPredict",
    page_icon="🫀",
    layout="centered"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.stApp { background-color: #0c0c0c; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    color: #f0ede8;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero p { color: #666; font-size: 0.95rem; }
.accent { color: #e05c5c; }

.card {
    background: #161616;
    border: 1px solid #242424;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
}

.section-label {
    color: #666;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.result-high {
    background: linear-gradient(135deg, #2a0f0f, #1a0808);
    border: 1px solid #c94a4a;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-low {
    background: linear-gradient(135deg, #0a1f14, #060f0b);
    border: 1px solid #2e9e62;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    margin-bottom: 0.4rem;
}
.prob-num {
    font-size: 2.5rem;
    font-weight: 600;
    margin: 0.5rem 0 0.25rem 0;
}
.prob-label { color: #666; font-size: 0.82rem; }

.divider { border: none; border-top: 1px solid #1f1f1f; margin: 1.5rem 0; }

.stButton > button {
    background-color: #e05c5c !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background-color: #c94a4a !important; }

.stTabs [data-baseweb="tab"] { color: #666; }
.stTabs [aria-selected="true"] {
    color: #e05c5c !important;
    border-bottom-color: #e05c5c !important;
}

.stDownloadButton > button {
    background-color: #1f1f1f !important;
    color: #f0ede8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    width: 100% !important;
}

.footer {
    text-align: center;
    color: #333;
    font-size: 0.75rem;
    padding: 2rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Constants
# ============================================================
FEATURE_COLS = [
    'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
    'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
    'exerciseangia', 'oldpeak', 'noofmajorvessels'
]
MEDIAN_COLS = ['restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
MODE_COLS   = ['chestpain', 'restingrelectro']

# ============================================================
# Load Artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model       = joblib.load('best_model.pkl')
    impute_vals = joblib.load('impute_vals.pkl')
    return model, impute_vals

try:
    model, impute_vals = load_artifacts()
    model_loaded = True
except:
    model_loaded = False

# ============================================================
# Helpers
# ============================================================
def preprocess(df, impute_vals):
    df = df.copy()
    df['serumcholestrol'] = df['serumcholestrol'].replace(0, np.nan)
    for col in MEDIAN_COLS:
        if col in df.columns:
            df[col].fillna(impute_vals.get(col, df[col].median()), inplace=True)
    for col in MODE_COLS:
        if col in df.columns:
            df[col].fillna(impute_vals.get(col, df[col].mode()[0]), inplace=True)
            df[col] = df[col].astype(int)
    return df[FEATURE_COLS]

def predict_single(data_dict):
    df   = pd.DataFrame([data_dict])
    df_p = preprocess(df, impute_vals)
    prob = model.predict_proba(df_p)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob

def predict_batch(df):
    df_p  = preprocess(df, impute_vals)
    probs = model.predict_proba(df_p)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Cardio<span class="accent">Predict</span></h1>
    <p>Heart disease risk prediction · Machine Learning</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model files not found. Place `best_model.pkl` and `impute_vals.pkl` in the same directory.")
    st.stop()

# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(["  Manual Input  ", "  Upload File  "])

# ──────────────────────────────────────────
# Tab 1 — Manual Input
# ──────────────────────────────────────────
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Patient Information</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age               = st.number_input("Age", 1, 100, 50)
        gender            = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        chestpain         = st.selectbox("Chest Pain Type", [0,1,2,3],
                                format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
        restingBP         = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 120)
        serumcholestrol   = st.number_input("Serum Cholesterol (mg/dl)", 0, 700, 200)
        fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
    with c2:
        restingrelectro  = st.selectbox("Resting ECG", [0,1,2],
                                format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
        maxheartrate     = st.number_input("Max Heart Rate", 60, 220, 150)
        exerciseangia    = st.selectbox("Exercise Induced Angina", [0,1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak          = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, step=0.1)
        noofmajorvessels = st.selectbox("No. of Major Vessels", [0,1,2,3])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict", key="btn_manual"):
        pred, prob = predict_single({
            'age': age, 'gender': gender, 'chestpain': chestpain,
            'restingBP': restingBP, 'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar, 'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate, 'exerciseangia': exerciseangia,
            'oldpeak': oldpeak, 'noofmajorvessels': noofmajorvessels
        })

        if pred == 1:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-title" style="color:#e05c5c;">⚠️ High Risk</div>
                <p style="color:#999;">Indicators of heart disease detected.</p>
                <div class="prob-num" style="color:#e05c5c;">{prob*100:.1f}%</div>
                <div class="prob-label">Probability of Heart Disease</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-title" style="color:#2e9e62;">✓ Low Risk</div>
                <p style="color:#999;">No significant indicators detected.</p>
                <div class="prob-num" style="color:#2e9e62;">{(1-prob)*100:.1f}%</div>
                <div class="prob-label">Probability of No Disease</div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Tab 2 — Upload File
# ──────────────────────────────────────────
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Upload Patient Data</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.85rem;'>Supported: CSV, Excel (.xlsx)</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=['csv', 'xlsx'], label_visibility="collapsed")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.markdown(f"<p style='color:#555; font-size:0.82rem;'>{len(df_up)} rows detected</p>", unsafe_allow_html=True)
            st.dataframe(df_up.head(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Run Prediction", key="btn_batch"):
                missing = [c for c in FEATURE_COLS if c not in df_up.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    preds, probs = predict_batch(df_up)
                    df_result = df_up.copy()
                    df_result['probability_%'] = (probs * 100).round(2)
                    df_result['prediction']    = preds
                    df_result['result']        = df_result['prediction'].map({1:'Heart Disease', 0:'No Disease'})

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='section-label'>Results</p>", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total", len(df_result))
                    m2.metric("High Risk", int(preds.sum()), f"{preds.mean()*100:.1f}%")
                    m3.metric("Low Risk", int((preds == 0).sum()))

                    st.dataframe(
                        df_result[['probability_%', 'prediction', 'result']],
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Results (CSV)",
                        data=df_result.to_csv(index=False).encode('utf-8'),
                        file_name="cardiopredict_results.csv",
                        mime='text/csv'
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<hr class="divider">
<div class="footer">
    CardioPredict — For academic purposes only. Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
