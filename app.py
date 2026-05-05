import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon="🧠",
    layout="wide"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .step-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #3a3f5c;
        margin-bottom: 20px;
    }
    .step-badge {
        background: #4f8ef7;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    .result-normal {
        background: linear-gradient(135deg, #0d3320, #1a5c38);
        border: 1px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-warning {
        background: linear-gradient(135deg, #332200, #5c3d00);
        border: 1px solid #f39c12;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-danger {
        background: linear-gradient(135deg, #330d0d, #5c1a1a);
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .mri-prompt {
        background: linear-gradient(135deg, #1a1a3e, #2d2d5e);
        border: 2px dashed #4f8ef7;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        animation: pulse-border 2s infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-color: #4f8ef7; }
        50% { border-color: #a78bfa; }
    }
    .workflow-arrow {
        text-align: center;
        font-size: 28px;
        color: #4f8ef7;
        margin: 10px 0;
    }
    .final-result {
        background: linear-gradient(135deg, #1e1e3a, #2a2a4a);
        border: 2px solid #7c3aed;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .confidence-bar-bg {
        background: #2a2d3e;
        border-radius: 8px;
        height: 12px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD MODELS
# =============================================
@st.cache_resource
def load_models():
    models = {}
    try:
        import tensorflow as tf
        from tensorflow import keras
        for name, fname in [
            ('VGG19',         'VGG19_best.keras'),
            ('ResNet50',       'ResNet50_best.keras'),
            ('EfficientNetB0', 'EfficientNetB0_best.keras'),
            ('1D-CNN',         '1DCNN_clinical_best.keras'),
        ]:
            path = os.path.join('saved_models', fname)
            if os.path.exists(path):
                models[name] = keras.models.load_model(path)
    except Exception as e:
        st.warning(f"TensorFlow models: {e}")

    for name, fname in [
        ('RandomForest', 'RandomForest_clinical.pkl'),
        ('XGBoost',      'XGBoost_clinical.pkl'),
    ]:
        path = os.path.join('saved_models', fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    scaler_path = os.path.join('saved_models', 'clinical_scaler.pkl')
    if os.path.exists(scaler_path):
        models['scaler'] = joblib.load(scaler_path)

    return models


MRI_CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
IMG_SIZE    = (224, 224)

# Threshold: if confidence of "No Alzheimer's" < 80% → request MRI
NORMAL_CONFIDENCE_THRESHOLD = 0.80

# Fusion weights: Clinical 40%, MRI 60%
CLINICAL_WEIGHT = 0.40
MRI_WEIGHT      = 0.60

# =============================================
# HELPERS
# =============================================
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_mri(models, img: Image.Image):
    arr      = preprocess_image(img)
    probas   = {}
    mri_keys = [k for k in ['VGG19', 'ResNet50', 'EfficientNetB0'] if k in models]
    for name in mri_keys:
        p = models[name].predict(arr, verbose=0)[0]
        probas[name] = p
    if not probas:
        return None, None, None
    fused    = np.mean(list(probas.values()), axis=0)
    pred_idx = int(np.argmax(fused))
    return MRI_CLASSES[pred_idx], fused, probas


def predict_clinical(models, features: np.ndarray):
    if 'scaler' not in models:
        return None, None, None
    scaled  = models['scaler'].transform(features)
    results = {}

    for name in ['RandomForest', 'XGBoost']:
        if name in models:
            results[name] = models[name].predict_proba(scaled)[0]

    if '1D-CNN' in models:
        arr = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)
        results['1D-CNN'] = models['1D-CNN'].predict(arr, verbose=0)[0]

    if not results:
        return None, None, None

    avg_proba = np.mean(list(results.values()), axis=0)
    pred_idx  = int(np.argmax(avg_proba))
    return pred_idx, avg_proba, results


def build_features(age, gender, edu, bmi, smoking, alcohol,
                   activity, family_hist, diabetes, hypertension,
                   depression, cholesterol, systolic, diastolic,
                   mmse, moca, adl, func_assess, models):
    gender_enc  = 1 if gender == "Male" else 0
    edu_enc     = ["None", "High School", "Bachelor's", "Higher"].index(edu)
    smoking_enc = 1 if smoking      == "Yes" else 0
    family_enc  = 1 if family_hist  == "Yes" else 0
    diab_enc    = 1 if diabetes     == "Yes" else 0
    hypert_enc  = 1 if hypertension == "Yes" else 0
    depres_enc  = 1 if depression   == "Yes" else 0

    features = np.array([[
        age, gender_enc, edu_enc, bmi, smoking_enc,
        alcohol, activity, family_enc, diab_enc,
        hypert_enc, depres_enc, cholesterol, systolic,
        diastolic, mmse, moca, adl, func_assess
    ]])

    expected = models['scaler'].n_features_in_
    if features.shape[1] < expected:
        features = np.pad(features, ((0, 0), (0, expected - features.shape[1])))
    elif features.shape[1] > expected:
        features = features[:, :expected]

    return features


def mri_to_binary(mri_class: str, mri_proba: np.ndarray):
    """Convert 4-class MRI result to binary (0=Normal, 1=Alzheimer's)."""
    non_demented_idx = MRI_CLASSES.index('NonDemented')
    prob_normal   = float(mri_proba[non_demented_idx])
    prob_alz      = 1.0 - prob_normal
    return np.array([prob_normal, prob_alz])


def fuse_results(clinical_proba: np.ndarray, mri_binary: np.ndarray):
    """Weighted fusion: Clinical 40% + MRI 60%."""
    fused    = CLINICAL_WEIGHT * clinical_proba + MRI_WEIGHT * mri_binary
    pred_idx = int(np.argmax(fused))
    return pred_idx, fused


def render_progress_bar(label, value, color="#4f8ef7"):
    pct = value * 100
    st.markdown(f"""
    <div style="margin: 6px 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
            <span style="color:#ccc; font-size:13px;">{label}</span>
            <span style="color:#fff; font-size:13px; font-weight:bold;">{pct:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div style="width:{pct}%; background:{color}; height:12px; border-radius:8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================
# SESSION STATE
# =============================================
if 'step' not in st.session_state:
    st.session_state.step = 1          # 1=Clinical, 2=MRI, 3=Final
if 'clinical_proba' not in st.session_state:
    st.session_state.clinical_proba = None
if 'clinical_pred' not in st.session_state:
    st.session_state.clinical_pred = None
if 'clinical_indiv' not in st.session_state:
    st.session_state.clinical_indiv = None


# =============================================
# HEADER
# =============================================
st.markdown("""
<div style="text-align:center; padding: 30px 0 10px 0;">
    <h1 style="font-size:2.8rem; font-weight:800; color:#fff; letter-spacing:-1px;">
        🧠 Alzheimer's Detection System
    </h1>
    <p style="color:#8892b0; font-size:1.1rem;">
        Multi-step AI Diagnosis — Clinical Data → MRI Scan → Final Result
    </p>
</div>
""", unsafe_allow_html=True)

# Workflow diagram
col_a, col_b, col_c, col_d, col_e = st.columns([2, 0.5, 2, 0.5, 2])
with col_a:
    active = st.session_state.step >= 1
    border = "#4f8ef7" if active else "#3a3f5c"
    st.markdown(f"""
    <div style="border:2px solid {border}; border-radius:12px; padding:14px; text-align:center;
                background:{'#1a2040' if active else '#1e2130'};">
        <div style="font-size:1.5rem;">📋</div>
        <div style="color:#fff; font-weight:700;">Step 1</div>
        <div style="color:#8892b0; font-size:12px;">Clinical Data</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="workflow-arrow" style="padding-top:22px;">→</div>', unsafe_allow_html=True)
with col_c:
    active = st.session_state.step >= 2
    border = "#4f8ef7" if active else "#3a3f5c"
    st.markdown(f"""
    <div style="border:2px solid {border}; border-radius:12px; padding:14px; text-align:center;
                background:{'#1a2040' if active else '#1e2130'};">
        <div style="font-size:1.5rem;">🖼️</div>
        <div style="color:#fff; font-weight:700;">Step 2</div>
        <div style="color:#8892b0; font-size:12px;">MRI Scan (if needed)</div>
    </div>""", unsafe_allow_html=True)
with col_d:
    st.markdown('<div class="workflow-arrow" style="padding-top:22px;">→</div>', unsafe_allow_html=True)
with col_e:
    active = st.session_state.step >= 3
    border = "#7c3aed" if active else "#3a3f5c"
    st.markdown(f"""
    <div style="border:2px solid {border}; border-radius:12px; padding:14px; text-align:center;
                background:{'#1a1a3e' if active else '#1e2130'};">
        <div style="font-size:1.5rem;">✅</div>
        <div style="color:#fff; font-weight:700;">Step 3</div>
        <div style="color:#8892b0; font-size:12px;">Final Diagnosis</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

models = load_models()
loaded = [k for k in models if k != 'scaler']
if not loaded:
    st.error("❌ No models found — make sure `saved_models/` folder exists next to app.py")
    st.stop()

# =============================================
# STEP 1 — CLINICAL DATA
# =============================================
st.markdown("""
<div class="step-card">
    <h2 style="color:#fff; margin:0 0 16px 0;">
        <span class="step-badge">1</span> Clinical Data Analysis
    </h2>
</div>
""", unsafe_allow_html=True)

clin_models = [k for k in ['RandomForest', 'XGBoost', '1D-CNN'] if k in models]

if not clin_models or 'scaler' not in models:
    st.warning("⚠️ Clinical models not loaded.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Demographics**")
        age    = st.number_input("Age",            min_value=50, max_value=100, value=70)
        gender = st.selectbox("Gender",            ["Male", "Female"])
        edu    = st.selectbox("Education Level",   ["None", "High School", "Bachelor's", "Higher"])

    with col2:
        st.markdown("**🏃 Lifestyle**")
        bmi      = st.number_input("BMI",                         min_value=15.0, max_value=45.0, value=25.0)
        smoking  = st.selectbox("Smoking",                        ["No", "Yes"])
        alcohol  = st.number_input("Alcohol (units/week)",        min_value=0.0,  max_value=20.0, value=0.0)
        activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=10.0, value=3.0)

    with col3:
        st.markdown("**🏥 Medical History**")
        family_hist  = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
        diabetes     = st.selectbox("Diabetes",                      ["No", "Yes"])
        hypertension = st.selectbox("Hypertension",                  ["No", "Yes"])
        depression   = st.selectbox("Depression",                    ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**🧪 Cognitive Assessments**")
        mmse = st.slider("MMSE Score (0–30)",  0, 30, 25, help="Higher = better cognition")
        moca = st.slider("MoCA Score (0–30)",  0, 30, 22)
        adl  = st.slider("ADL Score (0–10)",   0, 10,  8, help="Activities of Daily Living")

    with col5:
        st.markdown("**💊 Clinical Measurements**")
        cholesterol = st.number_input("Total Cholesterol", min_value=100.0, max_value=350.0, value=200.0)
        systolic    = st.number_input("Systolic BP",       min_value=80,    max_value=200,   value=120)
        diastolic   = st.number_input("Diastolic BP",      min_value=50,    max_value=130,   value=80)
        func_assess = st.slider("Functional Assessment (0–10)", 0, 10, 8)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyze Clinical Data", type="primary", use_container_width=True):
        # Reset everything before re-running
        for key in ['clinical_proba', 'clinical_pred', 'clinical_indiv', 'mri_class', 'mri_proba']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.step = 1

        features = build_features(
            age, gender, edu, bmi, smoking, alcohol, activity,
            family_hist, diabetes, hypertension, depression,
            cholesterol, systolic, diastolic, mmse, moca, adl,
            func_assess, models
        )
        with st.spinner("Running clinical models..."):
            try:
                pred_idx, avg_proba, indiv_results = predict_clinical(models, features)
                st.session_state.clinical_proba = avg_proba
                st.session_state.clinical_pred  = pred_idx
                st.session_state.clinical_indiv = indiv_results

                prob_normal = float(avg_proba[0]) if avg_proba is not None else 0

                if prob_normal >= NORMAL_CONFIDENCE_THRESHOLD:
                    st.session_state.step = 3
                else:
                    st.session_state.step = 2

                st.rerun()
            except Exception as e:
                st.error(f"Prediction error: {e}")

# =============================================
# CLINICAL RESULT DISPLAY (after step 1)
# =============================================
if st.session_state.clinical_proba is not None:
    avg_proba    = st.session_state.clinical_proba
    pred_idx     = st.session_state.clinical_pred
    indiv_results= st.session_state.clinical_indiv
    prob_normal  = float(avg_proba[0])
    prob_alz     = float(avg_proba[1]) if len(avg_proba) > 1 else 1 - prob_normal

    st.markdown("---")
    st.markdown("### 📊 Clinical Analysis Result")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        render_progress_bar("No Alzheimer's", prob_normal, "#2ecc71")
        render_progress_bar("Alzheimer's Risk", prob_alz, "#e74c3c")

    with col_r2:
        if indiv_results:
            with st.expander("Individual Model Details"):
                classes = ["No Alzheimer's", "Alzheimer's"]
                for mname, proba in indiv_results.items():
                    p_idx  = np.argmax(proba)
                    p_conf = float(np.max(proba)) * 100
                    p_lbl  = classes[p_idx] if p_idx < len(classes) else f"Class {p_idx}"
                    st.write(f"**{mname}** → {p_lbl} ({p_conf:.1f}%)")

    if st.session_state.step == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="mri-prompt">
            <div style="font-size:2.5rem;">🖼️</div>
            <h3 style="color:#4f8ef7; margin:10px 0 6px 0;">MRI Scan Required</h3>
            <p style="color:#ccc; margin:0;">
                Clinical confidence is <strong style="color:#f39c12;">{prob_normal*100:.1f}%</strong> Normal
                (threshold: {NORMAL_CONFIDENCE_THRESHOLD*100:.0f}%)<br>
                Please upload an MRI scan to confirm the diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# STEP 2 — MRI SCAN (only if suspicious)
# =============================================
if st.session_state.step >= 2 and st.session_state.clinical_proba is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card">
        <h2 style="color:#fff; margin:0 0 16px 0;">
            <span class="step-badge">2</span> MRI Brain Scan
        </h2>
    </div>
    """, unsafe_allow_html=True)

    mri_models_available = [k for k in ['VGG19', 'ResNet50', 'EfficientNetB0'] if k in models]

    if not mri_models_available:
        st.warning("⚠️ No MRI models loaded.")
    else:
        uploaded = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

        if uploaded:
            img = Image.open(uploaded)
            col_img, col_mri_res = st.columns([1, 1.5])

            with col_img:
                st.image(img, caption="Uploaded MRI", use_column_width=True)

            with col_mri_res:
                with st.spinner("Analyzing MRI..."):
                    mri_class, mri_proba, mri_indiv = predict_mri(models, img)

                if mri_class:
                    st.markdown("**MRI Classification:**")

                    colors = {
                        'NonDemented':      '#2ecc71',
                        'VeryMildDemented': '#f39c12',
                        'MildDemented':     '#e67e22',
                        'ModerateDemented': '#e74c3c',
                    }
                    for cls, prob in zip(MRI_CLASSES, mri_proba):
                        render_progress_bar(cls, float(prob), colors.get(cls, "#4f8ef7"))

                    if len(mri_indiv) > 1:
                        with st.expander("Individual MRI Model Details"):
                            for mname, proba in mri_indiv.items():
                                pred = MRI_CLASSES[np.argmax(proba)]
                                conf = float(np.max(proba)) * 100
                                st.write(f"**{mname}** → {pred} ({conf:.1f}%)")

                    # Save MRI result and go to final step
                    st.session_state.mri_class = mri_class
                    st.session_state.mri_proba = mri_proba
                    st.session_state.step      = 3

# =============================================
# STEP 3 — FINAL DIAGNOSIS
# =============================================
if st.session_state.step == 3 and st.session_state.clinical_proba is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card">
        <h2 style="color:#fff; margin:0 0 4px 0;">
            <span class="step-badge" style="background:#7c3aed;">3</span> Final Diagnosis
        </h2>
    </div>
    """, unsafe_allow_html=True)

    clinical_proba = st.session_state.clinical_proba
    prob_normal    = float(clinical_proba[0])

    # Case A: Normal with high confidence (no MRI was done)
    if prob_normal >= NORMAL_CONFIDENCE_THRESHOLD and not hasattr(st.session_state, 'mri_class'):
        confidence = prob_normal * 100
        st.markdown(f"""
        <div class="result-normal">
            <div style="font-size:3rem;">✅</div>
            <h2 style="color:#2ecc71; margin:10px 0;">No Alzheimer's Detected</h2>
            <p style="color:#aaa;">Based on clinical data alone — MRI not required</p>
            <div style="font-size:1.8rem; color:#fff; font-weight:800; margin-top:12px;">
                Confidence: {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Case B: MRI was done → fuse results
    elif hasattr(st.session_state, 'mri_proba'):
        mri_binary     = mri_to_binary(st.session_state.mri_class, st.session_state.mri_proba)
        final_idx, fused = fuse_results(clinical_proba, mri_binary)
        final_prob_normal = float(fused[0])
        final_prob_alz    = float(fused[1])
        confidence        = max(final_prob_normal, final_prob_alz) * 100

        if final_idx == 0:
            st.markdown(f"""
            <div class="result-normal">
                <div style="font-size:3rem;">✅</div>
                <h2 style="color:#2ecc71; margin:10px 0;">No Alzheimer's Detected</h2>
                <p style="color:#aaa;">Combined Clinical + MRI analysis</p>
                <div style="font-size:1.8rem; color:#fff; font-weight:800; margin-top:12px;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                <div style="font-size:3rem;">⚠️</div>
                <h2 style="color:#e74c3c; margin:10px 0;">Alzheimer's Detected</h2>
                <p style="color:#aaa;">Combined Clinical + MRI analysis</p>
                <div style="font-size:1.8rem; color:#fff; font-weight:800; margin-top:12px;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Fusion breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Fusion Breakdown (Clinical 40% + MRI 60%):**")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Clinical → Normal", f"{float(clinical_proba[0])*100:.1f}%")
        with col_f2:
            st.metric("MRI → Normal",      f"{float(mri_binary[0])*100:.1f}%")
        with col_f3:
            st.metric("Fused → Normal",    f"{final_prob_normal*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Analysis", use_container_width=True):
        for key in ['step', 'clinical_proba', 'clinical_pred', 'clinical_indiv',
                    'mri_class', 'mri_proba']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:12px; padding:10px 0;">
    🎓 Graduation Project — Alzheimer's Detection using Deep Learning & Machine Learning<br>
    ⚠️ For research purposes only. Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
