import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Multi-Risk Health Engine", layout="wide")

# --- CSS Styling ---



st.markdown("""



    <style>



    .main { background-color: #f5f7f9; }



    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

    /* 4. Force Metric VALUE (The percentage/number) to black */
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
    }

    </style>



    """, unsafe_allow_html=True)

# --- Logic: Data Cleaning & Preprocessing ---
def clean_dataset(df):
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

@st.cache_resource
def train_models():
    """Loads data and trains models. Cached to run only once."""
    try:
        # Loading datasets
        df_diabetes = pd.read_csv('diabetes_dataset_new.csv')
        df_heart = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
        df_ckd = pd.read_csv('CKD_NHANES_2021_2023.csv')
        
        # Preprocessing Diabetes
        df_diab_c = clean_dataset(df_diabetes)
        for col in ['bmi', 'hbA1c_level', 'blood_glucose_level']:
            q99 = df_diab_c[col].quantile(0.99)
            df_diab_c[col] = np.where(df_diab_c[col] > q99, q99, df_diab_c[col])
        df_diab_c = pd.get_dummies(df_diab_c, columns=['smoking_history', 'gender', 'location'], drop_first=True)
        
        X_diab = df_diab_c.drop(columns=['diabetes'])
        y_diab = df_diab_c['diabetes']
        scaler_diab = StandardScaler().fit(X_diab)
        model_diab = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42).fit(scaler_diab.transform(X_diab), y_diab)

        # Preprocessing Heart Disease
        df_heart_c = clean_dataset(df_heart)
        X_hrt = df_heart_c.drop(columns=['HeartDiseaseorAttack'])
        y_hrt = df_heart_c['HeartDiseaseorAttack']
        scaler_hrt = StandardScaler().fit(X_hrt)
        model_heart = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42).fit(scaler_hrt.transform(X_hrt), y_hrt)

        # Preprocessing CKD (Integrated Engine)
        df_ckd_c = clean_dataset(df_ckd)
        df_ckd_m = df_ckd_c.copy()
        # Derive synthetic risk for training the engine
        df_ckd_m['pred_risk_diabetes'] = model_diab.predict_proba(scaler_diab.transform(X_diab.iloc[:len(df_ckd_m)]))[:, 1] * 100
        df_ckd_m['pred_risk_heart'] = model_heart.predict_proba(scaler_hrt.transform(X_hrt.iloc[:len(df_ckd_m)]))[:, 1] * 100
        
        ckd_features = ['age', 'bmi', 'bp_systolic', 'bp_diastolic', 'serum_creatinine', 'egfr', 'pred_risk_diabetes', 'pred_risk_heart']
        X_ckd = df_ckd_m[ckd_features]
        y_ckd = df_ckd_m['ckd_present']
        
        model_ckd = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42).fit(X_ckd, y_ckd)
        
        return model_diab, model_heart, model_ckd, scaler_diab, scaler_hrt, X_diab, X_hrt, ckd_features, (X_ckd, y_ckd)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None

# --- Application UI ---
st.title("🏥 Multi-Risk Health Predictive Engine")
st.markdown("Interconnected Metabolic & Vascular Risk Assessment")

with st.spinner("Initializing Clinical Engine & Training Models..."):
    models = train_models()

if models:
    model_diab, model_heart, model_ckd, scaler_diab, scaler_hrt, X_diab_cols, X_hrt_cols, ckd_features, (X_ckd_test, y_ckd_test) = models

    # --- Sidebar: Patient Input ---
    st.sidebar.header("Patient Biomarkers")
    age = st.sidebar.slider("Age", 18, 100, 45)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    sys_bp = st.sidebar.number_input("Systolic Blood Pressure", 80, 200, 120)
    dia_bp = st.sidebar.number_input("Diastolic Blood Pressure", 40, 120, 80)
    creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.0)
    egfr = st.sidebar.number_input("eGFR (mL/min/1.73m²)", 10, 150, 90)

    # --- Main Dashboard ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Risk Profile")
        
        # Calculate Intermediate Risks (Using medians for missing categorical data)
        raw_diab = np.zeros((1, len(X_diab_cols.columns)))
        prob_diab = model_diab.predict_proba(scaler_diab.transform(raw_diab))[0][1] * 100

        raw_hrt = np.zeros((1, len(X_hrt_cols.columns)))
        prob_heart = model_heart.predict_proba(scaler_hrt.transform(raw_hrt))[0][1] * 100

        # Integrated CKD Prediction
        live_input = np.array([[age, bmi, sys_bp, dia_bp, creatinine, egfr, prob_diab, prob_heart]])
        final_ckd_prob = model_ckd.predict_proba(live_input)[0][1] * 100

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Diabetes Risk", f"{prob_diab:.1f}%")
        m2.metric("Heart Risk", f"{prob_heart:.1f}%")
        m3.metric("CKD Risk", f"{final_ckd_prob:.1f}%", delta_color="inverse")

        if final_ckd_prob > 50:
            st.error("⚠️ **High Risk Detected:** Clinical intervention recommended.")
        elif final_ckd_prob > 20:
            st.warning("Keep Monitoring: Moderate risk of chronic conditions.")
        else:
            st.success("Low Risk: Maintain current lifestyle and regular checkups.")

    with col2:
        st.subheader("Risk Engine Insights")
        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        importances = model_ckd.feature_importances_
        indices = np.argsort(importances)
        sns.barplot(x=importances[indices], y=np.array(ckd_features)[indices], palette="viridis", ax=ax)
        ax.set_title("Key Risk Drivers for CKD")
        st.pyplot(fig)

    # --- Visualization Section ---
    st.divider()
    if st.checkbox("Show Technical Model Analytics"):
        st.subheader("Model Performance Evaluation")
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            # Confusion Matrix
            preds_ckd = model_ckd.predict(X_ckd_test)
            cm = confusion_matrix(y_ckd_test, preds_ckd)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title("CKD Engine Confusion Matrix")
            st.pyplot(fig_cm)

        with t_col2:
            # ROC Curve
            probs_ckd = model_ckd.predict_proba(X_ckd_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_ckd_test, probs_ckd)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', label=f'AUC: {roc_auc_score(y_ckd_test, probs_ckd):.2f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax_roc.set_title("ROC Curve (CKD Engine)")
            ax_roc.legend()
            st.pyplot(fig_roc)
else:
    st.info("Please ensure the CSV datasets are in the same folder as this script to begin.")
