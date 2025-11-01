# ============================================================
# 🧠 Employee Performance Predictor - Streamlit Web App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎯 Employee Performance Predictor")
st.markdown("""
This tool helps HR professionals estimate an **employee's performance rating** 
based on key factors from historical data.  
Just fill in the details below and click **Predict Performance**!
""")

# ------------------------------------------------------------
# 2. Load Trained Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_employee_performance_model.pkl")
        return model
    except:
        st.error("Model file not found. Please ensure 'random_forest_employee_performance_model.pkl' is in the app directory.")
        return None

model = load_model()

# ------------------------------------------------------------
# 3. Sidebar or Main Form for Input
# ------------------------------------------------------------
st.subheader("📋 Enter Employee Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 10)
        EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        EmpJobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    with col2:
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        EmpWorkLifeBalance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        EmpJobInvolvement = st.slider("Job Involvement (1-4)", 1, 4, 3)

    submitted = st.form_submit_button("🚀 Predict Performance")

# ------------------------------------------------------------
# 4. Make Prediction
# ------------------------------------------------------------
if submitted:
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame({
            'EmpLastSalaryHikePercent': [EmpLastSalaryHikePercent],
            'EmpEnvironmentSatisfaction': [EmpEnvironmentSatisfaction],
            'EmpJobSatisfaction': [EmpJobSatisfaction],
            'YearsSinceLastPromotion': [YearsSinceLastPromotion],
            'EmpWorkLifeBalance': [EmpWorkLifeBalance],
            'EmpJobInvolvement': [EmpJobInvolvement]
        })

        # Align features (if model trained with more columns)
        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # ------------------------------------------------------------
        # 5. Display Result
        # ------------------------------------------------------------
        st.success(f"⭐ Predicted Performance Rating: **{prediction:.2f} / 4**")

        if prediction >= 3.5:
            st.markdown("### 🟢 Excellent Performance Potential!")
            st.balloons()
        elif prediction >= 3.0:
            st.markdown("### 🟡 Good Performer")
        else:
            st.markdown("### 🔴 Needs Development Support")

        st.markdown("---")
        st.caption("Prediction powered by machine learning model trained on INX Future Inc. employee data.")
