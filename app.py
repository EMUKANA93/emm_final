# employee_performance_app.py
# =============================================
# Streamlit App for Employee Performance Prediction
# =============================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =============================================
# Page Configuration
# =============================================
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="💼",
    layout="wide"
)

# =============================================
# Load Model with Caching
# =============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_employee_performance_model.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# =============================================
# Sidebar - App Navigation
# =============================================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Adjust input values and predict performance.")

# =============================================
# Main Title
# =============================================
st.title("💼 Employee Performance Prediction Dashboard")
st.markdown("Use this interactive tool to predict an employee’s performance rating based on workplace and personal factors.")

st.divider()

# =============================================
# Input Section
# =============================================

st.header("📋 Enter Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    last_salary_hike = st.slider("Last Salary Hike (%)", 0, 25, 10)
    environment_satisfaction = st.slider("Environment Satisfaction (1–5)", 1, 5, 3)

with col2:
    job_satisfaction = st.slider("Job Satisfaction (1–5)", 1, 5, 3)
    years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 3)

with col3:
    work_life_balance = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
    job_involvement = st.slider("Job Involvement (1–5)", 1, 5, 3)

# Collect inputs
input_data = pd.DataFrame({
    'LastSalaryHikePercent': [last_salary_hike],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'JobSatisfaction': [job_satisfaction],
    'YearsSinceLastPromotion': [years_since_promotion],
    'WorkLifeBalance': [work_life_balance],
    'JobInvolvement': [job_involvement]
})

st.markdown("### 🧩 Input Summary")
st.dataframe(input_data, use_container_width=True)

# =============================================
# Prediction Section
# =============================================
st.divider()
st.header("🎯 Prediction Result")

if st.button("Predict Performance"):
    if model is not None:
        try:
            prediction = model.predict(input_data)
            pred_class = int(prediction[0])

            # Human-readable mapping (example)
            perf_mapping = {
                1: "Low Performer 🚫",
                2: "Average Performer ⚖️",
                3: "High Performer 🌟"
            }

            st.success(f"Predicted Employee Performance: **{perf_mapping.get(pred_class, 'Unknown')}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not loaded. Please ensure the file 'best_employee_performance_model.joblib' is in the same directory.")

# =============================================
# Footer Section
# =============================================
st.divider()
st.caption("Developed by Alex Ndiritu — Powered by Streamlit and scikit-learn.")
