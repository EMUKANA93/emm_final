# =============================================================
# 🎯 Employee Performance Predictor - Streamlit Web App
# =============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
import sys
from pathlib import Path

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Performance Prediction App")
st.markdown("""
Estimate an **employee's performance rating** based on key HR metrics.
Fill in the details below and click **Predict Performance**.
""")

# -------------------------------------------------------------
# 2. Environment Info Sidebar
# -------------------------------------------------------------
st.sidebar.title("⚙️ Environment Info")
st.sidebar.write(f"**Python:** {sys.version.split()[0]}")
st.sidebar.write(f"**scikit-learn:** {sklearn.__version__}")
st.sidebar.write(f"**Streamlit:** {st.__version__}")

# -------------------------------------------------------------
# 3. Load the trained model safely
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("best_employee_performance_model.joblib")
    if not model_path.exists():
        st.error("❌ Model file not found. Upload or train the model first.")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Model failed to load: {e}")
        st.stop()

model = load_model()

# -------------------------------------------------------------
# 4. Collect Employee Input Data
# -------------------------------------------------------------
st.subheader("📋 Enter Employee Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
        EmpJobSatisfaction = st.slider("Job Satisfaction (1–5)", 1, 5, 3)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        EmpWorkLifeBalance = st.slider("Work-Life Balance (1–5)", 1, 5, 3)

    with col2:
        EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1–5)", 1, 5, 3)
        EmpJobInvolvement = st.slider("Job Involvement (1–5)", 1, 5, 3)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 2)
        ExperienceYearsAtThisCompany = st.slider("Years at Current Company", 0, 20, 5)
        TotalWorkExperienceInYears = st.slider("Total Work Experience (Years)", 0, 40, 10)

    submitted = st.form_submit_button("🚀 Predict Performance")

# -------------------------------------------------------------
# 5. Prepare Data and Make Prediction
# -------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame({
        "Age": [Age],
        "EmpLastSalaryHikePercent": [EmpLastSalaryHikePercent],
        "EmpJobSatisfaction": [EmpJobSatisfaction],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "EmpWorkLifeBalance": [EmpWorkLifeBalance],
        "EmpEnvironmentSatisfaction": [EmpEnvironmentSatisfaction],
        "EmpJobInvolvement": [EmpJobInvolvement],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "ExperienceYearsAtThisCompany": [ExperienceYearsAtThisCompany],
        "TotalWorkExperienceInYears": [TotalWorkExperienceInYears]
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"⭐ Predicted Employee Performance Rating: **{prediction} / 4**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # -------------------------------------------------------------
    # 6. Display Insights
    # -------------------------------------------------------------
    if prediction >= 4:
        st.markdown("### 🟢 High Performer")
        st.balloons()
        st.caption("This employee shows strong potential for leadership or growth roles.")
    elif prediction == 3:
        st.markdown("### 🟡 Consistent Performer")
        st.caption("Steady performer. Continued engagement and mentorship recommended.")
    else:
        st.markdown("### 🔴 Needs Support")
        st.caption("Consider training, performance improvement plan, or targeted mentoring.")

    st.markdown("---")
    st.markdown("### 💡 Key HR Factors Influencing Performance")
    st.markdown("""
    - 🏆 **TrainingTimesLastYear:** Frequent training correlates with higher performance.  
    - 📈 **YearsSinceLastPromotion:** Long gaps may reduce motivation and engagement.  
    - 👔 **Job Involvement:** High involvement drives stronger results.  
    - ⚖️ **Work-Life Balance:** Stable balance supports consistent productivity.  
    """)

