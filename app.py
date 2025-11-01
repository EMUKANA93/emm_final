# =============================================================
# STREAMLIT APP: EMPLOYEE PERFORMANCE PREDICTOR FOR HR
# Optimized for clarity, efficiency, and HR usability
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sys
from pathlib import Path

# -------------------------------
# 🧠 Sidebar: Environment Info
# -------------------------------
st.sidebar.title("⚙️ Environment Info")
st.sidebar.write(f"**Python:** {sys.version.split()[0]}")
st.sidebar.write(f"**Streamlit:** {st.__version__}")
st.sidebar.write(f"**scikit-learn:** {sklearn.__version__}")

# -------------------------------
# 🧩 Load the trained model safely
# -------------------------------
model_path = Path("best_employee_performance_model.joblib")

@st.cache_resource
def load_model():
    if not model_path.exists():
        st.error("🚫 Trained model not found. Please retrain and save as 'best_employee_performance_model.joblib'.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

model = load_model()

# -------------------------------
# 🎯 App Header
# -------------------------------
st.set_page_config(page_title="Employee Performance Predictor", page_icon="💼", layout="wide")
st.title("💼 Employee Performance Prediction App")
st.markdown("""
Predict an **employee's performance rating** based on their professional and personal factors.  
Use this tool to assist **hiring, internal assessments, and career development planning**.
""")

# -------------------------------
# 🧾 Candidate Inputs
# -------------------------------
st.header("🧾 Candidate Information")

with st.expander("Expand to Fill Candidate Details", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        EducationBackground = st.selectbox("Education Background", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree",
            "Human Resources", "Other"
        ])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        EmpDepartment = st.selectbox("Department", [
            "Sales", "Development", "Research & Development", "Human Resources"
        ])
        EmpJobRole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Manager", "Laboratory Technician",
            "Developer", "Manufacturing Director", "Healthcare Representative"
        ])

    with col2:
        BusinessTravelFrequency = st.selectbox("Business Travel Frequency", [
            "Non-Travel", "Travel_Rarely", "Travel_Frequently"
        ])
        DistanceFromHome = st.slider("Distance from Home (km)", 0, 30, 10)
        EmpEducationLevel = st.slider("Education Level", 1, 5, 3)
        EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 5, 3)
        EmpHourlyRate = st.slider("Hourly Rate", 20, 100, 60)
        EmpJobInvolvement = st.slider("Job Involvement", 1, 5, 3)
        EmpJobLevel = st.slider("Job Level", 1, 5, 2)
        EmpJobSatisfaction = st.slider("Job Satisfaction", 1, 5, 4)

    with col3:
        NumCompaniesWorked = st.slider("Companies Worked", 0, 10, 2)
        OverTime = st.selectbox("Overtime", ["Yes", "No"])
        EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
        EmpRelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        TotalWorkExperienceInYears = st.slider("Total Work Experience (Years)", 0, 40, 6)
        TrainingTimesLastYear = st.slider("Trainings Last Year", 0, 10, 2)
        EmpWorkLifeBalance = st.slider("Work-Life Balance", 1, 5, 3)
        ExperienceYearsAtThisCompany = st.slider("Years at Company", 0, 20, 3)
        ExperienceYearsInCurrentRole = st.slider("Years in Current Role", 0, 15, 2)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        YearsWithCurrManager = st.slider("Years with Current Manager", 0, 15, 2)
        Attrition = st.selectbox("Attrition", ["Yes", "No"])

# -------------------------------
# 🧮 Prepare input data
# -------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "EducationBackground": EducationBackground,
    "MaritalStatus": MaritalStatus,
    "EmpDepartment": EmpDepartment,
    "EmpJobRole": EmpJobRole,
    "BusinessTravelFrequency": BusinessTravelFrequency,
    "DistanceFromHome": DistanceFromHome,
    "EmpEducationLevel": EmpEducationLevel,
    "EmpEnvironmentSatisfaction": EmpEnvironmentSatisfaction,
    "EmpHourlyRate": EmpHourlyRate,
    "EmpJobInvolvement": EmpJobInvolvement,
    "EmpJobLevel": EmpJobLevel,
    "EmpJobSatisfaction": EmpJobSatisfaction,
    "NumCompaniesWorked": NumCompaniesWorked,
    "OverTime": OverTime,
    "EmpLastSalaryHikePercent": EmpLastSalaryHikePercent,
    "EmpRelationshipSatisfaction": EmpRelationshipSatisfaction,
    "TotalWorkExperienceInYears": TotalWorkExperienceInYears,
    "TrainingTimesLastYear": TrainingTimesLastYear,
    "EmpWorkLifeBalance": EmpWorkLifeBalance,
    "ExperienceYearsAtThisCompany": ExperienceYearsAtThisCompany,
    "ExperienceYearsInCurrentRole": ExperienceYearsInCurrentRole,
    "YearsSinceLastPromotion": YearsSinceLastPromotion,
    "YearsWithCurrManager": YearsWithCurrManager,
    "Attrition": Attrition
}])

# -------------------------------
# 🔮 Predict performance
# -------------------------------
if st.button("🔮 Predict Performance Rating"):
    try:
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        conf = float(max(probs))

        st.success("✅ Prediction complete!")
        st.subheader("📊 Predicted Employee Performance Rating:")
        st.metric("Performance Rating", int(prediction))
        st.progress(conf)
        st.caption(f"Confidence: {conf:.2%}")

        st.subheader("💬 HR Interpretation")
        if prediction == 4:
            st.success("⭐ High Performer — Ideal for leadership or growth-track roles.")
        elif prediction == 3:
            st.info("🧩 Consistent Performer — Good potential; can excel with targeted training.")
        else:
            st.warning("⚠️ Underperformer Risk — May require additional mentoring or onboarding support.")

        st.markdown("""
        ---
        **Key Drivers of Performance (based on historical model insights):**
        - 🏆 *TrainingTimesLastYear*: Frequent training correlates strongly with higher performance.  
        - 📈 *YearsSinceLastPromotion*: Longer gaps since promotion often signal declining engagement.  
        - 👔 *ExperienceYearsAtThisCompany*: Mid-tenure employees (2–6 years) perform most consistently.  
        """)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
