import numpy as np
import pandas as pd
import joblib

import sklearn
import streamlit as st

st.sidebar.write(f"🧩 Scikit-learn version: {sklearn.__version__}")


# --- Load Model and  ---
model_path = "best_employee_performance_model.joblib"
model = joblib.load(model_path)

st.set_page_config(page_title="Employee Performance Predictor", page_icon="💼", layout="wide")

st.title("💼 Employee Performance Prediction App")

st.markdown("""
This app predicts an **employee's performance rating** based on various job and personal attributes.  
Use it to
 **support hiring and internal performance evaluations**.
""")

# -------------------------------
# 2. Collect HR inputs
# -------------------------------
st.header("🧾 Candidate Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.slider("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    EducationBackground = st.selectbox("Education Background", [
        "Life Sciences", "Medical", "Marketing", "Technical Degree",
        "Human Resources", "Other"
    ])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    EmpDepartment = st.selectbox("Department", ["Sales", "Development", "Research & Development", "Human Resources"])
    EmpJobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Manager", "Laboratory Technician", 
        "Developer", "Manufacturing Director", "Healthcare Representative"
    ])

with col2:
    BusinessTravelFrequency = st.selectbox("Business Travel Frequency", [
        "Non-Travel", "Travel_Rarely", "Travel_Frequently"
    ])
    DistanceFromHome = st.slider("Distance from Home (km)", 0, 30, 10)
    EmpEducationLevel = st.slider("Education Level (1–5)", 1, 5, 3)
    EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1–5)", 1, 5, 3)
    EmpHourlyRate = st.slider("Hourly Rate", 20, 100, 60)
    EmpJobInvolvement = st.slider("Job Involvement (1–5)", 1, 5, 3)
    EmpJobLevel = st.slider("Job Level (1–5)", 1, 5, 2)
    EmpJobSatisfaction = st.slider("Job Satisfaction (1–5)", 1, 5, 4)

with col3:
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 2)
    OverTime = st.selectbox("Overtime", ["Yes", "No"])
    EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
    EmpRelationshipSatisfaction = st.slider("Relationship Satisfaction (1–5)", 1, 5, 3)
    TotalWorkExperienceInYears = st.slider("Total Experience (Years)", 0, 40, 6)
    TrainingTimesLastYear = st.slider("Trainings Attended Last Year", 0, 10, 2)
    EmpWorkLifeBalance = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
    ExperienceYearsAtThisCompany = st.slider("Years at Current Company", 0, 20, 3)
    ExperienceYearsInCurrentRole = st.slider("Years in Current Role", 0, 15, 2)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    YearsWithCurrManager = st.slider("Years with Current Manager", 0, 15, 2)
    Attrition = st.selectbox("Attrition", ["Yes", "No"])

    # -------------------------------
# 3. Prepare input data
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
# 4. Predict performance
# -------------------------------
if st.button("🔮 Predict Performance Rating"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Results")
    st.metric("Predicted Performance Rating", int(prediction))
    st.write("Confidence Levels:")
    st.progress(float(max(probs)))
    st.json({int(cls): float(round(prob, 3)) for cls, prob in zip(model.named_steps['clf'].classes_, probs)})

    # -------------------------------
    # 5. Interpret key influencing factors (static interpretation)
    # -------------------------------
    st.subheader("💡 Key Insights for HR")
    if prediction == 4:
        st.success("This candidate shows strong potential for **high performance** — consider leadership roles or accelerated development.")
    elif prediction == 3:
        st.info("This candidate is likely to be a **consistent performer** — focus on engagement and training to boost output further.")
    else:
        st.warning("This candidate might need **closer supervision or upskilling** to meet performance expectations.")

    st.markdown("""
    **Key Factors to Focus On:**
    - 🏆 *TrainingTimesLastYear*: More trainings are strongly linked with better performance.  
    - 📈 *YearsSinceLastPromotion*: Long gaps since promotion often correlate with lower performance.  
    - 👔 *ExperienceYearsAtThisCompany*: Moderate tenure (2–6 years) aligns with higher ratings.
    """)