import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- MAPPING & CONFIGURATION ---
# 1. Correct mapping for EmpEducationLevel (Ordinal feature expected as numeric by model)
EDUCATION_MAPPING = {
    "Below College": 1, "College": 2, "Bachelor": 3, "Master": 4, "Doctor": 5
}

# 2. Correct mapping for BusinessTravelFrequency labels
BUSINESS_TRAVEL_MAPPING = {
    "No Travel": "Non-Travel",
    "Travel Rarely": "Travel_Rarely",
    "Travel Frequently": "Travel_Frequently"
}

# 3. Define the exact feature lists used by the model (excluding 'AgeGroup')
NUMERIC_COLUMNS = [
    'Age', 'DistanceFromHome', 'EmpJobLevel', 'EmpEducationLevel', # EmpEducationLevel is now numeric/ordinal
    'EmpEnvironmentSatisfaction', 'EmpJobSatisfaction',
    'EmpRelationshipSatisfaction', 'EmpWorkLifeBalance',
    'EmpJobInvolvement', 'ExperienceYearsAtThisCompany',
    'ExperienceYearsInCurrentRole', 'YearsWithCurrManager',
    'YearsSinceLastPromotion', 'TotalWorkExperienceInYears',
    'NumCompaniesWorked', 'TrainingTimesLastYear',
    'EmpLastSalaryHikePercent', 'EmpHourlyRate'
]

CATEGORICAL_COLUMNS = [
    'Gender', 'MaritalStatus', 'EmpDepartment',
    'EmpJobRole', 'EducationBackground',
    'BusinessTravelFrequency', 'OverTime', 'Attrition'
]
# --- END MAPPING & CONFIGURATION ---


# Load the model
@st.cache_resource
def load_model():
    # Use the current working directory path for Streamlit Cloud compatibility
    model_path = os.path.join(os.getcwd(), "best_employee_performance_model.joblib")
    try:
        # Check if the file is in the current directory or the directory of the script
        if not os.path.exists(model_path):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             model_path = os.path.join(script_dir, "best_employee_performance_model.joblib")
             
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found. Ensure 'best_employee_performance_model.joblib' is in the same directory as app.py.")
        return None
    except Exception as e:
         st.error(f"Error loading model: {str(e)}")
         return None


def main():
    st.title("Employee Performance Predictor")
    st.write("Enter employee information to predict performance rating")

    # Create input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        department = st.selectbox("Department", ["Sales", "Development", "Data Science", 
                                                 "Research & Development", "Human Resources", "Finance"])
        
        # CORRECTED: Get string label, which will be mapped to numeric value later
        education_label = st.selectbox("Education Level", list(EDUCATION_MAPPING.keys()))
        
        education_background = st.selectbox("Education Background", ["Life Sciences", "Medical", "Marketing", 
                                                                     "Technical Degree", "Other", "Human Resources"])
        
    with col2:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Developer", "Data Scientist",
                                             "Research Scientist", "Manager", "Financial Analyst", 
                                             "Manufacturing Director", "Technical Lead"])
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=10)
        
        # CORRECTED: Get string label, which will be mapped to the model's required label
        business_travel_label = st.selectbox("Business Travel", list(BUSINESS_TRAVEL_MAPPING.keys()))
        
        overtime = st.selectbox("Over Time", ["Yes", "No"])
        attrition = st.selectbox("Attrition", ["Yes", "No"])

    with col3:
        environment_satisfaction = st.slider("Environment Satisfaction (1-Low, 4-High)", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction (1-Low, 4-High)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-Low, 4-High)", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance (1-Poor, 4-Best)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1-Low, 4-High)", 1, 4, 3)
        
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
        years_with_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        total_work_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=40, value=8)
        companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
        
        training_times = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
        salary_hike = st.number_input("Last Salary Hike Percent", min_value=0, max_value=25, value=15)
        hourly_rate = st.number_input("Hourly Rate", min_value=20, max_value=100, value=65)


    # Create a dataframe from inputs with corrected mapping
    data = {
        'Age': age,
        # Removed 'AgeGroup'
        'Gender': gender,
        'MaritalStatus': marital_status,
        'EmpDepartment': department,
        'EmpJobRole': job_role,
        'EmpJobLevel': job_level,
        'DistanceFromHome': distance_from_home,
        # CORRECTED: Map label to numeric value (1-5)
        'EmpEducationLevel': EDUCATION_MAPPING[education_label], 
        'EducationBackground': education_background,
        # CORRECTED: Map friendly label to model's expected label
        'BusinessTravelFrequency': BUSINESS_TRAVEL_MAPPING[business_travel_label], 
        'EmpEnvironmentSatisfaction': environment_satisfaction,
        'EmpJobSatisfaction': job_satisfaction,
        'EmpRelationshipSatisfaction': relationship_satisfaction,
        'EmpWorkLifeBalance': work_life_balance,
        'EmpJobInvolvement': job_involvement,
        'ExperienceYearsAtThisCompany': years_at_company,
        'ExperienceYearsInCurrentRole': years_in_role,
        'YearsWithCurrManager': years_with_manager,
        'YearsSinceLastPromotion': years_since_promotion,
        'TotalWorkExperienceInYears': total_work_years,
        'NumCompaniesWorked': companies_worked,
        'TrainingTimesLastYear': training_times,
        'EmpLastSalaryHikePercent': salary_hike,
        'EmpHourlyRate': hourly_rate,
        'OverTime': overtime,
        'Attrition': attrition
    }
    input_df = pd.DataFrame([data])

    # Reorder columns to match the training data (optional, but good practice)
    expected_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    
    # Filter the dataframe to only include the expected features
    # This also handles the removal of the now-unneeded 'AgeGroup'
    input_df = input_df[expected_columns]

    # Ensure numeric columns are float/int
    for col in NUMERIC_COLUMNS:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Ensure categorical columns are string
    for col in CATEGORICAL_COLUMNS:
        input_df[col] = input_df[col].astype(str)

    if st.button("Predict Performance"):
        model = load_model()
        if model is not None:
            try:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)
                
                # Human-readable mapping for results
                RATING_MAP = {2: 'Low Performance', 3: 'Good Performance', 4: 'Excellent Performance'}
                predicted_label = RATING_MAP.get(prediction[0], f"Rating {prediction[0]}")
                
                st.subheader("Prediction Results")
                st.write(f"Predicted Performance Rating: **{predicted_label}**")
                
                # The classes in probability often match the sorted unique classes: 2, 3, 4
                prob_index = sorted(RATING_MAP.keys())
                prob_df = pd.DataFrame(probability[0], 
                                     columns=['Probability'],
                                     index=[RATING_MAP[i] for i in prob_index])
                prob_df['Probability'] = (prob_df['Probability'] * 100).round(2).astype(str) + '%'
                st.dataframe(prob_df)

                if prediction[0] == 4:
                    st.success("Outstanding Performance! This employee shows excellent potential.")
                elif prediction[0] == 3:
                    st.info("Good Performance. This employee meets expectations.")
                else:
                    st.warning("Performance needs improvement. Consider additional support and training.")

                # Add feature importance visualization (Original logic preserved)
                st.subheader("Input Factors")
                try:
                    # Display the cleaned input for clarity
                    display_df = input_df.T.rename(columns={0: 'Value'})
                    st.dataframe(display_df)
                except:
                    st.info("Input factor display failed.")

            except Exception as e:
                # This error message is now more helpful
                st.error(f"Error making prediction: {str(e)}")
                st.error("Model input mismatch detected. Check feature types and values.")

if __name__ == "__main__":
    main()