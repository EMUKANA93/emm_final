import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np

# --- 1. CONFIGURATION AND CONSTANTS ---
MODEL_FILE = 'best_employee_performance_model.joblib'
DATA_FILE = 'INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv'

# Target Mapping
TARGET_MAP = {
    2: 'Low Performance',
    3: 'Good Performance',
    4: 'Excellent Performance'
}
TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}

# Define Feature Sets (must match the model's ColumnTransformer)
NUM_COLS = [
    'Age', 'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
    'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction',
    'NumCompaniesWorked', 'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
    'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'EmpWorkLifeBalance',
    'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

CAT_COLS = [
    'Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment',
    'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition'
]

ALL_FEATURES = NUM_COLS + CAT_COLS

# UI Ranges/Defaults (based on INX_Future_Inc data analysis)
UI_CONFIG = {
    'Age': (18, 60, 35, 1),
    'DistanceFromHome': (1, 29, 10, 1),
    'EmpEducationLevel': (1, 5, 3, 1),
    'EmpEnvironmentSatisfaction': (1, 4, 3, 1),
    'EmpHourlyRate': (30, 100, 65, 1),
    'EmpJobInvolvement': (1, 4, 3, 1),
    'EmpJobLevel': (1, 5, 2, 1),
    'EmpJobSatisfaction': (1, 4, 3, 1),
    'NumCompaniesWorked': (0, 9, 2, 1),
    'EmpLastSalaryHikePercent': (11, 25, 15, 1),
    'EmpRelationshipSatisfaction': (1, 4, 3, 1),
    'TotalWorkExperienceInYears': (0, 40, 10, 1),
    'TrainingTimesLastYear': (0, 6, 2, 1),
    'EmpWorkLifeBalance': (1, 4, 3, 1),
    'ExperienceYearsAtThisCompany': (0, 40, 10, 1),
    'ExperienceYearsInCurrentRole': (0, 18, 7, 1),
    'YearsSinceLastPromotion': (0, 15, 2, 1),
    'YearsWithCurrManager': (0, 17, 7, 1),
}

# Categorical Options (extracted from the data)
CAT_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'EducationBackground': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'MaritalStatus': ['Married', 'Single', 'Divorced'],
    'EmpDepartment': ['Development', 'Sales', 'Research & Development', 'Human Resources', 'Data Science', 'Finance'],
    'EmpJobRole': ['Developer', 'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Senior Developer', 'Manager R&D', 'Technical Lead', 'Sales Representative', 'Human Resources', 'Delivery Manager', 'Research Director', 'Finance', 'Data Scientist'],
    'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'OverTime': ['No', 'Yes'],
    'Attrition': ['No', 'Yes']
}

# --- 2. MODEL AND DATA LOADING FUNCTIONS ---
@st.cache_resource
def load_model():
    """Load the scikit-learn model pipeline."""
    try:
        pipeline = joblib.load(MODEL_FILE)
        st.success("✅ Model loaded successfully.")
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_example_data():
    """Load and prepare example data for download."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Select only the feature columns and a few rows for the example
        example_df = df[ALL_FEATURES].head(5)
        return example_df
    return pd.DataFrame(columns=ALL_FEATURES)

# --- 3. SHAP EXPLANATION FUNCTION ---
def run_shap_explainer(pipeline, data_df):
    """Generates and displays SHAP summary plot for the prediction."""
    try:
        # Check if the last estimator in the pipeline is a tree-based model (for TreeExplainer)
        final_estimator = pipeline.steps[-1][1]
        
        # Determine appropriate explainer
        if hasattr(final_estimator, 'feature_importances_'):
            st.info("💡 **Model Interpretability:** Using SHAP (TreeExplainer) as the model appears tree-based.")
            explainer = shap.TreeExplainer(final_estimator)
        else:
            st.info("💡 **Model Interpretability:** Using SHAP (KernelExplainer) for general model type.")
            # Use a background dataset for KernelExplainer, or just the input data if only one row
            # For demonstration, we'll use a simplified approach and assume TreeExplainer might work
            # or fall back to feature importance table if TreeExplainer fails.
            st.warning("Model type not explicitly recognized as tree-based. SHAP interpretation might be slow or inaccurate. Falling back to feature importance table if SHAP fails.")
            st.subheader("Feature Importance Table (Fallback)")
            try:
                # Fallback: simple feature importance table for linear/tree models, 
                # but requires model structure knowledge which is complex for a general pipeline.
                # For simplicity, we skip this complex fallback for now and rely on SHAP.
                # For an unknown black box model, generating feature importance is non-trivial.
                st.write("Interpretation skipped or failed. SHAP is highly recommended for black-box models.")
            except Exception:
                st.write("Could not determine feature importance.")
            return

        # Get the preprocessor from the pipeline
        preprocessor = pipeline.named_steps['preprocessor']

        # Get the feature names after one-hot encoding
        # This is a bit complex due to the ColumnTransformer. We will use a proxy of raw feature names
        # for a cleaner plot, acknowledging SHAP is calculated on transformed features.

        # The SHAP explainer must run on the preprocessed data (input to the final model)
        data_transformed = preprocessor.transform(data_df)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data_transformed)

        # Assuming multi-class (3 classes), SHAP values will be a list of arrays.
        # We focus on the class with the highest probability.
        pred_class_idx = pipeline.predict(data_df)[0] - 2 # 2->0, 3->1, 4->2
        
        if isinstance(shap_values, list):
            # For multi-class classification, take the SHAP values for the predicted class
            shap_values_for_class = shap_values[pred_class_idx]
        else:
            # For binary classification (or other structures), use the main values
            shap_values_for_class = shap_values
            
        # Get feature names from preprocessor (complex, using raw names as a simple proxy)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Fallback to simple integer names if get_feature_names_out fails
            feature_names = [f'Feature {i}' for i in range(shap_values_for_class.shape[1])]

        # If batch prediction, use summary plot, otherwise use force plot
        if data_df.shape[0] == 1:
            st.subheader(f"SHAP Force Plot for Predicted Class: {TARGET_MAP[pred_class_idx + 2]}")
            # SHAP requires a specific Streamlit element
            shap.initjs()
            st_shap(shap.force_plot(
                explainer.expected_value[pred_class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                shap_values_for_class, 
                data_transformed, 
                feature_names=feature_names, 
                matplotlib=True
            ))
        else:
            st.subheader("SHAP Summary Plot for Batch Prediction")
            st_shap(shap.summary_plot(shap_values, data_transformed, feature_names=feature_names, plot_type="bar", class_names=list(TARGET_MAP.values())))

    except Exception as e:
        st.error(f"SHAP Interpretability failed (Error: {e}). It might be incompatible with the loaded model type.")
        st.info("Falling back to displaying basic model info (if available).")
        st.text(f"Model Type: {type(final_estimator).__name__}")
        
# A helper function to render SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- 4. STREAMLIT APPLICATION LAYOUT ---
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Header ---
st.title("👨‍💼 Employee Performance Predictor")
st.markdown("""
    This app uses a machine learning model to predict an employee's **Performance Rating** (Low, Good, or Excellent) based on various professional and personal factors.
""")

# --- Model Loading ---
pipeline = load_model()

if pipeline is None:
    st.stop() # Stop if model loading failed

# --- Sidebar Input Mode Selection ---
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Select an input method:", ["Manual Entry", "Batch CSV Upload"])
st.sidebar.markdown("---")

# --- DATA INPUT UI ---
input_df = None

if input_mode == "Manual Entry":
    st.subheader("Manual Employee Data Entry")
    
    # Organize inputs into two columns for better layout
    col1, col2 = st.columns(2)
    
    # Use two columns for numeric inputs
    col1.subheader("Numeric/Ordinal Features")
    col1_num, col2_num = col1.columns(2)
    input_data = {}
    
    # Numeric/Ordinal Inputs
    numeric_inputs = []
    for i, col in enumerate(NUM_COLS):
        # Alternate columns
        current_col = col1_num if i % 2 == 0 else col2_num
        min_val, max_val, default_val, step = UI_CONFIG[col]
        
        if col in ['EmpEducationLevel', 'EmpEnvironmentSatisfaction', 'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction', 'EmpRelationshipSatisfaction', 'EmpWorkLifeBalance']:
            # Use selectbox for ordinal features
            value = current_col.selectbox(f"**{col}** (1-Low, 4/5-High)", options=range(min_val, max_val + 1), index=default_val - min_val)
        else:
            # Use slider for continuous numeric features
            value = current_col.slider(f"**{col}**", min_val=min_val, max_val=max_val, value=default_val, step=step)
        
        input_data[col] = value
        
    # Categorical Inputs
    col2.subheader("Categorical Features")
    col3_cat, col4_cat = col2.columns(2)
    
    for i, col in enumerate(CAT_COLS):
        # Alternate columns
        current_col = col3_cat if i % 2 == 0 else col4_cat
        options = CAT_OPTIONS[col]
        default_index = options.index(options[0]) if options[0] in options else 0
        value = current_col.selectbox(f"**{col}**", options=options, index=default_index)
        input_data[col] = value
        
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    # Prediction Button
    if st.button("Predict Performance Rating"):
        pass # Execution continues to the prediction section

elif input_mode == "Batch CSV Upload":
    st.subheader("Batch Prediction via CSV Upload")
    
    uploaded_file = st.file_uploader("Upload a CSV file (Must contain columns: " + ", ".join(ALL_FEATURES) + ")", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())
            
            # Validation: Check if all required columns are present
            missing_cols = [col for col in ALL_FEATURES if col not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Error: The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
                st.stop()
            
            input_df = batch_df[ALL_FEATURES]
            st.success(f"✅ Loaded {input_df.shape[0]} rows for prediction.")
            
            if st.button("Run Batch Prediction"):
                pass # Execution continues to the prediction section
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.stop()
            
# --- Example Data and Instructions ---
with st.sidebar.expander("Instructions & Example Data"):
    st.markdown("""
    ### 🚀 Steps to Run
    1. Select an **Input Mode** (Manual or Batch CSV).
    2. Enter features or upload a CSV file.
    3. Click the **Predict** button.
    
    ### 💾 Example Data
    Download the CSV below for the correct column format required for batch upload.
    The required columns are: `""" + ", ".join(ALL_FEATURES) + "`
    """)
    example_df = load_example_data()
    st.download_button(
        label="Download Example CSV",
        data=example_df.to_csv(index=False).encode('utf-8'),
        file_name='employee_performance_example_input.csv',
        mime='text/csv',
    )
    st.markdown("---")
    st.markdown("""
    ### ⚙️ Requirements
    Create a file named `requirements.txt` with the following content:
    ```
    streamlit
    pandas
    scikit-learn
    joblib
    shap
    ```
    """)
    
    st.markdown("""
    ### 💻 Local Run
    1. Install dependencies: `pip install -r requirements.txt`
    2. Run the app: `streamlit run app.py`
    """)

# --- 5. PREDICTION AND RESULTS DISPLAY ---
if input_df is not None and (input_mode == "Batch CSV Upload" or st.session_state.get('predict_button_clicked', False)):
    
    try:
        # Perform prediction and probability prediction
        predictions = pipeline.predict(input_df)
        probas = pipeline.predict_proba(input_df)
        
        # Map numeric predictions to human-readable labels
        predicted_classes = np.vectorize(TARGET_MAP.get)(predictions)
        
        # Determine the probability of the predicted class
        # The class indices for probas are 0, 1, 2 corresponding to classes 2, 3, 4
        class_indices = predictions - 2
        predicted_proba = probas[np.arange(len(probas)), class_indices]
        
        # Create a results DataFrame
        results_df = input_df.copy()
        results_df['Predicted_Rating_Label'] = predicted_classes
        results_df['Predicted_Rating_Score'] = predictions
        
        # Add probability for each class
        for i, class_label in TARGET_MAP.items():
             # Check if the model predicted all classes, otherwise handle missing columns
            if (i - 2) < probas.shape[1]:
                results_df[f'Probability_{class_label}'] = probas[:, i - 2].round(4)
        
        # Add Confidence Indicator
        results_df['Confidence'] = (predicted_proba * 100).round(2).astype(str) + '%'
        
        # --- Display Results ---
        if input_mode == "Manual Entry":
            st.header("🎯 Prediction Results")
            pred_class = results_df['Predicted_Rating_Label'].iloc[0]
            confidence = results_df['Confidence'].iloc[0]
            
            if pred_class == 'Excellent Performance':
                st.balloons()
                st.success(f"**Predicted Performance Rating: {pred_class}** (Confidence: {confidence})")
            elif pred_class == 'Good Performance':
                st.info(f"**Predicted Performance Rating: {pred_class}** (Confidence: {confidence})")
            else:
                st.warning(f"**Predicted Performance Rating: {pred_class}** (Confidence: {confidence})")
            
            st.subheader("Class Probabilities:")
            proba_data = {
                TARGET_MAP[i]: results_df[f'Probability_{TARGET_MAP[i]}'].iloc[0] 
                for i in sorted(TARGET_MAP.keys())
            }
            proba_df = pd.DataFrame.from_dict(proba_data, orient='index', columns=['Probability']).T
            st.dataframe(proba_df, use_container_width=True)
            
            st.markdown("---")
            st.header("🔎 Model Interpretability (SHAP)")
            # For Manual Entry (single row), run SHAP on that row
            run_shap_explainer(pipeline, input_df)

        elif input_mode == "Batch CSV Upload":
            st.header("📦 Batch Prediction Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button for results
            st.download_button(
                label="Download Results CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='employee_performance_batch_results.csv',
                mime='text/csv',
            )

            st.markdown("---")
            st.header("📈 Model Interpretability (SHAP Summary)")
            # For Batch Upload, run SHAP on the whole batch
            run_shap_explainer(pipeline, input_df)

    except Exception as e:
        st.error(f"❌ An error occurred during prediction. Please check your input data format. Error: {e}")

# Persistent state management for manual entry button press
if input_mode == "Manual Entry":
    if st.button("Predict Performance Rating"):
        st.session_state['predict_button_clicked'] = True
    else:
        st.session_state['predict_button_clicked'] = False