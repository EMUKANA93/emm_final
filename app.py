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
# Footer Section
# =============================================
st.divider()
st.caption("Developed by Alex Ndiritu — Powered by Streamlit and scikit-learn.")
