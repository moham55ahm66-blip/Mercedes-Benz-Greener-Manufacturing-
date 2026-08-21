import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Mercedes-Benz Greener Manufacturing",
    page_icon="🚗",
    layout="wide"
)

# App Title & Description
st.title("🚗 Mercedes-Benz Testing Time Prediction")
st.markdown("Select a trained model and input features to predict the testing time.")

st.sidebar.header("Model Selection")

# Path to the models folder
MODELS_DIR = "models"  # Change this to your models folder name if different

# List models inside the folder
if os.path.exists(MODELS_DIR):
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.pkl', '.joblib', '.h5', '.sav'))]
else:
    model_files = []

if model_files:
    selected_model_name = st.sidebar.selectbox("Choose a Model:", model_files)
    model_path = os.path.join(MODELS_DIR, selected_model_name)

    @st.cache_resource
    def load_selected_model(path):
        return joblib.load(path)

    try:
        model = load_selected_model(model_path)
        st.sidebar.success(f"Loaded: {selected_model_name}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model = None
else:
    st.sidebar.warning(f"No model files found in '{MODELS_DIR}' folder.")
    model = None

# Input Features Section
st.subheader("Input Features")

# Add input fields matching your model's required features
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

st.markdown("---")

# Prediction Button
if st.button("Predict Testing Time", type="primary"):
    if model is not None:
        try:
            # Prepare feature array (adjust column count to match your model training data)
            input_data = np.array([[feature1, feature2, feature3, feature4]])
            
            prediction = model.predict(input_data)
            
            st.balloons()
            st.success(f"Predicted Testing Time: **{prediction[0]:.2f} seconds**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Please load a valid model first.")