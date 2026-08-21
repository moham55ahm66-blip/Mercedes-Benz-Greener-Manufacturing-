import streamlit as st
import joblib
import os
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mercedes-Benz Testing Time Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "xgboost_tuned.joblib"
META_PATH = "meta.joblib"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_meta():
    return joblib.load(META_PATH)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

try:

    model = load_model()
    meta = load_meta()

    model_loaded = True

except Exception as e:

    model_loaded = False
    st.error(f"Error loading model: {e}")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Selection")

if model_loaded:

    st.sidebar.success("Loaded: xgboost_tuned.joblib")


# =========================================================
# TITLE
# =========================================================

st.title("🚗 Mercedes-Benz Testing Time Prediction")

st.markdown(
    "Select input features to predict the vehicle testing time."
)


# =========================================================
# INPUT FEATURES
# =========================================================

st.subheader("Input Features")


col1, col2 = st.columns(2)


with col1:

    X314 = st.number_input(
        "X314",
        value=0.0,
        step=1.0
    )

    X261 = st.number_input(
        "X261",
        value=0.0,
        step=1.0
    )


with col2:

    X118 = st.number_input(
        "X118",
        value=0.0,
        step=1.0
    )

    X127 = st.number_input(
        "X127",
        value=0.0,
        step=1.0
    )


st.markdown("---")


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Testing Time", type="primary"):

    if not model_loaded:

        st.error("Model could not be loaded.")

    else:

        try:

            # -------------------------------------------------
            # Get the original feature columns
            # -------------------------------------------------

            feature_columns = meta["feature_columns"]


            # -------------------------------------------------
            # Create one complete vehicle configuration
            # -------------------------------------------------

            input_data = {}

            for column in feature_columns:

                # Categorical features:
                # use the first known category from the metadata

                if column in meta["cat_cols"]:

                    input_data[column] = meta["cat_options"][column][0]

                # Binary / numeric features:
                # default = 0

                else:

                    input_data[column] = 0


            # -------------------------------------------------
            # Replace the four important UI features
            # -------------------------------------------------

            input_data["X314"] = X314
            input_data["X261"] = X261
            input_data["X118"] = X118
            input_data["X127"] = X127


            # -------------------------------------------------
            # Create DataFrame
            # -------------------------------------------------

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )


            # -------------------------------------------------
            # Predict
            #
            # xgboost_tuned.joblib is already a Pipeline
            # containing preprocessing + XGBoost.
            # -------------------------------------------------

            prediction = model.predict(input_df)[0]


            # -------------------------------------------------
            # Display result
            # -------------------------------------------------

            st.balloons()

            st.success(
                f"Predicted Testing Time: **{prediction:.2f} seconds**"
            )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
