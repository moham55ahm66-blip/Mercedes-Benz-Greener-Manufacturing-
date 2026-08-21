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
# PATHS
# =========================================================

MODELS_DIR = "models"

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "xgboost_final_corrected.joblib"
)

PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "final_preprocessor_corrected.joblib"
)

META_PATH = os.path.join(
    MODELS_DIR,
    "final_meta_corrected.joblib"
)

TRAIN_PATH = "train.csv"


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    meta = joblib.load(META_PATH)

    return model, preprocessor, meta


# =========================================================
# LOAD TRAINING DATA
# =========================================================

@st.cache_data
def load_training_data():

    return pd.read_csv(TRAIN_PATH)


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model, preprocessor, meta = load_artifacts()
    train = load_training_data()

    artifacts_loaded = True

except Exception as e:

    artifacts_loaded = False
    st.error(f"Error loading model artifacts: {e}")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Selection")

if artifacts_loaded:

    st.sidebar.success("Loaded: xgboost_final_corrected.joblib")


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

    if not artifacts_loaded:

        st.error("Please make sure the model files exist in the models folder.")

    else:

        try:

            # -------------------------------------------------
            # Get the original feature columns
            # -------------------------------------------------

            feature_columns = meta["feature_columns_before_encoding"]


            # -------------------------------------------------
            # Create default input row
            # -------------------------------------------------

            input_row = {}

            for column in feature_columns:

                # Categorical features
                if train[column].dtype == "object":

                    input_row[column] = train[column].mode()[0]

                # Numeric features
                else:

                    input_row[column] = 0.0


            # -------------------------------------------------
            # Replace the four important features
            # -------------------------------------------------

            input_row["X314"] = X314
            input_row["X261"] = X261
            input_row["X118"] = X118
            input_row["X127"] = X127


            # -------------------------------------------------
            # Convert to DataFrame
            # -------------------------------------------------

            input_df = pd.DataFrame(
                [input_row],
                columns=feature_columns
            )


            # -------------------------------------------------
            # Apply the same preprocessing used during training
            # -------------------------------------------------

            processed_input = preprocessor.transform(input_df)


            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            prediction = model.predict(processed_input)[0]


            # -------------------------------------------------
            # Display prediction
            # -------------------------------------------------

            st.balloons()

            st.success(
                f"Predicted Testing Time: **{prediction:.2f} seconds**"
            )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
