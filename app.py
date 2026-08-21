import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mercedes-Benz Testing Time Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILES = {
    "XGBoost Tuned": "xgboost_tuned.joblib",
    "XGBoost Early Stop": "xgboost_earlystop.joblib",
    "LightGBM": "lightgbm.joblib",
    "Lasso": "lasso_pipeline.joblib",
    "PCA + Ridge": "pca_ridge_pipeline.joblib"
}


# =========================================================
# PATHS
# =========================================================

META_PATH = "meta.joblib"
ORDINAL_ENCODER_PATH = "ordinal_encoder.joblib"


# =========================================================
# LOAD METADATA
# =========================================================

@st.cache_resource
def load_metadata():

    return joblib.load(META_PATH)


@st.cache_resource
def load_ordinal_encoder():

    return joblib.load(ORDINAL_ENCODER_PATH)


try:

    meta = load_metadata()
    ordinal_encoder = load_ordinal_encoder()

    artifacts_loaded = True

except Exception as e:

    artifacts_loaded = False

    st.error(
        f"Error loading metadata: {e}"
    )


# =========================================================
# LOAD SELECTED MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


# =========================================================
# SIDEBAR - MODEL SELECTION
# =========================================================

st.sidebar.header("Model Selection")


selected_model_name = st.sidebar.selectbox(
    "Choose a Model:",
    list(MODEL_FILES.keys())
)


selected_model_file = MODEL_FILES[selected_model_name]


if os.path.exists(selected_model_file):

    try:

        model = load_model(selected_model_file)

        st.sidebar.success(
            f"Loaded: {selected_model_file}"
        )

    except Exception as e:

        model = None

        st.sidebar.error(
            f"Error loading model: {e}"
        )

else:

    model = None

    st.sidebar.error(
        f"Model file not found: {selected_model_file}"
    )


# =========================================================
# TITLE
# =========================================================

st.title(
    "🚗 Mercedes-Benz Testing Time Prediction"
)

st.markdown(
    "Select a trained model and input features to predict "
    "the vehicle testing time."
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
# CREATE RAW INPUT
# =========================================================

def create_raw_input():

    feature_columns = meta["feature_columns"]
    cat_cols = meta["cat_cols"]

    input_data = {}

    # -----------------------------------------------------
    # Create a complete raw feature row
    # -----------------------------------------------------

    for column in feature_columns:

        if column in cat_cols:

            # Use the first known category as default
            input_data[column] = meta["cat_options"][column][0]

        else:

            # Numeric / binary features default to 0
            input_data[column] = 0.0


    # -----------------------------------------------------
    # Replace selected UI features
    # -----------------------------------------------------

    input_data["X314"] = X314
    input_data["X261"] = X261
    input_data["X118"] = X118
    input_data["X127"] = X127


    return pd.DataFrame(
        [input_data],
        columns=feature_columns
    )


# =========================================================
# ONE-HOT ENCODING
# =========================================================

def create_onehot_input(input_df):

    feature_columns = meta["feature_columns"]
    cat_cols = meta["cat_cols"]

    num_cols = [
        col for col in feature_columns
        if col not in cat_cols
    ]


    # -----------------------------------------------------
    # Recreate the same OneHotEncoder categories
    # from the training metadata
    # -----------------------------------------------------

    categories = [
        meta["cat_options"][column]
        for column in cat_cols
    ]


    encoder = OneHotEncoder(
        categories=categories,
        handle_unknown="ignore",
        sparse_output=False
    )


    # Fit using the predefined categories
    encoder.fit(
        input_df[cat_cols]
    )


    categorical_array = encoder.transform(
        input_df[cat_cols]
    )


    categorical_names = (
        encoder.get_feature_names_out(cat_cols)
    )


    categorical_df = pd.DataFrame(
        categorical_array,
        columns=categorical_names,
        index=input_df.index
    )


    numeric_df = input_df[
        num_cols
    ].astype(float)


    # Same order as the notebook:
    # categorical features + numeric features

    encoded_df = pd.concat(
        [
            categorical_df,
            numeric_df
        ],
        axis=1
    )


    return encoded_df


# =========================================================
# ORDINAL ENCODING
# =========================================================

def create_ordinal_input(input_df):

    ordinal_df = input_df.copy()

    cat_cols = meta["cat_cols"]


    ordinal_df[cat_cols] = ordinal_encoder.transform(
        ordinal_df[cat_cols]
    )


    return ordinal_df


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "Predict Testing Time",
    type="primary"
):

    if not artifacts_loaded:

        st.error(
            "Model metadata could not be loaded."
        )

    elif model is None:

        st.error(
            "Please select a valid model."
        )

    else:

        try:

            # -------------------------------------------------
            # Create complete raw vehicle configuration
            # -------------------------------------------------

            input_df = create_raw_input()


            # =================================================
            # XGBOOST TUNED
            # =================================================

            if selected_model_name == "XGBoost Tuned":

                # This model already contains:
                #
                # Raw Data
                #      ↓
                # OneHotEncoder
                #      ↓
                # XGBoost
                #
                prediction = model.predict(
                    input_df
                )[0]


            # =================================================
            # XGBOOST EARLY STOP
            # =================================================

            elif selected_model_name == "XGBoost Early Stop":

                ordinal_df = create_ordinal_input(
                    input_df
                )

                prediction = model.predict(
                    ordinal_df
                )[0]


            # =================================================
            # LIGHTGBM
            # =================================================

            elif selected_model_name == "LightGBM":

                ordinal_df = create_ordinal_input(
                    input_df
                )

                prediction = model.predict(
                    ordinal_df
                )[0]


            # =================================================
            # LASSO
            # =================================================

            elif selected_model_name == "Lasso":

                onehot_df = create_onehot_input(
                    input_df
                )

                prediction = model.predict(
                    onehot_df
                )[0]


            # =================================================
            # PCA + RIDGE
            # =================================================

            elif selected_model_name == "PCA + Ridge":

                onehot_df = create_onehot_input(
                    input_df
                )

                prediction = model.predict(
                    onehot_df
                )[0]


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.balloons()

            st.success(
                f"Predicted Testing Time: "
                f"**{prediction:.2f} seconds**"
            )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
