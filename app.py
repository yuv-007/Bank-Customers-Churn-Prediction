import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Card Churn Predictor", layout="wide")

ROOT = os.getcwd()

def find_file(name):
    candidates = [os.path.join("model", name), name]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

@st.cache_resource
def load_artifacts():
    artifacts = {}
    for name in [
        "churn_model.pkl",
        "feature_columns.pkl",
        "encoders.pkl",
        "scaler.pkl",
        "optimal_threshold.pkl",
    ]:
        path = find_file(name)
        if path:
            try:
                artifacts[name] = joblib.load(path)
            except Exception as e:
                artifacts[name] = None
        else:
            artifacts[name] = None
    return artifacts

art = load_artifacts()
model = art.get("churn_model.pkl")
feature_columns = art.get("feature_columns.pkl")
encoders = art.get("encoders.pkl") or {}
scaler = art.get("scaler.pkl")
opt_thr = art.get("optimal_threshold.pkl")

# sensible default categorical options when encoders are missing
default_cats = {
    "Gender": ["F", "M", "Female", "Male", "Unknown"],
    "Income_Category": [
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
        "Unknown",
    ],
    "Card_Category": ["Blue", "Silver", "Gold", "Platinum"],
    "Marital_Status": ["Single", "Married", "Divorced", "Unknown"],
    "Education_Level": [
        "Unknown",
        "Uneducated",
        "High School",
        "College",
        "Graduate",
        "Post-Graduate",
        "Doctorate",
    ],
}

st.title("Credit Card Churn — Interactive Predictor")

if model is None:
    st.warning("No saved model found. Place `churn_model.pkl` in the project root or `model/` folder.")

if feature_columns is None:
    st.info("No `feature_columns.pkl` found — using a sensible default feature set.")
    feature_columns = [
        "Customer_Age",
        "Gender",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Utilization_Ratio",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Total_Amt_Chng_Q4_Q1",
        "Income_Category",
        "Card_Category",
    ]

st.sidebar.header("Input Options")
batch_mode = st.sidebar.checkbox("Predict from CSV (batch)", value=False)

def input_widget_for(col):
    # If we have a saved LabelEncoder, show its classes
    if col in encoders:
        le = encoders[col]
        try:
            options = list(le.classes_)
        except Exception:
            options = ["Unknown"]

        # ensure NA is first/default option
        options = ["NA"] + [o for o in options if o != "NA"]
        val = st.selectbox(col, options, index=0)
        return val

    # If column is a known categorical name, present a sensible dropdown
    if col in default_cats:
        options = default_cats[col]
        # ensure NA is available and default
        options = ["NA"] + [o for o in options if o != "NA"]
        val = st.selectbox(col, options, index=0)
        return val

    # otherwise treat as numeric
    # For numeric fields use a single text input defaulting to 'NA'.
    # Users can type a numeric value; leaving as 'NA' or blank will be treated as missing.
    # show 'NA' as a placeholder so the field appears labeled but is emptied on focus
    val = st.text_input(col, value="", placeholder="NA", key=f"input_{col}")
    return val

if batch_mode:
    st.sidebar.markdown("Upload CSV with columns matching saved feature columns")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write("Sample input:")
        st.dataframe(df_in.head())
        if st.sidebar.button("Run batch predictions"):
            # prepare
            X = pd.DataFrame(columns=feature_columns)
            # reindex: ensure columns exist
            X = df_in.reindex(columns=feature_columns)
            # encode categorical columns
            for c, le in encoders.items():
                if c in X.columns:
                    X[c] = X[c].astype(str).fillna("NA").apply(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

            # apply default encoding for known categorical columns when encoders missing
            for c, opts in default_cats.items():
                if c in X.columns and c not in encoders:
                    opts_with_na = ["NA"] + [o for o in opts if o != "NA"]
                    le2 = LabelEncoder().fit(opts_with_na)
                    X[c] = X[c].astype(str).fillna("NA").apply(lambda v: le2.transform([v])[0] if v in le2.classes_ else -1)

            # fill numeric NAs with 0.0 to match model expectations
            X = X.fillna(0.0)
            # apply scaler for logistic if needed
            try:
                probs = model.predict_proba(X)[:, 1]
            except Exception:
                probs = model.predict(X)
            thr = float(opt_thr) if opt_thr is not None else 0.5
            labels = (probs >= thr).astype(int)
            out = df_in.copy()
            out["churn_probability"] = probs
            out["predicted_churn"] = labels
            st.write(out.head(20))
            st.success("Batch predictions completed")

else:
    st.markdown("Provide customer details below and click Predict.")
    inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_columns):
        with cols[i % 2]:
            inputs[col] = input_widget_for(col)

    if st.button("Predict"):
        row = {}
        for col in feature_columns:
            val = inputs.get(col, 0)
            if col in encoders:
                le = encoders[col]
                try:
                    if val in list(le.classes_):
                        row[col] = le.transform([val])[0]
                    else:
                        row[col] = -1
                except Exception:
                    row[col] = -1
            elif col in default_cats:
                # encode using default mapping
                opts = default_cats[col]
                try:
                    opts_with_na = ["NA"] + [o for o in opts if o != "NA"]
                    le2 = LabelEncoder().fit(opts_with_na)
                    if val in le2.classes_:
                        row[col] = int(le2.transform([val])[0])
                    else:
                        row[col] = -1
                except Exception:
                    row[col] = -1
            else:
                try:
                    if val == "NA":
                        row[col] = np.nan
                    else:
                        row[col] = float(val)
                except Exception:
                    row[col] = np.nan

        X = pd.DataFrame([row], columns=feature_columns)

        # prepare input and handle missing numeric values
        X = pd.DataFrame([row], columns=feature_columns)
        X = X.fillna(0.0)

        # If model expects scaled input and scaler is available, attempt to apply it
        try:
            from sklearn.linear_model import LogisticRegression
            if scaler is not None and model.__class__.__name__ == "LogisticRegression":
                X_scaled = scaler.transform(X)
                probs = model.predict_proba(X_scaled)[:, 1]
            else:
                probs = model.predict_proba(X)[:, 1]
        except Exception:
            try:
                probs = model.predict(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                probs = np.array([0.0])

        prob = float(probs[0])
        thr = float(opt_thr) if opt_thr is not None else 0.5
        pred = int(prob >= thr)

        st.metric("Churn Probability", f"{prob*100:.2f}%")
        st.write("Predicted Label:", "Churned" if pred == 1 else "Loyal")

        st.subheader("Input (model-ready)")
        st.dataframe(X.T)

        if st.checkbox("Show threshold details"):
            st.write({"threshold_used": thr})

        st.success("Prediction completed")
