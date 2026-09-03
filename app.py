import os
from datetime import datetime
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Credit Card Churn Predictor",
    page_icon=":material/credit_card:",
    layout="wide",
)

dark_mode = st.sidebar.toggle("Dark mode", value=False, key="dark_mode")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#14213d; --muted:#526176; --teal:#087f8c; --coral:#e76f51; --gold:#f4c95d; --mint:#dff5ee; --paper:#fbfaf7; }
    .stApp { background:radial-gradient(circle at 88% 4%, rgba(91,192,190,.24) 0, transparent 24%), radial-gradient(circle at 8% 88%, rgba(244,201,93,.16) 0, transparent 22%), linear-gradient(135deg, var(--paper) 0%, #edf5f2 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(251,250,247,.72); }
    h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:0 !important; color:var(--ink); }
    p, label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family:'DM Sans', sans-serif; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p { color:var(--ink); }
    .hero { position:relative; overflow:hidden; padding:2.2rem 2.4rem 1.8rem; border-radius:18px; background:linear-gradient(118deg,#14213d 0%,#164e63 48%,#087f8c 100%); color:white; box-shadow:0 18px 42px rgba(20,33,61,.2); animation:rise .65s ease-out both; }
    .hero::after { content:""; position:absolute; inset:-40% -15%; background:linear-gradient(105deg, transparent 42%, rgba(255,255,255,.16) 50%, transparent 58%); transform:translateX(-70%); animation:shine 5.5s ease-in-out 1s infinite; pointer-events:none; }
    .hero h1 { color:white !important; font-size:clamp(2rem,4vw,3.4rem); margin:0 0 .35rem; }
    .hero p { color:#e7f8f4; max-width:680px; margin:0; font-size:1.05rem; }
    .eyebrow { color:#a9ece0; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; margin-bottom:.65rem; }
    .section-label { color:#075e68; text-transform:uppercase; letter-spacing:.12em; font-weight:700; font-size:.72rem; margin:1.5rem 0 .3rem; }
    .risk-card { padding:1.2rem 1.35rem; border-radius:14px; border:1px solid #b8d9d2; background:rgba(255,255,255,.78); animation:rise .45s ease-out both; }
    .risk-number { font-family:'Space Grotesk'; font-size:2.35rem; font-weight:700; color:var(--teal); }
    .risk-copy { color:var(--muted); font-size:.9rem; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"]) { border:1px solid #c4dcd7 !important; background:rgba(255,255,255,.72); border-radius:14px; padding:.7rem .85rem .45rem; margin-bottom:.65rem; box-sizing:border-box; transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"]):hover { transform:translateY(-2px); border-color:#5bc0be !important; box-shadow:0 8px 20px rgba(20,33,61,.08); }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stSlider"]) { border-left:4px solid var(--teal) !important; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"] [data-baseweb="select"]) { border-left:4px solid var(--gold) !important; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"] [data-baseweb="input"]) { border-left:4px solid var(--coral) !important; }
    [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox [role="group"], [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInput"] input { background:#eef4fb; border:1px solid #8fa8c4; border-radius:9px; color:#14213d; }
    [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox input, [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInput"] input { color:#14213d; }
    [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInput"] input::placeholder { color:#14213d !important; opacity:1 !important; }
    [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInputRootElement"] { background:#eef4fb; border:1px solid #8fa8c4 !important; border-radius:9px; }
    [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInputRootElement"] input { background:transparent; border:0 !important; box-shadow:none !important; outline:0 !important; }
    [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox button { color:#14213d !important; }
    [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox [role="group"]:hover, [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInputRootElement"]:hover { border-color:#087f8c !important; background:#e5f0fa; }
    [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label p { color:#526176; font-size:.78rem; }
    [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label > div:first-of-type { width:1rem; height:1rem; border:1.5px solid #14213d; border-radius:4px; background:transparent; box-sizing:border-box; transition:background .18s ease, border-color .18s ease, transform .18s ease; }
    [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label:hover > div:first-of-type { transform:scale(1.08); border-color:#087f8c; }
    [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background:var(--coral); border-color:var(--coral); color:#ffffff; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-na_"]) [data-testid="stSlider"] { transition:opacity .2s ease, filter .2s ease; }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-na_"]) [data-testid="stSlider"]:has([data-disabled="true"]) { opacity:.48; filter:grayscale(.8); }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p { color:#ffffff !important; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.78); border:1px solid #b8d9d2; border-radius:14px; padding:1rem; box-shadow:0 6px 20px rgba(20,33,61,.05); }
    div.stButton > button { border-radius:10px; border:0; background:var(--coral); color:white; font-weight:700; transition:transform .18s ease, box-shadow .18s ease, background .18s ease; }
    div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(231,111,81,.3); background:#c9573c; color:white; }
    div.stButton > button:focus-visible { outline:3px solid rgba(244,201,93,.9); outline-offset:2px; }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within { border-color:var(--teal); box-shadow:0 0 0 2px rgba(8,127,140,.16); }
    [data-testid="stSlider"] [role="slider"] { background-color:var(--coral); border-color:var(--coral); }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] { background-color:var(--teal); }
    @keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    @keyframes shine { 0%, 18% { transform:translateX(-70%); } 48%, 100% { transform:translateX(70%); } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background:radial-gradient(circle at 88% 4%, rgba(8,127,140,.25) 0, transparent 24%), radial-gradient(circle at 8% 88%, rgba(231,111,81,.13) 0, transparent 22%), linear-gradient(135deg, #081525 0%, #10243b 100%); color:#edf5ff; }
        [data-testid="stHeader"] { background:rgba(8,21,37,.78); }
        .hero { background:linear-gradient(118deg,#063b2f 0%,#087f65 52%,#8bd8bd 100%); box-shadow:0 18px 42px rgba(0,0,0,.38), 0 0 0 1px rgba(244,201,93,.22); }
        h1, h2, h3 { color:#f4f8ff !important; }
        [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p { color:#edf5ff; }
        .section-label { color:#70d6c5; }
        .risk-card, div[data-testid="stMetric"] { background:rgba(18,39,62,.82); border-color:#31516f; color:#edf5ff; box-shadow:0 8px 24px rgba(0,0,0,.18); }
        .risk-number { color:#70d6c5; }
        .risk-copy { color:#b9cbe0; }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"]) { background:rgba(18,39,62,.78); border-color:#31516f !important; }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"][class*="st-key-input_"]):hover { border-color:#5bc0be !important; box-shadow:0 8px 20px rgba(0,0,0,.2); }
        [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox [role="group"], [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInputRootElement"] { background:#172d49; border-color:#557492 !important; }
        [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox input, [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInput"] input { color:#f4f8ff; }
        [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInput"] input::placeholder { color:#d1deed !important; }
        [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox [role="group"]:hover, [data-testid="stElementContainer"][class*="st-key-input_"] [data-testid="stTextInputRootElement"]:hover { background:#203c5d; border-color:#70d6c5 !important; }
        [data-testid="stElementContainer"][class*="st-key-input_"] .stSelectbox button { color:#f4f8ff !important; }
        [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label p { color:#b9cbe0; }
        [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label > div:first-of-type { border-color:#8fa8c4 !important; background:transparent; }
        [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label:hover > div:first-of-type { border-color:#70d6c5 !important; }
        [data-testid="stElementContainer"][class*="st-key-na_"] [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background:var(--coral); border-color:var(--coral) !important; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#09182b 0%,#102942 100%); }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#edf5ff !important; }
        [data-testid="stSidebar"] [data-testid="stToggle"] label p { color:#ffffff !important; }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#b9cbe0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        "model/churn_model.pkl",
        "model/feature_columns.pkl",
        "model/encoders.pkl",
        "model/scaler.pkl",
        "model/optimal_threshold.pkl",
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
model = art.get("model/churn_model.pkl")
feature_columns = art.get("model/feature_columns.pkl")
encoders = art.get("model/encoders.pkl") or {}
scaler = art.get("model/scaler.pkl")
opt_thr = art.get("model/optimal_threshold.pkl")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# sensible default categorical options when encoders are missing
default_cats = {
    "Gender": ["F", "M", "Others"],
    "Income_Category": [
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
        "Unknown",
    ],
    "Card_Category": ["Blue", "Silver", "Gold", "Platinum", "Unknown"],
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

numeric_slider_config = {
    "Customer_Age": (18, 100, 45, 1),
    "Dependent_count": (0, 5, 2, 1),
    "Months_on_book": (0, 100, 36, 1),
    "Total_Relationship_Count": (1, 6, 3, 1),
    "Months_Inactive_12_mon": (0, 6, 2, 1),
    "Contacts_Count_12_mon": (0, 6, 2, 1),
    "Credit_Limit": (0.0, 35_000.0, 5_000.0, 100.0),
    "Total_Revolving_Bal": (0.0, 3_000.0, 1_000.0, 50.0),
    "Avg_Utilization_Ratio": (0.0, 1.0, 0.3, 0.01),
    "Total_Trans_Amt": (0.0, 20_000.0, 4_000.0, 100.0),
    "Total_Trans_Ct": (0, 200, 60, 1),
    "Total_Ct_Chng_Q4_Q1": (0.0, 4.0, 0.7, 0.01),
    "Total_Amt_Chng_Q4_Q1": (0.0, 4.0, 0.7, 0.01),
}

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Retention intelligence / live workspace</div>
        <h1>Credit Card Churn Predictor</h1>
        <p>Turn customer signals into a clear retention decision with a fast, focused prediction workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)

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

st.sidebar.header("Input options")
batch_mode = st.sidebar.checkbox("Predict from CSV (batch)", value=False)

with st.sidebar:
    st.caption("Model status")
    if model is not None:
        st.badge("Ready", icon=":material/check_circle:", color="green")
    else:
        st.badge("Model missing", icon=":material/error:", color="red")
    st.caption(f"Decision threshold: {float(opt_thr) if opt_thr is not None else 0.5:.2f}")

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
        val = st.selectbox(col, options, index=0, key=f"input_{col}")
        return val

    # If column is a known categorical name, present a sensible dropdown
    if col in default_cats:
        options = default_cats[col]
        # ensure NA is available and default
        options = ["NA"] + [o for o in options if o != "NA"]
        val = st.selectbox(col, options, index=0, key=f"input_{col}")
        return val

    if col in numeric_slider_config:
        minimum, maximum, default, step = numeric_slider_config[col]
        use_na = st.checkbox("NA", value=False, key=f"na_{col}")
        slider_value = st.slider(
            col.replace("_", " "),
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step,
            disabled=use_na,
            key=f"input_{col}",
        )
        return "NA" if use_na else slider_value

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
        st.write("Sample input")
        st.dataframe(df_in.head(), width="stretch", hide_index=True)
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
            st.session_state.batch_output = out
            st.session_state.batch_summary = (float(np.mean(probs)), int(np.sum(labels)), len(labels))

        if "batch_output" in st.session_state:
            out = st.session_state.batch_output
            avg_prob, churn_count, total_count = st.session_state.batch_summary
            st.markdown('<div class="section-label">Batch pulse</div>', unsafe_allow_html=True)
            with st.container(horizontal=True):
                st.metric("Customers scored", f"{total_count:,}", border=True)
                st.metric("Likely to churn", f"{churn_count:,}", f"{churn_count / max(total_count, 1):.1%}", border=True)
                st.metric("Average risk", f"{avg_prob:.1%}", border=True)
            st.subheader("Prediction results")
            st.dataframe(out.head(50), width="stretch", hide_index=True)
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv",
                icon=":material/download:",
            )
            st.success("Batch predictions completed", icon=":material/check_circle:")

else:
    st.markdown("Provide customer details below and click Predict.")
    control_col, reset_col = st.columns([5, 1])
    with control_col:
        st.caption(":material/tune: Guided controls are enabled for common numeric signals. Drag a slider or use the value arrows for precision.")
    with reset_col:
        if st.button("Reset", icon=":material/refresh:"):
            for feature in feature_columns:
                st.session_state.pop(f"input_{feature}", None)
                st.session_state.pop(f"na_{feature}", None)
            st.rerun()
    inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_columns):
        with cols[i % 2]:
            with st.container(border=True):
                inputs[col] = input_widget_for(col)

    if st.button("Predict", icon=":material/bolt:"):
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

        label = "Churned" if pred == 1 else "Loyal"
        risk_band = "High attention" if prob >= thr else "Healthy signal"
        st.session_state.prediction_history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "probability": prob,
            "label": label,
        })
        st.session_state.prediction_history = st.session_state.prediction_history[:8]

        st.markdown('<div class="section-label">Decision signal</div>', unsafe_allow_html=True)
        result_col, action_col = st.columns([1, 1.35])
        with result_col:
            with st.container(border=True):
                st.metric("Churn probability", f"{prob*100:.2f}%", border=True)
                st.progress(min(max(prob, 0.0), 1.0), text=f"{risk_band} · threshold {thr:.2f}")
                st.markdown(f'<div class="risk-card"><div class="risk-number">{label}</div><div class="risk-copy">The model sees a {risk_band.lower()} for this customer.</div></div>', unsafe_allow_html=True)
        with action_col:
            with st.container(border=True):
                st.subheader("Signal profile")
                signal_data = pd.DataFrame({"signal": ["Churn risk", "Retention headroom"], "value": [prob, 1 - prob]})
                st.bar_chart(signal_data, x="signal", y="value", horizontal=True, height=190, color="#087f8c")

        with st.expander("Inspect model-ready input", icon=":material/table_chart:"):
            st.dataframe(X.T, width="stretch")
        if st.checkbox("Show threshold details"):
            st.write({"threshold_used": thr, "risk_band": risk_band})
        st.success("Prediction completed", icon=":material/check_circle:")

    if st.session_state.prediction_history:
        st.markdown('<div class="section-label">Recent decisions</div>', unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["probability"] = history_df["probability"].map(lambda value: f"{value:.2%}")
        st.dataframe(history_df, width="stretch", hide_index=True, column_config={"time": "Time", "probability": "Risk", "label": "Decision"})
