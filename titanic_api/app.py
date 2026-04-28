import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for Streamlit

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Titanic Survival Predictor",
    page_icon  = "🚢",
    layout     = "wide"
)

# ── Load model once at startup ─────────────────────────────────────────────────
# @st.cache_resource means the model loads ONCE and stays in memory.
# Without this, Streamlit reloads the model on every slider drag — very slow.

@st.cache_resource
def load_model():
    model = joblib.load("titanic_model.joblib")
    return model

@st.cache_resource
def load_explainer(_model):
    # Underscore prefix on _model tells Streamlit not to hash this argument
    preprocessor = _model.named_steps['preprocessor']
    rf_model     = _model.named_steps['model']
    explainer    = shap.TreeExplainer(rf_model)
    return explainer, preprocessor

model              = load_model()
explainer, preprocessor = load_explainer(model)

# ── Feature name list ──────────────────────────────────────────────────────────
# These must match EXACTLY what your ColumnTransformer produces
num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']

ohe_feature_names = (
    preprocessor
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(['sex', 'embarked', 'title'])
    .tolist()
)
all_feature_names = num_cols + ohe_feature_names

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.markdown(
    "Adjust the passenger details in the sidebar. "
    "The model predicts survival probability in real time and explains "
    "exactly which features drove the result."
)
st.divider()

# ── Sidebar — passenger inputs ─────────────────────────────────────────────────
st.sidebar.header("🧳 Passenger Details")
st.sidebar.markdown("Adjust the values below:")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
    index=2
)

sex = st.sidebar.radio(
    "Sex",
    options=["male", "female"],
    horizontal=True
)

age = st.sidebar.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=28,
    step=1
)

fare = st.sidebar.slider(
    "Fare (£)",
    min_value=0.0,
    max_value=520.0,
    value=32.0,
    step=0.5
)

embarked = st.sidebar.selectbox(
    "Port of Embarkation",
    options=["S", "C", "Q"],
    format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
)

sibsp = st.sidebar.number_input(
    "Siblings / Spouses aboard",
    min_value=0, max_value=8, value=0, step=1
)

parch = st.sidebar.number_input(
    "Parents / Children aboard",
    min_value=0, max_value=6, value=0, step=1
)

title = st.sidebar.selectbox(
    "Title",
    options=["Mr", "Mrs", "Miss", "Master", "Rare"],
    index=0
)

# ── Derived features — computed automatically ──────────────────────────────────
# These were engineered in Section 4 of the notebook.
# The user sets sibsp, parch, and the app derives family_size and is_alone.
family_size = sibsp + parch + 1
is_alone    = 1 if family_size == 1 else 0

st.sidebar.divider()
st.sidebar.markdown("**Derived features** (auto-calculated):")
st.sidebar.markdown(f"- `family_size` = {family_size}")
st.sidebar.markdown(f"- `is_alone` = {is_alone}")

# ── Build input DataFrame ──────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    'pclass':      pclass,
    'age':         age,
    'sibsp':       sibsp,
    'parch':       parch,
    'fare':        fare,
    'sex':         sex,
    'embarked':    embarked,
    'family_size': family_size,
    'is_alone':    is_alone,
    'title':       title
}])

# ── Prediction ─────────────────────────────────────────────────────────────────
prediction    = model.predict(input_df)[0]
probability   = model.predict_proba(input_df)[0][1]
survived      = prediction == 1

# ── Layout — two columns ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── Column 1: Prediction result ────────────────────────────────────────────────
with col1:
    st.subheader("Prediction")

    if survived:
        st.success(f"## ✅ Survived")
        st.markdown(f"### Survival probability: **{probability:.1%}**")
    else:
        st.error(f"## ❌ Did Not Survive")
        st.markdown(f"### Survival probability: **{probability:.1%}**")

    # Probability bar
    st.progress(float(probability))

    st.divider()

    # Passenger summary table
    st.subheader("Passenger Summary")
    summary = pd.DataFrame({
        "Feature": [
            "Class", "Sex", "Age", "Fare",
            "Embarked", "Siblings/Spouses", "Parents/Children",
            "Title", "Family Size", "Travelling Alone"
        ],
        "Value": [
            f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} Class",
            sex.capitalize(),
            f"{age} years",
            f"£{fare:.2f}",
            {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[embarked],
            sibsp, parch, title, family_size,
            "Yes" if is_alone else "No"
        ]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Column 2: SHAP explanation ─────────────────────────────────────────────────
with col2:
    st.subheader("Why this prediction?")
    st.markdown(
        "Each bar shows how much a feature pushed the survival probability "
        "**up** (red) or **down** (blue) from the baseline."
    )

    # Transform input through preprocessor for SHAP
    X_transformed = preprocessor.transform(input_df)
    X_shap_df     = pd.DataFrame(X_transformed, columns=all_feature_names)

    # Compute SHAP values for this single passenger
    shap_values_single = explainer.shap_values(X_shap_df)

    # Handle both old format (list) and new format (3D array)
    if isinstance(shap_values_single, list):
        sv = shap_values_single[1][0]               # old format
    else:
        sv = shap_values_single[:, :, 1][0]         # new format (3D array)

    # Build SHAP Explanation object for waterfall plot
    explanation = shap.Explanation(
        values        = sv,
        base_values   = explainer.expected_value[1]
            if isinstance(explainer.expected_value, np.ndarray)
            else explainer.expected_value,
        data          = X_shap_df.iloc[0].values,
        feature_names = all_feature_names
    )

    # Render waterfall plot
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Top 3 reasons in plain English
    st.divider()
    st.subheader("Top reasons")

    shap_dict   = dict(zip(all_feature_names, sv))
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    for feat, val in top_factors:
        direction = "increased" if val > 0 else "decreased"
        arrow     = "🔺" if val > 0 else "🔻"
        st.markdown(f"{arrow} **{feat}** {direction} survival probability by `{abs(val):.3f}`")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**Model:** Random Forest · **Tuning:** GridSearchCV (18 candidates) · "
    "**Test ROC-AUC:** 0.8951 · **Test Accuracy:** 0.8473 · "
    "**Explainability:** SHAP TreeExplainer"
)