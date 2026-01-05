import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Setup
st.set_page_config(page_title="Pred Pipeline Dashboard", layout="wide")
st.title("🏡 Housing Price AI Dashboard")

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    return model

try:
    model = load_assets()
    st.success("Model Loaded Successfully!")
except:
    st.error("Model not found. Run 'python pipeline.py' first.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("Input Features")
features = {}
for i in range(10):
    # Create sliders for each feature
    features[f'feat_{i}'] = st.sidebar.slider(f'Feature {i}', -3.0, 3.0, 0.0)

input_df = pd.DataFrame([features])

# 3. Main Display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button("Estimate Price"):
        pred = model.predict(input_df)[0]
        st.metric("Estimated Value", f"${pred:,.2f}")
        
        # Explainability
        st.subheader("Why this price?")
        # Note: SHAP KernelExplainer is slow, using TreeExplainer on the XGB part usually better
        # For Stacking, we explain the dominant base model (XGB usually)
        xgb_model = model.estimators_[0] # Assuming XGB is first
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

with col2:
    st.subheader("Model Architecture")
    st.code("""
    StackedRegressor:
    ├── Level 0:
    │   ├── XGBoost (Optuna Tuned)
    │   ├── RandomForest (100 trees)
    │   └── SVM (RBF Kernel)
    └── Level 1 (Meta):
        └── Ridge Regression
    """, language="text")

    st.info("The Stacked model combines 3 perspectives to minimize variance.")
