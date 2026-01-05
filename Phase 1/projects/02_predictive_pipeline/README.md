# 🏗️ End-to-End Predictive Pipeline (Core ML Project)

> **Officer Kael's Log**: "A model on a laptop is a toy. A model in an API is a tool. A model with a dashboard is a weapon."

This project implements a complete **Production ML Pipeline** for a housing price prediction system.
It goes beyond model fitting—it handles automation, deployment, and Explainability.

## 🛠️ Components

### 1. `pipeline.py` (The Factory)
*   **AutoML Logic**: Automatically trains **Linear Regression, Random Forest, XGBoost, SVM**.
*   **Hyperparameter Tuning**: Uses **Optuna** to find the best params for XGBoost.
*   **Stacking**: Builds a Meta-Learner on top of the best models.
*   **Artifacts**: Saves the best model to `model.pkl`.

### 2. `app.py` (The Service)
*   **FastAPI**: Serves the model as a REST API.
*   **Pydantic**: Validates input data types (prevents "Garbage In").
*   **Health Checks**: `/health` endpoint for Kubernetes/Docker.

### 3. `dashboard.py` (The Cockpit)
*   **Streamlit**: Interactive UI.
*   **Explainability**: Uses **SHAP** (SHapley Additive exPlanations) to explain *why* a house is priced locally.
*   **What-If Analysis**: Change the square footage sliders and see the price update real-time.

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install xgboost optuna fastapi uvicorn streamlit shap scikit-learn
```

### Step 2: Train the Model
```bash
python pipeline.py
# Output: Best params found... Saving model.pkl...
```

### Step 3: Run the API
```bash
uvicorn app:app --reload
# Open http://localhost:8000/docs
```

### Step 4: Launch Dashboard
```bash
streamlit run dashboard.py
```
