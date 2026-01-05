# Appendix E: The ML Project Lifecycle

## 🔄 The Flywheel (End-to-End Workflow)

> **"Amateurs worry about algorithms. Professionals worry about data pipelines."**

### Phase 1: Problem Scoping & Design
*   **The Question**: "Can we solve this with simple rules?" (If yes, don't use ML).
*   **Metric Definition**: "What does success look like?"
    *   *Business Metric*: Revenue, User Retention.
    *   *Proxy Metric*: Accuracy, F1-Score, Latency.
    *   *Gap*: Optimizing F1-Score doesn't always increase Revenue. Bridge this gap early.
*   **Feasibility Study**: Do we have the data? Is the signal predictive?

### Phase 2: The Data Engine
*   **Collection**: Scraping, Logging, Labeling.
*   **Cleaning**: Handling missing values, outliers, and formatting.
*   **Splitting**: Train / Validation / Test.
    *   *Critical*: Split by **Time** if the problem is temporal (e.g., Stock prediction). Random split causes "Look-ahead Bias".
*   **Feature Engineering**: Turning raw data into signal. "Domain Knowledge" lives here.

### Phase 3: Modeling & Experimentation
*   **Baselines**: Always start with a simple model (Logistic Regression, Random Forest).
*   **Complexity**: Only move to Deep Learning if the Baseline fails.
*   **Debugging**: Overfitting vs Underfitting.
    *   *High Bias*: Add features, increase model size.
    *   *High Variance*: Add data, regularization, simplify model.

### Phase 4: Deployment (The Production Gap)
*   **Serving**: Real-time (API) vs Batch (Offline).
*   **Optimization**: Quantization, Pruning, Caching.
*   **Testing**: Unit Tests for code, "Data Tests" for inputs (schema validation).

### Phase 5: Monitoring & Maintenance
*   **Drift Detection**: Is the world changing?
*   **Retraining**: Automated pipelines (CI/CD/CT).
*   **Feedback Loops**: Use model predictions to collect new labels (Active Learning).

---

## ⚠️ Common Challenges & Pitfalls

### 1. Data Leakage
using information in Training that won't be available at Inference.
*   *Example*: Using "Transaction Status" (Approved/Rejected) to predict "Fraud". The status is generated *after* the fraud check.
*   *Fix*: Strict timestamp-based splitting.

### 2. Training-Serving Skew
The code used to generate features in Python (Training) differs from the C++/Java code in Production.
*   *Fix*: Feature Store (Calculate once, serve everywhere).

### 3. The "SOTA" Trap
Chasing State-of-the-Art (SOTA) papers instead of solving the business problem.
*   *Reality*: A simple XGBoost that is robust & interpretable is usually better than a fragile Transformer that gains 0.1% accuracy.

### 4. Silent Failures
Code rarely crashes in ML. It just produces garbage predictions.
*   *Fix*: Monitor output distributions, not just system health (CPU/RAM).

---

## 📝 Implementation Checklist

- [ ] **Data Versioning**: DVC / LakeFS. (Can I reproduce the dataset from last month?)
- [ ] **Experiment Tracking**: MLflow / Weights & Biases. (What hyperparameters did I use?)
- [ ] **Model Registry**: Staging vs Production artifacts.
- [ ] **Rollback Strategy**: "Can I revert to the old model in 5 seconds if the new one fails?"
