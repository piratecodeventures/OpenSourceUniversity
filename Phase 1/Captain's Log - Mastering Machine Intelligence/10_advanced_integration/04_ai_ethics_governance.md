# ai_ethics_governance: Bias, Safety & Compliance (Deep Dive)

## 📜 Story Mode: The Guardian

> **Mission Date**: 2044.02.01
> **Location**: Compliance Office
> **Officer**: AI Safety Auditor
>
> **The Problem**: The Hiring Algo trained on 10 years of resumes.
> In the past 10 years, we mostly hired Men.
> Now the AI penalizes the word "Women's Chess Club".
>
> **The Solution**: **Algorithmic Fairness**.
> Bias Detection Pipelines.
> Demographic Parity Constraints.
> "Math cannot fix Culture, but it can stop propagating it."
>
> *"Computer. Run Fairness Indicators. Calculate Disparate Impact Ratio."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: Ensuring AI systems do not discriminate, cause harm, or violate laws.
2.  **WHY**: Bias + Scale = Mass Destruction. Legal Liability (EU AI Act).
3.  **WHEN**: Always, but critical in Regulated Industries (HR, Lending, Crime).
4.  **WHERE**: `Fairlearn`, `AIF360` (IBM), `TensorFlow Model Card`.
5.  **WHO**: Timnit Gebru (Gender Shades), Joy Buolamwini (Algorithmic Justice League).
6.  **HOW**: Pre-processing (Re-weighting) $\to$ In-processing (Constraint Loss) $\to$ Post-processing (Threshold Adjustment).

---

## 2. Mathematical Deep Dive: Fairness Metrics

### 2.1 Demographic Parity (Independence)
The probability of positive outcome should be equal across groups.
$$ P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1) $$
*   $A$: Sensitive Attribute (Gender, Race).
*   **Problem**: What if Group 1 is legitimately more qualified? (Simpson's Paradox).

### 2.2 Equal Opportunity (Separation)
The True Positive Rate should be equal.
$$ P(\hat{Y}=1 | Y=1, A=0) = P(\hat{Y}=1 | Y=1, A=1) $$
*   "If two people are qualified, they should have equal chance of selection."

---

## 3. The Ship's Code (Polyglot: Fairlearn)

```python
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.tree import DecisionTreeClassifier

# LEVEL 2: Mitigating Bias
def train_fair_model(X, y, sensitive_features):
    # 1. Measure Baseline Bias
    metric = MetricFrame(
        metrics=selection_rate, 
        y_true=y, 
        y_pred=y_pred, 
        sensitive_features=sensitive_features
    )
    print(f"Disparity: {metric.difference()}")
    
    # 2. Train with Fairness Constraint (Exponentiated Gradient)
    # Solves a Lagrangian optimization: Max Accuracy s.t. Constraint < Epsilon
    constraint = DemographicParity()
    clf = DecisionTreeClassifier()
    
    mitigator = ExponentiatedGradient(clf, constraint)
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    
    return mitigator
```

---

## 4. System Architecture: The Model Card

Every model must ship with a "Nutrition Label" (**Model Card**).

| Section | Content |
| :--- | :--- |
| **Model Details** | v1.0, ConvNet, Trained 2024-01-01. |
| **Intended Use** | Skin cancer detection helper. NOT for autonomous diagnosis. |
| **Training Data** | ISIC Dataset (80% Light Skin, 20% Dark Skin). |
| **Limitations** | Performance degrades on Dark Skin (Type V-VI). |
| **Ethical Considerations** | False Negatives may delay treatment. |

---

## 13. Industry Interview Corner

### ❓ Real World Questions

**Q1: "Can we have both Demographic Parity and Equal Opportunity?"**
*   **Answer**: "Mathematically, **No** (Impossibility Theorem). Unless the base rates ($P(Y=1|A)$) are identical across groups, you cannot satisfy both. You must choose one based on your ethical framework (Equality of Outcome vs Equality of Opportunity)."

**Q2: "What is disparate impact?"**
*   **Answer**: "A legal term (US Labor Law). If the selection rate for a protected group is less than 80% of the highest group (4/5ths rule), it is evidence of adverse impact. In ML, we code this as a hard constraint."

---

## 14. Debug Your Thinking (Misconceptions)

> [!WARNING]
> **"Removing the Gender column fixes bias."**
> *   **Correction**: **Proxy Variables**. Your zip code correlates with race. Your college correlates with gender. The model *will* reconstruct the sensitive attribute from proxies. You must measure bias explicitly.

> [!WARNING]
> **"Bias is a data problem."**
> *   **Correction**: It's also a **Metrics problem** (optimizing for Clickbait biases towards outrage) and a **Deployment problem** (using a US model in India).
