# Appendix B: ML Metrics & Errors Cheatsheet

## 🎯 Classification Metrics

| Metric | Formula | Interpretation | When to Use |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP+TN}{Total}$ | Overall correctness | Balanced classes only. |
| **Precision** | $\frac{TP}{TP+FP}$ | "Of all predicted Positives, how many are real?" | High cost of False Positive (Spam filter). |
| **Recall** | $\frac{TP}{TP+FN}$ | "Of all real Positives, how many did we find?" | High cost of False Negative (Cancer detection). |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of P & R | Imbalanced datasets. |
| **ROC-AUC** | Area under FPR vs TPR | Ability to rank positives above negatives | Binary classification comparison. |
| **Log Loss** | $-\frac{1}{N}\sum (y \log \hat{y} + (1-y)\log(1-\hat{y}))$ | Confidence penalty | Probabilistic outputs. |

## 📉 Regression Metrics

| Metric | Formula | Interpretation | Nuance |
| :--- | :--- | :--- | :--- |
| **MSE** | $\frac{1}{N}\sum (y - \hat{y})^2$ | Mean Squared Error | Punishes outliers heavily (Squared). |
| **RMSE** | $\sqrt{MSE}$ | Root Mean Squared Error | Same units as target. Interpret as "Avg error". |
| **MAE** | $\frac{1}{N}\sum |y - \hat{y}|$ | Mean Absolute Error | Robust to outliers. |
| **R2 (R-Squared)** | $1 - \frac{SS_{res}}{SS_{tot}}$ | "Explained Variance" | 1.0 = Perfect. 0.0 = Baseline Mean Model. <0 = Worse than random. |

## ⚠️ Types of Errors

### Statistical Errors
*   **Type I Error (False Positive)**: "Crying Wolf". Model says YES, Truth is NO.
    *   *Significance Level ($ \alpha $)*: Probability of Type I error (usually 0.05).
*   **Type II Error (False Negative)**: "Missing the Wolf". Model says NO, Truth is YES.
    *   *Power ($ 1 - \beta $)*: Ability to avoid Type II error.

### Bias-Variance Tradeoff
*   **Bias**: Error due to overly shifting simplistic assumptions. (Underfitting).
    *   *High Bias*: Train Error High, Test Error High.
*   **Variance**: Error due to sensitivity to small fluctuations in training set. (Overfitting).
    *   *High Variance*: Train Error Low, Test Error High.

## 🕸️ Cluster Metrics (Unsupervised)
*   **Silhouette Score**: -1 to +1. How similar point is to own cluster vs neighbor cluster.
*   **Davies-Bouldin**: Lower is better. Ratio of within-cluster scatter to separation.

## 📝 NLP Metrics
*   **BLEU**: N-Gram overlap precision. (Translation).
*   **ROUGE**: N-Gram recall. (Summarization).
*   **Perplexity**: $e^{H(P)}$. "How confused is the model?" Lower is better.
