# Advanced Core ML: Bayesian Methods (MCMC & VI)

## 📜 Story Mode: The Oracle

> **Mission Date**: 2042.11.15
> **Location**: Deep Space Outpost "Vector Prime"
> **Officer**: Science Officer Kael
>
> **The Problem**: We are landing on a rogue planet.
> The Neural Net says: "Landing Safety: 99%".
> But the Net has *never seen this terrain before*. It is confidently wrong (OOD).
>
> Standard Nets are Point Estimates. They give a single number.
> I need a **Distribution**.
> I don't want "99%". I want "Mean 99%, Variance 50%".
> I need to know *what the model doesn't know*.
>
> I need to integrate over all possible parameters $\theta$, not just the best one.
> But the integral is intractable.
> I have to simulate it. I have to walk through the parameter space randomly to build a map.
>
> *"Computer! Initiate Markov Chain Monte Carlo. Run the Metropolis-Hastings algorithm. Sample the Posterior. Quantify the Uncertainty."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**:
    *   **Bayesian Inference**: Calculating Posterior $P(\theta|D) \propto P(D|\theta)P(\theta)$.
    *   **MCMC**: Exact sampling from complex posteriors using random walks.
    *   **Variational Inference (VI)**: Approximate inference using optimization.
2.  **WHY**: Uncertainty Quantification. Small data handling. Prior knowledge incorporation.
3.  **WHEN**: Medical trials, High-stakes robotics, A/B testing.
4.  **WHERE**: `PyMC`, `Stan`, `TensorFlow Probability`.
5.  **WHO**: Statisticians, Physicists.
6.  **HOW**: `pm.sample(2000)`.

> [!NOTE]
> **🛑 Pause & Explain (In Simple Words)**
>
> **The Drunk Hiker (MCMC).**
>
> - **Goal**: Find the highest peaks of a mountain (Probability Density) in dense fog.
> - **Strategy**:
> - 1. Stand at a point.
> - 2. Pick a random direction.
> - 3. If it's higher, go there.
> - 4. If it's lower, *maybe* go there (Probabilistically) to explore.
> - 5. Repeat 10,000 times.
> - **Result**: You spend the most time at the peaks. Your footprint track *is* the distribution.

---

## 2. Mathematical Problem Formulation

### The Intractable Integral
$$ P(\theta | D) = \frac{P(D|\theta) P(\theta)}{\int P(D|\theta')P(\theta') d\theta'} $$
The denominator (Evidence) is an integral over high-dimensional space.
Impossible to compute analytically.

### Solutions
1.  **MCMC (Sampling)**: "Don't compute the integral. Just get samples from the distribution." (Accurate but Slow).
2.  **Variational Inference (Optimization)**: "Approximation the posterior with a Gaussian $q(\theta)$ and minimize KL Divergence." (Fast but Approximate).

---

## 3. Step-by-Step Derivation

### Metropolis-Hastings Algorithm
1.  Current state $\theta$.
2.  Propose $\theta' \sim N(\theta, \sigma)$.
3.  Calculate Acceptance Ratio $\alpha = \frac{P(\theta'|D)}{P(\theta|D)}$.
    *   Note: The denominator cancels out! We don't need the integral!
4.  Accept $\theta'$ with probability $\min(1, \alpha)$.

---

## 4. Algorithm Construction

### Map to Memory (NUTS)
Standard MCMC (Random Walk) is slow in high dimensions.
**NUTS (No-U-Turn Sampler)**: Uses gradients (Hamiltonian Monte Carlo) to slide along probability contours like a hockey puck.
Converges much faster. Used by PyMC.

---

## 5. Optimization & Convergence Intuition

### Variational Inference (VI)
We turn Integration into Optimization.
Minimize $KL(q(\theta) || P(\theta|D))$.
Equivalent to Maximizing the **ELBO (Evidence Lower Bound)**.
$$ \text{ELBO} = E_q[\log P(D, \theta)] - E_q[\log q(\theta)] $$
This works with SGD! We can train Bayesian Neural Nets on GPUs.

---

## 6. Worked Examples

### Example 1: Coin Flip
**Data**: 2 Heads, 8 Tails.
**Frequentist**: $P(H) = 0.2$. (Confident).
**Bayesian**:
*   Prior: Beta(2,2) (Values near 0.5 likely).
*   Posterior: Beta(4, 10).
*   Mean: 0.28.
*   Credible Interval: [0.1, 0.5].
**Result**: "I think it's 0.28, but I admit it could be 0.5."

---

## 7. Production-Grade Code

### PyMC Probabilistic Programming

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# 1. Data
# True slope=2, True intercept=1
X = np.random.randn(100)
y = 2 * X + 1 + np.random.randn(100) * 0.5

# 2. Model
with pm.Model() as model:
    # Priors
    slope = pm.Normal("slope", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=slope*X + intercept, sigma=sigma, observed=y)
    
    # 3. Inference (MCMC)
    trace = pm.sample(1000, return_inferencedata=True)

# 4. Results
pm.plot_posterior(trace)
pm.plot_posterior(trace)
plt.show()
```

> [!TIP]
> **👁️ Visualizing the Uncertainty**
> A Bayesian model doesn't just draw one line. It draws Infinite Lines, weighted by probability.
>
> ```python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.stats import norm
>
> def plot_bayesian_regression():
>     # 1. Generate Data
>     np.random.seed(42)
>     X = np.linspace(0, 10, 10)
>     y_true = 2 * X + 1
>     y_obs = y_true + np.random.normal(0, 2, 10) # Noisy observations
>     
>     # 2. Bayesian Posterior Simulation (Simplified)
>     # Instead of MCMC, we just sample lines consistent with the data noise
>     # In reality, you'd use the Posterior distribution of Slope/Intercept
>     
>     plt.figure(figsize=(10, 6))
>     plt.scatter(X, y_obs, color='black', label='Data')
>     
>     # Sample 50 possible lines from a hypothetical posterior
>     # (Simulation for visual intuition)
>     for i in range(50):
>         slope_sample = np.random.normal(2, 0.2)
>         intercept_sample = np.random.normal(1, 1.0)
>         y_sample = slope_sample * X + intercept_sample
>         plt.plot(X, y_sample, color='red', alpha=0.1)
>         
>     plt.plot(X, y_true, color='blue', linewidth=2, linestyle='--', label='True Truth')
>     plt.title(f"Bayesian Regression: 50 Possible Worlds")
>     plt.xlabel("X")
>     plt.ylabel("y")
>     plt.legend()
>     plt.show()
>
> # Uncomment to run:
> # plot_bayesian_regression()
> ```

> [!CAUTION]
> **🛑 Production Warning**
>
> **MCMC Speed**:
> MCMC is **Orders of Magnitude** slower than SGD.
> Don't use it for ImageNet.
> Use it for A/B testing, Marketing Mix Modeling, or small-data medical problems.
> For Neural Nets, use **Monte Carlo Dropout** (Cheap Approximation).

---

## 8. System-Level Integration

```mermaid
graph TD
    Data --> |Priors| Model
    Model --> |Sampling| HamiltonianMC
    HamiltonianMC --> |Trace| Posterior
    Posterior --> |Integration| Prediction Interval
```

**Where it lives**:
**SpaceX/NASA**: Trajectory estimation uses Kalman Filters (Bayesian).
**Drug Discovery**: Estimating toxicity with limited samples.

---

## 9. Evaluation & Failure Analysis

### Failure Mode: Bad Priors
If Prior is strong and wrong (e.g., $P(\theta) = 0$ for the true value).
The Posterior will never find the truth.
**Fix**: Use Weakly Informative Priors (Wide Gaussians).

---

## 10. Ethics, Safety & Risk Analysis

### Certainty is Safety
A standard AI says "Use this treatment" (51% confidence).
A Bayesian AI says "Using this treatment has 49% risk of death."
Bayesian methods are mandatory for Ethical AI in high-stakes domains.

---

## 11. Advanced Theory & Research Depth

### Bayesian Neural Networks (BNN)
Every weight $w$ is a Gaussian $(\mu, \sigma)$.
Doubles the parameters. Use VI to train.
Allows "Uncertainty Awareness" in Deep Learning.

---

## 12. Career & Mastery Signals

### Interview Pitfall
Q: "What is the difference between Confidence Interval and Credible Interval?"
**Bad Answer**: "They are the same."
**Good Answer**: "Confidence Interval (Frequentist): If we repeat the experiment, 95% of intervals will contain True Parameter.
Credible Interval (Bayesian): There is 95% probability the True Parameter is in *this* interval."

---

## 13. Assessment & Mastery Checks

**Q1: The Prior**
What is a Conjugate Prior?
*   *Answer*: A Prior that, when multiplied by Likelihood, results in a Posterior of the *same family* (e.g. Beta-Binomial). Allows analytic solution.

**Q2: Trace Plot**
What indicates MCMC convergence?
*   *Answer*: The Trace looks like a "Fuzzy Caterpillar" (Mixing well). If it meanders, it hasn't converged (R-hat > 1.05).

---

## 14. Further Reading & Tooling

*   **Lib**: **PyMC** (The gold standard).
*   **Book**: *"Bayesian Methods for Hackers"* (Cam Davidson-Pilon).

---

## 15. Concept Graph Integration

*   **Previous**: [Statistical Learning Theory](02_core_ml/05_advanced/02_theory.md).
> *   **Next**: [Deep Learning](03_deep_learning/01_neural_nets.md) (The Neural Era).
> 
> ### Concept Map
> ```mermaid
> graph LR
>     Inference[Inference Methods] --> Exact
>     Inference --> Approximate
>     
>     Exact --> Enumeration
>     Exact --> MCMC[MCMC Sampling]
>     
>     Approximate --> VI[Variational Inference]
>     Approximate --> Laplace[Laplace Approx]
>     
>     MCMC --> MH[Metropolis-Hastings]
>     MCMC --> NUTS[NUTS / HMC]
>     
>     VI -- "Optimizes" --> ELBO[ELBO Lower Bound]
>     VI -- "Uses" --> KL[KL Divergence]
>     
>     Bayes -- "Key Formula" --> Posterior
>     Posterior -- "Is" --> Likelihood_x_Prior
>     
>     style Inference fill:#f9f,stroke:#333
>     style MCMC fill:#bbf,stroke:#333
>     style VI fill:#bfb,stroke:#333
> ```
