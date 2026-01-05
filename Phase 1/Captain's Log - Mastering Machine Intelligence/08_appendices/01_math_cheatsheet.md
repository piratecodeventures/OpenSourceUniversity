# Appendix A: The Math Cheatsheet

## 📐 Linear Algebra

### Vectors & Matrices
*   **Dot Product**: $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i = ||a|| ||b|| \cos \theta$.
    *   *Code*: `np.dot(a, b)` or `a @ b`
    *   *Meaning*: Similarity. 0 if orthogonal ($90^\circ$).
*   **Norm (L2)**: $||\mathbf{x}||_2 = \sqrt{\sum x_i^2}$.
    *   *Code*: `np.linalg.norm(x)`
*   **Matrix Multiplication**: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$.
    *   *Shape Rule*: $(m \times n) \cdot (n \times p) \to (m \times p)$.
*   **Determinant**: $\det(A)$.
    *   *Meaning*: Volume scaling factor. If 0, matrix is "singular" (squashes space flat).
*   **Eigenvalue**: $A \mathbf{v} = \lambda \mathbf{v}$.
    *   *Meaning*: Directions where $A$ only stretches, doesn't rotate.

### Decompositions
*   **Eigendecomposition**: $A = Q \Lambda Q^{-1}$ (Only for square matrices).
*   **SVD**: $A = U \Sigma V^T$ (For any matrix).
    *   $U$: Left singular vectors.
    *   $\Sigma$: Singular values (diagonal).
    *   $V$: Right singular vectors.

---

## 📈 Calculus

### Derivatives
*   **Power Rule**: $\frac{d}{dx} x^n = n x^{n-1}$.
*   **Chain Rule**: $\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}$.
    *   *Neural Nets*: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial h} \frac{\partial h}{\partial w}$.
*   **Product Rule**: $(uv)' = u'v + uv'$.
*   **Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$.
    *   $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

### Gradients
*   **Gradient**: $\nabla f = [\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}]$.
    *   *Direction*: Points to steepest **ascent**.
*   **Hessian**: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$.
    *   *Meaning*: Curvature. Positive definite -> Local Minima.

---

## 🎲 Probability

### Basics
*   **Bayes Theorem**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
    *   *Posterior = (Likelihood $\times$ Prior) / Evidence*.
*   **Expectation**: $\mathbb{E}[X] = \sum x P(x)$.
*   **Variance**: $\text{Var}(X) = \mathbb{E}[(X - \mu)^2]$.

### Distributions
*   **Gaussian (Normal)**: $\mathcal{N}(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$.
*   **Bernoulli**: $P(k) = p^k (1-p)^{1-k}$ (Coin flip).
*   **Softmax**: $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$. (Probability distribution sum to 1).

### Information Theory
*   **Entropy**: $H(P) = - \sum P(x) \log P(x)$. (Uncertainty).
*   **KL Divergence**: $D_{KL}(P||Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$. (Distance between dists).
*   **Cross Entropy**: $H(P, Q) = H(P) + D_{KL}(P||Q) = -\sum P(x) \log Q(x)$. (Loss function).
