# Linear Algebra: Norms and Inner Products

## 📜 Story Mode: The Target Lock

> **Mission Date**: 2042.03.20
> **Location**: Deep Space Outpost "Vector Prime"
> **Officer**: Lead Engineer Kael
>
> **The Problem**: We can translate coordinates (Linear Transforms) and navigate span (Vector Spaces).
> But the ship's targeting system is blind. It knows *where* the enemy is, but not *how far*.
>
> It says: "Enemy at $[10, 20]$".
> I ask: "Is that close enough to fire?"
> It says: "I don't know what 'Close' means."
>
> We need a Ruler. We need to measure **Distance** (Norms).
>
> And we need a Compass. We need to measure **Direction** relative to us (Inner Products).
>
> *"Computer! Define the L2 Metric. Calculate Cosine Similarity. Lock onto the nearest hostile vector!"*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**:
    *   **Norm ($||\mathbf{v}||$)**: A function that assigns a positive "Length" to a vector.
    *   **Inner Product ($\mathbf{u} \cdot \mathbf{v}$)**: A function that measures "Alignment" (Angle) between two vectors.
2.  **WHY**: Without them, "Optimization" is impossible. You can't minimize "Error" if you can't measure how big the error is.
3.  **WHEN**: Every time you train a model. Loss Functions (MSE, Cross-Entropy) are just fancy Norms.
4.  **WHERE**:
    *   **Search**: "Are these two documents similar?" (Cosine Similarity).
    *   **Physics**: Work = Force $\cdot$ Displacement (Inner Product).
5.  **WHO**: ML Engineers (Loss functions), Physicists (Metrics).
6.  **HOW**: `L2 = sqrt(sum(x^2))`. `Dot = sum(a*b)`.

> [!NOTE]
> **🛑 Pause & Explain (In Simple Words)**
>
> **Norm = Size.**
> "How big is this file/error/signal?"
>
> **Inner Product = Agreement.**
> "Do these two vectors point in the same direction?"
>
> - Product > 0: They agree.
> - Product = 0: They are indifferent (Orthogonal/Perpendicular).
> - Product < 0: They disagree (Opposite).

---

## 2. Mathematical Problem Formulation

### The Inner Product (Dot Product)
$$ \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = ||\mathbf{u}|| ||\mathbf{v}|| \cos(\theta) $$
This connects geometry (Angle $\theta$) to algebra (Multiplication).

### The Norms (L1 vs L2)
*   **L2 Norm (Euclidean)**: Direct line. $\sqrt{x^2 + y^2}$. Sensitive to outliers (Squares equate to massive errors).
*   **L1 Norm (Manhattan)**: City blocks. $|x| + |y|$. Robust to outliers, encourages sparsity.

---

## 3. Step-by-Step Derivation

### Deriving Cosine Similarity
**Goal**: Measure similarity independent of magnitude. "Is this huge text about 'Cats' similar to this tiny tweet about 'Cats'?"
**Step 1**: Start with Dot Product.
$$ \mathbf{u} \cdot \mathbf{v} = ||\mathbf{u}|| ||\mathbf{v}|| \cos(\theta) $$
**Step 2**: Solve for $\cos(\theta)$.
$$ \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| ||\mathbf{v}||} $$
**Result**: A score from -1 (Opposite) to 1 (Identical). Size doesn't matter.

> [!TIP]
> **🧠 Intuition Behind the Math**
>
> "Normalization" just means "Make the arrow length 1".
> Once all arrows are length 1, the only difference between them is **Direction**.
> Cosine Similarity effectively normalizes your vectors before comparing them.

---

## 4. Algorithm Construction

### Map to Memory (Sparse vs Dense)
In Recommender Systems, vectors are **Sparse** (mostly zeros).
Dot product of sparse vectors takes $O(\text{non-zeros})$, not $O(N)$.
Libraries like SciPy store only indices `(index, value)`.

### Algorithm: Nearest Neighbor Search
**Goal**: Find vector most similar to Query $\mathbf{q}$.
**Naive**: Scan all 1M items. $O(N)$. Slow.
**Optimized**: **HNSW (Hierarchical Navigable Small World)**. A graph structure that acts like a "highway" to the target neighborhood. $O(\log N)$.

---

## 5. Optimization & Convergence Intuition

### Regularization (L1 vs L2 Ridge/Lasso)
We add a Norm penalty to the Loss function: $Loss + \lambda ||\mathbf{w}||$.
*   **L2 (Ridge)**: Shrinks weights small. "Spread the blame."
*   **L1 (Lasso)**: Shrinks weights to ZERO. "Find the few key factors."
*   **Optimization View**: L2 ball is round; L1 ball is a diamond. The diamond hits axes first (Sparsity).

---

## 6. Worked Examples

### Example 1: The Lazy Employee (Cosine Sim)
**Story**: HR is matching resumes to jobs.
*   **Job**: `[Python, SQL]` (Vector directions).
*   **Resume A**: `[Python, SQL]` (Perfect alignment).
*   **Resume B**: `[Python, SQL, Cooking, Dancing]` (Diluted).
**Math**:
A: `[1, 1]`. Norm $\sqrt{2}$.
B: `[1, 1, 1, 1]`. Norm 2.
Dot Product $(A \cdot Job)$ is same for both.
But Cosine Similarity favors A. B is "distracted".

### Example 2: The Sniper (L2 Norm)
**Story**: Targeting laser error.
Error X = 10m. $L2 = \sqrt{100} = 10$.
Error X = 100m. $L2 = \sqrt{10000} = 100$.
The system fights 10x harder to fix the 100m error vs the 10m error? No, gradients scale.
**Lesson**: Squared errors make the model panic on large mistakes.

---

## 7. Production-Grade Code

### The Ship's Code (Polyglot: Pure Python + Libraries)

```python
import numpy as np
import torch
import tensorflow as tf
import math

# LEVEL 0: Pure Python (The Math Logic)
# Math: ||v|| = sqrt(sum(x^2))
def norm_l2_pure(vector):
    """
    Manual L2 norm implementation.
    Demonstrates the raw accumulation of squared errors.
    """
    sum_sq = 0.0
    for x in vector:
        sum_sq += x * x
    return math.sqrt(sum_sq)

def cosine_similarity_pure(v1, v2):
    # Dot Product sum(a*b)
    dot = sum(a*b for a, b in zip(v1, v2))
    # Norms
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Safety check
    return dot / (norm1 * norm2)

# LEVEL 1: NumPy (Batch CPU)
def batch_cosine_numpy(users, items):
    # users: (N, D), items: (M, D)
    # 1. Normalize Rows (Axis 1)
    # keepdims=True is crucial for broadcasting
    users_n = users / np.linalg.norm(users, axis=1, keepdims=True)
    items_n = items / np.linalg.norm(items, axis=1, keepdims=True)
    
    # 2. Dot Product
    return users_n @ items_n.T

# LEVEL 2: PyTorch (Research/GPU)
def pairwise_sim_torch(users, items):
    # users: Tensor(N, D)
    # F.normalize is the standard way to do L2 normalization
    users_n = torch.nn.functional.normalize(users, p=2, dim=1)
    items_n = torch.nn.functional.normalize(items, p=2, dim=1)
    return users_n @ items_n.T

# LEVEL 3: TensorFlow (Production)
def pairwise_sim_tf(users, items):
    # tf.math.l2_normalize handles the epsilon safety automatically
    users_n = tf.math.l2_normalize(users, axis=1)
    items_n = tf.math.l2_normalize(items, axis=1)
    return tf.linalg.matmul(users_n, items_n, transpose_b=True)

# LEVEL 4: Visualization (Angle vs Distance)
def visualize_metrics():
    """
    Shows why Cosine Similarity != Euclidean Distance.
    """
    import matplotlib.pyplot as plt
    
    # Vector A: Short
    v_a = np.array([1, 1])
    # Vector B: Long (Same direction)
    v_b = np.array([3, 3])
    # Vector C: Medium (Different direction)
    v_c = np.array([2, -0.5])
    
    origin = [0, 0]
    
    plt.figure(figsize=(6,6))
    plt.quiver(*origin, *v_a, color='r', scale=10, label='A (1,1)')
    plt.quiver(*origin, *v_b, color='b', scale=10, label='B (3,3)')
    plt.quiver(*origin, *v_c, color='g', scale=10, label='C (2,-0.5)')
    
    plt.title("Cosine Sim: A & B are Identical (Angle=0)\nEuclidean Dist: A & B are far apart!")
    plt.xlim(-1, 4); plt.ylim(-2, 4)
    plt.grid()
    plt.legend()
    plt.show()
```

> [!CAUTION]
> **🛑 Production Warning**
>
> If a vector is all zeros (e.g., a new user with no clicks), its Norm is 0.
> Dividing by zero creates `NaN`.
> **Fix**: Add a tiny epsilon `1e-9` to the denominator, or mask out zero vectors.

---

## 8. System-Level Integration

```mermaid
graph TD
    Query["Search: 'Cheap Shoes'"] --> Encoder[BERT Encoder]
    Encoder --> Vector[Query Vector]
    Vector --> ANN[ANN Index (Faiss)]
    ANN --> |Dot Product| Candidates[Top-100 IDs]
    Candidates --> Ranker[Re-Ranking Model]
    Ranker --> Results
```

**Where it lives**:
**Vector Databases** (Pinecone, Milvus). They are purely engines for computing Norms and Inner Products at scale.

---

## 9. Evaluation & Failure Analysis

### Metric: Precision @ K
Did the "True Match" appear in the top K dot products?

### Failure Mode: The Curse of Dimensionality (Distance Concentration)
In very high dimensions (d > 1000), *all* vectors define themselves as "Orthogonal". The ratio of "Nearest" to "Farthest" distance approaches 1.
**Result**: Nearest Neighbor meaningless.
**Fix**: Dimensionality Reduction (PCA) before search.

---

## 10. Ethics, Safety & Risk Analysis

### The Norm of Fairness
If we want a model to be fair, we might want the **Distance** between "Protected Group Prediction" and "General Prediction" to be small.
We minimize $|| f(x_{male}) - f(x_{female}) ||$.
**Risk**: Forcing this equality ("Statistical Parity") might reduce accuracy for both groups.

---

## 11. Advanced Theory & Research Depth

### Hilbert Spaces
A Vector Space + An Inner Product + Complete (no holes) = **Hilbert Space**.
This is the math setting for Quantum Mechanics and Kernel Support Vector Machines.

### 📚 Deep Dive Resources
*   **Paper**: "FaceNet: A Unified Embedding for Face Recognition and Clustering" (Schroff et al., 2015) - Using Triplet Loss (Euclidean Distance comparisons) to learn identity. [ArXiv:1503.03832](https://arxiv.org/abs/1503.03832)
*   **Concept**: **Minkowski Distance**. The generalization of distance ($L_p$ norm). $L_1$ is diamond, $L_2$ is circle. As $p \to \infty$, it becomes a square (Max Norm).


---

## 12. Career & Mastery Signals

### Cadet (Junior)
*   Can calculate `np.linalg.norm(x)`.
*   Knows that "Cosine Similarity" ignores magnitude.

### Commander (Senior)
*   Uses **faiss** or **ScaNN** for billion-scale similarity search (MIPS).
*   Understands the geometry of high-dimensional spheres (everything is on the surface).

---

## 13. Industry Interview Corner

### ❓ Real World Questions
**Q1: "When would you use L1 over L2 loss?"**
*   **Interviewer's Intent**: Checking if you know about Robustness vs Stability.
*   **Good Answer**: "L1 (MAE) is robust to outliers because the gradient is constant; a huge error doesn't explode the update. L2 (MSE) is sensitive to outliers because errors are squared, but it converges faster and smoother near zero (gradients get smaller)."

**Q2: "What is the 'Curse of Dimensionality' relative to distance?"**
*   **Answer**: "In high dimensions, the volume of space explodes. All points become equidistant. The ratio of distance to the nearest neighbor vs farthest neighbor approaches 1. This breaks Euclidean distance-based algorithms like K-Means or KNN."

**Q3: "How do you optimize dot product search for 100 million vectors?"**
*   **Answer**: "You cannot do brute force ($O(N)$). You use Approximate Nearest Neighbors (ANN) like HNSW (Graph-based) or IVFPQ (Quantization-based), which trade a little accuracy for $O(\log N)$ speed."

---

## 14. Debug Your Thinking (Common Misconceptions)

### ❌ Myth: "Correlation implies Causation." (Classic)
**✅ Truth**: Inner Product measures **Correlation** (Alignment). It says "A happens when B happens". It never says "A causes B".

### ❌ Myth: "Euclidean Distance is always the best metric."
**✅ Truth**: On a sphere (like Earth or high-dim embeddings), Euclidean distance cuts *through* the sphere. Geodesic distance (Great Circle / Cosine Sim) is correct.
*   If your vectors are normalized, L2 distance and Cosine Distance are ranking-equivalent ($L2^2 = 2(1 - \cos)$), but conceptually different.


---

## 15. Assessment & Mastery Checks

**Q1: Orthogonality**
If $\mathbf{u} \cdot \mathbf{v} = 0$, what is the angle?
*   *Answer*: 90 degrees (Perpendicular). They are uncorrelated.

**Q2: Triangle Inequality**
Does taking a detour ever save distance?
*   *Answer*: No. $||\mathbf{a} + \mathbf{b}|| \le ||\mathbf{a}|| + ||\mathbf{b}||$. The straight line is always shortest.

---

## 16. Further Reading & Tooling

*   **Paper**: *"FaceNet: A Unified Embedding for Face Recognition and Clustering"* (Triplet Loss based on distances).
*   **Tool**: **Pinecone** - A managed vector database.

---

## 17. Concept Graph Integration

*   **Previous**: [Linear Transformations](01_foundation_math_cs/01_linear_algebra/05_linear_transformations.md).
*   **Next**: [Singular Value Decomposition](01_foundation_math_cs/01_linear_algebra/07_svd.md).

### Concept Map
```mermaid
graph TD
    Metric[Metric Space] --> Norm[Norm (Length)]
    Metric --> Inner[Inner Product (Angle)]
    
    Norm --> L1[L1 Manhattan (Diamond)]
    Norm --> L2[L2 Euclidean (Circle)]
    Norm --> P[Lp Norm]
    
    Inner --> Dot[Dot Product]
    Dot --> Cos[Cosine Similarity]
    Dot --> Ortho[Orthogonality]
    
    Norm --> Loss[Loss Functions]
    Loss --> MSE[MSE (L2)]
    Loss --> MAE[MAE (L1)]
    
    style Metric fill:#f9f,stroke:#333
    style Norm fill:#bbf,stroke:#333
```
