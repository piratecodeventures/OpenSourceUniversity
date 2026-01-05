# Algorithms: Advanced Structures (The Vault)

## 📜 Story Mode: The Vault

> **Mission Date**: 2043.01.15
> **Optimization Log**: Section 9
>
> **The Problem**: Linear Search ($O(N)$) is too slow for the Star Map (10 Billion stars).
> I need to find "The nearest gas station".
> A Hydro-Spanner is just data.
>
> **The Solution**: **Spatial Partitioning**.
> I will cut space in half. Then half again.
> This is the **KD-Tree**.
>
> *"Computer! Balance the Tree. Index the Galaxy sector."*

---

## 1. Problem Setup

### The Engineering Questions
1.  **WHAT**: Advanced structures (KD-Trees, Tries, Bloom Filters).
2.  **WHY**: $O(\log N)$ or $O(1)$ isn't good enough if constant factor is high. We need structure-specific speedups.
3.  **WHEN**: Nearest Neighbors, Autocomplete, Set Membership.

---

## 2. Mathematical Formulation

### KD-Tree (k-dimensional Tree)
Cycle through axes ($x, y, z$) at each depth.
Depth 0: Split on X median.
Depth 1: Split on Y median.
**Complexity**: Build $O(N \log N)$. Search $O(\log N)$ avg.

---

## 3. The Ship's Code (Polyglot)

```python
import numpy as np

# LEVEL 0: Pure Python (KD-Tree Build)
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None
    
    k = len(points[0]) # Dimensions
    axis = depth % k
    
    # Sort and pick median
    points.sort(key=lambda x: x[axis])
    mid = len(points) // 2
    
    return Node(
        point=points[mid],
        left=build_kdtree(points[:mid], depth + 1),
        right=build_kdtree(points[mid+1:], depth + 1)
    )

# LEVEL 1: Scikit-Learn (Ball Tree / KD Tree)
"""
from sklearn.neighbors import KDTree
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
"""
```

### PageRank (Graph Deep Dive)
(See [Graphs](../05_data_structures/04_graphs_networks.md) for theory).

```python
# Iterative PageRank Implementation
def pagerank(M, num_iterations=100, d=0.85):
    N = M.shape[0]
    v = np.ones(N) / N
    M_hat = d * M + (1 - d) / N
    for i in range(num_iterations):
        v = M_hat @ v
    return v
```

---

## 13. Assessment & Mastery Checks

**Q1: Cursor Dimensionality**
Why do KD-Trees fail in high dimensions (e.g., 1000)?
*   *Answer*: Curse of Dimensionality. In high D, everything is far apart. The concept of "Nearest" breaks down. Search becomes $O(N)$.

### 14. Common Misconceptions

> [!WARNING]
> **"Trees are always log(N)."**
> *   **Correction**: Only if **Balanced**. An unbalanced tree is a Linked List ($O(N)$).
