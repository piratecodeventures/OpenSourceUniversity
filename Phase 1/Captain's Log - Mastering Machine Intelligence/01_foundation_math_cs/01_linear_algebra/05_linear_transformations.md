# Linear Algebra: Linear Transformations

## 📜 Story Mode: The Holodeck Calibration

> **Mission Date**: 2042.04.01
> **Location**: Deep Space Outpost "Vector Prime" - Recreation Deck
> **Officer**: Lead Engineer Kael
>
> **The Glitch**: The ship's hologram room (Holodeck) is acting up. I try to simulate a simple coffee cup, but it looks... stretched. It's twice as wide as it should be, and it's tilted.
>
> **The Analysis**: The mapping function $T(\mathbf{v})$ that turns the digital wireframe into projected light is broken.
> - **Input**: A point on the wireframe $\mathbf{v} = [x, y]$.
> - **Output**: A point on the wall $\mathbf{v}' = T(\mathbf{v})$.
>
> Somewhere in the code, a matrix is multiplying our coordinates wrong. It's turning squares into parallelograms.
>
> **The Fix**: We need to perform the inverse transformation to "untilt" the room. We need to understand exactly how the space is being warped.
>
> *"Computer! Visualize the transformation grid. Isolate the shear factor."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: A Linear Transformation is a function $T(\mathbf{v})$ that maps vectors to vectors while keeping grid lines parallel and the origin fixed.
2.  **WHY**: It is the fundamental operation of Neural Networks. A layer $y = Wx + b$ is just a linear transformation (plus a shift). It's how we morph data to make it separable.
3.  **WHEN**: Used in Image processing (rotation, scaling), Data Augmentation, and 3D Graphics.
4.  **WHERE**:
    *   **Math**: $\mathbf{y} = \mathbf{A}\mathbf{x}$
    *   **Code**: `cv2.warpAffine` or `torch.nn.Linear`.
5.  **WHO**: Computer Vision Engineers and Graphics Programmers.
6.  **HOW**: By defining a "Transformation Matrix" that creates the new coordinate system.

> [!NOTE]
> **🛑 Pause & Explain**
>
> **What makes it "Linear"?**
>
> It's linear if it follows two rules:
> 1.  **Additivity**: Transform(A + B) = Transform(A) + Transform(B).
> 2.  **Homogeneity**: Transform(c * A) = c * Transform(A).
>
> **In English**:
> - It doesn't curve lines (Squaring $x^2$ is NOT linear).
> - It doesn't move the zero point (Adding $+5$ is technically Affine, not Linear, though we lump them together often).

---

## 2. Mathematical Problem Formulation

### Definitions & Axioms
A transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ is linear if:
$$ T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v}) $$

This implies that $T$ can **always** be represented by a matrix multiplication.

### The Standard Basis
If we know where $T$ sends the basis vectors $\mathbf{i} (1,0)$ and $\mathbf{j} (0,1)$, we know everything.
$$ \mathbf{A} = \begin{bmatrix} T(\mathbf{i}) & T(\mathbf{j}) \end{bmatrix} $$

---

## 3. Step-by-Step Derivation

### Deriving a "Shear" Transformation
**Goal**: We want to "slant" the image to the right.
*   The vertical line $\mathbf{j} (0,1)$ stays vertical. $\to [0, 1]$.
*   The horizontal line $\mathbf{i} (1,0)$ is pushed up. $\to [1, k]$. (Wait... classic shear pushes top right).
*   Correct Shear: Top moves right.
    *   $\mathbf{i} (1,0)$ stays fixed $\to [1, 0]$.
    *   $\mathbf{j} (0,1)$ moves right $\to [k, 1]$.

**Step 1: Construct Columns**
Col 1 (New i): $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$
Col 2 (New j): $\begin{bmatrix} k \\ 1 \end{bmatrix}$

**Step 2: The Matrix**
$$ \mathbf{S} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix} $$

**Step 3: Apply to a point $(0, 1)$**
$$ \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} k \\ 1 \end{bmatrix} $$
Result: The point $(0,1)$ moved to $(k,1)$. Success.

---

## 4. Algorithm Construction

### Map to Memory
Transformations are just Matrix Multiplications.
However, in Computer Vision, we often do **Inverse Mapping**.
*   **Forward**: For each input pixel, find where it goes. (Leaves holes).
*   **Inverse**: For each output pixel, ask "Where did I come from?" and interpolate value.

---

## 5. Worked Examples

### Example 1: Rotating an Image
**Story**: Rotate a face detection box by $90^\circ$.
**Matrix**:
$$ \mathbf{R} = \begin{bmatrix} \cos(90) & -\sin(90) \\ \sin(90) & \cos(90) \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $$
**Point**: Right Eye at $[2, 1]$.
**New Point**:
$$ \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix} $$
(x became -y, y became x).

---

## 6. Production-Grade Code

### The Ship's Code (Polyglot: Pure Python + Libraries)

```python
import numpy as np
import torch
import tensorflow as tf
import math

# LEVEL 0: Pure Python (The Math Logic)
# Math: y_1 = a*x_1 + b*x_2
#       y_2 = c*x_1 + d*x_2
def transform_point_pure(point, matrix):
    """
    Applies a 2x2 Linear Transformation manually.
    point: [x, y]
    matrix: [[a, b], [c, d]]
    """
    x, y = point
    # Row 0 dot Point
    new_x = matrix[0][0] * x + matrix[0][1] * y
    # Row 1 dot Point
    new_y = matrix[1][0] * x + matrix[1][1] * y
    
    return [new_x, new_y]

# LEVEL 1: NumPy (Batch CPU)
def transform_batch_numpy(points, matrix):
    # points: (N, 2)
    # matrix: (2, 2)
    # We transpose matrix to align (N, 2) @ (2, 2)
    # Or usually we do Vectors @ Matrix.T
    return points @ matrix.T

# LEVEL 2: PyTorch (Grid Sample - The CV Way)
def rotate_image_torch(image_tensor, degrees):
    # PyTorch handles transformations differently for images (Affine Grid)
    theta = math.radians(degrees)
    # The Rotation Matrix (2x3 for Affine)
    rot_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0]
    ]).unsqueeze(0) # Batch dim
    
    # Create the coordinate grid
    grid = torch.nn.functional.affine_grid(rot_matrix, image_tensor.size(), align_corners=False)
    # Sample pixels
    return torch.nn.functional.grid_sample(image_tensor, grid, align_corners=False)

# LEVEL 3: TensorFlow (Keras Layer)
def transform_layer_tf():
    # In Keras, transformations are often layers
    # In Keras, transformations are often layers
    return tf.keras.layers.RandomRotation(0.25) # Random rotation

# LEVEL 4: Visualization (Rotating a Face)
def plot_rotation_demo():
    """
    Visualizes rotation of a set of points (a square face).
    """
    import matplotlib.pyplot as plt
    
    # Define a square "Face"
    face = np.array([
        [1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1], # Box
        [0.5, 0.5], [-0.5, 0.5], # Eyes
        [0, -0.2], [-0.3, -0.5], [0.3, -0.5] # Nose & Smile
    ]).T # Shape (2, N)
    
    # Rotation Matrix (45 degrees)
    theta = np.radians(45)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate
    rotated_face = R @ face
    
    plt.figure(figsize=(6,6))
    plt.plot(face[0], face[1], 'o-', label='Original')
    plt.plot(rotated_face[0], rotated_face[1], 'x-', label='Rotated 45')
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.title("Linear Transformation: Rotation")
    plt.show()
```

> [!CAUTION]
> **🛑 Python Logic vs Matrix Logic**
> In Level 0, we see the individual scalings ($ax + by$).
> In Level 1, we stop caring about $x$ and $y$ names and just treat them as indices 0 and 1.

---

## 7. System-Level Integration

```mermaid
graph TD
    Raw[Raw Image] --> Aug[Augmentation Pipeline]
    Aug --> Rotation[Random Rotation (Matrix)]
    Aug --> Scale[Random Scale (Matrix)]
    Rotation --> Train[Training Batch]
    Scale --> Train
```

---

## 8. Evaluation & Failure Analysis

### Failure Mode: Determinant Collapse
If your transformation matrix has $\det(\mathbf{A}) = 0$, you have flattened your data.
*   **Scenario**: You unintentionally set one column to zeros.
*   **Result**: Your 2D image becomes a 1D line. The Neural Net loses all spatial info.

---

## 9. Ethics, Safety & Risk Analysis

### The "Shear" Attack
Adversarial attacks often use tiny linear transformations (rotating by $1^\circ$ or shearing by 1%) to fool classifiers. A human eye corrects for this instantly (invariance), but simple CNNs can be fragile to it without proper Data Augmentation.

---

## 10. Advanced Theory & Research Depth

### Eigenvectors (Invariant Directions)
An **Eigenvector** is a vector that does *not* rotate during the transformation. It only stretches.
$$ \mathbf{A}\mathbf{v} = \lambda \mathbf{v} $$
These are the "Axes of Rotation" or proper axes of the object.

### 📚 Deep Dive Resources
*   **Paper**: "Spatial Transformer Networks" (Jaderberg et al., 2015) - Teaching Neural Nets to learn their own transformation matrices. [ArXiv:1506.02025](https://arxiv.org/abs/1506.02025)

---

## 11. Career & Mastery Signals

### Cadet (Junior)
*   Can rotate an image using `PIL` or `OpenCV`.

### Commander (Senior)
*   Can implement `affine_grid` and `grid_sample` from scratch to create a custom differentiable warping layer for a specific biological cell shape.

---

## 12. Industry Interview Corner

### ❓ Real World Questions
**Q1: "Explain Affine vs Linear Transformations."**
*   **Answer**: "Linear fixes the origin ($f(0)=0$). Affine includes a translation (shift). $y = Ax$ is linear; $y = Ax + b$ is affine. Neural Network layers are technically Affine."

**Q2: "How do you represent a translation (shift) as a matrix multiplication?"**
*   **Trick**: You can't in 2x2. You need **Homogeneous Coordinates** (add a 1 to the vector, making it 3D).
*   $\begin{bmatrix} 1 & 0 & dx \\ 0 & 1 & dy \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x+dx \\ y+dy \\ 1 \end{bmatrix}$

**Q3: "What is the inverse of a rotation matrix?"**
*   **Answer**: "Its transpose! Rotation matrices are **Orthogonal**. $\mathbf{R}^{-1} = \mathbf{R}^T$. This is computationally very cheap."

---

## 13. Debug Your Thinking (Common Misconceptions)

### ❌ Myth: "Transformations move the object."
**✅ Truth**: This is a matter of perspective (Active vs Passive).
*   You can think of it as moving the *object* (Active).
*   OR moving the *camera/coordinate system* (Passive).
*   They are mathematically inverse. Be sure which one your code assumes.

### ❌ Myth: "Non-linear is always better."
**✅ Truth**: Linear transformations are computationally cheap and invertible. We use them for standardizing inputs (Whitening/PCA) before we ever hit the expensive non-linear ReLU.

---

## 14. Assessment & Mastery Checks

**Q1: Determinant**
If $\det(A) = -1$, what happened?
*   *Answer*: The image was flipped (mirrored). Area is preserved ($|1|$), but orientation is reversed.

---

## 15. Further Reading & Tooling
*   **Tool**: **OpenCV (`cv2`)** - The industry standard for applying these matrices to pixels.
*   **Visualizer**: **3Blue1Brown (Essence of Linear Algebra)** - The best visual intuition for this topic.

---

## 16. Concept Graph Integration
*   **Previous**: [Matrix Operations](01_foundation_math_cs/01_linear_algebra/02_matrix_operations.md).
*   **Next**: [Norms and Inner Products](01_foundation_math_cs/01_linear_algebra/06_norms_inner_products.md) (Measuring the results).

### Concept Map
```mermaid
graph TD
    Trans[Linear Transformation] --> Matrix[Matrix A]
    Matrix --> Det[Determinant (Area Scale)]
    Matrix --> Trace[Trace]
    
    Trans --> Rotation[Rotation]
    Trans --> Scale[Scaling]
    Trans --> Shear[Shear]
    Trans --> Reflection[Reflection]
    
    Trans --> Image[Image Processing]
    Trans --> 3D[3D Graphics]
    
    style Trans fill:#f9f,stroke:#333
    style Matrix fill:#bbf,stroke:#333
```
