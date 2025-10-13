This cheatsheet provides a concise reference for **NumPy**, the core library for numerical computing in Python. It covers array creation, operations, indexing, mathematical functions, linear algebra, random numbers, file I/O, performance tips, debugging, and integration with machine learning (ML) and visualization. Ideal for data scientists, it assumes basic Python knowledge.

---

## 1. Installation and Setup

```bash
pip install numpy  # or conda install numpy
```

```python
import numpy as np
print(np.__version__)  # Check version
print(np.show_config())  # Check BLAS setup
```

---

## 2. Core Concepts

- **ndarray**: Multi-dimensional array with shape, dtype, strides.
    
- **Key Attributes**:
    
    ```python
    arr = np.array([[1, 2], [3, 4]])
    arr.shape    # (2, 2)
    arr.ndim     # 2
    arr.size     # 4
    arr.dtype    # int64
    arr.strides  # (16, 8) for int64
    ```
    
- **Data Types**:
    
    - Numeric: `np.int8`, `np.int32`, `np.float32`, `np.float64`
    - Other: `np.bool_`, `np.complex64`, `np.string_`
    
    ```python
    arr = np.array([1.5, 2.5], dtype=np.float32)
    ```
    

---

## 3. Array Creation

- From lists: `np.array([[1, 2], [3, 4]])`
- Zeros/Ones: `np.zeros((2, 3))`, `np.ones((2, 3))`
- Range: `np.arange(0, 10, 2)` → `[0, 2, 4, 6, 8]`
- Linear space: `np.linspace(0, 1, 5)` → `[0. , 0.25, 0.5 , 0.75, 1. ]`
- Identity: `np.eye(3)` (3x3 identity matrix)
- Empty/Full: `np.empty((2, 2))`, `np.full((2, 2), 7)`
- Meshgrid: `x, y = np.meshgrid(np.arange(3), np.arange(2))`

---

## 4. Array Manipulation

- Reshape: `arr.reshape(4)` → `[1, 2, 3, 4]`
- Flatten: `arr.ravel()` (view) or `arr.flatten()` (copy)
- Concatenate: `np.concatenate([arr1, arr2], axis=0)`
- Stack: `np.vstack([arr1, arr2])`, `np.hstack([arr1, arr2])`
- Split: `np.split(arr, 2)` → two sub-arrays
- Transpose: `arr.T` or `np.transpose(arr)`
- Swap axes: `np.swapaxes(arr, 0, 1)`

---

## 5. Array Operations

- **Element-Wise**:
    
    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    a + b  # [5, 7, 9]
    a * 2  # [2, 4, 6]
    np.sin(a)  # [0.841, 0.909, 0.141]
    ```
    
- **Broadcasting**:
    
    ```python
    a = np.array([[1, 2], [3, 4]])  # (2, 2)
    b = np.array([10, 20])  # (2,) → (2, 2)
    a + b  # [[11, 22], [13, 24]]
    ```
    
- **Reduction**:
    
    ```python
    np.sum(a, axis=0)  # Column sums
    np.prod(a)  # Product of all elements
    ```
    

---

## 6. Indexing and Slicing

- Basic: `arr[0, 1]` → `2`
- Slicing (views): `arr[:, 1]` → `[2, 5]`
- Fancy Indexing:
    
    ```python
    arr[[0, 1], [1, 0]]  # [2, 3]
    arr[arr > 2]  # [3, 4]
    ```
    
- Boolean: `arr[arr > 2] = 0` → mask assignment

---

## 7. Mathematical and Statistical Functions

- **Math**:
    
    ```python
    np.sqrt(arr)  # Element-wise square root
    np.exp(arr)   # e^x
    np.log1p(arr) # log(1+x) for small x
    np.clip(arr, 0, 3)  # Limit values
    ```
    
- **Stats**:
    
    ```python
    np.mean(arr, axis=0)  # Column means
    np.std(arr)  # Standard deviation
    np.median(arr)  # Median
    np.percentile(arr, 75)  # 75th percentile
    np.corrcoef(arr1, arr2)  # Correlation
    ```
    

---

## 8. Linear Algebra (np.linalg)

- Dot product: `np.dot(a, b)` or `a @ b`
- Matrix inverse: `np.linalg.inv(A)`
- Eigenvalues: `np.linalg.eig(A)` → `(eigvals, eigvecs)`
- SVD: `U, S, Vt = np.linalg.svd(A)`
- Solve Ax = b: `x = np.linalg.solve(A, b)`
- Norm: `np.linalg.norm(A)`

**Example: PCA**:

```python
cov = np.cov(data.T)
eigvals, eigvecs = np.linalg.eig(cov)
```

---

## 9. Random Number Generation

- Modern: `rng = np.random.default_rng(seed=42)`
    
    ```python
    rng.random(3)  # [0.374, 0.950, 0.731]
    rng.normal(0, 1, 1000)  # Normal distribution
    rng.choice([1, 2, 3], size=2)  # Random sample
    ```
    

---

## 10. File Input/Output

- Binary: `np.save('file.npy', arr)`, `np.load('file.npy')`
- Compressed: `np.savez('file.npz', arr1=arr1, arr2=arr2)`
- Text: `np.savetxt('file.csv', arr, delimiter=',')`, `np.loadtxt('file.csv', delimiter=',')`
- Memory Map: `mmap = np.memmap('file.dat', dtype=np.float32, mode='r')`

---

## 11. Advanced Features

- **Masked Arrays**:
    
    ```python
    import numpy.ma as ma
    arr = ma.array([1, 2, 3], mask=[0, 1, 0])
    ma.mean(arr)  # 2.0
    ```
    
- **Structured Arrays**:
    
    ```python
    dt = np.dtype([('name', 'S10'), ('age', 'i4')])
    arr = np.array([('Alice', 25)], dtype=dt)
    arr['name']  # [b'Alice']
    ```
    
- **FFT**:
    
    ```python
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 10 * t)
    fft = np.fft.fft(signal)
    ```
    
- **Polynomials**: `np.polyfit(x, y, deg=2)` (fit quadratic)
- **Sorting**: `np.sort(arr)`, `np.argsort(arr)`

---

## 12. Performance and Parallel Processing

- **Vectorization**: Replace loops with `np.where()`, `np.einsum()`.
- **Contiguous Arrays**: `np.ascontiguousarray(arr)`
- **Multi-Threading**: Set `OMP_NUM_THREADS=4` for BLAS.
- **Numba**:
    
    ```python
    from numba import jit
    @jit(nopython=True)
    def fast_sum(arr):
        return np.sum(arr)
    ```
    
- **Dask**: For out-of-core arrays:
    
    ```python
    import dask.array as da
    x = da.from_array(np.random.rand(1000000), chunks='auto')
    ```
    

---

## 13. Debugging and Tracebacks

- **Common Errors**:
    - `ValueError`: Shape mismatch.
    - `IndexError`: Out-of-bounds.
    - `TypeError`: Invalid dtype.
- **Handling**:
    
    ```python
    import traceback
    try:
        arr = np.array([1, 2]) + np.array([3])
    except ValueError:
        print(traceback.format_exc())
    ```
    
- **Set Error Behavior**: `np.seterr(all='raise')`
- **Testing**: `np.testing.assert_array_equal(arr1, arr2)`

---

## 14. Machine Learning and Deep Learning

- **Data Prep**:
    
    ```python
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # Normalize
    ```
    
- **One-Hot Encoding**:
    
    ```python
    labels = np.eye(3)[np.array([0, 1, 2])]  # [[1,0,0], [0,1,0], [0,0,1]]
    ```
    
- **To Tensor**:
    
    ```python
    import torch
    tensor = torch.from_numpy(np_array.copy())
    ```
    
- **K-Means**:
    
    ```python
    def kmeans(X, k):
        centroids = X[np.random.choice(len(X), k)]
        for _ in range(100):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            centroids = np.array([X[labels == i].mean(0) for i in range(k)])
        return labels
    ```
    

---

## 15. Visualization

- **Matplotlib**:
    
    ```python
    import matplotlib.pyplot as plt
    plt.plot(np.arange(10), np.sin(np.arange(10)))
    plt.imshow(np.random.rand(10, 10), cmap='viridis')
    plt.show()
    ```
    
- **Seaborn** (NumPy-based):
    
    ```python
    import seaborn as sns
    sns.heatmap(np.corrcoef(data.T))
    ```
    

---

## 16. Sample Use Cases

- **Finance**: Volatility calculation:
    
    ```python
    prices = np.array([100, 102, 98, 105])
    returns = np.diff(np.log(prices))
    vol = np.std(returns) * np.sqrt(252)
    ```
    
- **Image Processing**: Gaussian blur:
    
    ```python
    from scipy.ndimage import gaussian_filter
    img = np.random.rand(100, 100)
    blurred = gaussian_filter(img, sigma=2)
    ```
    
- **ML**: Logistic regression gradient:
    
    ```python
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)
    w = np.zeros(2)
    for _ in range(100):
        pred = sigmoid(X @ w)
        grad = X.T @ (pred - y) / len(y)
        w -= 0.1 * grad
    ```
    

---

## 17. Best Practices

- Use vectorized ops over loops.
- Check shapes: `arr.shape` before operations.
- Copy when needed: `arr.copy()`.
- Optimize dtypes: `np.float32` for ML.
- Handle NaNs: `np.isnan()`, `np.nanmean()`.

**Pitfalls**:

- Views vs copies: `arr[:] = 0` modifies original.
- Broadcasting errors: Use `np.newaxis`.
- Memory: Use `del arr` for large arrays.

---

## 18. Quick Reference

- **Array Info**: `arr.shape`, `arr.dtype`, `arr.flags`
- **Fast Ops**: `np.einsum('ij,jk->ik', A, B)` for matrix multiply
- **Debugging**: `np.seterr(all='warn')`
- **File I/O**: `np.savez_compressed('data.npz', arr=arr)`
- **Random**: `rng = np.random.default_rng(42)`

This cheatsheet covers NumPy’s essentials for data science, with practical snippets and tips for efficient coding.