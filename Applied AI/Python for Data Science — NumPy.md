# Chapter 4: Python for Data Science — NumPy

## Introduction

NumPy (Numerical Python) is the foundational library for scientific computing in Python. It provides:

- **Multidimensional array objects (ndarray)** for efficient storage and vectorized computation.
    
- **Mathematical functions** that operate on arrays without Python loops.
    
- **Linear algebra, Fourier transforms, and random number generation** capabilities.
    
- Seamless integration with C/C++ and Fortran code.
    

In data science, NumPy underpins libraries like pandas, SciPy, scikit-learn, and TensorFlow. Understanding NumPy is crucial for high-performance numerical work.

---

## 4.1 Getting Started

### Installation

```bash
pip install numpy
```

### Importing

```python
import numpy as np
```

The alias `np` is a community convention.

---

## 4.2 NumPy Arrays

### Creating Arrays

- From Python lists or tuples:
    

```python
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])
```

- Special arrays:
    

```python
np.zeros((2,3))        # 2x3 of zeros
np.ones((3,))          # 1D array of ones
np.arange(0,10,2)      # even numbers from 0 to 8
np.linspace(0,1,5)     # 5 points from 0 to 1
```

- Random arrays:
    

```python
np.random.rand(2,3)
np.random.randn(3,3)
```

### Array Attributes

```python
a = np.array([[1,2,3],[4,5,6]])
a.shape      # (2,3)
a.ndim       # 2
a.size       # 6
a.dtype      # dtype('int64')
```

---

## 4.3 Array Operations

### Element-wise arithmetic

```python
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b      # array([5,7,9])
a * b      # array([4,10,18])
```

### Broadcasting

Broadcasting allows operations between arrays of different shapes when compatible.

```python
A = np.ones((3,3))
b = np.array([1,2,3])
A + b  # b is broadcast across rows
```

### Universal Functions (ufuncs)

```python
np.sqrt(a)
np.exp(a)
np.log(a)
```

---

## 4.4 Indexing and Slicing

```python
a = np.arange(10)
a[2:7:2]    # elements at 2,4,6
```

- Multi-dimensional indexing:
    

```python
b = np.array([[1,2,3],[4,5,6]])
b[0,1]  # 2
b[:,1]  # column 1
b[1,:]  # row 1
```

- Boolean indexing:
    

```python
a[a > 5]    # elements greater than 5
```

---

## 4.5 Array Manipulation

- Reshape:
    

```python
a.reshape((2,5))
```

- Stack:
    

```python
np.hstack((a,b))
np.vstack((a,b))
```

- Transpose:
    

```python
b.T
```

---

## 4.6 Linear Algebra

```python
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
np.dot(A,B)          # matrix multiplication
np.linalg.inv(A)     # inverse
np.linalg.eig(A)     # eigenvalues/vectors
```

---

## 4.7 Random Module

```python
np.random.seed(42)
np.random.randint(0,10,5)
np.random.normal(loc=0, scale=1, size=(2,3))
```

---

## 4.8 Performance Tips

- Prefer vectorized operations over Python loops.
    
- Use appropriate `dtype` for memory/performance balance (e.g., `float32` vs `float64`).
    
- Leverage `np.memmap` for large arrays that don’t fit in RAM.
    

---

## 4.9 Interoperability

- **With pandas**: DataFrame values are stored as NumPy arrays.
    
- **With C/C++/Fortran**: Use `numpy.ctypeslib` or Cython for efficient interfacing.
    

---

## 4.10 Debugging & Best Practices

- Check shapes with `array.shape` before operations.
    
- Use `assert` statements to validate assumptions.
    
- Profile with `%timeit` in Jupyter to spot bottlenecks.
    

---

## Summary

NumPy provides the core data structure and computational tools for Python data science. Mastery of array creation, broadcasting, indexing, and vectorized operations is essential for building efficient machine learning and analytical workflows.