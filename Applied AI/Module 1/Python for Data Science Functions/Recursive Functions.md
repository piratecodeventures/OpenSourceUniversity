## Overview

Recursion is a programming technique where a function calls itself (directly or indirectly) to solve smaller instances of the same problem. It's a natural fit for problems that exhibit _self-similarity_ or can be decomposed into identical subproblems (trees, divide-and-conquer algorithms, backtracking). In data science, recursion appears in tree traversals, parsing nested structures, divide-and-conquer algorithms (like quicksort/merge sort), and some dynamic programming solutions.

This document covers recursion patterns, implementation tips, common algorithms, Python-specific considerations (recursion depth, lack of tail-call optimization), memoization, and debugging strategies.

---

## Key concepts

- **Base case:** The condition where the recursion stops. Every recursive function must have a correct base case to avoid infinite recursion.
    
- **Recursive case:** The part where the function calls itself with a smaller/simpler input.
    
- **Depth:** How many nested calls are on the call stack at a given point.
    
- **Overlapping subproblems:** When recursive calls compute the same results repeatedly — memoization or dynamic programming can help.
    
- **Divide and conquer:** Split a problem into independent subproblems, solve them recursively, and combine results.
    

---

## Simple examples

### Factorial (direct recursion)

```python
def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

**Time complexity:** O(n). **Space complexity (stack):** O(n).

### Fibonacci (naive recursion — exponential)

```python
def fib_naive(n: int) -> int:
    if n < 2:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
```

This naive approach does a lot of repeated work: time grows exponentially (≈φ^n). Use memoization or iterative methods for efficiency.

---

## Memoization and dynamic programming

When recursion recomputes the same subproblems, memoization (caching results) converts exponential time to polynomial or linear time.

### Top-down memoization with `functools.lru_cache`

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    if n < 2:
        return n
    return fib_memo(n-1) + fib_memo(n-2)
```

**Time complexity:** O(n). **Space complexity:** O(n) for recursion depth + cache.

### Bottom-up (iterative) DP

```python
def fib_iter(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

Bottom-up often uses less stack space and can be faster in practice.

---

## Divide and conquer examples

### Merge sort

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # merge left and right
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out
```

**Time complexity:** O(n log n). **Space complexity:** O(n) due to merging and recursion frames.

---

## Tree recursion and traversal

Recursion is a natural fit for tree structures (binary trees, general trees, nested JSON). Typical traversals (preorder, inorder, postorder) are implemented recursively.

```python
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

def dfs(node):
    if node is None:
        return
    # preorder
    print(node.value)
    for child in node.children:
        dfs(child)
```

For very deep trees, consider iterative traversals using an explicit stack to avoid recursion limits.

---

## Mutual recursion

Functions can call each other recursively.

```python
def is_even(n):
    if n == 0:
        return True
    return is_odd(n-1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n-1)
```

Mutual recursion is elegant but can obscure flow and increase call depth.

---

## Tail recursion & Python's limitation

**Tail recursion** is when the recursive call is the last operation in a function, allowing some languages (but not CPython) to optimize and reuse stack frames (tail-call optimization, TCO).

Python **does not** implement TCO in CPython. Deep tail recursion can still overflow the call stack. If you need repetitive tail-like recursion, rewrite to an iterative loop or use trampolines manually.

Example of tail-recursive style (not optimized in CPython):

```python
def tail_fact(n, acc=1):
    if n == 0:
        return acc
    return tail_fact(n-1, acc * n)
```

Rewrite iteratively to avoid depth issues.

---

## Python-specific limits and controls

- **Recursion limit:** Use `sys.getrecursionlimit()` and `sys.setrecursionlimit()` to query/change the recursion depth limit. Increasing the limit can permit deeper recursion but risks crashing the interpreter if the C stack overflows — use with caution.
    

```python
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)  # use carefully
```

- **No TCO:** CPython does not perform tail-call elimination.
    
- **Stack frames & memory:** Each call consumes memory for frame objects and locals.
    

---

## When to use recursion

- The problem has natural recursive structure (trees, nested data, fractal decomposition).
    
- Divide-and-conquer algorithms where simpler recursive code improves clarity and correctness (e.g., quicksort, mergesort).
    
- Backtracking search (permutations, combinations, sudoku solvers, constraint search).
    

When not to use recursion:

- In extremely deep recursions unless you control the environment and stack (prefer iterative alternatives).
    
- Where overlapping subproblems cause exponential blow-up and memoization is not feasible.
    

---

## Debugging recursive functions

- **Add assertions** to verify input invariants and catch incorrect calls early.
    
- **Use logging** with a depth parameter to trace calls; include a maximum depth guard.
    
- **Use `pdb` / `breakpoint()`** to step through recursion; consider printing a depth value to avoid overwhelming output.
    
- **Inspect tracebacks** in exceptions to see the chain of calls.
    

Example debug wrapper:

```python
import functools

def debug_rec(func):
    @functools.wraps(func)
    def wrapper(*args, _depth=0, **kwargs):
        print('  ' * _depth + f'Calling {func.__name__}({args}, {kwargs})')
        result = func(*args, _depth=_depth+1, **kwargs)
        print('  ' * _depth + f'{func.__name__} -> {result}')
        return result
    return wrapper
```

Note: this wrapper expects the recursive function to accept an optional `_depth` argument; it's a helpful technique for learning/debugging but not recommended in production.

---

## Performance tips

- Replace naive recursion with memoization (`functools.lru_cache`) when overlapping subproblems exist.
    
- Convert deep recursion to iterative algorithms or use explicit stacks for large depths.
    
- Use bottom-up dynamic programming when applicable to minimize stack use.
    
- Use C-accelerated libraries (NumPy, Pandas) for heavy numeric processing; recursion at Python-level is rarely optimal for large numeric loops.
    

---

## Advanced patterns

### Generators + recursion (`yield from`)

You can write recursive generators and use `yield from` to yield values from subcalls.

```python
def walk(node):
    yield node.value
    for child in node.children:
        yield from walk(child)
```

This composes naturally for streaming traversal of nested structures.

### Trampolines (simulating TCO)

A trampoline converts recursion into iteration by returning a callable for the next step.

```python
def trampoline(f, *args, **kwargs):
    result = f(*args, **kwargs)
    while callable(result):
        result = result()
    return result

# Example: tail-recursive factorial using trampolines omitted for brevity
```

Trampolines can be useful but add complexity.

---

## Examples: useful recursive algorithms in data science

- **Tree traversals**: feature extraction, tree-based models, hierarchical clustering postprocessing.
    
- **Divide & conquer sorting**: merge sort-like algorithms when custom comparators are needed.
    
- **Spatial partitioning**: recursive KD-tree building and nearest-neighbor searches.
    
- **Backtracking**: combinatorial feature generation, subset selection, structure search.
    
- **Parsing nested JSON or XML**: walk and transform nested dictionaries/lists.
    

---

## Exercises

1. **Tail recursion refactor:** Write a tail-recursive version of factorial and then rewrite it iteratively. Compare stack usage.
    
2. **Memoization vs naive:** Implement `fib_naive` and `fib_memo` and measure timings for n=30, 35, 40.
    
3. **Tree traversal:** Given a nested dictionary representing a filesystem, write a recursive generator `walk_paths` that yields file paths.
    
4. **Backtracking:** Implement a recursive solver for the N-Queens problem and measure how many solutions exist for N=8.
    
5. **Mutual recursion:** Implement even/odd mutual recursion and then replace it with a simple iterative approach.
    

---

## Mini-project idea

Build a small tool that processes nested JSON logs from a web service:

- Parse deeply-nested event objects.
    
- Extract features recursively (timestamps, user IDs, metrics).
    
- Stream results to a CSV using recursive generators to keep memory low.
    
- Add memoization for repeated substructure processing.
    

This project demonstrates recursion for parsing, generators for streaming, and memoization for efficiency.

---

## References

- Python docs: Recursion — [https://docs.python.org/3/glossary.html#term-recursion](https://docs.python.org/3/glossary.html#term-recursion)
    
- `functools.lru_cache` docs
    
- Articles: tail-call optimization, trampolines, and recursion patterns
    

_Would you like this section turned into runnable notebooks with tests and performance measurements, or expanded with more advanced algorithms (KD-tree, recursive descent parser)?_