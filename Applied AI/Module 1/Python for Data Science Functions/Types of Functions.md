## Overview

Python supports many kinds of functions and callable objects. Understanding the differences helps you choose the right tool for a task, reason about performance and memory, and write clearer, more idiomatic code. This document catalogs the common types you will encounter in data science code, explains how each works, shows practical examples, highlights pros/cons, and provides exercises to solidify learning.

---

## 1. User-defined functions (`def`)

### What

The usual way to create named functions in Python. They are defined with the `def` keyword and can include arbitrary logic, annotations, defaults, and closures.

### Example

```python
def mean(values: list[float]) -> float:
    """Return the arithmetic mean."""
    return sum(values) / len(values)
```

### When to use

Most of the time — for readable, testable, and named functionality. Use for library code and complex logic.

### Pros / Cons

- Pros: Clear, readable, supports docstrings, annotations, closures.
    
- Cons: Call overhead for very hot small operations compared to inlining or vectorized code.
    

---

## 2. Anonymous (lambda) functions

### What

Short, single-expression anonymous functions defined with `lambda`. Good for small callbacks or inline operations.

### Example

```python
add = lambda x, y: x + y
print(map(lambda x: x**2, range(5)))
```

### When to use

Small inline operations in contexts like `sorted(key=...)`, `map`, or as short callbacks in higher-order functions.

### Pros / Cons

- Pros: Concise for small expressions; convenient inline usage.
    
- Cons: Limited to single expressions, harder to debug (no docstring, default `__name__` is `<lambda>`), can reduce readability if overused.
    

---

## 3. Generator functions (`yield`)

### What

Functions that yield values lazily, producing iterators without building full lists in memory.

### Example

```python
def batched(iterable, n=100):
    it = iter(iterable)
    while True:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        if not batch:
            break
        yield batch
```

### When to use

Streaming data processing, large datasets, pipelines that should be memory-efficient.

### Pros / Cons

- Pros: Low memory footprint, composable with other iterators.
    
- Cons: Slightly more complex control flow; debugging can be harder; once consumed, generators are exhausted.
    

---

## 4. Coroutines and async functions (`async def`)

### What

Asynchronous functions defined with `async def` that return coroutine objects. They are scheduled by an event loop (e.g., `asyncio`) and allow non-blocking I/O and concurrency.

### Example

```python
import aiohttp
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            return await resp.text()

async def main(urls):
    tasks = [asyncio.create_task(fetch(u)) for u in urls]
    return await asyncio.gather(*tasks)
```

### When to use

I/O-bound concurrency (network, disks) where many tasks wait on I/O and you want efficient multiplexing.

### Pros / Cons

- Pros: Efficient for many concurrent I/O operations; avoids blocking threads.
    
- Cons: Requires async-aware libraries; complicates code flow; not inherently parallel for CPU-bound tasks.
    

---

## 5. Methods (bound vs unbound) — functions on classes

### What

Functions defined inside a class become methods. When accessed via an instance they become **bound methods** (receive the instance as first argument). When accessed from a class they are **function objects** or descriptors.

### Types

- **Instance method** (`def method(self, ...)`) — typical method, takes instance.
    
- **Class method** (`@classmethod`) — receives class (`cls`) instead of instance.
    
- **Static method** (`@staticmethod`) — behaves like a plain function namespaced in the class.
    

### Example

```python
class Counter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

    @classmethod
    def from_value(cls, v):
        obj = cls()
        obj.count = v
        return obj

    @staticmethod
    def help_text():
        return 'Use inc to increment.'
```

### When to use

Use instance methods to operate on instance state, class methods for alternate constructors or behavior tied to class-level data, and static methods for utilities that logically belong under the class namespace.

---

## 6. Higher-order functions

### What

Functions that accept other functions as arguments or return functions. Core to functional programming and useful for building flexible APIs.

### Examples

```python
def apply_twice(f, x):
    return f(f(x))

def make_multiplier(n):
    def mul(x):
        return x * n
    return mul
```

**Stdlib examples:** `map`, `filter`, `functools.reduce`, `functools.partial`.

### When to use

When you want configurable behavior, function pipelines, or to build decorators and factories.

---

## 7. Decorators (function wrappers)

### What

Decorators are higher-order functions that wrap another function to modify or enhance its behavior (e.g., timing, caching, logging, access control).

### Example

```python
import functools, time

def timed(func):
    @functools.wraps(func)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        result = func(*a, **kw)
        print(func.__name__, 'took', time.perf_counter() - t0)
        return result
    return wrapper

@timed
def compute(n):
    return sum(range(n))
```

### When to use

Cross-cutting concerns that should be separated from business logic: logging, auth, memoization, retries.

---

## 8. Built-in & C-extension functions

### What

Functions implemented in C (or other compiled languages) and exposed to Python. These include many standard library functions (e.g., `len`, `map` in CPython), and functions from extension modules (`numpy`, `pandas`) that are heavily optimized.

### When to use

Prefer these for heavy numeric computations: vectorized operations in `numpy`/`pandas` avoid Python-level loops and are much faster.

### Pros / Cons

- Pros: Highly optimized; performance and memory advantages.
    
- Cons: Sometimes less flexible; can have surprising broadcasting rules; debugging C-level code requires special tools.
    

---

## 9. Callable objects (objects implementing `__call__`)

### What

Any class that implements `__call__` becomes callable like a function. This pattern allows stateful callables and is often used for parameterized operations.

### Example

```python
class Scaler:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, x):
        return x * self.factor

scale2 = Scaler(2)
print(scale2(3))  # 6
```

### When to use

When you want a function-like object that carries state (e.g., learned parameters, configuration) but behaves as a callable.

---

## 10. Partial functions (`functools.partial`)

### What

Create a new callable with some arguments pre-bound.

### Example

```python
from functools import partial
pow2 = partial(pow, exp=2)
print(pow2(3))  # 9
```

### When to use

When adapting APIs that require callables with fewer arguments, or when wiring functions into frameworks that expect a specific signature.

---

## 11. Dynamic / runtime-created functions

### What

Functions can be created at runtime using `types.FunctionType`, `exec`, or by returning closures. This supports metaprogramming but should be used cautiously.

### Example

```python
def make_adder(n):
    code = f"def add(x):\n    return x + {n}\n"
    ns = {}
    exec(code, ns)
    return ns['add']
```

### When to use

Code generation, DSLs, or when you must produce specialized functions dynamically. Avoid unless necessary.

---

## Internals & performance considerations (summary)

- **Call overhead:** Every call creates a frame and performs argument binding; for very small functions in tight loops, prefer local inlined code or vectorized approaches.
    
- **Generator overhead:** Generators yield lazily but incur the cost of creating generator objects and context switching between caller and generator.
    
- **Async coroutines:** Lightweight compared to threads for many concurrent I/O tasks, but add complexity.
    
- **C-extensions:** Moving computation to C (NumPy, Pandas) drastically reduces Python-level overhead — prefer this for numeric-heavy work.
    
- **Closures & `__closure__`:** Closures allocate cell objects; keep closure sizes small to avoid extra memory.
    

---

## Best practices across function types

- Prefer plain `def` functions with clear docstrings for most APIs.
    
- Use `lambda` sparingly — only for short, throwaway callbacks.
    
- Use generators to keep memory usage low for streaming data, but document that the function returns an iterator.
    
- Use `async` only when you need concurrency for I/O; prefer synchronous code for simple scripts.
    
- Use decorators for orthogonal concerns and keep them transparent with `functools.wraps`.
    
- Prefer built-in and vectorized functions (`numpy`, `pandas`) for heavy numeric operations.
    
- Keep callable classes minimal — prefer stateless functions unless state is required.
    

---

## Common pitfalls

- Mutable defaults in any function type that supports defaults.
    
- Expecting generator functions to be reusable (they're single-use).
    
- Mixing sync and async code incorrectly (calling `await`-ables without an event loop).
    
- Overusing decorators that change call signatures unexpectedly.
    

---

## Small exercises

1. **Lambda vs def:** Replace a small `def` with a `lambda` in a `sorted(key=...)` call; explain why `lambda` is appropriate or not.
    
2. **Generator practice:** Implement `stream_csv(path)` that yields rows as dictionaries lazily.
    
3. **Async practice:** Write a small `async` function to fetch three URLs concurrently using `aiohttp` (or a mock) and time the difference vs serial `requests` calls.
    
4. **Decorator building:** Build a `@retry` decorator that retries a function on failure with exponential backoff.
    
5. **Callable class:** Implement a `BatchNormalizer` callable that learns mean/std in `fit()` and is callable to normalize arrays.
    

---

## Mini-project idea

Build a small data ingestion pipeline that demonstrates multiple function types:

```
ingestor/
├─ ingest.py          # generator functions that stream raw files
├─ transform.py       # pure functions + higher-order funcs for transformations
├─ async_fetch.py     # async functions to fetch remote data
├─ model.py           # callable model objects or functions
└─ run.py             # orchestrates pipeline using partials and decorators
```

This project shows when to use generators for streaming, async for remote IO, and callables for stateful models.

---

## References & further reading

- Python docs: Functions — [https://docs.python.org/3/tutorial/controlflow.html#defining-functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
    
- PEP 8 — Function and variable naming conventions
    
- `asyncio` documentation and guides
    
- `functools` module docs (partial, wraps, lru_cache)
    

_Would you like this section converted into a runnable notebook, expanded with more examples of each type, or split into slides for teaching?_