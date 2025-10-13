
## Overview

Functions are the primary building blocks of structured and modular code in Python. A **function** is a named block of reusable code that performs a specific task. In data science, functions let you encapsulate data loading, cleaning, feature engineering, model training, evaluation, and plotting into concise, testable units. This makes pipelines repeatable, easier to debug, and simpler to share.

---

## What is a function?

- **Definition:** A function is a callable object that groups statements and returns a value (or `None`).
    
- **Purpose:** Encapsulation, reuse, abstraction, separation of concerns, and easier testing.
    

**Example (simple):**

```python
def add(a, b):
    """Return the sum of a and b."""
    return a + b

print(add(2, 3))  # 5
```

---

## How to implement functions (practical syntax)

### Basic function

```python
def greet(name: str) -> str:
    """Return a greeting for name."""
    return f"Hello, {name}!"
```

### Arguments and return

- Positional and keyword arguments
    
- Default arguments
    
- `*args` and `**kwargs`
    
- Type hints (optional, highly recommended for readability & tooling)
    

```python
def summarize(values, /, *, method='mean'):
    # slash `/` indicates positional-only arguments (Python 3.8+)
    # star `*` prior indicates keyword-only arguments
    if method == 'mean':
        return sum(values) / len(values)
    elif method == 'sum':
        return sum(values)
    raise ValueError("Unknown method")
```

### Lambda (anonymous) function

```python
square = lambda x: x * x
print(square(5))  # 25
```

### Generator function (lazy sequences)

```python
def window(seq, n=2):
    it = iter(seq)
    from collections import deque
    buf = deque([], maxlen=n)
    for _ in range(n):
        buf.append(next(it))
    yield tuple(buf)
    for item in it:
        buf.append(item)
        yield tuple(buf)
```

---

## How functions work internally (interpreter-level view)

> This section describes typical CPython behavior (most common Python implementation). Details will vary across interpreters (PyPy, Jython, etc.) but concepts are stable.

### Compilation and function objects

1. **Source → AST → Bytecode:** When Python loads a module, the source is parsed into an AST and compiled into bytecode (`.pyc` file may be written). A `code` object (`PyCodeObject`) is created representing the compiled body.
    
2. **Function object creation:** When the `def` statement executes, it wraps that `code` object into a `function` object (`PyFunctionObject`) with references to:
    
    - the bytecode (`__code__`)
        
    - default argument values (`__defaults__` / `__kwdefaults__`)
        
    - the global namespace where it was defined (`__globals__`)
        
    - closure cells (`__closure__`) if it closes over variables
        
3. **Calling a function:** When called, the interpreter:
    
    - Creates a **frame** (a `PyFrameObject`) which holds local variables, the instruction pointer, evaluation stack, and reference to the function's code object.
        
    - Pushes the frame on the call stack and executes bytecode in the frame through the **evaluation loop** (the bytecode interpreter). The loop consumes operations like `LOAD_FAST`, `CALL_FUNCTION`, `RETURN_VALUE`.
        
    - When the function returns, the frame is popped and the returned value is passed back to the caller.
        

### Important implementation details

- **Local vs global lookups:** Local variables are stored in fast arrays indexed by position in `co_varnames` for quick access (fast locals). Globals use dictionary lookups (`__globals__`). Builtins are checked last.
    
- **Closures:** If an inner function refers to variables from the outer scope, Python creates _cell_ objects to hold references. The function object has `__closure__` linking to these cells.
    
- **Default arguments:** Default values are evaluated at **function definition time**, not at call time. This explains the famous mutable-default pitfall.
    
- **Bytecode & optimization:** The interpreter executes bytecode instructions — small, stack-based steps. There is no automatic tail-call optimization in CPython.
    
- **Reference counting & GC:** CPython uses reference counting plus cyclic GC. Each frame and function object increases reference counts while alive.
    

---

## Low-level design (brief)

- **Key C structs in CPython:** `PyFunctionObject`, `PyCodeObject`, `PyFrameObject` — these underpin how functions are represented in memory.
    
- **Eval loop:** `ceval.c` (the evaluation loop) implements the virtual machine that executes bytecode.
    
- **Performance considerations:** Function call overhead is not free — creating frames and argument binding have costs. For hot inner loops, consider inlining small operations or using local loops.
    

---

## When to use functions (guidelines)

Use functions to:

- Encapsulate a single responsibility (do one thing).
    
- Make code reusable in multiple places or projects.
    
- Break complex algorithms into understandable steps.
    
- Isolate side effects (I/O, randomness) from pure computation for easier testing.
    
- Provide clear interfaces in modules and packages.
    

Avoid making extremely tiny functions that harm readability by scattering logic; balance granularity.

---

## Best practices

- Keep functions small and focused (single-responsibility).
    
- Use descriptive names and include docstrings (explain inputs, outputs, exceptions, side effects).
    
- Prefer **pure functions** for logic (deterministic outputs for given inputs) — they are easier to test and cache (`functools.lru_cache`).
    
- Avoid mutable default arguments. Use `None` and create the object inside the function.
    

```python
# Bad
def append_item(x, lst=[]):
    lst.append(x)
    return lst

# Good
def append_item(x, lst=None):
    if lst is None:
        lst = []
    lst.append(x)
    return lst
```

- Use type hints and small docstrings for clarity and to help linters and IDEs.
    
- Use `functools.wraps` when writing decorators to preserve metadata.
    
- Prefer exceptions over sentinel return values for error handling in libraries.
    
- Keep I/O, plotting, and heavy side effects at the top-level (or in separate functions) so core logic can be tested.
    

---

## Pros and cons of using functions

**Pros**

- Reuse and DRY (Don't Repeat Yourself).
    
- Easier testing and debugging.
    
- Better readability and organization.
    
- Facilitates collaboration (clear APIs).
    

**Cons / caveats**

- Call overhead for very small functions in tight loops.
    
- Too many tiny functions can make flow harder to follow.
    
- Misuse of mutable default arguments and globals can introduce subtle bugs.
    

---

## Different types of functions

- **User-defined functions** (`def`) — the normal case.
    
- **Anonymous (lambda) functions** — single-expression functions.
    
- **Generator functions** (`yield`) — produce lazy sequences.
    
- **Coroutines** (`async def`) — asynchronous functions.
    
- **Bound / unbound methods** — functions attached to class instances (`def` in class body).
    
- **Static and class methods** (`@staticmethod`, `@classmethod`).
    
- **Higher-order functions** — accept or return other functions (e.g., `map`, `filter`, custom factory functions).
    
- **Decorator functions** — wrap other functions to modify behavior.
    
- **Built-in functions** — implemented in C and available in the interpreter.
    

---

## Different ways to create/call functions

- `def` statement
    
- `lambda` expression
    
- Callable objects (class with `__call__`)
    
- `functools.partial` to pre-bind arguments
    
- Dynamic creation: `types.FunctionType` or `exec` (use sparingly and with care)
    

**Example: callable class**

```python
class Multiplier:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        return x * self.n

double = Multiplier(2)
print(double(5))  # 10
```

---

## Small data-science sample project (modular functions)

### Goal

A tiny pipeline: load CSV, simple cleaning, feature engineering, train a linear regression, evaluate.

### Project structure

```
ds_project/
├─ data/
│  └─ sample.csv
├─ ds_project/
│  ├─ __init__.py
│  ├─ io.py            # loading/saving
│  ├─ preprocess.py    # cleaning, imputing
│  ├─ features.py      # feature engineering
│  ├─ model.py         # model training & evaluation
│  └─ utils.py         # small utilities
└─ run_pipeline.py
```

### Example `io.py`

```python
import csv
from typing import List, Dict


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV file and yield rows as dicts (header-aware)."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        return list(reader)
```

### Example `preprocess.py`

```python
from typing import List, Dict


def drop_na_rows(rows: List[Dict[str, str]], required_cols):
    return [r for r in rows if all(r.get(c) not in (None, '', 'NA') for c in required_cols)]


def to_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default
```

### Example `features.py`

```python
from typing import List, Dict


def add_ratio_feature(rows, a, b, new_name='ratio'):
    for r in rows:
        av = r.get(a)
        bv = r.get(b)
        try:
            r[new_name] = float(av) / float(bv)
        except Exception:
            r[new_name] = None
    return rows
```

### Example `model.py` (sketch)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


def train_linear(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    return {'mse': mse}
```

`run_pipeline.py` would orchestrate these modules into a readable script that imports modular functions.

---

## Small exercises (practical, per subtopic)

1. **Basic function exercise** — write `median(values)` without using `statistics` module. Add docstring and type hints.
    
2. **Mutable default pitfall** — write a function that uses a list as default; then fix it properly. Explain behavior across two calls.
    
3. **Closures** — implement `make_scaler(factor)` that returns a function multiplying by `factor`. Show closure `__closure__` contents.
    
4. **Generator** — implement `batched(iterable, n)` yielding n-sized tuples for streaming data.
    
5. **Decorator** — write a `@timed` decorator that prints how long a function took to run.
    
6. **Refactor mini-project** — take a one-file messy script that loads data, cleans it, and trains a model; split it into `io.py`, `preprocess.py`, `model.py`, and a `run_pipeline.py` that ties them together.
    

---

## Quick debugging tips for functions

- Add assertions and small unit tests for pure functions.
    
- Use `pdb` (or `breakpoint()`) to step into a function and inspect locals.
    
- Use logging instead of `print()` for production code.
    
- For performance issues, use `cProfile` and `line_profiler` (where available) to find expensive function calls.
    

### Expanded debugging techniques

#### 1. Interactive debugging

- **`breakpoint()` / `pdb`**: Use `breakpoint()` (Python 3.7+) which drops you into the `pdb` console. Example:
    

```python
def compute(x):
    breakpoint()
    return x * 2

compute(5)
```

Inside the prompt you can inspect variables, step (`n`), step into (`s`), continue (`c`), and evaluate expressions.

- **IDE debuggers**: PyCharm, VS Code, and other IDEs provide graphical debuggers with conditional breakpoints, watch expressions, call-stack navigation, and variable viewers. Use them for complex flows or when debugging GUIs or long-running jobs.
    
- **Remote debugging**: Tools like `debugpy` let you attach an IDE to a remote process (useful for code running on servers, containers or cloud instances). Remember to secure the debugging port (avoid exposing it publicly).
    

#### 2. Logging & tracing

- **Structured logging**: Use the `logging` module with properly named loggers (`logging.getLogger(__name__)`) and different log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Configure handlers and formatters for file or console output.
    

```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug('Starting computation with %s', x)
```

- **Avoid excessive prints** in libraries — prefer logging so the user of your module can control verbosity.
    
- **Tracing**: The `traceback` module can format stack traces for logging. Use `traceback.format_exc()` inside an `except` block to log the full stack trace.
    

#### 3. Profiling & performance

- **`timeit`**: For microbenchmarks of small snippets.
    
- **`cProfile` / `pstats`**: Good for function-level profiling of complete scripts. Save results and inspect the heaviest callers.
    
- **`line_profiler`**: Profiles per-line execution time; excellent for bottlenecks in hot functions (requires `@profile` decorator and installation).
    
- **Memory profiling**: `memory_profiler` and `tracemalloc` help find memory leaks or heavy allocations.
    
- **Sampling profilers**: `py-spy`, `scalene` are non-invasive and work on running Python processes (great for production diagnostics).
    

**Typical workflow**: reproduce the slow case → run `cProfile` to find the costly function → use `line_profiler` on that function to find slow lines → optimize and re-profile.

#### 4. Automated testing & TDD

- **`pytest`**: Write unit tests for pure functions and small integration tests for IO-bound functions. Use fixtures to create reproducible inputs.
    
- **Test-driven development**: Write failing tests first to pin down the exact behavior you expect, then implement minimal code to pass tests.
    
- **Property-based testing**: Tools like `hypothesis` generate random inputs to find edge cases automatically.
    
- **Continuous Integration (CI)**: Run tests in CI pipelines (GitHub Actions, GitLab CI, etc.) to catch regressions early.
    

#### 5. Static analysis and type checking

- **`mypy`**: Static type checker for catching type errors early.
    
- **Linters**: `flake8`, `pylint`, or `ruff` to enforce style and catch common bugs.
    
- **IDE hints**: Type hints aid IDE auto-completion and improve readability.
    

#### 6. Debugging concurrency & async code

- **`asyncio` debugging**: Set `PYTHONASYNCIODEBUG=1` and use `asyncio.get_running_loop().set_debug(True)` to get more helpful warnings.
    
- **Threading issues**: Race conditions can be subtle — use locks, or prefer multiprocessing for CPU-bound tasks to avoid subtle shared-state bugs. Tools like `faulthandler` can dump stacks for deadlocks.
    

#### 7. Helpful stdlib modules

- `inspect` — introspect functions, get source, signature, and closure cells.
    
- `traceback` — format and retrieve tracebacks.
    
- `faulthandler` — dump Python tracebacks on crashes or signals.
    
- `warnings` — emit and filter warnings for deprecations and misuse.
    

### Practical examples

**Profiling example (cProfile + pstats)**

```python
import cProfile
import pstats

def expensive():
    for _ in range(1000000):
        pass

cProfile.run('expensive()', 'prof.out')
ps = pstats.Stats('prof.out')
ps.strip_dirs().sort_stats('cumtime').print_stats(10)
```

**Logging with traceback in exception**

```python
import logging, traceback
logger = logging.getLogger(__name__)

try:
    risky()
except Exception:
    logger.error('risky failed: %s', traceback.format_exc())
```

**Using `inspect` to debug a closure**

```python
import inspect

def make_scaler(f):
    def scale(x):
        return x * f
    return scale

s = make_scaler(3)
print(inspect.getsource(s))
print(s.__closure__)
```

### Debugging exercises

1. **Find the bug:** Given a function that computes a rolling average, use `pytest` and `breakpoint()` to reproduce and fix the off-by-one error.
    
2. **Profile & optimize:** Profile a small script that computes pairwise distances on a large list. Identify the hot function and optimize it (e.g., vectorize with NumPy).
    
3. **Memory leak detective:** Simulate accumulating large objects across function calls and use `tracemalloc` to find which function is holding references.
    

---

## Summary & next steps

Functions are the core mechanism to write modular, testable Python. You now know how to write functions, the interpreter-level story of how they work, practical best-practices, and how to apply them in a small data-science pipeline. The next sections should cover function **argument styles**, **recursion**, **lambda** usage, and how to split code across **modules** and **packages** with concrete examples and exercises.

Functions are the core mechanism to write modular, testable Python. You now know how to write functions, the interpreter-level story of how they work, practical best-practices, and how to apply them in a small data-science pipeline. The next sections should cover function **argument styles**, **recursion**, **lambda** usage, and how to split code across **modules** and **packages** with concrete examples and exercises.

---

### Appendix: Cheat-sheet (common attributes & introspection)

```python
def example(a=1, *, kw=2) -> int:
    return a + kw

print(example.__name__)
print(example.__doc__)
print(example.__defaults__)
print(example.__kwdefaults__)
print(example.__annotations__)
print(example.__code__.co_varnames)
```

---

_If you'd like, I can also expand this into:_

- A slide deck (outline + speaker notes) for teaching this section.
    
- A set of Jupyter notebook cells that users can run and experiment with.
    
- Full sample project files populated with runnable code and small synthetic dataset.