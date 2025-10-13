## Overview

Function arguments are how data flows into functions. Mastering argument syntax and semantics lets you design clear, flexible APIs, write robust wrappers, and avoid common pitfalls. This document covers: calling conventions, argument binding rules, positional-only and keyword-only parameters, default values, variable-length arguments (`*args`, `**kwargs`), argument unpacking, annotations and type hints, forwarding arguments, and practical patterns used in data science code.

---

## Argument kinds & their order

Python functions can accept several kinds of parameters. The order in a `def` signature is important:

1. **Positional-or-keyword parameters** (e.g., `a`, `b`) — can be passed by position or by keyword.
    
2. **Positional-only parameters** (`/` marker, Python 3.8+) — can only be passed by position.
    
3. **Var-positional (`*args`)** — captures extra positional args as a tuple.
    
4. **Keyword-only parameters** (after a `*` or `*args`) — must be passed by keyword.
    
5. **Var-keyword (`**kwargs`)** — captures extra keyword args as a dict.
    

A canonical signature combining these looks like:

```python
def fn(pos1, pos2, /, pos_or_kw, *args, kw_only1, kw_only2=None, **kwargs):
    pass
```

**Notes:**

- The slash `/` indicates the end of positional-only parameters.
    
- A bare asterisk `*` can be used to mark the start of keyword-only parameters when you don't need `*args`.
    

---

## Positional-only parameters (`/`)

### What

Parameters before `/` must be supplied positionally. They cannot be passed by keyword.

### Why

- Useful for APIs where parameter names are implementation details, or to match C-API semantics.
    
- The standard library uses them in places (e.g., `divmod(x, y)`-style functions).
    

### Example

```python
def concat(a, b, /, sep=' '):
    return f"{a}{sep}{b}"

concat('hello', 'world')            # OK
concat('hello', 'world', sep='-')   # OK
concat(a='hello', b='world')        # TypeError (positional-only)
```

---

## Keyword-only parameters (`*` or after `*args`)

### What

Parameters after `*` or after `*args` must be passed by keyword.

### Why

- Make calls self-documenting; avoid accidental positional mistakes.
    
- Common in functions with many optional flags.
    

### Example

```python
def plot(xs, ys, *, title=None, xlabel='x', ylabel='y'):
    ...

plot(xs, ys, title='My plot')      # OK
plot(xs, ys, 'My plot')            # Wrong — title must be keyword
```

---

## Defaults & evaluation time

### Rules

- Default argument expressions are evaluated **once** at function definition time (not on each call).
    
- Use immutable defaults or `None` sentinel for mutable objects.
    

### Mutable default pitfall

```python
def append_item(x, lst=[]):
    lst.append(x)
    return lst

print(append_item(1))  # [1]
print(append_item(2))  # [1, 2]  <- surprising! shared list

# Fix:
def append_item(x, lst=None):
    if lst is None:
        lst = []
    lst.append(x)
    return lst
```

### Best practice

Use `None` and create fresh objects within the body, or use immutable defaults.

---

## `*args` and `**kwargs`

### What

- `*args` captures extra positional arguments as a tuple.
    
- `**kwargs` captures extra keyword arguments as a dict.
    

### Use cases

- Wrapping functions and forwarding extra parameters to nested calls.
    
- Flexible APIs where callers may pass optional named parameters.
    

### Example: forwarding

```python
def log_and_call(func, *args, **kwargs):
    print('Calling', func.__name__)
    return func(*args, **kwargs)
```

### Avoiding shadowing

Don't name parameters `args`/`kwargs` if you also accept `*args`/`**kwargs` — prefer `*extras` and `**extras_kw` only if necessary for clarity.

---

## Argument unpacking in calls

You can unpack sequences and mappings into call arguments using `*` and `**`:

```python
vals = [1, 2]
print(add(*vals))           # same as add(1, 2)
params = {'sep': '-'}
print(concat('a', 'b', **params))
```

This is extremely handy when building dynamic calls or wiring hyperparameters into training functions.

---

## Keyword-only + required parameters

Use keyword-only parameters to force callers to pass critical flags explicitly. Mark required keyword-only parameters by omitting defaults.

```python
def train(X, y, *, model, epochs, lr=1e-3):
    ...

# Caller must say model=..., epochs=... which improves clarity
```

---

## Annotations & type hints for arguments

### Syntax

```python
def load_csv(path: str, header: bool = True) -> list[dict]:
    ...
```

### Using `typing`

For richer types (pre-3.9 or complex generics):

```python
from typing import Iterable, Sequence

def compute_stats(values: Iterable[float]) -> dict[str, float]:
    ...
```

### Benefits

- Documentation & readability
    
- IDE autocompletion
    
- `mypy` static checking
    

---

## Inspecting signatures and binding

Use `inspect.signature()` and `inspect.BoundArguments` to introspect or bind arguments programmatically.

```python
import inspect

def f(a, b=2, *args, c=3, **kwargs):
    pass

sig = inspect.signature(f)
print(sig)
bound = sig.bind(1, 4, 5, c=7)
print(bound.arguments)
```

This is powerful for building wrappers, decorators, or dynamic dispatchers.

---

## Forwarding arguments cleanly

When wrapping functions, forward `*args`/`**kwargs` and use `functools.wraps` to preserve metadata.

```python
import functools

def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('calling', func.__name__)
        return func(*args, **kwargs)
    return wrapper
```

For Python 3.8+, you can also use `inspect.signature` to create wrappers that modify or validate arguments before calling the inner function.

---

## Argument patterns in data science

- **Config objects vs kwargs:** For many hyperparameters, group them into a config dataclass or dictionary instead of long argument lists.
    

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32

def train(X, y, cfg: TrainConfig):
    ...
```

- **Using `**kwargs` with sklearn-like APIs:** Pass arbitrary parameters to underlying estimators.
    

```python
def build_model(hidden=64, *, activation='relu', **kwargs):
    # pass kwargs to optimizer, regularizer, etc.
    pass
```

- **Unpacking hyperparameter grids** when performing grid search:
    

```python
params = {'hidden': 64, 'lr': 1e-3}
model = build_model(**params)
```

---

## Common pitfalls & gotchas

- Mutable default arguments (already covered).
    
- Shadowing names with `*args`/`**kwargs`.
    
- Incorrect ordering of parameters — Python enforces the positional/keyword rules.
    
- Passing unexpected keys in `**kwargs` can silently hide typos unless validated.
    
- Over-accepting `**kwargs` in public APIs makes static analysis and IDE completion harder.
    

**Defensive programming:** validate unexpected kwargs or pop them and raise helpful errors.

```python
def fn(a, **kwargs):
    allowed = {'x', 'y'}
    extra = set(kwargs) - allowed
    if extra:
        raise TypeError(f"Unexpected kwargs: {extra}")
```

---

## Performance considerations

- Argument binding has cost; many small calls with large `**kwargs` can be slower than tightly-packed positional args.
    
- Avoid heavy use of `**kwargs` in inner loops — resolve parameters once outside the loop.
    

---

## Debugging argument-related issues

- Use `inspect.signature()` to print expected signature.
    
- Use `functools.wraps` to avoid wrappers hiding signatures.
    
- When `TypeError` says unexpected keyword arg, inspect function definition and trace call sites.
    

---

## Exercises

1. **Signature practice:** Write functions that use positional-only and keyword-only parameters, and show valid and invalid calls.
    
2. **Wrapper:** Build a decorator that enforces types at runtime using annotations and `inspect.signature`.
    
3. **Config refactor:** Convert a function with many optional keyword arguments into one that accepts a dataclass config.
    
4. **Safe forwarding:** Implement a wrapper that forwards only a whitelist of kwargs to an inner function and raises on others.
    

---

## Mini-project snippet

A small training pipeline that demonstrates clean argument patterns:

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32

def train(X, y, cfg: TrainConfig, *, verbose=False, **trainer_opts):
    # validate unexpected trainer opts
    allowed = {'shuffle', 'device'}
    extra = set(trainer_opts) - allowed
    if extra:
        raise TypeError(f"Unexpected trainer options: {extra}")

    # training loop uses cfg values
    for epoch in range(cfg.epochs):
        if verbose:
            print('epoch', epoch)
        # ...
```

---

## References

- Python docs: Defining Functions — [https://docs.python.org/3/tutorial/controlflow.html#defining-functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
    
- `inspect` module docs — [https://docs.python.org/3/library/inspect.html](https://docs.python.org/3/library/inspect.html)
    
- PEP 570: Positional-only parameters
    

_Would you like this converted into a runnable notebook or expanded with more real-world examples and tests?_