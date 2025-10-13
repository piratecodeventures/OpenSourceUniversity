## Overview

Lambda functions (also called _anonymous functions_) are small, single-expression functions created with the `lambda` keyword. They are useful for short throwaway functions, inline callbacks, and functional-style programming. In this section we cover syntax, idiomatic uses, trade-offs, how they interact with closures and scoping, alternatives (named functions, `functools.partial`, `operator` functions), debugging tips, and practical data-science examples.

---

## Syntax & basic examples

```python
# basic lambda that squares its input
square = lambda x: x * x
print(square(5))  # 25

# inline usage
pairs = [(1, 'a'), (3, 'c'), (2, 'b')]
sorted_pairs = sorted(pairs, key=lambda p: p[0])
```

General form:

```python
lambda arg1, arg2, ..., kw=default: expression_using_args
```

- Lambdas are expressions and evaluate to function objects.
    
- The `expression` must be a single expression — no statements, `try/except`, `yield`, or multiple lines.
    

---

## When to use lambdas

- Small one-off callbacks: `sorted(..., key=...)`, `map`, `filter`.
    
- Simple transformations passed into higher-order functions.
    
- Quick inline behavior where defining a named function would be verbose.
    

**Examples**

```python
# map
squares = list(map(lambda x: x*x, range(6)))

# filter
evens = list(filter(lambda x: x % 2 == 0, range(10)))

# sorted by computed key
names = ['alice', 'Bob', 'charlie']
names_sorted = sorted(names, key=lambda s: s.lower())
```

---

## Lambda vs `def` — trade-offs

**Lambda advantages**

- Concise inline definition.
    
- Good for tiny, throwaway functions.
    

**Lambda disadvantages**

- Limited to a single expression — complex logic is awkward or impossible.
    
- No docstring — harder to document intent.
    
- `__name__` is `<lambda>`, making debugging/readability worse.
    
- Harder to set breakpoints and step through in debuggers.
    

**Rule of thumb:** If the function is nontrivial, reused, or benefits from a descriptive name or docstring — use `def`.

---

## Scoping and closures with lambdas

Lambdas behave like normal functions regarding scope and closures — they capture variables from enclosing scopes.

```python
def make_adder(n):
    return lambda x: x + n

add5 = make_adder(5)
print(add5(10))  # 15
```

Be careful with late binding when creating lambdas in loops:

```python
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])  # surprising: [2, 2, 2]

# Fix using default argument binding:
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])  # [0, 1, 2]
```

The default argument trick evaluates the value at definition time, avoiding the late-binding trap.

---

## Common functional patterns with lambdas

- `map(func, iterable)` — apply func to each item and return an iterator (use list/tuple to materialize).
    
- `filter(pred, iterable)` — keep items where pred(item) is true.
    
- `functools.reduce(func, iterable)` — accumulate values using binary func.
    
- `sorted(iterable, key=...)`, `min`, `max` — accept `key` functions.
    

**Example pipeline using builtins**

```python
from functools import reduce

data = [1, 2, 3, 4]
result = reduce(lambda a, b: a + b, map(lambda x: x*x, filter(lambda x: x%2==0, data)))
```

While compact, such pipelines can become hard to read; consider intermediate variables or named functions for clarity.

---

## Lambdas in libraries & data science

- **Pandas:** `df['col'].apply(lambda x: ...)` — convenient for row-wise or element-wise custom transforms. Prefer vectorized operations (Pandas/Numpy) when possible for performance.
    

```python
import pandas as pd

df = pd.DataFrame({'x': [1,2,3]})
df['y'] = df['x'].apply(lambda v: v**2)
```

- **scikit-learn pipelines:** Small lambdas can be used in `FunctionTransformer` or for quick feature extraction, but for production prefer named transformers with `fit`/`transform` methods.
    
- **matplotlib / callbacks:** lambdas often appear in GUI callbacks or small event handlers.
    

**Performance note:** lambdas are normal Python functions; performing heavy operations inside lambda will be as slow as the equivalent `def`. For numeric-heavy transforms, prefer vectorized NumPy/Pandas operations or compiled code.

---

## Alternatives to lambda (when to prefer them)

- **Named `def` function:** when reuse, tests, or docs help.
    
- **`functools.partial`**: when you want to pre-bind arguments of an existing function.
    

```python
from functools import partial
pow2 = partial(pow, exp=2)
```

- **`operator` module:** provides fast function equivalents for common ops (e.g., `operator.itemgetter`, `operator.attrgetter`, `operator.add`) and can be clearer and faster than small lambdas.
    

```python
from operator import itemgetter
sorted_pairs = sorted(pairs, key=itemgetter(0))
```

- **Callable classes:** if you need stateful behavior or want a descriptive `__repr__`.
    

---

## Debugging and readability tips

- Replace complex lambdas with named functions to improve stack traces and allow docstrings.
    
- Use `functools.wraps` only with decorators — it doesn't apply to lambdas; name the function if you need preserved metadata.
    
- Avoid deeply nested lambda expressions — they hurt readability.
    
- When using lambdas in data transformations, prefer small, well-named helper functions when the transformation is non-trivial.
    

---

## Serialization & pickling considerations

- Standard `pickle` cannot serialize locally defined lambdas or functions defined in the REPL; serializing code objects is fragile. Use named, top-level functions to maximize portability across processes and clusters.
    
- Tools like `dill` or `cloudpickle` can serialize lambdas and closures for distributed computing, but they increase dependency complexity and reduce portability.
    

---

## Performance considerations

- A lambda is just a function object created at runtime — creation cost is similar to `def`, and calling cost is the same as a normal function call.
    
- For tight loops, avoid constructing lambdas repeatedly inside the loop; construct them once outside.
    
- Prefer vectorized ops (NumPy/Pandas) for large numeric workloads rather than Python-level lambdas and loops.
    

---

## Advanced: composing lambdas and functional utilities

You can compose small lambda functions manually or use helper libraries.

```python
# simple composition example
compose = lambda f, g: lambda x: f(g(x))
inc = lambda x: x + 1
double = lambda x: x * 2
inc_then_double = compose(double, inc)
print(inc_then_double(3))  # (3+1)*2 = 8
```

For pipeline-style composition prefer readable helper functions or libraries like `toolz` / `fn` in production code.

---

## Pitfalls and gotchas

- Late binding in closures (common in loops) — fix with default args.
    
- Overuse leads to unreadable, terse code.
    
- Inability to include statements (no multi-line logic or assignments) — if you need them, use `def`.
    
- Debugging stack traces often show `<lambda>` which is less informative than named functions.
    

---

## Exercises

1. **Sorting practice:** Use a lambda to sort a list of strings by their last character.
    
2. **Map/filter:** Use `map` and `filter` with lambdas to compute the squares of odd numbers in a list.
    
3. **Late binding bug:** Create a list of lambdas in a loop that capture the loop variable wrongly; then fix it using default arguments.
    
4. **Pandas apply:** Given a DataFrame of timestamps, use a lambda to extract the hour into a new column, then replace with a vectorized solution using `dt.hour` and compare performance.
    
5. **Compose:** Implement `compose` and `pipe` helpers using lambdas and test them on small transformations.
    

---

## Mini-project idea

Build a small ETL script that ingests CSV rows and applies a configurable pipeline of transformations. Allow the user to supply small lambda expressions for quick experimentation (in a REPL or config), but provide a way to register named transformers for production runs. This shows when lambdas are useful for exploration and when named functions are better for reproducibility.

---

## References & further reading

- Python docs: Lambda expressions — [https://docs.python.org/3/reference/expressions.html#lambda](https://docs.python.org/3/reference/expressions.html#lambda)
    
- `functools`, `operator`, and `itertools` docs for functional helpers
    

_Would you like this section converted to runnable notebook cells or packaged into slides for teaching?_