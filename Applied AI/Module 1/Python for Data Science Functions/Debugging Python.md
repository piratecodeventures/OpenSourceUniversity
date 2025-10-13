## Overview

**Debugging** is the process of identifying, analyzing, and fixing bugs or unexpected behavior in code. For data-science projects—where datasets are large and workflows complex—debugging ensures reproducibility, correctness, and performance. Python provides a rich ecosystem of built-in and external tools to locate problems quickly.

This section explores:

- Recognizing bugs and gathering information
    
- Built-in debugging techniques
    
- Using `pdb` and modern debuggers
    
- Logging and assertions
    
- Profiling for performance issues
    
- Best practices to create maintainable, debuggable code
    

---

## Recognizing and Reproducing Bugs

- **Symptoms**: unexpected outputs, exceptions, performance bottlenecks.
    
- **Steps to reproduce**: Always isolate and reliably reproduce the problem before fixing.
    
- **Minimal example**: Reduce the failing code to the smallest snippet to confirm the bug.
    

---

## Quick Debugging Techniques

1. **Print statements**
    
    - Insert temporary `print()` calls to check variable states.
        
    - Use f-strings for clarity: `print(f"x={x}, y={y}")`.
        
    - Pros: fast and simple. Cons: clutter, risk of leaving prints in production.
        
2. **Assertions**
    
    - Ensure invariants hold: `assert len(data) > 0, "Data is empty"`.
        
    - Useful for catching logic errors early.
        
    - Disabled with the `-O` flag, so don’t rely on them for production validation.
        

---

## Built-in Debugger: `pdb`

`pdb` is Python’s interactive source-level debugger.

```bash
python -m pdb script.py
```

Inside `pdb` you can:

- `l` (list source)
    
- `n` (next line)
    
- `s` (step into)
    
- `c` (continue)
    
- `p var` (print variable)
    
- `q` (quit)
    

Set breakpoints in code:

```python
import pdb; pdb.set_trace()
```

Run until the breakpoint and inspect variables interactively.

### Enhanced `pdb` Alternatives

- **ipdb**: `pip install ipdb`, offers IPython features and better tracebacks.
    
- **pdb++**: Colorized output and sticky mode.
    

---

## Modern IDE Debuggers

- VS Code, PyCharm, JupyterLab provide GUI breakpoints, variable inspection, call-stack navigation.
    
- For Jupyter notebooks, use `%debug` magic after an exception to open a post-mortem session.
    
- `%pdb on` automatically starts the debugger on errors.
    

---

## Logging for Diagnostics

Logging provides a non-intrusive way to trace execution.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Variable x=%s", x)
```

- Different levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    
- Use rotating or timed file handlers for long-running jobs.
    
- Logs persist beyond program termination and support structured analysis.
    

---

## Tracing and Profiling

For performance bottlenecks:

- **cProfile**: Built-in profiler to measure function-level timing.
    

```bash
python -m cProfile -o stats.out script.py
```

Analyze with `pstats` or tools like SnakeViz.

- **line_profiler**: Decorator-based line-level profiling.
    
- **memory_profiler**: Track memory usage of functions.
    

---

## Common Debugging Patterns in Data Science

- **Data validation**: Assert schema and value ranges before processing.
    
- **Unit tests**: Use `pytest` to catch regressions automatically.
    
- **Isolation**: Debug using a small sample of large datasets.
    
- **Deterministic seeds**: Set random seeds to reproduce results.
    
- **Logging experiments**: Track hyperparameters, environment, and data versions.
    

---

## Best Practices

- Write small, pure functions with clear inputs/outputs—easier to test and debug.
    
- Commit failing test cases before fixing a bug (test-driven debugging).
    
- Avoid catching exceptions without logging (swallowing errors).
    
- Keep dependencies minimal to reduce complexity.
    
- Document known issues and workarounds.
    

---

## Remote and Distributed Debugging

- **Remote `pdb`**: Tools like `rpdb` or debug adapters let you attach to running servers.
    
- **Cluster jobs**: Redirect logs to files; use sampling logs and structured JSON for analysis.
    
- **Distributed data processing** (Spark, Dask): Use local mode with small data for reproductions.
    

---

## Example Mini-Project: Debugging a Data Pipeline

Scenario: A CSV ingestion pipeline sometimes produces missing values.

```python
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def load_data(path):
    logging.debug("Loading %s", path)
    df = pd.read_csv(path)
    assert 'value' in df.columns, "Missing 'value' column"
    return df

def transform(df):
    df['value'] = df['value'].fillna(df['value'].mean())
    return df

def main():
    df = load_data('input.csv')
    df = transform(df)
    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()
```

Debugging steps:

1. Reproduce with a small CSV containing edge cases.
    
2. Add `logging.debug` calls to track data shapes.
    
3. Use `pdb.set_trace()` inside `transform` to inspect `df` at runtime.
    

---

## Exercises

1. Insert breakpoints into a numerical optimization routine and inspect variable changes.
    
2. Use `%pdb on` in Jupyter to debug a failing machine-learning model training.
    
3. Profile a script that takes too long to run; identify the slowest functions with `cProfile`.
    
4. Add structured logging to a web-scraping pipeline and analyze logs after a failure.
    

---

## References

- Python Debugger (`pdb`): [https://docs.python.org/3/library/pdb.html](https://docs.python.org/3/library/pdb.html)
    
- Logging: [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
    
- cProfile: [https://docs.python.org/3/library/profile.html](https://docs.python.org/3/library/profile.html)
    
- PyCharm & VS Code Debugging Guides
    
- Real Python: [Python Debugging with Pdb](https://realpython.com/python-debugging-pdb/)