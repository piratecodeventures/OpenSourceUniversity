## Overview

**Exception handling** enables Python programs to respond gracefully to unexpected situations—file errors, invalid inputs, network issues, and more—without crashing. For data science, robust error management ensures long-running experiments, data pipelines, and model training jobs can continue or fail cleanly with helpful diagnostics.

This section covers:

- What exceptions are
    
- The `try`/`except`/`else`/`finally` flow
    
- Built-in exception hierarchy
    
- Creating custom exceptions
    
- Raising exceptions deliberately
    
- Best practices for reliable, maintainable code
    
- Patterns for data-science pipelines
    

---

## What is an Exception?

An **exception** is an event that interrupts normal program flow when an error occurs. Instead of halting execution abruptly, Python lets you _catch_ these events and decide how to handle them.

For example:

```python
f = open('data.txt')  # FileNotFoundError if file does not exist
```

Without handling, Python prints a traceback and stops the program.

---

## Basic Handling: `try`/`except`

```python
try:
    f = open('data.txt', 'r', encoding='utf-8')
    content = f.read()
except FileNotFoundError as e:
    print("File missing:", e)
```

- The `except` block catches matching exceptions and prevents a crash.
    
- You can handle multiple exception types:
    

```python
try:
    num = int(input('Enter a number: '))
    print(10 / num)
except (ValueError, ZeroDivisionError) as e:
    print("Invalid input or division by zero:", e)
```

---

## `else` and `finally`

```python
try:
    f = open('data.txt', 'r')
except FileNotFoundError:
    print("Missing file")
else:
    print(f.read())  # runs only if no exception
finally:
    f.close()        # always runs, even if exception occurred
```

- **`else`** runs only when no exception occurs.
    
- **`finally`** runs no matter what, ideal for cleanup.
    

With context managers (`with`), you often don’t need `finally` for closing files.

---

## Exception Hierarchy

All exceptions inherit from `BaseException`:

```
BaseException
 ├── SystemExit
 ├── KeyboardInterrupt
 └── Exception
      ├── ArithmeticError
      │    ├── ZeroDivisionError
      │    └── OverflowError
      ├── LookupError
      │    ├── IndexError
      │    └── KeyError
      └── ...
```

Catch specific subclasses for fine-grained control.

---

## Raising Exceptions

Use `raise` to signal an error intentionally:

```python
def sqrt(x):
    if x < 0:
        raise ValueError("x must be non-negative")
    return x ** 0.5
```

Raising exceptions early makes bugs easier to detect and prevents silent failures.

---

## Custom Exceptions

Create your own by subclassing `Exception`:

```python
class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

def validate(df):
    if df.isnull().any().any():
        raise DataValidationError("Missing values detected")
```

Custom exceptions improve readability and let callers distinguish specific error cases.

---

## Logging Exceptions

For production code, log exceptions instead of just printing:

```python
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

try:
    risky_operation()
except Exception as e:
    logging.exception("Unexpected error")
```

`logging.exception` records the full traceback.

---

## Chaining Exceptions

Python automatically chains exceptions when one causes another. Use `raise ... from ...` to provide context:

```python
try:
    connect_db()
except ConnectionError as e:
    raise RuntimeError("Database unavailable") from e
```

This preserves the original traceback.

---

## Best Practices

- **Catch specific exceptions**: Avoid bare `except:` which swallows all errors.
    
- **Fail fast**: Raise exceptions early when invalid state is detected.
    
- **Log and re-raise**: Log details, then re-raise if you can’t handle.
    
- **Cleanup resources**: Use `finally` or context managers.
    
- **Don’t hide bugs**: Overly broad handlers can mask real problems.
    
- **Graceful degradation**: In data pipelines, skip bad records but report them.
    

---

## Patterns for Data Science

- **Retry loops** for transient network or database errors:
    

```python
import time
for i in range(3):
    try:
        fetch_data()
        break
    except TimeoutError:
        time.sleep(2 ** i)
else:
    raise RuntimeError("Failed after retries")
```

- **Validation**: Raise custom exceptions when data schema mismatches occur.
    
- **Pipeline resilience**: Wrap each stage (ingest, transform, train) in its own try/except to isolate failures.
    

---

## Common Pitfalls

- Catching `Exception` or using bare `except` can mask KeyboardInterrupt and SystemExit.
    
- Swallowing exceptions silently (e.g., `pass` in except block) leads to debugging nightmares.
    
- Raising strings instead of Exception subclasses is not allowed in Python 3.
    
- Not re-raising after logging when the error must propagate.
    

---

## Example Mini-Project: Robust ETL Pipeline

```python
import logging, pandas as pd

logging.basicConfig(filename='etl.log', level=logging.INFO)

def extract(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        logging.error("Missing source file: %s", e)
        raise

def transform(df):
    if 'value' not in df.columns:
        raise ValueError("Expected 'value' column")
    return df.dropna()

def load(df, out):
    try:
        df.to_csv(out, index=False)
    except PermissionError as e:
        logging.error("Cannot write output: %s", e)
        raise

def etl(src, dest):
    try:
        df = extract(src)
        df = transform(df)
        load(df, dest)
    except Exception as e:
        logging.exception("ETL failed: %s", e)
        raise

if __name__ == "__main__":
    etl('data/input.csv', 'data/output.csv')
```

This demonstrates granular handling and comprehensive logging.

---

## Exercises

1. Wrap a machine learning training script with exception handling and retry logic for data-loading errors.
    
2. Create a custom exception hierarchy (`DataError`, `SchemaMismatchError`, `MissingColumnError`) and raise appropriate errors during preprocessing.
    
3. Implement a context manager that logs exceptions to a file and re-raises them.
    
4. Write a script that catches `KeyboardInterrupt` and saves intermediate results before exiting.
    

---

## References

- Python Exceptions: [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
    
- Logging: [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
    
- PEP 3134: Exception Chaining
    
- Real Python: [Python Exception Handling](https://realpython.com/python-exceptions/)