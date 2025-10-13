## Overview

File handling in Python allows you to **read**, **write**, and **manage files** directly from your programs. For data science, it is crucial for importing datasets, logging experiment results, or saving models. Python’s built-in `open()` function and related libraries give you a powerful yet simple interface to interact with text, CSV, JSON, binary, and other file types.

In this section, we cover:

- File types and modes
    
- Reading and writing text/binary files
    
- Using context managers for safe file operations
    
- Common data formats (CSV, JSON, etc.)
    
- Efficient large-file handling
    
- Best practices and pitfalls
    

---

## Opening Files

The `open()` function is your gateway to files.

```python
file_object = open('data.txt', mode='r', encoding='utf-8')
```

### Modes

- `'r'` : Read (default)
    
- `'w'` : Write (truncate if exists)
    
- `'a'` : Append
    
- `'x'` : Exclusive creation (fails if exists)
    
- `'b'` : Binary (combine with others, e.g. `'rb'`)
    
- `'+'` : Read/Write
    

Always specify an **encoding** (e.g., `utf-8`) for text files to avoid cross-platform issues.

---

## Reading Files

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()        # entire file as string
    lines = f.readlines()  # list of lines
```

- Use `.read(size)` to read a specific number of bytes.
    
- Iterate directly over the file object for memory efficiency:
    

```python
for line in f:
    process(line)
```

---

## Writing Files

```python
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('Hello World!\n')
    f.writelines(['Line1\n', 'Line2\n'])
```

- `'w'` overwrites; `'a'` appends.
    
- Flush frequently or close properly to avoid data loss.
    

---

## Context Managers

Always use `with` to ensure files are closed automatically, even on errors:

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    data = f.read()
```

This pattern prevents file descriptor leaks and locks.

---

## Working with Paths

Use the `pathlib` module for modern, object-oriented file paths:

```python
from pathlib import Path
p = Path('data') / 'dataset.csv'
if p.exists():
    print(p.read_text())
```

Benefits:

- Cross-platform (Windows/Unix)
    
- Convenient methods: `.read_text()`, `.write_text()`, `.iterdir()`
    

---

## Handling Large Files

- Process line by line to avoid loading entire files.
    
- Use buffered reading (`with open(..., buffering=8192)`).
    
- For huge datasets, consider libraries like **pandas** or chunked processing:
    

```python
import pandas as pd
for chunk in pd.read_csv('bigdata.csv', chunksize=10000):
    analyze(chunk)
```

---

## Common Data Formats

### CSV

```python
import csv
with open('data.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['name'], row['age'])
```

Use `pandas.read_csv()` for quick analysis.

### JSON

```python
import json
with open('data.json', 'r', encoding='utf-8') as f:
    obj = json.load(f)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(obj, f, indent=2)
```

### Binary

```python
with open('image.png', 'rb') as f:
    content = f.read()
```

Use `'wb'` for writing.

### Pickle (Python objects)

```python
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

> **Warning**: Only load pickle files from trusted sources (security risk).

---

## File System Operations

`os` and `shutil` provide utilities:

```python
import os, shutil
os.rename('old.txt', 'new.txt')
os.remove('new.txt')
shutil.copy('src.txt', 'dst.txt')
```

Prefer `pathlib` for modern workflows.

---

## Error & Exception Handling

Always handle I/O errors gracefully:

```python
from pathlib import Path
try:
    data = Path('file.txt').read_text()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
```

---

## Logging and Data Science

- Log experiment metrics to files using the `logging` module.
    
- Store results in structured formats (CSV, JSON) for reproducibility.
    
- Use rotating log files (`logging.handlers.RotatingFileHandler`) for long-running jobs.
    

---

## Best Practices

- **Always use `with`** for resource management.
    
- **Explicit encoding**: Avoid platform-dependent defaults.
    
- **Validate input paths**: Use `pathlib` and check existence.
    
- **Avoid hard-coded paths**: Use configuration files or environment variables.
    
- **Close files**: Even if using `with`, be mindful in complex code.
    
- **Handle exceptions**: Prevent crashes on missing files or permission issues.
    
- **Backup important data**: before overwriting.
    

---

## Common Pitfalls

- Forgetting to close files -> file locks, resource leaks.
    
- Relying on default encoding -> inconsistent behavior across OSes.
    
- Overwriting data unintentionally with `'w'`.
    
- Reading huge files entirely into memory.
    

---

## Example Mini-Project: Log Analyzer

**Goal**: Parse a web server log, summarize requests, and output a CSV report.

```python
from pathlib import Path
import csv

def analyze_log(log_path, out_csv):
    summary = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            ip = line.split()[0]
            summary[ip] = summary.get(ip, 0) + 1
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['IP', 'Count'])
        for ip, count in summary.items():
            writer.writerow([ip, count])

if __name__ == '__main__':
    analyze_log('access.log', 'report.csv')
```

This project demonstrates reading a large log file line by line, processing text, and writing structured output.

---

## Exercises

1. Write a script to copy a directory recursively using `shutil`.
    
2. Create a CSV summarizing word counts in a large text corpus.
    
3. Implement a function that safely writes JSON to disk with automatic backups.
    
4. Use `pathlib` to find and delete all `.tmp` files older than 7 days.
    

---

## References

- Python `io` and `pathlib` docs: [https://docs.python.org/3/library/](https://docs.python.org/3/library/)
    
- CSV Module: [https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html)
    
- JSON Module: [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)
    
- Pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)
    
- Logging: [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)