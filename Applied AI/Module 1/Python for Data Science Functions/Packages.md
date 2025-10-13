## Overview

A **package** is a way of structuring Python’s module namespace by using _dotted module names_. A package is essentially a directory that contains a special file `__init__.py` (or is a [namespace package](https://peps.python.org/pep-0420/) without it). Packages enable you to organize related modules together, share code across projects, and distribute libraries. They are the building blocks for creating reusable, maintainable, and large-scale Python applications and libraries.

This section explores package layout, import mechanics, namespace packages, best practices, distribution, versioning, and common pitfalls relevant to data-science projects.

---

## What is a package?

- A **module** is a single `.py` file.
    
- A **package** is a directory that Python treats as a single namespace.
    
- Packages can contain subpackages and modules.
    
- Importing a package executes its `__init__.py` (if present) and creates a package object.
    

Example directory structure:

```
myproject/
│
└─ mypkg/
   ├─ __init__.py
   ├─ io.py
   ├─ preprocess.py
   └─ models/
      ├─ __init__.py
      ├─ regression.py
      └─ tree.py
```

Usage:

```python
from mypkg import io
from mypkg.models import regression
```

---

## `__init__.py`

- Marks a directory as a _regular package_ (not required for namespace packages).
    
- Executes when the package is first imported.
    
- Can be empty or used to set up package-level state, expose public API, or perform initialization.
    

Example minimal `__init__.py`:

```python
"""mypkg: data-science utilities"""
__all__ = ["io", "preprocess"]
```

Use `__all__` to control `from mypkg import *` behavior.

---

## Namespace packages (PEP 420)

- Allow packages to span multiple directories or distributions.
    
- No `__init__.py` is required.
    
- Useful for plugin systems, large organizations, or when multiple projects contribute modules to the same namespace.
    

Example layout across repos:

```
acme.analytics/plot/
acme.analytics/stats/
```

Both subdirectories provide parts of `acme.analytics`.

To create: simply omit `__init__.py` and ensure the parent directory is on `sys.path`.

---

## Absolute vs. relative imports

- **Absolute**: full path from project root: `from mypkg.io import read_csv`.
    
- **Relative**: relative to current package: `from .io import read_csv` or `from ..utils import helper`.
    
- Prefer absolute imports for clarity in large projects.
    

Relative imports only work inside packages. Top-level scripts executed directly (`python script.py`) aren’t treated as part of a package; run them with `python -m` or restructure.

---

## Organizing a package for data science

Recommended layout:

```
project/
├─ pyproject.toml
├─ src/
│  └─ mypkg/
│     ├─ __init__.py
│     ├─ io.py
│     ├─ preprocess.py
│     ├─ features.py
│     ├─ model/
│     │   ├─ __init__.py
│     │   ├─ train.py
│     │   └─ evaluate.py
└─ tests/
   └─ test_*.py
```

### Tips

- **Single responsibility**: Each module or subpackage handles one concern.
    
- **Lightweight `__init__.py`**: Avoid heavy computation or imports.
    
- **Explicit API**: Re-export key functions/classes in `__init__.py` for a clean user experience.
    
- **Config & logging**: Keep configuration separate and use `logging.getLogger(__name__)`.
    

---

## Import mechanics for packages

- When you import a package, Python creates a module object with the package name and sets its `__path__` attribute to a list of directories.
    
- Submodules are found by searching each entry in `__path__`.
    
- For namespace packages, `__path__` can span multiple directories.
    

`mypkg.__path__` reveals where submodules are loaded from.

---

## Packaging and distribution

Modern packaging uses **pyproject.toml**:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mypkg"
version = "0.1.0"
description = "Data science utilities"
authors = [{ name="Your Name", email="you@example.com" }]
```

Build and publish:

```bash
python -m build      # creates sdist and wheel
python -m twine upload dist/*
```

For data-science projects, you can distribute private wheels on an internal index or via Git repositories.

---

## Versioning

Follow [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH. Increment MAJOR for breaking changes, MINOR for backward-compatible features, PATCH for bug fixes.

Expose the version in your package:

```python
# mypkg/__init__.py
__version__ = "0.1.0"
```

---

## Testing packages

- Place tests in a separate `tests/` directory.
    
- Use `pytest` and run with `pytest` or `python -m pytest`.
    
- Ensure imports work as installed by running tests against the installed package (`pip install -e .` for editable mode).
    

---

## Data files and resources

If your package includes data (templates, CSVs, models), use `importlib.resources` to read them:

```python
from importlib import resources
with resources.files('mypkg.data').joinpath('schema.json').open('r') as f:
    schema = f.read()
```

This is safer and more portable than using `__file__` paths.

---

## Best practices

- Keep package hierarchy shallow and intuitive.
    
- Minimize top-level imports in `__init__.py` to reduce import time.
    
- Document the public API clearly.
    
- Avoid circular dependencies; factor shared utilities into separate submodules.
    
- For large packages, consider splitting into subpackages (e.g., `mypkg.io`, `mypkg.ml`).
    
- Use namespace packages if multiple distributions must share a top-level namespace.
    

---

## Common pitfalls

- **Running scripts inside the package directly**: May break relative imports. Use `python -m mypkg.module` instead.
    
- **Overpopulated `__init__.py`**: Heavy code in `__init__.py` slows imports and can cause side effects.
    
- **Accidental name clashes**: Don’t name a module the same as a standard library module (e.g., `json.py`).
    
- **Mutable globals**: Be careful with global state shared across submodules.
    

---

## Example mini-project

Create a package `mlpipe` for a machine-learning pipeline:

```
mlpipe/
├─ __init__.py
├─ data.py
├─ preprocess.py
├─ model/
│  ├─ __init__.py
│  ├─ train.py
│  └─ evaluate.py
└─ cli.py
```

Expose a command-line entry point in `pyproject.toml`:

```toml
[project.scripts]
mlpipe = "mlpipe.cli:main"
```

This lets users run `mlpipe` directly after installation.

---

## Exercises

1. **Refactor**: Convert a multi-thousand-line analysis script into a package with submodules for I/O, preprocessing, and modeling.
    
2. **Namespace package**: Create two separate distributions (`acme.analytics.plot` and `acme.analytics.stats`) that contribute to the same namespace `acme.analytics`.
    
3. **Resource access**: Include a CSV schema file in your package and load it using `importlib.resources`.
    
4. **Version management**: Implement a `__version__` attribute and write a script that prints it.
    

---

## References

- Python Packaging User Guide — [https://packaging.python.org/](https://packaging.python.org/)
    
- PEP 420 — Implicit Namespace Packages
    
- importlib.resources — [https://docs.python.org/3/library/importlib.resources.html](https://docs.python.org/3/library/importlib.resources.html)
    
- Semantic Versioning — [https://semver.org/](https://semver.org/)
    

---

_Would you like runnable packaging examples (with a real pyproject.toml and wheel build), slide-ready notes, or a sample CI pipeline for this package layout?_