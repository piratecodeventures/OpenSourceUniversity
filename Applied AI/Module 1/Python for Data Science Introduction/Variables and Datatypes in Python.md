### What Are Variables and Data Types?

**Variables in Python**  
Variables are like labeled boxes that store data. In Python, they’re names that refer to values, such as numbers or text. You can think of them as trucks carrying cargo: the truck (variable) can hold different items (data) at different times.

**Data Types in Python**  
Data types classify the kind of data a variable holds, like integers (whole numbers, e.g., 5), strings (text, e.g., "hello"), or lists (ordered collections, e.g., [1, 2, 3]). Python has many built-in types, and each determines what operations you can perform.

![[Variable and data type.png]]
---

### How Python Handles Dynamic Typing

Python is dynamically typed, meaning you don’t declare a variable’s type upfront. Instead, the type is set by the value assigned. For example:
- `x = 5` makes `x` an integer.
- Later, `x = "hello"` makes `x` a string. It’s like swapping the truck’s cargo from apples to books.

This flexibility is powerful but requires caution: if you try to add 3 to `"hello"`, Python will raise a TypeError, as it can’t add a number to text.
![[dynamic typing.png]]

---

### Why Be Careful with Variable Assignments?

Since variables can change types, you must ensure they hold the expected data before using them. For instance:
- `x = 5; print(x + 3)` works (outputs 8).
- But if `x = "hello"; print(x + 3)` fails, as strings and numbers don’t add.

Use `type(x)` to check: `print(type(x))` shows `<class 'int'>` or `<class 'str'>`. This helps avoid errors, especially in complex programs where variables might be reassigned elsewhere.
![[variable type check.png]]

---

[[Comprehensive Guide - Variables and Data Types in Python]]

![[python data types.png]]
#### Key Citations
- [Python Documentation: Variables](https://docs.python.org/3/reference/executionmodel.html#naming-and-binding)
- [Python Documentation: Data Types](https://docs.python.org/3/library/stdtypes.html)
- [Real Python: Understanding Variables in Python](https://realpython.com/courses/introduction-to-python/03-introduction-to-python/04-variables/)


