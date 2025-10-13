
**What Are Standard Input and Output?**  
Standard input (stdin) is how your program receives data, usually from the keyboard. Standard output (stdout) is where it sends output, typically to the console. In Python, you use `input()` to get input and `print()` to show output.

**Using `print()` for Output**  
The `print()` function displays text or values on the screen. For example:
```python
print("Hello, World!")  # Shows "Hello, World!"
print("Age:", 25)  # Shows "Age: 25"
```
You can customize it with `sep` (separator) and `end` (ending), like:
```python
print("Hello", "World", sep="-", end="!")  # Shows "Hello-World!"
```
This is great for formatting output clearly.

**Using `input()` for Input**  
The `input()` function waits for you to type something and press enter, returning it as a string. For example:
```python
name = input("Enter your name: ")  # Waits for input, stores as "Alice"
print("Hello,", name)  # Shows "Hello, Alice"
```
Since it always returns a string, convert it for numbers:
```python
age = int(input("Enter your age: "))  # Convert to integer
```
If the input is invalid, it might crash, so use `try-except`:
```python
try:
    age = int(input("Enter age: "))
except ValueError:
    print("Please enter a number.")
```

**Unexpected Detail**: You can read input from files or pipes, not just the keyboard, using `sys.stdin`, but that’s more advanced.

![[Std InpOut.png]]

---


[[Comprehensive Guide -Standard Input and Output in Python]]

#### Key Citations
- [Python Documentation: Built-in Functions](https://docs.python.org/3/library/functions.html)
- [Real Python: Python Input and Output](https://realpython.com/python-input-output/)
- [Python Official Documentation: The print() Function](https://docs.python.org/3/library/functions.html#print)
- [Python Official Documentation: The input() Function](https://docs.python.org/3/library/functions.html#input)

---

