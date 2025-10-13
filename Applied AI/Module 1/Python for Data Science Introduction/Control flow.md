
### If-Else Statements
If-else statements let you run different code based on whether a condition is true or false. For example:
```python
x = 5
if x > 0:
    print("Positive")
else:
    print("Zero or negative")
```
You can add `elif` for multiple conditions, like checking grades:
```python
grade = 85
if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
else:
    print("C or below")
```
Ensure conditions are clear and use proper indentation.
![[img/if else.png]]
### While Loops
While loops repeat code while a condition is true, like counting:
```python
count = 0
while count < 5:
    print(count)
    count += 1
```
Add an `else` to run code after the loop ends:
```python
count = 0
while count < 5:
    print(count)
    count += 1
else:
    print("Done")
```
Watch for infinite loops by ensuring the condition changes.
![[while loop.png]]
### For Loops
For loops iterate over sequences, like lists or ranges:
```python
fruits = ["apple", "banana"]
for fruit in fruits:
    print(fruit)
```
Use `range()` for counting:
```python
for i in range(3):
    print(i)  # Prints 0, 1, 2
```
Use `enumerate()` for indices:
```python
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```
![[for loops.png]]
### Break and Continue
- **Break**: Exits the loop early, like stopping at 5:
  ```python
  for i in range(10):
      if i == 5:
          break
      print(i)  # Prints 0 to 4
  ```
- **Continue**: Skips to the next iteration, like skipping even numbers:
  ```python
  for i in range(5):
      if i % 2 == 0:
          continue
      print(i)  # Prints 1, 3
  ```
Use these sparingly to keep code readable.
![[break and contunie.png]]

---

[[Comprehensive Note - Control Flow in Python – If-Else, While Loops, For Loops, and Break Continue]]

![[loops.png]]

#### Key Citations
- [Python Documentation: Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [Real Python: Python If-Else Statements](https://realpython.com/python-if-else/)
- [Real Python: Python Loops](https://realpython.com/python-for-loop/)
- [GeeksforGeeks: Python Break and Continue](https://www.geeksforgeeks.org/python-break-continue-pass/)
