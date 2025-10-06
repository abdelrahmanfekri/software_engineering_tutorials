# Python Programming Tutorial 01: Python Fundamentals

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand Python syntax and basic structure
- Work with variables and data types
- Use operators and expressions
- Handle input and output
- Write comments and documentation
- Understand Python's indentation system

## Introduction to Python

### What is Python?
Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and scientific computing.

### Why Python for Mathematics?
- **Readable syntax**: Easy to understand and write
- **Rich libraries**: NumPy, SciPy, Matplotlib, Pandas
- **Interactive environment**: Jupyter notebooks
- **Cross-platform**: Works on Windows, Mac, Linux
- **Large community**: Extensive documentation and support

### Python Installation
```bash
# Check if Python is installed
python --version

# Install Python from python.org or use Anaconda
# Anaconda includes many scientific libraries
```

## Basic Syntax

### Hello World
```python
print("Hello, World!")
```

### Python as Calculator
```python
# Basic arithmetic
print(2 + 3)        # 5
print(10 - 4)       # 6
print(3 * 4)        # 12
print(15 / 3)       # 5.0
print(2 ** 3)       # 8 (exponentiation)
print(17 % 5)       # 2 (modulo)
```

## Variables and Data Types

### Variables
Variables store data and can be reassigned.

```python
# Variable assignment
x = 5
y = 3.14
name = "Alice"
is_student = True

# Variable reassignment
x = x + 1  # x is now 6
```

### Data Types

#### Numbers
```python
# Integers
age = 25
print(type(age))  # <class 'int'>

# Floats
pi = 3.14159
print(type(pi))   # <class 'float'>

# Complex numbers
z = 3 + 4j
print(type(z))    # <class 'complex'>
```

#### Strings
```python
# String creation
message = "Hello, Python!"
quote = 'Single quotes work too'
multiline = """This is a
multiline string"""

# String operations
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  # John Doe

# String formatting
age = 25
print(f"My name is {first_name} and I'm {age} years old")
```

#### Booleans
```python
# Boolean values
is_true = True
is_false = False

# Boolean operations
result = True and False  # False
result = True or False   # True
result = not True        # False
```

#### Lists
```python
# List creation
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# List operations
numbers.append(6)        # Add to end
numbers.insert(0, 0)     # Insert at index
numbers.remove(3)        # Remove first occurrence
last = numbers.pop()     # Remove and return last element

print(numbers)  # [0, 1, 2, 4, 5]
```

#### Dictionaries
```python
# Dictionary creation
student = {
    "name": "Alice",
    "age": 20,
    "major": "Mathematics"
}

# Dictionary operations
student["gpa"] = 3.8      # Add new key-value pair
age = student["age"]      # Access value
student.pop("major")      # Remove key-value pair

print(student)  # {'name': 'Alice', 'age': 20, 'gpa': 3.8}
```

## Operators

### Arithmetic Operators
```python
a, b = 10, 3

print(a + b)   # 13 (addition)
print(a - b)   # 7 (subtraction)
print(a * b)   # 30 (multiplication)
print(a / b)   # 3.333... (division)
print(a // b)  # 3 (floor division)
print(a % b)   # 1 (modulo)
print(a ** b)  # 1000 (exponentiation)
```

### Comparison Operators
```python
x, y = 5, 10

print(x == y)  # False (equal)
print(x != y)  # True (not equal)
print(x < y)   # True (less than)
print(x > y)   # False (greater than)
print(x <= y)  # True (less than or equal)
print(x >= y)  # False (greater than or equal)
```

### Logical Operators
```python
p, q = True, False

print(p and q)  # False
print(p or q)   # True
print(not p)    # False
```

### Assignment Operators
```python
x = 10
x += 5    # x = x + 5 = 15
x -= 3    # x = x - 3 = 12
x *= 2    # x = x * 2 = 24
x /= 4    # x = x / 4 = 6.0
```

## Input and Output

### Input
```python
# Get user input
name = input("Enter your name: ")
age = int(input("Enter your age: "))  # Convert to integer
height = float(input("Enter your height: "))  # Convert to float

print(f"Hello {name}, you are {age} years old and {height} feet tall")
```

### Output
```python
# Print function
print("Hello")                    # Simple text
print("Hello", "World")           # Multiple arguments
print("Hello", "World", sep="-")  # Custom separator
print("Hello", end="!")           # Custom end character

# Formatted output
x = 3.14159
print(f"Pi is approximately {x:.2f}")  # Pi is approximately 3.14
```

## Comments and Documentation

### Comments
```python
# This is a single-line comment

"""
This is a multi-line comment
or docstring
"""

def calculate_area(radius):
    """
    Calculate the area of a circle.
    
    Args:
        radius (float): The radius of the circle
        
    Returns:
        float: The area of the circle
    """
    return 3.14159 * radius ** 2
```

## Indentation and Code Blocks

### Indentation Rules
Python uses indentation to define code blocks. Use 4 spaces (not tabs) for each level.

```python
# Correct indentation
if x > 0:
    print("Positive")
    if x > 10:
        print("Large positive")
else:
    print("Non-positive")
```

### Common Indentation Errors
```python
# Wrong - inconsistent indentation
if x > 0:
    print("Positive")
  print("This will cause an error")

# Wrong - mixing tabs and spaces
if x > 0:
    print("Positive")  # Uses spaces
    print("Also positive")  # Uses tabs
```

## Mathematical Operations

### Basic Math Functions
```python
import math

# Common mathematical functions
print(math.sqrt(16))      # 4.0
print(math.pow(2, 3))     # 8.0
print(math.exp(1))        # 2.718...
print(math.log(10))       # 2.302...
print(math.sin(math.pi/2)) # 1.0
print(math.cos(0))        # 1.0

# Constants
print(math.pi)            # 3.14159...
print(math.e)             # 2.71828...
```

### Working with Numbers
```python
# Type conversion
x = "123"
y = int(x)        # Convert string to integer
z = float(x)      # Convert string to float

# Rounding
number = 3.14159
print(round(number, 2))    # 3.14
print(math.floor(number))  # 3
print(math.ceil(number))   # 4
```

## Practice Problems

### Problem 1
Create variables for a student's information and display it.

```python
# Solution
name = "John Smith"
age = 20
gpa = 3.7
major = "Mathematics"

print(f"Student: {name}")
print(f"Age: {age}")
print(f"GPA: {gpa}")
print(f"Major: {major}")
```

### Problem 2
Calculate the area and circumference of a circle.

```python
# Solution
import math

radius = float(input("Enter the radius: "))
area = math.pi * radius ** 2
circumference = 2 * math.pi * radius

print(f"Area: {area:.2f}")
print(f"Circumference: {circumference:.2f}")
```

### Problem 3
Create a list of numbers and perform various operations.

```python
# Solution
numbers = [1, 2, 3, 4, 5]

# Calculate sum
total = sum(numbers)
print(f"Sum: {total}")

# Calculate average
average = total / len(numbers)
print(f"Average: {average}")

# Find maximum and minimum
print(f"Maximum: {max(numbers)}")
print(f"Minimum: {min(numbers)}")
```

## Key Takeaways
- Python uses simple, readable syntax
- Variables can store different data types
- Indentation defines code blocks
- Comments help document code
- Built-in functions provide mathematical operations
- Type conversion is important for calculations

## Next Steps
In the next tutorial, we'll explore control structures, learning about conditional statements and loops to make programs more dynamic and powerful.
