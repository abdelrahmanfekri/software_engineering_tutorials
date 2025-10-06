# Python Programming Tutorial 04: Functions and Modules

## Learning Objectives
By the end of this tutorial, you will be able to:
- Define and call functions with different parameter types
- Use return values effectively
- Understand scope and namespaces
- Work with lambda functions
- Create and import modules and packages
- Organize code into reusable components
- Apply functions to mathematical problems

## Function Definition and Calling

### Basic Function Syntax
```python
def greet():
    """A simple function that prints a greeting"""
    print("Hello, World!")

# Calling the function
greet()  # Hello, World!
```

### Functions with Parameters
```python
def greet_person(name):
    """Greet a specific person"""
    print(f"Hello, {name}!")

def add_numbers(a, b):
    """Add two numbers and return the result"""
    return a + b

# Function calls
greet_person("Alice")  # Hello, Alice!
result = add_numbers(5, 3)
print(result)  # 8
```

### Functions with Multiple Parameters
```python
def calculate_area(length, width):
    """Calculate the area of a rectangle"""
    return length * width

def calculate_volume(length, width, height):
    """Calculate the volume of a box"""
    return length * width * height

# Using the functions
area = calculate_area(10, 5)
volume = calculate_volume(10, 5, 3)
print(f"Area: {area}, Volume: {volume}")  # Area: 50, Volume: 150
```

## Parameters and Arguments

### Positional Arguments
```python
def power(base, exponent):
    """Calculate base raised to the power of exponent"""
    return base ** exponent

# Positional arguments
result1 = power(2, 3)    # 2^3 = 8
result2 = power(3, 2)     # 3^2 = 9
print(f"2^3 = {result1}, 3^2 = {result2}")
```

### Keyword Arguments
```python
def create_student(name, age, major, gpa=3.0):
    """Create a student record with optional GPA"""
    return {
        "name": name,
        "age": age,
        "major": major,
        "gpa": gpa
    }

# Using keyword arguments
student1 = create_student(name="Alice", age=20, major="Mathematics")
student2 = create_student(name="Bob", age=22, major="Physics", gpa=3.8)

print(student1)  # {'name': 'Alice', 'age': 20, 'major': 'Mathematics', 'gpa': 3.0}
print(student2)  # {'name': 'Bob', 'age': 22, 'major': 'Physics', 'gpa': 3.8}
```

### Default Parameters
```python
def greet_with_title(name, title="Mr."):
    """Greet someone with an optional title"""
    return f"Hello, {title} {name}!"

# Using default parameter
greeting1 = greet_with_title("Smith")  # Hello, Mr. Smith!
greeting2 = greet_with_title("Johnson", "Dr.")  # Hello, Dr. Johnson!
print(greeting1)
print(greeting2)
```

### Variable Arguments (*args)
```python
def sum_numbers(*args):
    """Sum any number of arguments"""
    total = 0
    for number in args:
        total += number
    return total

# Using variable arguments
result1 = sum_numbers(1, 2, 3)           # 6
result2 = sum_numbers(1, 2, 3, 4, 5)     # 15
result3 = sum_numbers(10)                # 10

print(f"Sum of 1,2,3: {result1}")
print(f"Sum of 1,2,3,4,5: {result2}")
print(f"Sum of 10: {result3}")
```

### Keyword Arguments (**kwargs)
```python
def create_profile(**kwargs):
    """Create a user profile from keyword arguments"""
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

# Using keyword arguments
profile1 = create_profile(name="Alice", age=25, city="New York")
profile2 = create_profile(name="Bob", occupation="Engineer", salary=75000)

print(profile1)  # {'name': 'Alice', 'age': 25, 'city': 'New York'}
print(profile2)  # {'name': 'Bob', 'occupation': 'Engineer', 'salary': 75000}
```

### Combining Different Argument Types
```python
def complex_function(required_arg, optional_arg="default", *args, **kwargs):
    """Function with all types of arguments"""
    print(f"Required: {required_arg}")
    print(f"Optional: {optional_arg}")
    print(f"Variable args: {args}")
    print(f"Keyword args: {kwargs}")

# Example usage
complex_function("hello", "world", 1, 2, 3, name="Alice", age=25)
```

## Return Values

### Single Return Value
```python
def square(number):
    """Return the square of a number"""
    return number ** 2

result = square(5)
print(result)  # 25
```

### Multiple Return Values
```python
def get_stats(numbers):
    """Return multiple statistics"""
    if not numbers:
        return 0, 0, 0
    
    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    
    return total, average, maximum

# Unpacking multiple return values
numbers = [1, 2, 3, 4, 5]
total, avg, max_val = get_stats(numbers)
print(f"Total: {total}, Average: {avg}, Max: {max_val}")
```

### Conditional Returns
```python
def find_element(lst, target):
    """Find element in list, return index or -1 if not found"""
    for i, element in enumerate(lst):
        if element == target:
            return i
    return -1

# Using the function
numbers = [10, 20, 30, 40, 50]
index = find_element(numbers, 30)
if index != -1:
    print(f"Found 30 at index {index}")
else:
    print("30 not found")
```

## Scope and Namespaces

### Local vs Global Scope
```python
# Global variable
global_var = "I'm global"

def test_scope():
    # Local variable
    local_var = "I'm local"
    print(f"Inside function: {global_var}")
    print(f"Inside function: {local_var}")

test_scope()
print(f"Outside function: {global_var}")
# print(local_var)  # This would cause an error
```

### Modifying Global Variables
```python
counter = 0

def increment_counter():
    global counter
    counter += 1

def get_counter():
    return counter

# Using the functions
increment_counter()
increment_counter()
print(get_counter())  # 2
```

### Nested Functions
```python
def outer_function(x):
    """Outer function with nested function"""
    def inner_function(y):
        """Inner function that can access outer function's variables"""
        return x + y
    
    return inner_function

# Using nested functions
add_five = outer_function(5)
result = add_five(3)
print(result)  # 8
```

## Lambda Functions

### Basic Lambda Functions
```python
# Regular function
def square(x):
    return x ** 2

# Lambda function
square_lambda = lambda x: x ** 2

# Both work the same way
print(square(5))           # 25
print(square_lambda(5))    # 25
```

### Lambda with Multiple Parameters
```python
# Lambda for addition
add = lambda a, b: a + b

# Lambda for finding maximum
max_val = lambda a, b: a if a > b else b

print(add(3, 4))      # 7
print(max_val(10, 5)) # 10
```

### Lambda with Built-in Functions
```python
# Using lambda with map
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x ** 2, numbers))
print(squares)  # [1, 4, 9, 16, 25]

# Using lambda with filter
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4]

# Using lambda with sorted
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
sorted_by_grade = sorted(students, key=lambda x: x[1], reverse=True)
print(sorted_by_grade)  # [('Bob', 92), ('Alice', 85), ('Charlie', 78)]
```

## Modules and Packages

### Creating a Module
```python
# File: math_utils.py
"""Mathematical utility functions"""

def factorial(n):
    """Calculate factorial of n"""
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def fibonacci(n):
    """Generate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

### Importing Modules
```python
# Import entire module
import math_utils

# Using functions from the module
print(math_utils.factorial(5))  # 120
print(math_utils.fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(math_utils.is_prime(17))   # True

# Import specific functions
from math_utils import factorial, fibonacci

print(factorial(6))  # 720
print(fibonacci(8))  # [0, 1, 1, 2, 3, 5, 8, 13]

# Import with alias
import math_utils as mu

print(mu.is_prime(23))  # True
```

### Built-in Modules
```python
# Math module
import math

print(math.pi)           # 3.141592653589793
print(math.sqrt(16))     # 4.0
print(math.sin(math.pi/2))  # 1.0

# Random module
import random

print(random.randint(1, 10))  # Random integer between 1 and 10
print(random.choice(['a', 'b', 'c']))  # Random choice from list

# Datetime module
import datetime

now = datetime.datetime.now()
print(now)  # Current date and time

# OS module
import os

print(os.getcwd())  # Current working directory
```

### Creating Packages
```python
# File structure:
# my_package/
#     __init__.py
#     math_functions.py
#     string_functions.py

# File: my_package/__init__.py
"""My custom package for mathematical and string operations"""

from .math_functions import add, multiply, power
from .string_functions import reverse_string, count_words

__version__ = "1.0.0"
__all__ = ['add', 'multiply', 'power', 'reverse_string', 'count_words']

# File: my_package/math_functions.py
"""Mathematical functions"""

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def power(a, b):
    return a ** b

# File: my_package/string_functions.py
"""String manipulation functions"""

def reverse_string(text):
    return text[::-1]

def count_words(text):
    return len(text.split())
```

## Mathematical Applications

### Numerical Methods
```python
def newton_raphson(f, f_prime, x0, tolerance=1e-6, max_iterations=100):
    """Find root using Newton-Raphson method"""
    x = x0
    for i in range(max_iterations):
        fx = f(x)
        if abs(fx) < tolerance:
            return x
        
        fpx = f_prime(x)
        if fpx == 0:
            raise ValueError("Derivative is zero")
        
        x = x - fx / fpx
    
    raise ValueError("Method did not converge")

# Example: Find root of x^2 - 2 = 0
def f(x):
    return x**2 - 2

def f_prime(x):
    return 2*x

root = newton_raphson(f, f_prime, 1.0)
print(f"Square root of 2: {root:.6f}")  # 1.414214
```

### Statistical Functions
```python
def mean(numbers):
    """Calculate arithmetic mean"""
    return sum(numbers) / len(numbers)

def variance(numbers):
    """Calculate variance"""
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)

def standard_deviation(numbers):
    """Calculate standard deviation"""
    return variance(numbers) ** 0.5

def correlation(x, y):
    """Calculate correlation coefficient"""
    if len(x) != len(y):
        raise ValueError("Lists must have same length")
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = (sum((x[i] - mean_x) ** 2 for i in range(n)) * 
                   sum((y[i] - mean_y) ** 2 for i in range(n))) ** 0.5
    
    return numerator / denominator if denominator != 0 else 0

# Example usage
data_x = [1, 2, 3, 4, 5]
data_y = [2, 4, 6, 8, 10]

print(f"Mean of x: {mean(data_x)}")
print(f"Standard deviation of x: {standard_deviation(data_x):.2f}")
print(f"Correlation: {correlation(data_x, data_y):.2f}")
```

### Linear Algebra Functions
```python
def dot_product(a, b):
    """Calculate dot product of two vectors"""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return sum(a[i] * b[i] for i in range(len(a)))

def matrix_multiply(a, b):
    """Multiply two matrices"""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    
    if cols_a != rows_b:
        raise ValueError("Cannot multiply matrices: dimensions don't match")
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    
    return result

def transpose(matrix):
    """Transpose a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Example usage
vector_a = [1, 2, 3]
vector_b = [4, 5, 6]
dot = dot_product(vector_a, vector_b)
print(f"Dot product: {dot}")  # 32

matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]
result_matrix = matrix_multiply(matrix_a, matrix_b)
print(f"Matrix multiplication result: {result_matrix}")
```

## Practice Problems

### Problem 1
Create a function that calculates the area of different geometric shapes.

```python
# Solution
import math

def calculate_area(shape, **kwargs):
    """Calculate area of different shapes"""
    if shape == "circle":
        radius = kwargs.get('radius', 0)
        return math.pi * radius ** 2
    
    elif shape == "rectangle":
        length = kwargs.get('length', 0)
        width = kwargs.get('width', 0)
        return length * width
    
    elif shape == "triangle":
        base = kwargs.get('base', 0)
        height = kwargs.get('height', 0)
        return 0.5 * base * height
    
    elif shape == "trapezoid":
        base1 = kwargs.get('base1', 0)
        base2 = kwargs.get('base2', 0)
        height = kwargs.get('height', 0)
        return 0.5 * (base1 + base2) * height
    
    else:
        raise ValueError(f"Unknown shape: {shape}")

# Example usage
circle_area = calculate_area("circle", radius=5)
rectangle_area = calculate_area("rectangle", length=10, width=6)
triangle_area = calculate_area("triangle", base=8, height=4)

print(f"Circle area: {circle_area:.2f}")
print(f"Rectangle area: {rectangle_area}")
print(f"Triangle area: {triangle_area}")
```

### Problem 2
Create a module for data analysis functions.

```python
# Solution - File: data_analysis.py
"""Data analysis utility functions"""

def describe_data(data):
    """Provide statistical description of data"""
    if not data:
        return None
    
    n = len(data)
    sorted_data = sorted(data)
    
    stats = {
        'count': n,
        'mean': sum(data) / n,
        'median': sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2,
        'min': min(data),
        'max': max(data),
        'range': max(data) - min(data)
    }
    
    # Standard deviation
    mean = stats['mean']
    variance = sum((x - mean) ** 2 for x in data) / n
    stats['std_dev'] = variance ** 0.5
    
    return stats

def find_outliers(data, method='iqr'):
    """Find outliers in data"""
    if method == 'iqr':
        sorted_data = sorted(data)
        q1_index = len(sorted_data) // 4
        q3_index = 3 * len(sorted_data) // 4
        
        q1 = sorted_data[q1_index]
        q3 = sorted_data[q3_index]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    elif method == 'zscore':
        stats = describe_data(data)
        mean = stats['mean']
        std_dev = stats['std_dev']
        
        outliers = [x for x in data if abs(x - mean) > 2 * std_dev]
        return outliers
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

# Example usage
sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # 100 is an outlier
description = describe_data(sample_data)
outliers = find_outliers(sample_data)

print("Data Description:")
for key, value in description.items():
    print(f"{key}: {value:.2f}")

print(f"Outliers: {outliers}")
```

### Problem 3
Create a function that implements the bisection method for finding roots.

```python
# Solution
def bisection_method(f, a, b, tolerance=1e-6, max_iterations=100):
    """Find root using bisection method"""
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    for i in range(max_iterations):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tolerance or (b - a) / 2 < tolerance:
            return c
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    raise ValueError("Method did not converge")

# Example: Find root of x^3 - x - 1 = 0
def cubic_function(x):
    return x**3 - x - 1

root = bisection_method(cubic_function, 1, 2)
print(f"Root of x^3 - x - 1 = 0: {root:.6f}")

# Verify the result
print(f"f({root:.6f}) = {cubic_function(root):.6f}")
```

## Key Takeaways
- Functions help organize code into reusable components
- Parameters can be positional, keyword, or variable
- Return values can be single or multiple
- Scope determines variable accessibility
- Lambda functions provide concise syntax for simple operations
- Modules and packages organize related functionality
- Functions are essential for mathematical and scientific computing

## Next Steps
In the next tutorial, we'll explore object-oriented programming, learning about classes, objects, inheritance, and how to design programs using OOP principles.
