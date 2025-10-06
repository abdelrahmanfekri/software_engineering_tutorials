# Python Programming Tutorial 03: Data Structures

## Learning Objectives
By the end of this tutorial, you will be able to:
- Work with lists and list methods effectively
- Use tuples and understand their immutability
- Manipulate dictionaries and their methods
- Understand sets and set operations
- Work with strings and string methods
- Choose appropriate data structures for different problems
- Apply data structures to mathematical problems

## Lists

### Creating Lists
```python
# Different ways to create lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []
from_range = list(range(10))
from_string = list("Python")

print(numbers)  # [1, 2, 3, 4, 5]
print(mixed)    # [1, 'hello', 3.14, True]
print(empty)    # []
print(from_range)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(from_string)  # ['P', 'y', 't', 'h', 'o', 'n']
```

### List Indexing and Slicing
```python
numbers = [10, 20, 30, 40, 50]

# Indexing
print(numbers[0])   # 10 (first element)
print(numbers[-1])  # 50 (last element)
print(numbers[2])   # 30 (third element)

# Slicing
print(numbers[1:4])    # [20, 30, 40]
print(numbers[:3])     # [10, 20, 30]
print(numbers[2:])     # [30, 40, 50]
print(numbers[::2])    # [10, 30, 50] (every second element)
print(numbers[::-1])   # [50, 40, 30, 20, 10] (reverse)
```

### List Methods
```python
fruits = ["apple", "banana"]

# Adding elements
fruits.append("orange")        # Add to end
fruits.insert(1, "grape")      # Insert at index
fruits.extend(["kiwi", "mango"])  # Add multiple elements

print(fruits)  # ['apple', 'grape', 'banana', 'orange', 'kiwi', 'mango']

# Removing elements
fruits.remove("grape")          # Remove first occurrence
last_fruit = fruits.pop()       # Remove and return last element
second_fruit = fruits.pop(1)    # Remove and return element at index

print(fruits)  # ['apple', 'banana', 'orange']

# Other useful methods
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(len(numbers))             # 8 (length)
print(numbers.count(1))         # 2 (count occurrences)
print(numbers.index(4))        # 2 (find index)
numbers.sort()                  # Sort in place
print(numbers)                  # [1, 1, 2, 3, 4, 5, 6, 9]
numbers.reverse()               # Reverse in place
print(numbers)                  # [9, 6, 5, 4, 3, 2, 1, 1]
```

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Nested list comprehension
matrix = [[i + j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

# Multiple variables
coordinates = [(x, y) for x in range(3) for y in range(3)]
print(coordinates)  # [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
```

## Tuples

### Creating Tuples
```python
# Different ways to create tuples
coordinates = (3, 4)
single_element = (42,)  # Note the comma
empty_tuple = ()
from_list = tuple([1, 2, 3])
from_string = tuple("hello")

print(coordinates)    # (3, 4)
print(single_element) # (42,)
print(empty_tuple)    # ()
print(from_list)      # (1, 2, 3)
print(from_string)    # ('h', 'e', 'l', 'l', 'o')
```

### Tuple Operations
```python
point = (3, 4)
x, y = point  # Unpacking

print(f"x = {x}, y = {y}")  # x = 3, y = 4

# Tuple methods
numbers = (1, 2, 3, 2, 4, 2)
print(numbers.count(2))    # 3
print(numbers.index(3))   # 2

# Concatenation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2
print(combined)  # (1, 2, 3, 4, 5, 6)

# Repetition
repeated = tuple1 * 3
print(repeated)  # (1, 2, 3, 1, 2, 3, 1, 2, 3)
```

### When to Use Tuples
```python
# Use tuples for:
# 1. Coordinates
point = (10, 20)

# 2. RGB colors
color = (255, 128, 0)

# 3. Database records (immutable)
student = ("Alice", 20, "Mathematics", 3.8)

# 4. Function return values
def get_stats(numbers):
    return (min(numbers), max(numbers), sum(numbers) / len(numbers))

min_val, max_val, avg_val = get_stats([1, 2, 3, 4, 5])
print(f"Min: {min_val}, Max: {max_val}, Avg: {avg_val}")
```

## Dictionaries

### Creating Dictionaries
```python
# Different ways to create dictionaries
student = {"name": "Alice", "age": 20, "major": "Mathematics"}
empty_dict = {}
from_pairs = dict([("a", 1), ("b", 2), ("c", 3)])
from_keys = dict.fromkeys(["x", "y", "z"], 0)

print(student)      # {'name': 'Alice', 'age': 20, 'major': 'Mathematics'}
print(empty_dict)   # {}
print(from_pairs)   # {'a': 1, 'b': 2, 'c': 3}
print(from_keys)    # {'x': 0, 'y': 0, 'z': 0}
```

### Dictionary Operations
```python
grades = {"Alice": 85, "Bob": 92, "Charlie": 78}

# Accessing values
print(grades["Alice"])        # 85
print(grades.get("Bob"))      # 92
print(grades.get("David", 0)) # 0 (default if key doesn't exist)

# Adding/updating values
grades["David"] = 88          # Add new key-value pair
grades["Alice"] = 90          # Update existing value

# Removing values
del grades["Charlie"]         # Remove key-value pair
removed_grade = grades.pop("Bob")  # Remove and return value

print(grades)  # {'Alice': 90, 'David': 88}
print(removed_grade)  # 92
```

### Dictionary Methods
```python
student = {"name": "Alice", "age": 20, "major": "Mathematics", "gpa": 3.8}

# Keys, values, and items
print(list(student.keys()))    # ['name', 'age', 'major', 'gpa']
print(list(student.values()))  # ['Alice', 20, 'Mathematics', 3.8]
print(list(student.items()))   # [('name', 'Alice'), ('age', 20), ('major', 'Mathematics'), ('gpa', 3.8)]

# Iterating through dictionary
for key in student:
    print(f"{key}: {student[key]}")

for key, value in student.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

## Sets

### Creating Sets
```python
# Different ways to create sets
numbers = {1, 2, 3, 4, 5}
empty_set = set()  # Note: {} creates empty dict, not set
from_list = set([1, 2, 3, 2, 1])  # Removes duplicates
from_string = set("hello")

print(numbers)     # {1, 2, 3, 4, 5}
print(empty_set)   # set()
print(from_list)   # {1, 2, 3}
print(from_string) # {'h', 'e', 'l', 'o'}
```

### Set Operations
```python
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union
union = set1 | set2
print(union)  # {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection
intersection = set1 & set2
print(intersection)  # {4, 5}

# Difference
difference = set1 - set2
print(difference)  # {1, 2, 3}

# Symmetric difference
symmetric_diff = set1 ^ set2
print(symmetric_diff)  # {1, 2, 3, 6, 7, 8}

# Set methods
set1.add(6)           # Add element
set1.remove(1)        # Remove element (raises error if not found)
set1.discard(10)      # Remove element (no error if not found)
set1.update([7, 8])   # Add multiple elements

print(set1)  # {2, 3, 4, 5, 6, 7, 8}
```

### Set Applications
```python
# Finding unique elements
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_numbers = set(numbers)
print(unique_numbers)  # {1, 2, 3, 4}

# Checking membership
primes = {2, 3, 5, 7, 11, 13, 17, 19}
print(7 in primes)    # True
print(8 in primes)    # False

# Set operations for data analysis
students_math = {"Alice", "Bob", "Charlie", "David"}
students_physics = {"Bob", "Charlie", "Eve", "Frank"}

# Students taking both subjects
both_subjects = students_math & students_physics
print(both_subjects)  # {'Bob', 'Charlie'}

# Students taking only math
only_math = students_math - students_physics
print(only_math)  # {'Alice', 'David'}

# All students
all_students = students_math | students_physics
print(all_students)  # {'Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'}
```

## Strings

### String Creation and Basic Operations
```python
# String creation
name = "Alice"
message = 'Hello, World!'
multiline = """This is a
multiline string"""

# String concatenation
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  # John Doe

# String repetition
separator = "-" * 20
print(separator)  # --------------------
```

### String Methods
```python
text = "  Hello, World!  "

# Case methods
print(text.upper())        # "  HELLO, WORLD!  "
print(text.lower())        # "  hello, world!  "
print(text.title())        # "  Hello, World!  "
print(text.capitalize())   # "  hello, world!  "

# Whitespace methods
print(text.strip())        # "Hello, World!"
print(text.lstrip())       # "Hello, World!  "
print(text.rstrip())       # "  Hello, World!"

# Search methods
print(text.find("World"))  # 8
print(text.count("l"))     # 3
print(text.startswith("Hello"))  # False (due to leading spaces)
print(text.endswith("!"))  # False (due to trailing spaces)

# Replace and split
new_text = text.strip().replace("World", "Python")
print(new_text)  # "Hello, Python!"

words = new_text.split(", ")
print(words)  # ['Hello', 'Python!']
```

### String Formatting
```python
name = "Alice"
age = 25
gpa = 3.8

# f-strings (Python 3.6+)
message = f"My name is {name}, I'm {age} years old, and my GPA is {gpa:.1f}"
print(message)

# .format() method
message = "My name is {}, I'm {} years old, and my GPA is {:.1f}".format(name, age, gpa)
print(message)

# % formatting (older style)
message = "My name is %s, I'm %d years old, and my GPA is %.1f" % (name, age, gpa)
print(message)

# String methods for formatting
text = "hello world"
print(text.center(20))     # "    hello world     "
print(text.ljust(15))      # "hello world    "
print(text.rjust(15))      # "    hello world"
```

## Mathematical Applications

### Working with Matrices
```python
# Representing matrices as lists of lists
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Accessing elements
print(matrix[1][2])  # 6 (row 1, column 2)

# Matrix operations
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

# Example usage
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = matrix_multiply(A, B)
print(C)  # [[19, 22], [43, 50]]
```

### Statistical Operations
```python
def calculate_statistics(numbers):
    """Calculate basic statistics"""
    if not numbers:
        return None
    
    # Sort for median calculation
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # Mean
    mean = sum(numbers) / n
    
    # Median
    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]
    
    # Mode (most frequent)
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    mode = max(frequency, key=frequency.get)
    
    # Standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'std_dev': std_dev,
        'min': min(numbers),
        'max': max(numbers)
    }

# Example usage
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
stats = calculate_statistics(data)
print(stats)
```

### Data Processing
```python
# Processing student grades
student_data = [
    {"name": "Alice", "grades": [85, 90, 78, 92]},
    {"name": "Bob", "grades": [78, 82, 88, 85]},
    {"name": "Charlie", "grades": [92, 88, 95, 90]},
    {"name": "David", "grades": [75, 80, 82, 78]}
]

# Calculate average grades
for student in student_data:
    grades = student["grades"]
    average = sum(grades) / len(grades)
    student["average"] = round(average, 2)

# Sort by average grade
student_data.sort(key=lambda x: x["average"], reverse=True)

# Display results
print("Student Rankings:")
for i, student in enumerate(student_data, 1):
    print(f"{i}. {student['name']}: {student['average']}")

# Find students above class average
all_grades = [grade for student in student_data for grade in student["grades"]]
class_average = sum(all_grades) / len(all_grades)

above_average = [student for student in student_data if student["average"] > class_average]
print(f"\nStudents above class average ({class_average:.2f}):")
for student in above_average:
    print(f"- {student['name']}: {student['average']}")
```

## Practice Problems

### Problem 1
Create a program that finds the most frequent word in a text.

```python
# Solution
def most_frequent_word(text):
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Count word frequencies
    word_count = {}
    for word in words:
        # Remove punctuation
        word = word.strip(".,!?;:")
        word_count[word] = word_count.get(word, 0) + 1
    
    # Find most frequent word
    most_frequent = max(word_count, key=word_count.get)
    return most_frequent, word_count[most_frequent]

# Example usage
text = "The quick brown fox jumps over the lazy dog. The fox is quick."
word, count = most_frequent_word(text)
print(f"Most frequent word: '{word}' (appears {count} times)")
```

### Problem 2
Implement a simple inventory management system.

```python
# Solution
class Inventory:
    def __init__(self):
        self.items = {}
    
    def add_item(self, name, quantity, price):
        if name in self.items:
            self.items[name]["quantity"] += quantity
        else:
            self.items[name] = {"quantity": quantity, "price": price}
    
    def remove_item(self, name, quantity):
        if name in self.items:
            self.items[name]["quantity"] -= quantity
            if self.items[name]["quantity"] <= 0:
                del self.items[name]
    
    def get_total_value(self):
        total = 0
        for item in self.items.values():
            total += item["quantity"] * item["price"]
        return total
    
    def display_inventory(self):
        print("Current Inventory:")
        for name, details in self.items.items():
            print(f"{name}: {details['quantity']} units @ ${details['price']:.2f} each")

# Example usage
inventory = Inventory()
inventory.add_item("Laptop", 5, 999.99)
inventory.add_item("Mouse", 20, 25.50)
inventory.add_item("Keyboard", 15, 75.00)

inventory.display_inventory()
print(f"Total inventory value: ${inventory.get_total_value():.2f}")
```

### Problem 3
Create a program that analyzes text statistics.

```python
# Solution
def analyze_text(text):
    # Basic statistics
    characters = len(text)
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    # Character frequency
    char_freq = {}
    for char in text.lower():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Most common character
    most_common_char = max(char_freq, key=char_freq.get) if char_freq else None
    
    # Word length analysis
    word_lengths = [len(word.strip(".,!?;:")) for word in text.split()]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    return {
        'characters': characters,
        'words': words,
        'sentences': sentences,
        'most_common_char': most_common_char,
        'avg_word_length': avg_word_length,
        'char_frequency': char_freq
    }

# Example usage
sample_text = "The quick brown fox jumps over the lazy dog. The fox is very quick!"
analysis = analyze_text(sample_text)
print(f"Characters: {analysis['characters']}")
print(f"Words: {analysis['words']}")
print(f"Sentences: {analysis['sentences']}")
print(f"Most common character: '{analysis['most_common_char']}'")
print(f"Average word length: {analysis['avg_word_length']:.2f}")
```

## Key Takeaways
- Lists are mutable and versatile for most data storage needs
- Tuples are immutable and useful for fixed data like coordinates
- Dictionaries provide key-value mapping for structured data
- Sets are useful for unique elements and set operations
- Strings have many built-in methods for text processing
- Choose the right data structure based on your specific needs
- Data structures are fundamental for mathematical and scientific computing

## Next Steps
In the next tutorial, we'll explore functions and modules, learning how to organize code into reusable components and work with Python's module system.
