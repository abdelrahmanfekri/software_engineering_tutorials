# Python Programming Tutorial 02: Control Structures

## Learning Objectives
By the end of this tutorial, you will be able to:
- Use conditional statements (if, elif, else)
- Implement different types of loops (for, while)
- Control loop execution (break, continue, pass)
- Work with nested structures
- Use list comprehensions
- Apply control structures to solve problems

## Conditional Statements

### Basic If Statement
```python
# Simple if statement
age = 18
if age >= 18:
    print("You are an adult")
```

### If-Else Statement
```python
# If-else statement
score = 85
if score >= 90:
    grade = "A"
else:
    grade = "B"
print(f"Your grade is {grade}")
```

### If-Elif-Else Statement
```python
# Multiple conditions
score = 75
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Your grade is {grade}")
```

### Nested Conditionals
```python
# Nested if statements
age = 20
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need a license to drive")
else:
    print("You are too young to drive")
```

### Logical Operators
```python
# Using logical operators
age = 25
income = 50000
has_job = True

if age >= 18 and income >= 30000:
    print("You qualify for the loan")
elif age >= 18 or has_job:
    print("You might qualify with additional requirements")
else:
    print("You don't qualify")
```

## For Loops

### Basic For Loop
```python
# Iterating through a list
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)
```

### For Loop with Range
```python
# Using range function
for i in range(5):
    print(f"Number: {i}")

# Range with start and stop
for i in range(2, 8):
    print(f"Number: {i}")

# Range with step
for i in range(0, 10, 2):
    print(f"Even number: {i}")
```

### For Loop with Enumerate
```python
# Getting both index and value
fruits = ["apple", "banana", "orange"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

### For Loop with Dictionary
```python
# Iterating through dictionary
student_grades = {"Alice": 85, "Bob": 92, "Charlie": 78}
for name, grade in student_grades.items():
    print(f"{name}: {grade}")
```

## While Loops

### Basic While Loop
```python
# Simple while loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1
```

### While Loop with User Input
```python
# Getting user input until condition is met
total = 0
while True:
    number = int(input("Enter a number (0 to stop): "))
    if number == 0:
        break
    total += number
print(f"Total: {total}")
```

### While Loop with Condition
```python
# Guessing game
import random
secret_number = random.randint(1, 100)
guesses = 0
max_guesses = 7

while guesses < max_guesses:
    guess = int(input("Guess a number between 1 and 100: "))
    guesses += 1
    
    if guess == secret_number:
        print(f"Congratulations! You guessed it in {guesses} tries.")
        break
    elif guess < secret_number:
        print("Too low!")
    else:
        print("Too high!")
else:
    print(f"Sorry, the number was {secret_number}")
```

## Loop Control Statements

### Break Statement
```python
# Breaking out of loop
for i in range(10):
    if i == 5:
        break
    print(i)
# Output: 0, 1, 2, 3, 4
```

### Continue Statement
```python
# Skipping current iteration
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)
# Output: 1, 3, 5, 7, 9
```

### Pass Statement
```python
# Placeholder for future code
for i in range(5):
    if i == 2:
        pass  # Do nothing
    else:
        print(i)
# Output: 0, 1, 3, 4
```

## Nested Loops

### Nested For Loops
```python
# Multiplication table
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i} × {j} = {i * j}", end="\t")
    print()  # New line after each row
```

### Nested While Loops
```python
# Pattern printing
row = 1
while row <= 5:
    col = 1
    while col <= row:
        print("*", end="")
        col += 1
    print()  # New line
    row += 1
```

## List Comprehensions

### Basic List Comprehension
```python
# Creating list of squares
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### List Comprehension with Condition
```python
# Even squares only
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
```

### Nested List Comprehension
```python
# Matrix creation
matrix = [[i + j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

## Mathematical Applications

### Finding Prime Numbers
```python
# Check if number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Find all primes up to 50
primes = [n for n in range(2, 51) if is_prime(n)]
print(primes)
```

### Fibonacci Sequence
```python
# Generate Fibonacci sequence
def fibonacci(n):
    fib_sequence = []
    a, b = 0, 1
    while len(fib_sequence) < n:
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Factorial Calculation
```python
# Calculate factorial
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(5))  # 120
```

## Practice Problems

### Problem 1
Write a program that finds the largest number in a list.

```python
# Solution
numbers = [3, 7, 2, 9, 1, 5]
largest = numbers[0]
for num in numbers:
    if num > largest:
        largest = num
print(f"Largest number: {largest}")
```

### Problem 2
Create a program that counts vowels in a string.

```python
# Solution
text = "Hello, World!"
vowels = "aeiouAEIOU"
count = 0
for char in text:
    if char in vowels:
        count += 1
print(f"Number of vowels: {count}")
```

### Problem 3
Write a program that prints a right triangle pattern.

```python
# Solution
height = 5
for i in range(1, height + 1):
    for j in range(i):
        print("*", end="")
    print()
```

### Problem 4
Create a simple calculator using loops.

```python
# Solution
while True:
    print("\nCalculator Menu:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Exit")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "5":
        print("Goodbye!")
        break
    elif choice in ["1", "2", "3", "4"]:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        
        if choice == "1":
            result = num1 + num2
        elif choice == "2":
            result = num1 - num2
        elif choice == "3":
            result = num1 * num2
        elif choice == "4":
            if num2 != 0:
                result = num1 / num2
            else:
                print("Error: Division by zero!")
                continue
        
        print(f"Result: {result}")
    else:
        print("Invalid choice!")
```

## Key Takeaways
- Conditional statements control program flow based on conditions
- For loops iterate over sequences or ranges
- While loops repeat until a condition is met
- Break, continue, and pass control loop execution
- Nested structures allow complex logic
- List comprehensions provide concise ways to create lists
- Control structures are essential for problem-solving

## Next Steps
In the next tutorial, we'll explore data structures in Python, learning about lists, dictionaries, sets, and tuples, and how to use them effectively in mathematical and scientific applications.
