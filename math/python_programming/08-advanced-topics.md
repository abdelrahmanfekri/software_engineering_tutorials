# Python Programming Tutorial 08: Advanced Topics

## Learning Objectives
By the end of this tutorial, you will be able to:
- Use decorators and closures effectively
- Work with generators and iterators
- Apply regular expressions for text processing
- Implement multithreading and multiprocessing
- Understand memory management concepts
- Apply performance optimization techniques
- Use advanced features for mathematical computing

## Decorators and Closures

### Understanding Closures
```python
def outer_function(x):
    """Outer function that creates a closure"""
    def inner_function(y):
        """Inner function that captures x from outer scope"""
        return x + y
    return inner_function

# Creating closures
add_five = outer_function(5)
add_ten = outer_function(10)

print(add_five(3))   # 8
print(add_ten(3))    # 13

# Closures with mutable objects
def create_counter():
    """Create a counter using closure"""
    count = [0]  # Using list to make it mutable
    
    def increment():
        count[0] += 1
        return count[0]
    
    def decrement():
        count[0] -= 1
        return count[0]
    
    def get_count():
        return count[0]
    
    return increment, decrement, get_count

# Using the counter
inc, dec, get = create_counter()
print(get())  # 0
print(inc())  # 1
print(inc())  # 2
print(dec())  # 1
print(get())  # 1
```

### Basic Decorators
```python
def my_decorator(func):
    """Simple decorator that adds functionality"""
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

@my_decorator
def greet(name):
    """Greet someone"""
    print(f"Hello, {name}!")
    return f"Greeted {name}"

# Using the decorated function
result = greet("Alice")
print(f"Result: {result}")
```

### Decorators with Parameters
```python
def repeat(times):
    """Decorator that repeats function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"Execution {i+1}/{times}")
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(3)
def calculate_square(x):
    """Calculate square of a number"""
    return x ** 2

# Using the decorator
results = calculate_square(5)
print(f"Results: {results}")
```

### Built-in Decorators
```python
class MathOperations:
    """Class demonstrating built-in decorators"""
    
    def __init__(self):
        self._value = 0
    
    @property
    def value(self):
        """Get the current value"""
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Set the value with validation"""
        if not isinstance(new_value, (int, float)):
            raise TypeError("Value must be a number")
        self._value = new_value
    
    @staticmethod
    def add(a, b):
        """Static method for addition"""
        return a + b
    
    @classmethod
    def from_string(cls, value_str):
        """Class method to create instance from string"""
        return cls(float(value_str))

# Using the class
math_ops = MathOperations()
math_ops.value = 42
print(f"Value: {math_ops.value}")

print(f"Static add: {MathOperations.add(5, 3)}")
new_ops = MathOperations.from_string("100")
print(f"From string: {new_ops.value}")
```

### Advanced Decorators
```python
import functools
import time

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def cache_decorator(func):
    """Simple caching decorator"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = str(args) + str(sorted(kwargs.items()))
        
        if key in cache:
            print(f"Cache hit for {func.__name__}")
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        print(f"Cache miss for {func.__name__}")
        return result
    
    return wrapper

@timing_decorator
@cache_decorator
def fibonacci(n):
    """Calculate Fibonacci number with caching"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the decorated function
print(f"fibonacci(10) = {fibonacci(10)}")
print(f"fibonacci(10) = {fibonacci(10)}")  # Should use cache
```

## Generators and Iterators

### Basic Generators
```python
def simple_generator():
    """Simple generator function"""
    yield 1
    yield 2
    yield 3

# Using the generator
gen = simple_generator()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3

# Using in a loop
for value in simple_generator():
    print(f"Value: {value}")
```

### Generator Expressions
```python
# Generator expression (similar to list comprehension)
squares_gen = (x**2 for x in range(10))
print(list(squares_gen))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Memory efficient for large datasets
large_gen = (x**2 for x in range(1000000))
print(f"Generator object: {large_gen}")
print(f"First few values: {[next(large_gen) for _ in range(5)]}")
```

### Advanced Generators
```python
def fibonacci_generator(n):
    """Generate Fibonacci sequence up to n terms"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

def prime_generator(limit):
    """Generate prime numbers up to limit"""
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    for num in range(2, limit):
        if is_prime(num):
            yield num

# Using generators
print("Fibonacci sequence:")
for fib in fibonacci_generator(10):
    print(fib, end=" ")
print()

print("\nPrime numbers up to 50:")
for prime in prime_generator(50):
    print(prime, end=" ")
print()
```

### Generator Methods
```python
def advanced_generator():
    """Generator with send, throw, and close methods"""
    value = yield "Started"
    while True:
        try:
            value = yield f"Received: {value}"
        except ValueError as e:
            yield f"Caught ValueError: {e}"
        except GeneratorExit:
            yield "Generator closing"
            break

# Using generator methods
gen = advanced_generator()
print(next(gen))  # "Started"

print(gen.send("Hello"))  # "Received: Hello"
print(gen.send("World"))  # "Received: World"

print(gen.throw(ValueError, "Test error"))  # "Caught ValueError: Test error"

gen.close()
```

## Regular Expressions

### Basic Pattern Matching
```python
import re

# Basic pattern matching
text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"

# Search for pattern
match = re.search(pattern, text)
if match:
    print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")

# Find all matches
matches = re.findall(r"\b\w{4}\b", text)  # Find all 4-letter words
print(f"4-letter words: {matches}")

# Split by pattern
words = re.split(r"\s+", text)  # Split by whitespace
print(f"Words: {words}")
```

### Advanced Pattern Matching
```python
import re

# Email validation
def validate_email(email):
    """Validate email address using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Phone number validation
def validate_phone(phone):
    """Validate phone number"""
    pattern = r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
    return re.match(pattern, phone) is not None

# Extract data from text
def extract_data(text):
    """Extract various data patterns from text"""
    data = {}
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    data['emails'] = re.findall(email_pattern, text)
    
    # Extract phone numbers
    phone_pattern = r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    data['phones'] = re.findall(phone_pattern, text)
    
    # Extract URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    data['urls'] = re.findall(url_pattern, text)
    
    return data

# Test the functions
test_text = """
Contact us at john@example.com or call (555) 123-4567.
Visit our website at https://www.example.com for more information.
You can also reach us at jane.doe@company.org or +1-800-555-0199.
"""

print("Email validation:")
print(f"john@example.com: {validate_email('john@example.com')}")
print(f"invalid-email: {validate_email('invalid-email')}")

print("\nPhone validation:")
print(f"(555) 123-4567: {validate_phone('(555) 123-4567')}")
print(f"123-456-7890: {validate_phone('123-456-7890')}")

print("\nExtracted data:")
extracted = extract_data(test_text)
for key, value in extracted.items():
    print(f"{key}: {value}")
```

### Text Processing with Regex
```python
import re

def clean_text(text):
    """Clean and normalize text using regex"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'[''']', "'", text)
    
    return text.strip()

def extract_numbers(text):
    """Extract all numbers from text"""
    # Find integers and floats
    pattern = r'-?\d+\.?\d*'
    numbers = re.findall(pattern, text)
    return [float(num) if '.' in num else int(num) for num in numbers]

def word_frequency(text):
    """Calculate word frequency using regex"""
    # Convert to lowercase and find words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count frequency
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    
    return frequency

# Test text processing
sample_text = """
The quick brown fox jumps over the lazy dog. The fox is very quick!
It jumps over the dog multiple times. The dog is quite lazy.
"""

print("Cleaned text:")
print(clean_text(sample_text))

print("\nExtracted numbers:")
numbers = extract_numbers("I have 5 apples and 3.14 pi")
print(numbers)

print("\nWord frequency:")
freq = word_frequency(sample_text)
for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {count}")
```

## Multithreading and Multiprocessing

### Basic Threading
```python
import threading
import time

def worker_function(name, duration):
    """Worker function for threading example"""
    print(f"Worker {name} starting")
    time.sleep(duration)
    print(f"Worker {name} finished")

# Create and start threads
threads = []
for i in range(3):
    thread = threading.Thread(target=worker_function, args=(f"Thread-{i}", 2))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All threads completed")
```

### Thread Synchronization
```python
import threading
import time

class ThreadSafeCounter:
    """Thread-safe counter using locks"""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        """Increment counter safely"""
        with self._lock:
            self._value += 1
    
    def get_value(self):
        """Get current value safely"""
        with self._lock:
            return self._value

def worker(counter, iterations):
    """Worker function that increments counter"""
    for _ in range(iterations):
        counter.increment()

# Test thread-safe counter
counter = ThreadSafeCounter()
threads = []

# Create multiple threads
for i in range(5):
    thread = threading.Thread(target=worker, args=(counter, 1000))
    threads.append(thread)
    thread.start()

# Wait for completion
for thread in threads:
    thread.join()

print(f"Final counter value: {counter.get_value()}")  # Should be 5000
```

### Multiprocessing
```python
import multiprocessing
import time

def cpu_intensive_task(n):
    """CPU-intensive task for multiprocessing"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def parallel_processing():
    """Demonstrate parallel processing"""
    # Sequential processing
    start_time = time.time()
    sequential_results = [cpu_intensive_task(1000000) for _ in range(4)]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        parallel_results = pool.map(cpu_intensive_task, [1000000] * 4)
    parallel_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")

# Run parallel processing example
if __name__ == "__main__":
    parallel_processing()
```

## Memory Management

### Memory Profiling
```python
import sys
import gc

def memory_usage():
    """Get current memory usage"""
    return sys.getsizeof(gc.get_objects())

def demonstrate_memory_usage():
    """Demonstrate memory usage patterns"""
    print(f"Initial memory usage: {memory_usage()} bytes")
    
    # Create large list
    large_list = list(range(1000000))
    print(f"After creating large list: {memory_usage()} bytes")
    
    # Delete reference
    del large_list
    print(f"After deleting reference: {memory_usage()} bytes")
    
    # Force garbage collection
    gc.collect()
    print(f"After garbage collection: {memory_usage()} bytes")

# Memory-efficient alternatives
def memory_efficient_range(n):
    """Memory-efficient range using generator"""
    for i in range(n):
        yield i

def demonstrate_memory_efficiency():
    """Compare memory usage of different approaches"""
    print("Memory usage comparison:")
    
    # List comprehension (uses more memory)
    start_mem = memory_usage()
    squares_list = [x**2 for x in range(100000)]
    list_mem = memory_usage() - start_mem
    
    # Generator expression (uses less memory)
    start_mem = memory_usage()
    squares_gen = (x**2 for x in range(100000))
    gen_mem = memory_usage() - start_mem
    
    print(f"List comprehension: {list_mem} bytes")
    print(f"Generator expression: {gen_mem} bytes")

# Run memory demonstrations
demonstrate_memory_usage()
demonstrate_memory_efficiency()
```

### Weak References
```python
import weakref
import gc

class Data:
    """Class for demonstrating weak references"""
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"Data({self.value})"

def demonstrate_weak_references():
    """Demonstrate weak references"""
    # Create object
    data = Data("important data")
    print(f"Created: {data}")
    
    # Create weak reference
    weak_ref = weakref.ref(data)
    print(f"Weak reference: {weak_ref()}")
    
    # Delete strong reference
    del data
    print(f"After deleting strong reference: {weak_ref()}")
    
    # Force garbage collection
    gc.collect()
    print(f"After garbage collection: {weak_ref()}")

# Run weak reference demonstration
demonstrate_weak_references()
```

## Performance Optimization

### Profiling and Timing
```python
import time
import cProfile
import pstats

def slow_function(n):
    """Slow function for profiling"""
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result

def fast_function(n):
    """Optimized version of slow function"""
    return sum(i * j for i in range(n) for j in range(n))

def profile_functions():
    """Profile function performance"""
    n = 100
    
    # Time slow function
    start_time = time.time()
    result1 = slow_function(n)
    slow_time = time.time() - start_time
    
    # Time fast function
    start_time = time.time()
    result2 = fast_function(n)
    fast_time = time.time() - start_time
    
    print(f"Slow function: {slow_time:.4f} seconds, result: {result1}")
    print(f"Fast function: {fast_time:.4f} seconds, result: {result2}")
    print(f"Speedup: {slow_time / fast_time:.2f}x")

def detailed_profiling():
    """Detailed profiling with cProfile"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run function
    slow_function(50)
    
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Run profiling
profile_functions()
print("\nDetailed profiling:")
detailed_profiling()
```

### Optimization Techniques
```python
import numpy as np
from functools import lru_cache

# Memoization with lru_cache
@lru_cache(maxsize=128)
def fibonacci_cached(n):
    """Fibonacci with memoization"""
    if n < 2:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

# Vectorized operations with NumPy
def vectorized_operations():
    """Demonstrate vectorized operations"""
    # Create large arrays
    size = 1000000
    a = np.random.random(size)
    b = np.random.random(size)
    
    # Vectorized operations (fast)
    start_time = time.time()
    result_vectorized = a + b
    vectorized_time = time.time() - start_time
    
    # Python loops (slow)
    start_time = time.time()
    result_loop = [a[i] + b[i] for i in range(size)]
    loop_time = time.time() - start_time
    
    print(f"Vectorized time: {vectorized_time:.4f} seconds")
    print(f"Loop time: {loop_time:.4f} seconds")
    print(f"Speedup: {loop_time / vectorized_time:.2f}x")

# List comprehensions vs loops
def comprehension_vs_loop():
    """Compare list comprehensions with loops"""
    data = list(range(10000))
    
    # List comprehension
    start_time = time.time()
    squares_comp = [x**2 for x in data]
    comp_time = time.time() - start_time
    
    # Traditional loop
    start_time = time.time()
    squares_loop = []
    for x in data:
        squares_loop.append(x**2)
    loop_time = time.time() - start_time
    
    print(f"List comprehension: {comp_time:.4f} seconds")
    print(f"Traditional loop: {loop_time:.4f} seconds")
    print(f"Speedup: {loop_time / comp_time:.2f}x")

# Run optimization demonstrations
print("Fibonacci with caching:")
print(f"fibonacci_cached(40) = {fibonacci_cached(40)}")

print("\nVectorized operations:")
vectorized_operations()

print("\nList comprehension vs loop:")
comprehension_vs_loop()
```

## Mathematical Applications

### Advanced Mathematical Computing
```python
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

class MathematicalSolver:
    """Advanced mathematical solver using optimization techniques"""
    
    def __init__(self):
        self.results = {}
    
    def solve_equation(self, func, initial_guess):
        """Solve equation using numerical methods"""
        try:
            result = optimize.fsolve(func, initial_guess)
            return result[0] if len(result) == 1 else result
        except Exception as e:
            print(f"Error solving equation: {e}")
            return None
    
    def minimize_function(self, func, bounds=None):
        """Minimize function using optimization"""
        try:
            if bounds:
                result = optimize.minimize_scalar(func, bounds=bounds)
            else:
                result = optimize.minimize_scalar(func)
            return result.x, result.fun
        except Exception as e:
            print(f"Error minimizing function: {e}")
            return None, None
    
    def integrate_function(self, func, a, b):
        """Integrate function numerically"""
        try:
            result, error = integrate.quad(func, a, b)
            return result, error
        except Exception as e:
            print(f"Error integrating function: {e}")
            return None, None
    
    def solve_system(self, equations, initial_guess):
        """Solve system of equations"""
        try:
            def system_func(vars):
                return [eq(*vars) for eq in equations]
            
            result = optimize.fsolve(system_func, initial_guess)
            return result
        except Exception as e:
            print(f"Error solving system: {e}")
            return None

# Example usage
solver = MathematicalSolver()

# Solve equation: x^2 - 4 = 0
def equation(x):
    return x**2 - 4

root = solver.solve_equation(equation, 1.0)
print(f"Root of x^2 - 4 = 0: {root}")

# Minimize function: x^2 - 4x + 3
def quadratic(x):
    return x**2 - 4*x + 3

min_x, min_val = solver.minimize_function(quadratic)
print(f"Minimum of x^2 - 4x + 3: x={min_x}, f(x)={min_val}")

# Integrate function: x^2
def integrand(x):
    return x**2

integral, error = solver.integrate_function(integrand, 0, 2)
print(f"Integral of x^2 from 0 to 2: {integral} (error: {error})")
```

### Parallel Mathematical Computing
```python
import multiprocessing
import numpy as np

def parallel_matrix_operations():
    """Demonstrate parallel matrix operations"""
    
    def matrix_multiply_chunk(args):
        """Multiply matrix chunk"""
        matrix_a, matrix_b, start_row, end_row = args
        result_chunk = np.zeros((end_row - start_row, matrix_b.shape[1]))
        
        for i in range(start_row, end_row):
            for j in range(matrix_b.shape[1]):
                for k in range(matrix_a.shape[1]):
                    result_chunk[i - start_row, j] += matrix_a[i, k] * matrix_b[k, j]
        
        return result_chunk, start_row, end_row
    
    # Create matrices
    size = 500
    matrix_a = np.random.random((size, size))
    matrix_b = np.random.random((size, size))
    
    # Sequential multiplication
    start_time = time.time()
    result_seq = np.dot(matrix_a, matrix_b)
    seq_time = time.time() - start_time
    
    # Parallel multiplication
    num_processes = multiprocessing.cpu_count()
    chunk_size = size // num_processes
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(matrix_a, matrix_b, i*chunk_size, (i+1)*chunk_size) 
                for i in range(num_processes)]
        chunks = pool.map(matrix_multiply_chunk, args)
    
    # Combine results
    result_par = np.zeros((size, size))
    for chunk, start_row, end_row in chunks:
        result_par[start_row:end_row, :] = chunk
    
    par_time = time.time() - start_time
    
    print(f"Sequential time: {seq_time:.4f} seconds")
    print(f"Parallel time: {par_time:.4f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    print(f"Results match: {np.allclose(result_seq, result_par)}")

# Run parallel matrix operations
parallel_matrix_operations()
```

## Practice Problems

### Problem 1
Create a decorator-based caching system for expensive mathematical computations.

```python
# Solution
import functools
import time
import hashlib
import json

class MathCache:
    """Advanced caching system for mathematical computations"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, func_name, args, kwargs):
        """Generate cache key from function arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None
    
    def set(self, key, value):
        """Set value in cache with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# Global cache instance
math_cache = MathCache()

def cached_computation(func):
    """Decorator for caching mathematical computations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        key = math_cache._generate_key(func.__name__, args, kwargs)
        
        # Check cache
        cached_result = math_cache.get(key)
        if cached_result is not None:
            print(f"Cache hit for {func.__name__}")
            return cached_result
        
        # Compute result
        print(f"Cache miss for {func.__name__}")
        result = func(*args, **kwargs)
        
        # Store in cache
        math_cache.set(key, result)
        
        return result
    
    return wrapper

# Example usage
@cached_computation
def expensive_fibonacci(n):
    """Expensive Fibonacci calculation"""
    if n < 2:
        return n
    return expensive_fibonacci(n-1) + expensive_fibonacci(n-2)

@cached_computation
def expensive_factorial(n):
    """Expensive factorial calculation"""
    if n <= 1:
        return 1
    return n * expensive_factorial(n-1)

# Test caching
print("Testing cached computations:")
print(f"fibonacci(10) = {expensive_fibonacci(10)}")
print(f"fibonacci(10) = {expensive_fibonacci(10)}")  # Should use cache
print(f"factorial(20) = {expensive_factorial(20)}")
print(f"factorial(20) = {expensive_factorial(20)}")  # Should use cache

print("\nCache statistics:")
stats = math_cache.get_stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

### Problem 2
Create a parallel data processing pipeline using generators and multiprocessing.

```python
# Solution
import multiprocessing
import time
from queue import Queue
import threading

class DataProcessor:
    """Parallel data processing pipeline"""
    
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.results = []
    
    def data_generator(self, data_source):
        """Generate data for processing"""
        for item in data_source:
            yield item
    
    def process_item(self, item):
        """Process a single data item"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Example processing: calculate statistics
        if isinstance(item, list):
            return {
                'sum': sum(item),
                'mean': sum(item) / len(item),
                'max': max(item),
                'min': min(item),
                'count': len(item)
            }
        else:
            return {'value': item, 'processed': True}
    
    def worker_process(self, input_queue, output_queue):
        """Worker process function"""
        while True:
            try:
                item = input_queue.get(timeout=1)
                if item is None:  # Sentinel value to stop
                    break
                
                result = self.process_item(item)
                output_queue.put(result)
                input_queue.task_done()
            except:
                break
    
    def process_data(self, data_source):
        """Process data using parallel workers"""
        # Start worker processes
        for _ in range(self.num_workers):
            worker = multiprocessing.Process(
                target=self.worker_process,
                args=(self.input_queue, self.output_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        # Add data to input queue
        for item in self.data_generator(data_source):
            self.input_queue.put(item)
        
        # Collect results
        results = []
        for _ in range(len(list(data_source))):
            try:
                result = self.output_queue.get(timeout=5)
                results.append(result)
            except:
                break
        
        # Stop workers
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        for worker in self.workers:
            worker.join()
        
        return results

# Test the data processor
def test_parallel_processing():
    """Test parallel data processing"""
    # Generate test data
    test_data = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300],
        [5, 10, 15, 20, 25, 30],
        [1, 1, 2, 3, 5, 8, 13]
    ]
    
    # Process data
    processor = DataProcessor(num_workers=2)
    results = processor.process_data(test_data)
    
    print("Processing results:")
    for i, result in enumerate(results):
        print(f"Data {i+1}: {result}")

# Run the test
test_parallel_processing()
```

### Problem 3
Create an advanced mathematical expression parser and evaluator using decorators and generators.

```python
# Solution
import re
import math
from functools import wraps

class MathExpressionParser:
    """Advanced mathematical expression parser and evaluator"""
    
    def __init__(self):
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'factorial': math.factorial
        }
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau
        }
    
    def validate_expression(self, expression):
        """Validate mathematical expression"""
        # Check for dangerous patterns
        dangerous = ['import', 'exec', 'eval', '__', 'open', 'file']
        for pattern in dangerous:
            if pattern in expression.lower():
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        # Check parentheses balance
        if expression.count('(') != expression.count(')'):
            raise ValueError("Unbalanced parentheses")
        
        return True
    
    def tokenize(self, expression):
        """Tokenize mathematical expression"""
        # Regular expression for tokens
        token_pattern = r'''
            \d+\.?\d*          # Numbers
            |[a-zA-Z_]\w*      # Variables and functions
            |[+\-*/^()]        # Operators and parentheses
            |\s+               # Whitespace
        '''
        
        tokens = re.findall(token_pattern, expression, re.VERBOSE)
        return [token.strip() for token in tokens if token.strip()]
    
    def parse_expression(self, tokens):
        """Parse tokens into mathematical expression"""
        # Simple recursive descent parser
        def parse_term():
            if not tokens:
                raise ValueError("Unexpected end of expression")
            
            token = tokens.pop(0)
            
            if token.isdigit() or '.' in token:
                return float(token)
            elif token in self.constants:
                return self.constants[token]
            elif token in self.functions:
                if tokens and tokens[0] == '(':
                    tokens.pop(0)  # Remove '('
                    arg = parse_expression(tokens)
                    if tokens and tokens[0] == ')':
                        tokens.pop(0)  # Remove ')'
                    else:
                        raise ValueError("Missing closing parenthesis")
                    return self.functions[token](arg)
                else:
                    raise ValueError(f"Function {token} requires parentheses")
            elif token == '(':
                result = parse_expression(tokens)
                if tokens and tokens[0] == ')':
                    tokens.pop(0)
                else:
                    raise ValueError("Missing closing parenthesis")
                return result
            else:
                raise ValueError(f"Unknown token: {token}")
        
        # Parse addition and subtraction
        result = parse_term()
        while tokens and tokens[0] in '+-':
            op = tokens.pop(0)
            term = parse_term()
            if op == '+':
                result += term
            else:
                result -= term
        
        return result
    
    def evaluate(self, expression, variables=None):
        """Evaluate mathematical expression"""
        try:
            # Validate expression
            self.validate_expression(expression)
            
            # Add variables to constants
            if variables:
                self.constants.update(variables)
            
            # Tokenize and parse
            tokens = self.tokenize(expression)
            result = self.parse_expression(tokens)
            
            # Check for remaining tokens
            if tokens:
                raise ValueError(f"Unexpected tokens: {tokens}")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

# Decorator for expression evaluation
def math_expression(func):
    """Decorator for mathematical expression evaluation"""
    parser = MathExpressionParser()
    
    @wraps(func)
    def wrapper(expression, *args, **kwargs):
        try:
            result = parser.evaluate(expression, kwargs)
            return func(expression, result, *args)
        except Exception as e:
            return func(expression, None, error=str(e), *args)
    
    return wrapper

# Generator for expression evaluation
def expression_generator(expressions, variables=None):
    """Generator for evaluating multiple expressions"""
    parser = MathExpressionParser()
    
    for expression in expressions:
        try:
            result = parser.evaluate(expression, variables)
            yield {'expression': expression, 'result': result, 'error': None}
        except Exception as e:
            yield {'expression': expression, 'result': None, 'error': str(e)}

# Example usage
@math_expression
def evaluate_expression(expr, result, error=None):
    """Evaluate and display expression result"""
    if error:
        print(f"Error in '{expr}': {error}")
    else:
        print(f"'{expr}' = {result}")

# Test expressions
test_expressions = [
    "2 + 3 * 4",
    "sin(pi/2)",
    "sqrt(16)",
    "log(e)",
    "factorial(5)",
    "2^3",
    "invalid_function(1)"
]

print("Expression evaluation:")
for expr in test_expressions:
    evaluate_expression(expr)

print("\nGenerator-based evaluation:")
expressions_gen = expression_generator(test_expressions)
for result in expressions_gen:
    if result['error']:
        print(f"Error: {result['expression']} -> {result['error']}")
    else:
        print(f"Success: {result['expression']} = {result['result']}")
```

## Key Takeaways
- Decorators provide powerful ways to modify function behavior
- Generators enable memory-efficient processing of large datasets
- Regular expressions are essential for text processing and validation
- Multithreading and multiprocessing enable parallel computation
- Memory management is crucial for performance optimization
- Advanced techniques can significantly improve mathematical computing performance
- Always profile before optimizing to identify real bottlenecks

## Conclusion
This tutorial covered advanced Python topics that are essential for sophisticated programming and mathematical computing. These concepts enable you to write more efficient, maintainable, and powerful Python programs. Practice these techniques with real-world problems to master them effectively.
