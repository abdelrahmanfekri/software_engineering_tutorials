# Python Programming Tutorial 07: Error Handling and Debugging

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand different types of exceptions
- Use try-except blocks effectively
- Raise and create custom exceptions
- Implement proper error handling strategies
- Use debugging techniques and tools
- Write robust, error-resistant code
- Apply error handling to mathematical applications

## Exception Types

### Built-in Exception Hierarchy
```python
# Common exception types
try:
    # ZeroDivisionError
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Division by zero: {e}")

try:
    # ValueError
    number = int("not_a_number")
except ValueError as e:
    print(f"Value error: {e}")

try:
    # TypeError
    result = "hello" + 5
except TypeError as e:
    print(f"Type error: {e}")

try:
    # IndexError
    numbers = [1, 2, 3]
    value = numbers[10]
except IndexError as e:
    print(f"Index error: {e}")

try:
    # KeyError
    dictionary = {"a": 1, "b": 2}
    value = dictionary["c"]
except KeyError as e:
    print(f"Key error: {e}")

try:
    # FileNotFoundError
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Exception Hierarchy
```python
# Understanding exception hierarchy
def demonstrate_exception_hierarchy():
    """Show how exception hierarchy works"""
    
    # BaseException is the base class
    exceptions_to_test = [
        ZeroDivisionError(),
        ValueError(),
        TypeError(),
        IndexError(),
        KeyError(),
        FileNotFoundError()
    ]
    
    for exc in exceptions_to_test:
        print(f"{type(exc).__name__} is instance of:")
        print(f"  Exception: {isinstance(exc, Exception)}")
        print(f"  BaseException: {isinstance(exc, BaseException)}")
        print()

demonstrate_exception_hierarchy()
```

## Try-Except Blocks

### Basic Try-Except
```python
# Basic exception handling
def safe_divide(a, b):
    """Safely divide two numbers"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None

# Using the function
result1 = safe_divide(10, 2)  # 5.0
result2 = safe_divide(10, 0)   # None (with error message)
print(f"Result 1: {result1}")
print(f"Result 2: {result2}")
```

### Multiple Exception Handling
```python
def process_number(user_input):
    """Process user input with multiple exception handling"""
    try:
        number = float(user_input)
        result = 100 / number
        return result
    except ValueError:
        print("Error: Invalid number format")
        return None
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Testing different inputs
test_inputs = ["10", "0", "abc", "3.14"]
for inp in test_inputs:
    result = process_number(inp)
    print(f"Input: {inp}, Result: {result}")
```

### Try-Except-Else-Finally
```python
def file_operation(filename):
    """Demonstrate try-except-else-finally"""
    file = None
    try:
        file = open(filename, "r")
        content = file.read()
        print("File opened successfully")
    except FileNotFoundError:
        print("File not found")
        return None
    except PermissionError:
        print("Permission denied")
        return None
    else:
        print("No exceptions occurred")
        return content
    finally:
        if file:
            file.close()
            print("File closed")

# Test the function
result = file_operation("example.txt")
```

## Raising Exceptions

### Raising Built-in Exceptions
```python
def validate_age(age):
    """Validate age and raise appropriate exceptions"""
    if not isinstance(age, (int, float)):
        raise TypeError("Age must be a number")
    
    if age < 0:
        raise ValueError("Age cannot be negative")
    
    if age > 150:
        raise ValueError("Age cannot be greater than 150")
    
    return True

# Testing validation
test_ages = [25, -5, 200, "twenty", 0]
for age in test_ages:
    try:
        validate_age(age)
        print(f"Age {age} is valid")
    except (TypeError, ValueError) as e:
        print(f"Age {age} is invalid: {e}")
```

### Raising with Custom Messages
```python
def calculate_square_root(number):
    """Calculate square root with proper error handling"""
    if not isinstance(number, (int, float)):
        raise TypeError(f"Expected number, got {type(number).__name__}")
    
    if number < 0:
        raise ValueError(f"Cannot calculate square root of negative number: {number}")
    
    import math
    return math.sqrt(number)

# Testing the function
test_numbers = [16, -4, "25", 0, 2.25]
for num in test_numbers:
    try:
        result = calculate_square_root(num)
        print(f"√{num} = {result:.2f}")
    except (TypeError, ValueError) as e:
        print(f"Error calculating √{num}: {e}")
```

## Custom Exceptions

### Creating Custom Exception Classes
```python
class MathError(Exception):
    """Base class for mathematical errors"""
    pass

class DivisionByZeroError(MathError):
    """Raised when attempting to divide by zero"""
    def __init__(self, message="Division by zero is not allowed"):
        self.message = message
        super().__init__(self.message)

class InvalidInputError(MathError):
    """Raised when input is invalid for mathematical operation"""
    def __init__(self, input_value, operation):
        self.input_value = input_value
        self.operation = operation
        self.message = f"Invalid input '{input_value}' for operation '{operation}'"
        super().__init__(self.message)

class MatrixDimensionError(MathError):
    """Raised when matrix dimensions are incompatible"""
    def __init__(self, matrix1_dims, matrix2_dims, operation):
        self.matrix1_dims = matrix1_dims
        self.matrix2_dims = matrix2_dims
        self.operation = operation
        self.message = f"Cannot perform {operation} on matrices with dimensions {matrix1_dims} and {matrix2_dims}"
        super().__init__(self.message)

# Using custom exceptions
def safe_divide(a, b):
    """Division with custom exception"""
    if b == 0:
        raise DivisionByZeroError()
    return a / b

def validate_matrix_dimensions(matrix1, matrix2, operation):
    """Validate matrix dimensions for operations"""
    if len(matrix1[0]) != len(matrix2):
        raise MatrixDimensionError(
            (len(matrix1), len(matrix1[0])),
            (len(matrix2), len(matrix2[0])),
            operation
        )

# Testing custom exceptions
try:
    result = safe_divide(10, 0)
except DivisionByZeroError as e:
    print(f"Custom error: {e}")

try:
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6, 7], [8, 9, 10]]
    validate_matrix_dimensions(matrix1, matrix2, "multiplication")
except MatrixDimensionError as e:
    print(f"Matrix error: {e}")
```

## Error Handling Strategies

### Defensive Programming
```python
def robust_calculator(operation, a, b):
    """Robust calculator with comprehensive error handling"""
    try:
        # Input validation
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both operands must be numbers")
        
        # Operation validation
        if operation not in ['+', '-', '*', '/', '**']:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Perform operation
        if operation == '+':
            return a + b
        elif operation == '-':
            return a - b
        elif operation == '*':
            return a * b
        elif operation == '/':
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a / b
        elif operation == '**':
            return a ** b
            
    except TypeError as e:
        print(f"Type Error: {e}")
        return None
    except ValueError as e:
        print(f"Value Error: {e}")
        return None
    except ZeroDivisionError as e:
        print(f"Division Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

# Testing the robust calculator
test_cases = [
    ('+', 5, 3),
    ('-', 10, 4),
    ('*', 6, 7),
    ('/', 15, 3),
    ('/', 10, 0),
    ('**', 2, 8),
    ('%', 10, 3),  # Unsupported operation
    ('+', "5", 3),  # Invalid type
]

for op, a, b in test_cases:
    result = robust_calculator(op, a, b)
    print(f"{a} {op} {b} = {result}")
```

### Error Recovery
```python
def read_config_file(filename, default_config=None):
    """Read configuration file with error recovery"""
    if default_config is None:
        default_config = {"debug": False, "timeout": 30, "retries": 3}
    
    try:
        import json
        with open(filename, 'r') as file:
            config = json.load(file)
        print(f"Configuration loaded from {filename}")
        return config
    except FileNotFoundError:
        print(f"Config file {filename} not found, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filename}: {e}")
        print("Using default configuration")
        return default_config
    except PermissionError:
        print(f"Permission denied reading {filename}")
        print("Using default configuration")
        return default_config
    except Exception as e:
        print(f"Unexpected error reading {filename}: {e}")
        print("Using default configuration")
        return default_config

# Test configuration loading
config = read_config_file("config.json")
print(f"Final config: {config}")
```

## Debugging Techniques

### Print Debugging
```python
def debug_function(data):
    """Function with debug prints"""
    print(f"DEBUG: Input data: {data}")
    print(f"DEBUG: Data type: {type(data)}")
    
    if isinstance(data, list):
        print(f"DEBUG: List length: {len(data)}")
        for i, item in enumerate(data):
            print(f"DEBUG: Item {i}: {item} (type: {type(item)})")
    
    # Process data
    try:
        result = sum(data) if isinstance(data, list) else data
        print(f"DEBUG: Result: {result}")
        return result
    except Exception as e:
        print(f"DEBUG: Error occurred: {e}")
        raise

# Using debug function
debug_function([1, 2, 3, 4, 5])
```

### Assertions for Debugging
```python
def calculate_statistics(numbers):
    """Calculate statistics with assertions for debugging"""
    assert isinstance(numbers, list), "Input must be a list"
    assert len(numbers) > 0, "List cannot be empty"
    assert all(isinstance(n, (int, float)) for n in numbers), "All elements must be numbers"
    
    # Calculate statistics
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    
    assert count > 0, "Count should be positive"
    assert isinstance(average, (int, float)), "Average should be a number"
    
    return {
        "total": total,
        "count": count,
        "average": average,
        "min": min(numbers),
        "max": max(numbers)
    }

# Test with assertions
try:
    stats = calculate_statistics([1, 2, 3, 4, 5])
    print(f"Statistics: {stats}")
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

### Logging for Debugging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

def process_data(data):
    """Process data with logging"""
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Starting to process data: {data}")
    
    try:
        # Validate input
        if not isinstance(data, list):
            logger.error(f"Invalid input type: {type(data)}")
            raise TypeError("Input must be a list")
        
        logger.info(f"Processing {len(data)} items")
        
        # Process each item
        results = []
        for i, item in enumerate(data):
            logger.debug(f"Processing item {i}: {item}")
            
            try:
                # Some processing
                if isinstance(item, (int, float)):
                    result = item * 2
                    results.append(result)
                    logger.debug(f"Item {i} processed successfully: {result}")
                else:
                    logger.warning(f"Item {i} is not a number, skipping: {item}")
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
        
        logger.info(f"Processing complete. {len(results)} items processed")
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in process_data: {e}")
        raise

# Test with logging
test_data = [1, 2, "three", 4, 5.5]
try:
    results = process_data(test_data)
    print(f"Results: {results}")
except Exception as e:
    print(f"Error: {e}")
```

## Mathematical Error Handling

### Numerical Stability
```python
import math

def safe_sqrt(number, tolerance=1e-10):
    """Safely calculate square root with numerical stability"""
    try:
        if not isinstance(number, (int, float)):
            raise TypeError("Input must be a number")
        
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        if abs(number) < tolerance:
            return 0.0
        
        result = math.sqrt(number)
        
        # Check for numerical issues
        if math.isnan(result):
            raise ValueError("Result is NaN (Not a Number)")
        if math.isinf(result):
            raise ValueError("Result is infinite")
        
        return result
        
    except (TypeError, ValueError) as e:
        print(f"Error in safe_sqrt: {e}")
        return None

# Test numerical stability
test_values = [16, -4, 0, 1e-20, "25", float('inf'), float('nan')]
for val in test_values:
    result = safe_sqrt(val)
    print(f"√{val} = {result}")
```

### Matrix Operations with Error Handling
```python
class Matrix:
    """Matrix class with comprehensive error handling"""
    
    def __init__(self, data):
        """Initialize matrix with validation"""
        try:
            if not isinstance(data, list):
                raise TypeError("Matrix data must be a list")
            
            if not data:
                raise ValueError("Matrix cannot be empty")
            
            # Validate all rows have same length
            row_length = len(data[0])
            for i, row in enumerate(data):
                if not isinstance(row, list):
                    raise TypeError(f"Row {i} must be a list")
                if len(row) != row_length:
                    raise ValueError(f"All rows must have same length. Row {i} has length {len(row)}, expected {row_length}")
            
            self.data = data
            self.rows = len(data)
            self.cols = row_length
            
        except (TypeError, ValueError) as e:
            print(f"Matrix initialization error: {e}")
            raise
    
    def __str__(self):
        """String representation of matrix"""
        return '\n'.join([' '.join([str(item) for item in row]) for row in self.data])
    
    def add(self, other):
        """Add two matrices with error handling"""
        try:
            if not isinstance(other, Matrix):
                raise TypeError("Can only add Matrix objects")
            
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError(f"Matrix dimensions must match. Got {self.rows}x{self.cols} and {other.rows}x{other.cols}")
            
            result_data = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    try:
                        sum_val = self.data[i][j] + other.data[i][j]
                        row.append(sum_val)
                    except TypeError:
                        raise TypeError(f"Cannot add non-numeric values at position ({i}, {j})")
                result_data.append(row)
            
            return Matrix(result_data)
            
        except (TypeError, ValueError) as e:
            print(f"Matrix addition error: {e}")
            return None
    
    def multiply(self, other):
        """Multiply two matrices with error handling"""
        try:
            if not isinstance(other, Matrix):
                raise TypeError("Can only multiply Matrix objects")
            
            if self.cols != other.rows:
                raise ValueError(f"Cannot multiply matrices: {self.rows}x{self.cols} and {other.rows}x{other.cols}")
            
            result_data = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    sum_val = 0
                    for k in range(self.cols):
                        try:
                            sum_val += self.data[i][k] * other.data[k][j]
                        except TypeError:
                            raise TypeError(f"Cannot multiply non-numeric values at position ({i}, {k}) and ({k}, {j})")
                    row.append(sum_val)
                result_data.append(row)
            
            return Matrix(result_data)
            
        except (TypeError, ValueError) as e:
            print(f"Matrix multiplication error: {e}")
            return None

# Test matrix operations
try:
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[5, 6], [7, 8]])
    
    print("Matrix 1:")
    print(matrix1)
    print("\nMatrix 2:")
    print(matrix2)
    
    result_add = matrix1.add(matrix2)
    if result_add:
        print("\nAddition result:")
        print(result_add)
    
    result_mult = matrix1.multiply(matrix2)
    if result_mult:
        print("\nMultiplication result:")
        print(result_mult)
        
except Exception as e:
    print(f"Matrix test error: {e}")
```

## Practice Problems

### Problem 1
Create a robust mathematical expression evaluator with comprehensive error handling.

```python
# Solution
import re
import math

class ExpressionEvaluator:
    """Mathematical expression evaluator with error handling"""
    
    def __init__(self):
        self.allowed_functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor
        }
    
    def validate_expression(self, expression):
        """Validate mathematical expression"""
        if not isinstance(expression, str):
            raise TypeError("Expression must be a string")
        
        if not expression.strip():
            raise ValueError("Expression cannot be empty")
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+',
            r'__\w+__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression):
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        return True
    
    def safe_eval(self, expression, variables=None):
        """Safely evaluate mathematical expression"""
        try:
            # Validate expression
            self.validate_expression(expression)
            
            # Prepare safe namespace
            safe_dict = {
                '__builtins__': {},
                'math': math,
                'pi': math.pi,
                'e': math.e
            }
            
            # Add allowed functions
            safe_dict.update(self.allowed_functions)
            
            # Add variables if provided
            if variables:
                safe_dict.update(variables)
            
            # Evaluate expression
            result = eval(expression, safe_dict)
            
            # Check result validity
            if math.isnan(result):
                raise ValueError("Result is NaN (Not a Number)")
            if math.isinf(result):
                raise ValueError("Result is infinite")
            
            return result
            
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax in expression: {e}")
        except NameError as e:
            raise ValueError(f"Undefined variable or function: {e}")
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")
    
    def evaluate_with_error_handling(self, expression, variables=None):
        """Evaluate expression with comprehensive error handling"""
        try:
            result = self.safe_eval(expression, variables)
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

# Test the evaluator
evaluator = ExpressionEvaluator()

test_expressions = [
    "2 + 3 * 4",
    "sin(pi/2)",
    "sqrt(16)",
    "log(e)",
    "1 / 0",
    "sqrt(-1)",
    "x + y",
    "import os",
    "invalid_function(1)"
]

variables = {"x": 5, "y": 3}

for expr in test_expressions:
    result = evaluator.evaluate_with_error_handling(expr, variables)
    if result["success"]:
        print(f"'{expr}' = {result['result']}")
    else:
        print(f"'{expr}' -> Error: {result['error']}")
```

### Problem 2
Create a data validation system with detailed error reporting.

```python
# Solution
from datetime import datetime
import re

class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.validation_rules = {}
    
    def add_rule(self, field_name, rule_func, error_message):
        """Add validation rule for a field"""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append({
            'function': rule_func,
            'message': error_message
        })
    
    def validate_field(self, field_name, value, row_number=None):
        """Validate a single field"""
        if field_name not in self.validation_rules:
            return True
        
        for rule in self.validation_rules[field_name]:
            try:
                if not rule['function'](value):
                    error_msg = rule['message']
                    if row_number is not None:
                        error_msg = f"Row {row_number}: {error_msg}"
                    self.errors.append(error_msg)
                    return False
            except Exception as e:
                error_msg = f"Validation error for {field_name}: {e}"
                if row_number is not None:
                    error_msg = f"Row {row_number}: {error_msg}"
                self.errors.append(error_msg)
                return False
        
        return True
    
    def validate_record(self, record, row_number=None):
        """Validate a complete record"""
        record_valid = True
        
        for field_name, value in record.items():
            field_valid = self.validate_field(field_name, value, row_number)
            if not field_valid:
                record_valid = False
        
        return record_valid
    
    def get_validation_report(self):
        """Get comprehensive validation report"""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "validation_passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }

# Define validation functions
def is_not_empty(value):
    """Check if value is not empty"""
    return value is not None and str(value).strip() != ""

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_age(age):
    """Validate age range"""
    try:
        age_int = int(age)
        return 0 <= age_int <= 150
    except (ValueError, TypeError):
        return False

def is_valid_date(date_str):
    """Validate date format"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def is_positive_number(value):
    """Check if value is a positive number"""
    try:
        num = float(value)
        return num > 0
    except (ValueError, TypeError):
        return False

# Test the validator
validator = DataValidator()

# Add validation rules
validator.add_rule("name", is_not_empty, "Name cannot be empty")
validator.add_rule("email", is_valid_email, "Invalid email format")
validator.add_rule("age", is_valid_age, "Age must be between 0 and 150")
validator.add_rule("birth_date", is_valid_date, "Invalid date format (YYYY-MM-DD)")
validator.add_rule("salary", is_positive_number, "Salary must be a positive number")

# Test data
test_records = [
    {"name": "Alice", "email": "alice@example.com", "age": 25, "birth_date": "1998-05-15", "salary": 50000},
    {"name": "", "email": "invalid-email", "age": -5, "birth_date": "invalid-date", "salary": -1000},
    {"name": "Bob", "email": "bob@company.com", "age": 30, "birth_date": "1993-12-01", "salary": 75000}
]

# Validate records
for i, record in enumerate(test_records, 1):
    print(f"\nValidating record {i}: {record}")
    validator.validate_record(record, i)

# Get validation report
report = validator.get_validation_report()
print(f"\nValidation Report:")
print(f"Total errors: {report['total_errors']}")
print(f"Validation passed: {report['validation_passed']}")
print("\nErrors:")
for error in report['errors']:
    print(f"  - {error}")
```

### Problem 3
Create a robust file processing system with error recovery.

```python
# Solution
import os
import json
from datetime import datetime

class RobustFileProcessor:
    """File processor with comprehensive error handling and recovery"""
    
    def __init__(self, log_file="processing.log"):
        self.log_file = log_file
        self.processed_files = []
        self.failed_files = []
        self.error_log = []
    
    def log_error(self, message, file_path=None):
        """Log error with timestamp"""
        timestamp = datetime.now().isoformat()
        error_entry = {
            "timestamp": timestamp,
            "message": message,
            "file": file_path
        }
        self.error_log.append(error_entry)
        
        # Write to log file
        try:
            with open(self.log_file, "a") as log:
                log.write(f"[{timestamp}] {message}")
                if file_path:
                    log.write(f" (File: {file_path})")
                log.write("\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    def process_text_file(self, file_path, output_path=None):
        """Process text file with error handling"""
        try:
            # Validate input file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            if not os.path.isfile(file_path):
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            # Process content
            processed_content = self.process_content(content)
            
            # Write output
            if output_path is None:
                output_path = file_path.replace(".txt", "_processed.txt")
            
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(processed_content)
            
            self.processed_files.append({
                "input": file_path,
                "output": output_path,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            self.log_error(error_msg, file_path)
            self.failed_files.append({
                "file": file_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    def process_content(self, content):
        """Process file content (example: convert to uppercase)"""
        try:
            # Simple processing example
            processed = content.upper()
            return processed
        except Exception as e:
            raise ValueError(f"Content processing error: {e}")
    
    def process_multiple_files(self, file_list, output_dir=None):
        """Process multiple files with batch error handling"""
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                self.log_error(f"Failed to create output directory {output_dir}: {e}")
                return False
        
        results = []
        for file_path in file_list:
            if output_dir:
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, filename)
            else:
                output_path = None
            
            success = self.process_text_file(file_path, output_path)
            results.append({"file": file_path, "success": success})
        
        return results
    
    def generate_processing_report(self, report_path="processing_report.json"):
        """Generate comprehensive processing report"""
        report = {
            "summary": {
                "total_files_processed": len(self.processed_files),
                "total_files_failed": len(self.failed_files),
                "success_rate": len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100 if (len(self.processed_files) + len(self.failed_files)) > 0 else 0,
                "generated_at": datetime.now().isoformat()
            },
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "error_log": self.error_log
        }
        
        try:
            with open(report_path, "w") as file:
                json.dump(report, file, indent=2)
            print(f"Processing report saved to {report_path}")
            return True
        except Exception as e:
            print(f"Failed to save processing report: {e}")
            return False
    
    def retry_failed_files(self, max_retries=3):
        """Retry processing failed files"""
        retry_results = []
        
        for failed_file in self.failed_files[:]:  # Copy list to avoid modification during iteration
            file_path = failed_file["file"]
            retry_count = 0
            
            while retry_count < max_retries:
                retry_count += 1
                self.log_error(f"Retrying file {file_path} (attempt {retry_count})", file_path)
                
                success = self.process_text_file(file_path)
                if success:
                    # Remove from failed files
                    self.failed_files.remove(failed_file)
                    retry_results.append({
                        "file": file_path,
                        "success": True,
                        "attempts": retry_count
                    })
                    break
                else:
                    retry_results.append({
                        "file": file_path,
                        "success": False,
                        "attempts": retry_count
                    })
        
        return retry_results

# Test the robust file processor
processor = RobustFileProcessor()

# Create test files
test_files = ["test1.txt", "test2.txt", "nonexistent.txt"]

# Create test files
for i, filename in enumerate(test_files[:2]):  # Create first two files
    with open(filename, "w") as file:
        file.write(f"This is test file {i+1}\nWith some content to process.")

# Process files
results = processor.process_multiple_files(test_files, "output")

print("Processing results:")
for result in results:
    print(f"  {result['file']}: {'Success' if result['success'] else 'Failed'}")

# Generate report
processor.generate_processing_report()

# Retry failed files
retry_results = processor.retry_failed_files()
print(f"\nRetry results: {len(retry_results)} files retried")
```

## Key Takeaways
- Always use try-except blocks for error handling
- Create custom exceptions for specific error conditions
- Implement defensive programming practices
- Use logging and debugging tools effectively
- Handle mathematical errors like division by zero and invalid inputs
- Provide meaningful error messages to users
- Implement error recovery strategies when possible
- Test error handling paths thoroughly

## Next Steps
In the next tutorial, we'll explore advanced Python topics, including decorators, generators, regular expressions, and performance optimization techniques.
