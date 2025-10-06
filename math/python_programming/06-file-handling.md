# Python Programming Tutorial 06: File Handling

## Learning Objectives
By the end of this tutorial, you will be able to:
- Read from and write to text files
- Work with different file modes and operations
- Handle CSV and JSON data formats
- Use exception handling with files
- Implement context managers for file operations
- Serialize and deserialize data
- Process large files efficiently

## Basic File Operations

### Opening and Closing Files
```python
# Basic file operations
file = open("example.txt", "w")  # Open file for writing
file.write("Hello, World!")
file.close()  # Always close files

# Reading from file
file = open("example.txt", "r")  # Open file for reading
content = file.read()
print(content)  # Hello, World!
file.close()
```

### File Modes
```python
# Different file modes
# 'r' - Read mode (default)
# 'w' - Write mode (overwrites existing file)
# 'a' - Append mode (adds to end of file)
# 'x' - Exclusive creation (fails if file exists)
# 'b' - Binary mode (for images, videos, etc.)
# 't' - Text mode (default)
# '+' - Read and write mode

# Examples
file1 = open("read_file.txt", "r")      # Read only
file2 = open("write_file.txt", "w")    # Write only
file3 = open("append_file.txt", "a")   # Append only
file4 = open("binary_file.bin", "wb")  # Binary write
file5 = open("read_write.txt", "r+")  # Read and write

# Close all files
for file in [file1, file2, file3, file4, file5]:
    file.close()
```

### Context Managers (Recommended Approach)
```python
# Using 'with' statement (recommended)
with open("example.txt", "w") as file:
    file.write("Hello, World!")
    # File is automatically closed when exiting the block

# Reading with context manager
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Multiple operations in one context
with open("data.txt", "w") as file:
    file.write("Line 1\n")
    file.write("Line 2\n")
    file.write("Line 3\n")
```

## Reading Files

### Reading Entire File
```python
# Read entire file as string
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Read entire file as list of lines
with open("example.txt", "r") as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        print(f"Line {i+1}: {line.strip()}")
```

### Reading Line by Line
```python
# Read line by line (memory efficient for large files)
with open("large_file.txt", "r") as file:
    for line_number, line in enumerate(file, 1):
        print(f"Line {line_number}: {line.strip()}")
        # Process each line here
```

### Reading Specific Amount
```python
# Read specific number of characters
with open("example.txt", "r") as file:
    first_10_chars = file.read(10)
    print(f"First 10 characters: {first_10_chars}")
    
    # Read next 10 characters
    next_10_chars = file.read(10)
    print(f"Next 10 characters: {next_10_chars}")
```

## Writing Files

### Writing Text
```python
# Write text to file
with open("output.txt", "w") as file:
    file.write("This is line 1\n")
    file.write("This is line 2\n")
    file.write("This is line 3\n")

# Write multiple lines at once
lines = ["Line 1", "Line 2", "Line 3"]
with open("output.txt", "w") as file:
    file.writelines([line + "\n" for line in lines])
```

### Appending to Files
```python
# Append to existing file
with open("log.txt", "a") as file:
    file.write("New log entry\n")
    file.write("Another log entry\n")

# Append with timestamp
import datetime
with open("log.txt", "a") as file:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"[{timestamp}] User logged in\n")
```

## File Path Operations

### Working with Paths
```python
import os

# Get current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# List files in directory
files = os.listdir(".")
print("Files in current directory:")
for file in files:
    print(f"  {file}")

# Check if file exists
if os.path.exists("example.txt"):
    print("File exists")
else:
    print("File does not exist")

# Get file information
if os.path.exists("example.txt"):
    file_size = os.path.getsize("example.txt")
    print(f"File size: {file_size} bytes")
```

### Path Manipulation
```python
import os

# Join paths (cross-platform)
file_path = os.path.join("data", "files", "example.txt")
print(f"File path: {file_path}")

# Split path
directory, filename = os.path.split(file_path)
print(f"Directory: {directory}")
print(f"Filename: {filename}")

# Get file extension
name, extension = os.path.splitext(filename)
print(f"Name: {name}")
print(f"Extension: {extension}")

# Create directories
os.makedirs("new_directory", exist_ok=True)
```

## CSV File Handling

### Reading CSV Files
```python
import csv

# Reading CSV file
with open("students.csv", "r") as file:
    csv_reader = csv.reader(file)
    
    # Skip header row
    next(csv_reader)
    
    for row in csv_reader:
        name, age, grade = row
        print(f"Name: {name}, Age: {age}, Grade: {grade}")

# Reading CSV with headers
with open("students.csv", "r") as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        print(f"Name: {row['name']}, Age: {row['age']}, Grade: {row['grade']}")
```

### Writing CSV Files
```python
import csv

# Writing CSV file
students = [
    ["Alice", 20, "A"],
    ["Bob", 22, "B"],
    ["Charlie", 19, "A"]
]

with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["name", "age", "grade"])  # Header
    writer.writerows(students)

# Writing CSV with dictionaries
students_dict = [
    {"name": "Alice", "age": 20, "grade": "A"},
    {"name": "Bob", "age": 22, "grade": "B"},
    {"name": "Charlie", "age": 19, "grade": "A"}
]

with open("students_dict.csv", "w", newline="") as file:
    fieldnames = ["name", "age", "grade"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students_dict)
```

## JSON File Handling

### Reading JSON Files
```python
import json

# Reading JSON file
with open("data.json", "r") as file:
    data = json.load(file)
    print(data)

# Reading JSON string
json_string = '{"name": "Alice", "age": 25, "city": "New York"}'
data = json.loads(json_string)
print(data)
```

### Writing JSON Files
```python
import json

# Writing JSON file
data = {
    "students": [
        {"name": "Alice", "age": 20, "grades": [85, 90, 78]},
        {"name": "Bob", "age": 22, "grades": [92, 88, 95]}
    ],
    "class": "Mathematics 101"
}

with open("students.json", "w") as file:
    json.dump(data, file, indent=2)  # indent for pretty printing

# Writing JSON string
json_string = json.dumps(data, indent=2)
print(json_string)
```

## Exception Handling with Files

### Basic Exception Handling
```python
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("Permission denied!")
except Exception as e:
    print(f"An error occurred: {e}")
```

### Robust File Operations
```python
def safe_read_file(filename):
    """Safely read a file with error handling"""
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Permission denied for '{filename}'")
        return None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None

# Using the safe function
content = safe_read_file("example.txt")
if content:
    print("File read successfully")
else:
    print("Failed to read file")
```

## Binary File Handling

### Working with Binary Files
```python
# Writing binary data
data = b"Hello, Binary World!"
with open("binary_file.bin", "wb") as file:
    file.write(data)

# Reading binary data
with open("binary_file.bin", "rb") as file:
    binary_data = file.read()
    print(binary_data)

# Working with images (example)
def copy_binary_file(source, destination):
    """Copy a binary file"""
    try:
        with open(source, "rb") as src_file:
            with open(destination, "wb") as dest_file:
                # Read and write in chunks for large files
                chunk_size = 1024
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    dest_file.write(chunk)
        print(f"File copied from {source} to {destination}")
    except Exception as e:
        print(f"Error copying file: {e}")

# Example usage
copy_binary_file("binary_file.bin", "copy_of_binary_file.bin")
```

## Data Serialization

### Pickle for Python Objects
```python
import pickle

# Serializing Python objects
data = {
    "numbers": [1, 2, 3, 4, 5],
    "text": "Hello, World!",
    "nested": {"key": "value"}
}

# Save to file
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)

# Load from file
with open("data.pkl", "rb") as file:
    loaded_data = pickle.load(file)
    print(loaded_data)

# Serialize to string
serialized = pickle.dumps(data)
deserialized = pickle.loads(serialized)
print(deserialized)
```

### Custom Serialization
```python
import json
from datetime import datetime

class Student:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades
        self.created_at = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "age": self.age,
            "grades": self.grades,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        student = cls(data["name"], data["age"], data["grades"])
        student.created_at = datetime.fromisoformat(data["created_at"])
        return student

# Using custom serialization
student = Student("Alice", 20, [85, 90, 78])

# Save to JSON
with open("student.json", "w") as file:
    json.dump(student.to_dict(), file, indent=2)

# Load from JSON
with open("student.json", "r") as file:
    data = json.load(file)
    loaded_student = Student.from_dict(data)
    print(f"Loaded student: {loaded_student.name}")
```

## Processing Large Files

### Memory-Efficient Processing
```python
def process_large_file(filename, chunk_size=1024):
    """Process large file in chunks"""
    try:
        with open(filename, "r") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                
                # Process chunk here
                print(f"Processing chunk: {len(chunk)} characters")
                
    except Exception as e:
        print(f"Error processing file: {e}")

def count_lines_large_file(filename):
    """Count lines in large file efficiently"""
    line_count = 0
    try:
        with open(filename, "r") as file:
            for line in file:
                line_count += 1
        return line_count
    except Exception as e:
        print(f"Error counting lines: {e}")
        return 0

# Example usage
lines = count_lines_large_file("large_file.txt")
print(f"Total lines: {lines}")
```

### Streaming JSON Processing
```python
import json

def process_large_json(filename):
    """Process large JSON file line by line"""
    try:
        with open(filename, "r") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    # Process each JSON object
                    print(f"Line {line_number}: {data}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON on line {line_number}")
                    
    except Exception as e:
        print(f"Error processing JSON file: {e}")
```

## Mathematical Data Processing

### Reading Numerical Data
```python
def read_matrix_from_file(filename):
    """Read matrix from text file"""
    matrix = []
    try:
        with open(filename, "r") as file:
            for line in file:
                row = [float(x) for x in line.strip().split()]
                matrix.append(row)
        return matrix
    except Exception as e:
        print(f"Error reading matrix: {e}")
        return None

def write_matrix_to_file(matrix, filename):
    """Write matrix to text file"""
    try:
        with open(filename, "w") as file:
            for row in matrix:
                file.write(" ".join([str(x) for x in row]) + "\n")
        print(f"Matrix written to {filename}")
    except Exception as e:
        print(f"Error writing matrix: {e}")

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
write_matrix_to_file(matrix, "matrix.txt")
loaded_matrix = read_matrix_from_file("matrix.txt")
print("Loaded matrix:", loaded_matrix)
```

### Data Analysis with Files
```python
import csv
import json

def analyze_student_data(csv_filename, json_filename):
    """Analyze student data from CSV and save results to JSON"""
    students = []
    
    # Read CSV data
    try:
        with open(csv_filename, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                students.append({
                    "name": row["name"],
                    "age": int(row["age"]),
                    "grades": [int(x) for x in row["grades"].split(",")]
                })
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Analyze data
    analysis = {
        "total_students": len(students),
        "average_age": sum(s["age"] for s in students) / len(students),
        "grade_statistics": {}
    }
    
    # Calculate grade statistics
    all_grades = []
    for student in students:
        all_grades.extend(student["grades"])
    
    analysis["grade_statistics"] = {
        "average": sum(all_grades) / len(all_grades),
        "highest": max(all_grades),
        "lowest": min(all_grades)
    }
    
    # Save analysis to JSON
    try:
        with open(json_filename, "w") as file:
            json.dump(analysis, file, indent=2)
        print("Analysis saved successfully")
    except Exception as e:
        print(f"Error saving analysis: {e}")

# Example usage (assuming students.csv exists)
analyze_student_data("students.csv", "analysis.json")
```

## Practice Problems

### Problem 1
Create a log file processor that reads log entries and generates a summary report.

```python
# Solution
import re
from datetime import datetime
from collections import defaultdict

def process_log_file(log_filename, report_filename):
    """Process log file and generate summary report"""
    log_entries = []
    error_count = 0
    warning_count = 0
    info_count = 0
    
    # Regular expressions for log parsing
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
    
    try:
        with open(log_filename, "r") as file:
            for line in file:
                match = re.match(log_pattern, line.strip())
                if match:
                    timestamp_str, level, message = match.groups()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    log_entry = {
                        "timestamp": timestamp,
                        "level": level,
                        "message": message
                    }
                    log_entries.append(log_entry)
                    
                    # Count by level
                    if level == "ERROR":
                        error_count += 1
                    elif level == "WARNING":
                        warning_count += 1
                    elif level == "INFO":
                        info_count += 1
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    # Generate report
    report = {
        "summary": {
            "total_entries": len(log_entries),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count
        },
        "time_range": {
            "start": min(entry["timestamp"] for entry in log_entries).isoformat(),
            "end": max(entry["timestamp"] for entry in log_entries).isoformat()
        },
        "recent_errors": [
            entry["message"] for entry in log_entries 
            if entry["level"] == "ERROR"
        ][-5:]  # Last 5 errors
    }
    
    # Save report
    try:
        with open(report_filename, "w") as file:
            json.dump(report, file, indent=2)
        print(f"Log analysis complete. Report saved to {report_filename}")
    except Exception as e:
        print(f"Error saving report: {e}")

# Example usage
process_log_file("application.log", "log_report.json")
```

### Problem 2
Create a CSV data validator that checks data integrity and generates validation reports.

```python
# Solution
import csv
import json
from datetime import datetime

class DataValidator:
    """CSV data validator"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_email(self, email):
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_age(self, age):
        """Validate age"""
        try:
            age_int = int(age)
            return 0 <= age_int <= 150
        except ValueError:
            return False
    
    def validate_date(self, date_str):
        """Validate date format"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def validate_row(self, row, row_number):
        """Validate a single row"""
        # Check required fields
        required_fields = ["name", "email", "age", "birth_date"]
        for field in required_fields:
            if field not in row or not row[field].strip():
                self.errors.append(f"Row {row_number}: Missing required field '{field}'")
        
        # Validate email
        if "email" in row and row["email"]:
            if not self.validate_email(row["email"]):
                self.errors.append(f"Row {row_number}: Invalid email format '{row['email']}'")
        
        # Validate age
        if "age" in row and row["age"]:
            if not self.validate_age(row["age"]):
                self.errors.append(f"Row {row_number}: Invalid age '{row['age']}'")
        
        # Validate birth date
        if "birth_date" in row and row["birth_date"]:
            if not self.validate_date(row["birth_date"]):
                self.errors.append(f"Row {row_number}: Invalid date format '{row['birth_date']}'")
    
    def validate_file(self, filename):
        """Validate entire CSV file"""
        try:
            with open(filename, "r") as file:
                csv_reader = csv.DictReader(file)
                
                for row_number, row in enumerate(csv_reader, 1):
                    self.validate_row(row, row_number)
        
        except Exception as e:
            self.errors.append(f"File error: {e}")
    
    def generate_report(self, report_filename):
        """Generate validation report"""
        report = {
            "validation_summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "validation_status": "PASS" if len(self.errors) == 0 else "FAIL"
            },
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        try:
            with open(report_filename, "w") as file:
                json.dump(report, file, indent=2)
            print(f"Validation report saved to {report_filename}")
        except Exception as e:
            print(f"Error saving report: {e}")

# Example usage
validator = DataValidator()
validator.validate_file("students.csv")
validator.generate_report("validation_report.json")
```

### Problem 3
Create a file backup utility that creates timestamped backups of important files.

```python
# Solution
import os
import shutil
from datetime import datetime

class FileBackup:
    """File backup utility"""
    
    def __init__(self, backup_directory="backups"):
        self.backup_directory = backup_directory
        self.ensure_backup_directory()
    
    def ensure_backup_directory(self):
        """Create backup directory if it doesn't exist"""
        if not os.path.exists(self.backup_directory):
            os.makedirs(self.backup_directory)
    
    def create_backup(self, source_file, description=""):
        """Create a backup of a file"""
        if not os.path.exists(source_file):
            print(f"Source file '{source_file}' does not exist")
            return False
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_file)
        name, extension = os.path.splitext(filename)
        
        if description:
            backup_filename = f"{name}_{description}_{timestamp}{extension}"
        else:
            backup_filename = f"{name}_{timestamp}{extension}"
        
        backup_path = os.path.join(self.backup_directory, backup_filename)
        
        try:
            shutil.copy2(source_file, backup_path)
            print(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def backup_multiple_files(self, file_list, description=""):
        """Backup multiple files"""
        results = []
        for file_path in file_list:
            result = self.create_backup(file_path, description)
            results.append((file_path, result))
        return results
    
    def list_backups(self):
        """List all backup files"""
        if not os.path.exists(self.backup_directory):
            print("No backup directory found")
            return []
        
        backups = []
        for filename in os.listdir(self.backup_directory):
            file_path = os.path.join(self.backup_directory, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                backups.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def cleanup_old_backups(self, days_to_keep=30):
        """Remove backups older than specified days"""
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        removed_count = 0
        
        try:
            for filename in os.listdir(self.backup_directory):
                file_path = os.path.join(self.backup_directory, filename)
                if os.path.isfile(file_path):
                    if os.path.getctime(file_path) < cutoff_time:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"Removed old backup: {filename}")
            
            print(f"Cleanup complete. Removed {removed_count} old backups")
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage
backup_util = FileBackup()

# Backup single file
backup_util.create_backup("important_data.txt", "daily")

# Backup multiple files
files_to_backup = ["config.json", "data.csv", "report.pdf"]
backup_util.backup_multiple_files(files_to_backup, "weekly")

# List backups
backups = backup_util.list_backups()
print("Available backups:")
for backup in backups:
    print(f"  {backup['filename']} - {backup['created']}")

# Cleanup old backups
backup_util.cleanup_old_backups(7)  # Keep backups for 7 days
```

## Key Takeaways
- Always use context managers (`with` statement) for file operations
- Handle exceptions properly when working with files
- Choose appropriate file modes for different operations
- Use CSV and JSON modules for structured data
- Process large files in chunks to avoid memory issues
- Implement proper error handling and logging
- File operations are essential for data persistence and processing

## Next Steps
In the next tutorial, we'll explore error handling and debugging, learning how to handle exceptions gracefully, debug code effectively, and implement robust error handling strategies.
