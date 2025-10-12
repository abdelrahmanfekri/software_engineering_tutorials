# Chapter 10: Error Handling

## ❌ The Error Interface

In Go, errors are values that implement the `error` interface.

```go
type error interface {
    Error() string
}
```

### Basic Error Handling

```go
import (
    "errors"
    "fmt"
)

func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func main() {
    result, err := divide(10, 0)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println("Result:", result)
}
```

### fmt.Errorf (Formatted Errors)

```go
func validateAge(age int) error {
    if age < 0 {
        return fmt.Errorf("age cannot be negative: %d", age)
    }
    if age > 150 {
        return fmt.Errorf("age %d is unrealistic", age)
    }
    return nil
}
```

## 🎯 Custom Errors

### Simple Custom Error

```go
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

func validateUser(username string) error {
    if len(username) < 3 {
        return ValidationError{
            Field:   "username",
            Message: "must be at least 3 characters",
        }
    }
    return nil
}
```

### Error with Context

```go
type FileError struct {
    Path string
    Op   string
    Err  error
}

func (e *FileError) Error() string {
    return fmt.Sprintf("%s %s: %v", e.Op, e.Path, e.Err)
}

func (e *FileError) Unwrap() error {
    return e.Err
}

func readConfig(path string) error {
    _, err := os.Open(path)
    if err != nil {
        return &FileError{
            Path: path,
            Op:   "open",
            Err:  err,
        }
    }
    return nil
}
```

## 🔍 Error Checking (Go 1.13+)

### errors.Is

```go
import "errors"

var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrInvalidInput = errors.New("invalid input")
)

func findUser(id int) error {
    if id < 0 {
        return ErrInvalidInput
    }
    if id > 100 {
        return fmt.Errorf("user lookup failed: %w", ErrNotFound)
    }
    return nil
}

func main() {
    err := findUser(101)
    
    // Check if error is or wraps ErrNotFound
    if errors.Is(err, ErrNotFound) {
        fmt.Println("User not found")
    }
}
```

### errors.As

```go
type TemporaryError interface {
    Temporary() bool
}

type NetworkError struct {
    Code      int
    Message   string
    temporary bool
}

func (e *NetworkError) Error() string {
    return fmt.Sprintf("network error %d: %s", e.Code, e.Message)
}

func (e *NetworkError) Temporary() bool {
    return e.temporary
}

func doRequest() error {
    return &NetworkError{
        Code:      503,
        Message:   "service unavailable",
        temporary: true,
    }
}

func main() {
    err := doRequest()
    
    var netErr *NetworkError
    if errors.As(err, &netErr) {
        fmt.Printf("Network error %d: %s\n", netErr.Code, netErr.Message)
        if netErr.Temporary() {
            fmt.Println("Can retry")
        }
    }
}
```

### Error Wrapping

```go
func processFile(path string) error {
    data, err := readFile(path)
    if err != nil {
        return fmt.Errorf("process file: %w", err)
    }
    
    if err := validateData(data); err != nil {
        return fmt.Errorf("validate data: %w", err)
    }
    
    return nil
}

// Unwrap manually
func main() {
    err := processFile("config.json")
    if err != nil {
        fmt.Println("Error:", err)
        
        // Unwrap chain
        for err != nil {
            fmt.Printf("  -> %v\n", err)
            err = errors.Unwrap(err)
        }
    }
}
```

## 💥 Panic and Recover

### When to Use Panic

Panic is for unrecoverable errors (exceptional situations).

```go
// Good use: Programming errors
func setAge(age int) {
    if age < 0 {
        panic("age cannot be negative")  // This is a bug
    }
}

// Bad use: Expected errors (use error return instead)
func findUser(id int) *User {
    // DON'T do this:
    // if user == nil {
    //     panic("user not found")
    // }
    
    // DO this:
    // return nil, fmt.Errorf("user not found")
}
```

### Recover from Panic

```go
func safeDivide(a, b int) (result int, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()
    
    result = a / b  // May panic on division by zero
    return result, nil
}

func main() {
    result, err := safeDivide(10, 0)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", result)
    }
}
```

### HTTP Handler Recovery

```go
func recoverMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic: %v\n%s", err, debug.Stack())
                http.Error(w, "Internal Server Error", 500)
            }
        }()
        
        next(w, r)
    }
}

func handler(w http.ResponseWriter, r *http.Request) {
    // May panic
    data := processRequest(r)
    json.NewEncoder(w).Encode(data)
}

func main() {
    http.HandleFunc("/", recoverMiddleware(handler))
    http.ListenAndServe(":8080", nil)
}
```

## 🎨 Error Handling Patterns

### Sentinel Errors

```go
var (
    ErrNotFound = errors.New("not found")
    ErrExists   = errors.New("already exists")
    ErrInvalid  = errors.New("invalid")
)

func GetUser(id int) (*User, error) {
    user, exists := users[id]
    if !exists {
        return nil, ErrNotFound
    }
    return &user, nil
}

// Usage
user, err := GetUser(123)
if err != nil {
    switch {
    case errors.Is(err, ErrNotFound):
        fmt.Println("User not found")
    case errors.Is(err, ErrInvalid):
        fmt.Println("Invalid user ID")
    default:
        fmt.Println("Unknown error:", err)
    }
}
```

### Error Types

```go
type ValidationError struct {
    Field   string
    Value   interface{}
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation failed: %s (%v) - %s",
        e.Field, e.Value, e.Message)
}

type DatabaseError struct {
    Operation string
    Err       error
}

func (e DatabaseError) Error() string {
    return fmt.Sprintf("database %s failed: %v", e.Operation, e.Err)
}

func (e DatabaseError) Unwrap() error {
    return e.Err
}
```

### Multi-Error

```go
type MultiError struct {
    errors []error
}

func (m *MultiError) Add(err error) {
    if err != nil {
        m.errors = append(m.errors, err)
    }
}

func (m *MultiError) Error() string {
    if len(m.errors) == 0 {
        return ""
    }
    
    var sb strings.Builder
    sb.WriteString("multiple errors occurred:\n")
    for i, err := range m.errors {
        sb.WriteString(fmt.Sprintf("  %d. %v\n", i+1, err))
    }
    return sb.String()
}

func (m *MultiError) HasErrors() bool {
    return len(m.errors) > 0
}

// Usage
func validateUser(user User) error {
    errs := &MultiError{}
    
    if user.Name == "" {
        errs.Add(errors.New("name is required"))
    }
    if user.Age < 0 {
        errs.Add(errors.New("age cannot be negative"))
    }
    if user.Email == "" {
        errs.Add(errors.New("email is required"))
    }
    
    if errs.HasErrors() {
        return errs
    }
    return nil
}
```

### Retry Logic

```go
func Retry(attempts int, sleep time.Duration, fn func() error) error {
    var err error
    
    for i := 0; i < attempts; i++ {
        err = fn()
        if err == nil {
            return nil
        }
        
        // Check if error is retryable
        var tempErr interface{ Temporary() bool }
        if errors.As(err, &tempErr) && !tempErr.Temporary() {
            return err  // Don't retry permanent errors
        }
        
        if i < attempts-1 {
            time.Sleep(sleep)
            sleep *= 2  // Exponential backoff
        }
    }
    
    return fmt.Errorf("after %d attempts, last error: %w", attempts, err)
}

// Usage
err := Retry(3, time.Second, func() error {
    return makeHTTPRequest()
})
```

## 🛡️ Defensive Programming

### Input Validation

```go
func CreateUser(name string, age int, email string) (*User, error) {
    // Validate all inputs
    if name == "" {
        return nil, fmt.Errorf("name cannot be empty")
    }
    if age < 0 || age > 150 {
        return nil, fmt.Errorf("age must be between 0 and 150, got %d", age)
    }
    if !strings.Contains(email, "@") {
        return nil, fmt.Errorf("invalid email: %s", email)
    }
    
    user := &User{
        Name:  name,
        Age:   age,
        Email: email,
    }
    
    return user, nil
}
```

### Nil Checks

```go
func ProcessUser(user *User) error {
    if user == nil {
        return errors.New("user cannot be nil")
    }
    
    // Safe to use user now
    fmt.Println(user.Name)
    return nil
}
```

## 💼 Complete Example: Service with Error Handling

```go
package main

import (
    "errors"
    "fmt"
    "log"
)

// Custom errors
var (
    ErrUserNotFound   = errors.New("user not found")
    ErrUserExists     = errors.New("user already exists")
    ErrInvalidInput   = errors.New("invalid input")
    ErrDatabase       = errors.New("database error")
)

// Custom error type
type ServiceError struct {
    Code    string
    Message string
    Err     error
}

func (e *ServiceError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Err)
    }
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *ServiceError) Unwrap() error {
    return e.Err
}

// Domain model
type User struct {
    ID    int
    Name  string
    Email string
}

// Service
type UserService struct {
    users map[int]User
}

func NewUserService() *UserService {
    return &UserService{
        users: make(map[int]User),
    }
}

func (s *UserService) Create(name, email string) (*User, error) {
    // Validation
    if name == "" || email == "" {
        return nil, &ServiceError{
            Code:    "VALIDATION_ERROR",
            Message: "name and email are required",
            Err:     ErrInvalidInput,
        }
    }
    
    // Check if exists
    for _, user := range s.users {
        if user.Email == email {
            return nil, &ServiceError{
                Code:    "DUPLICATE_ERROR",
                Message: fmt.Sprintf("user with email %s already exists", email),
                Err:     ErrUserExists,
            }
        }
    }
    
    // Create user
    user := User{
        ID:    len(s.users) + 1,
        Name:  name,
        Email: email,
    }
    
    s.users[user.ID] = user
    return &user, nil
}

func (s *UserService) Get(id int) (*User, error) {
    user, exists := s.users[id]
    if !exists {
        return nil, &ServiceError{
            Code:    "NOT_FOUND",
            Message: fmt.Sprintf("user %d not found", id),
            Err:     ErrUserNotFound,
        }
    }
    return &user, nil
}

func main() {
    service := NewUserService()
    
    // Create user
    user, err := service.Create("Alice", "alice@example.com")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Created user: %+v\n", user)
    
    // Try to create duplicate
    _, err = service.Create("Alice", "alice@example.com")
    if err != nil {
        var serviceErr *ServiceError
        if errors.As(err, &serviceErr) {
            fmt.Printf("Service error [%s]: %s\n", serviceErr.Code, serviceErr.Message)
        }
        
        if errors.Is(err, ErrUserExists) {
            fmt.Println("User already exists")
        }
    }
    
    // Get user
    foundUser, err := service.Get(1)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Found user: %+v\n", foundUser)
    
    // Get non-existent user
    _, err = service.Get(999)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            fmt.Println("User not found")
        }
    }
}
```

## 🎯 Exercises

### Exercise 1: Validation Library
Create a validation library with chainable validators.

### Exercise 2: Circuit Breaker with Errors
Extend the circuit breaker to classify errors as temporary or permanent.

### Exercise 3: Error Logger
Build an error logger that categorizes and formats errors for logging.

### Solutions

```go
// Exercise 1: Validation
type Validator struct {
    errors []error
}

func (v *Validator) Required(value string, field string) *Validator {
    if value == "" {
        v.errors = append(v.errors, fmt.Errorf("%s is required", field))
    }
    return v
}

func (v *Validator) MinLength(value string, min int, field string) *Validator {
    if len(value) < min {
        v.errors = append(v.errors, 
            fmt.Errorf("%s must be at least %d characters", field, min))
    }
    return v
}

func (v *Validator) Validate() error {
    if len(v.errors) > 0 {
        return &MultiError{errors: v.errors}
    }
    return nil
}

// Usage
validator := &Validator{}
err := validator.
    Required(user.Name, "name").
    MinLength(user.Name, 3, "name").
    Required(user.Email, "email").
    Validate()
```

## 🔑 Key Takeaways

- Errors are values in Go (not exceptions)
- Always handle errors explicitly
- Use `errors.Is` and `errors.As` for error checking
- Wrap errors with context using `fmt.Errorf("...: %w", err)`
- Panic only for programmer errors, not expected errors
- Use custom error types for rich error information
- Sentinel errors for common cases
- Multi-errors for validation

## 📖 Next Steps

Continue to [Chapter 11: Testing](11-testing.md) to learn about testing in Go.

