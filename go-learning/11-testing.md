# Chapter 11: Testing

## 🧪 Basic Testing

Go has built-in testing support with the `testing` package.

### Writing Tests

```go
// math.go
package math

func Add(a, b int) int {
    return a + b
}

func Multiply(a, b int) int {
    return a * b
}
```

```go
// math_test.go
package math

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5
    
    if result != expected {
        t.Errorf("Add(2, 3) = %d; want %d", result, expected)
    }
}

func TestMultiply(t *testing.T) {
    result := Multiply(3, 4)
    expected := 12
    
    if result != expected {
        t.Fatalf("Multiply(3, 4) = %d; want %d", result, expected)
    }
}
```

### Running Tests

```bash
# Run all tests in current package
go test

# Run with verbose output
go test -v

# Run specific test
go test -run TestAdd

# Run tests in all packages
go test ./...

# Run with coverage
go test -cover

# Generate coverage report
go test -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## 📋 Table-Driven Tests

Best practice for testing multiple scenarios.

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -2, -3, -5},
        {"mixed signs", -2, 3, 1},
        {"with zero", 5, 0, 5},
        {"both zero", 0, 0, 0},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("Add(%d, %d) = %d; want %d",
                    tt.a, tt.b, result, tt.expected)
            }
        })
    }
}
```

### Testing Error Cases

```go
func Divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func TestDivide(t *testing.T) {
    tests := []struct {
        name        string
        a, b        float64
        expected    float64
        expectError bool
    }{
        {"normal division", 10, 2, 5, false},
        {"division by zero", 10, 0, 0, true},
        {"negative result", 10, -2, -5, false},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result, err := Divide(tt.a, tt.b)
            
            if tt.expectError {
                if err == nil {
                    t.Error("expected error, got nil")
                }
                return
            }
            
            if err != nil {
                t.Errorf("unexpected error: %v", err)
            }
            
            if result != tt.expected {
                t.Errorf("got %f, want %f", result, tt.expected)
            }
        })
    }
}
```

## 🎯 Test Helpers

### Helper Functions

```go
func assertEqual(t *testing.T, got, want interface{}) {
    t.Helper()  // Marks this as helper function
    if got != want {
        t.Errorf("got %v, want %v", got, want)
    }
}

func TestWithHelper(t *testing.T) {
    result := Add(2, 3)
    assertEqual(t, result, 5)
}
```

### Setup and Teardown

```go
func TestMain(m *testing.M) {
    // Setup
    fmt.Println("Setting up tests...")
    setup()
    
    // Run tests
    code := m.Run()
    
    // Teardown
    fmt.Println("Tearing down...")
    teardown()
    
    os.Exit(code)
}

func setup() {
    // Initialize database, mock services, etc.
}

func teardown() {
    // Clean up resources
}
```

### Test Fixtures

```go
type UserServiceTest struct {
    service *UserService
    db      *MockDatabase
}

func newUserServiceTest(t *testing.T) *UserServiceTest {
    db := &MockDatabase{}
    service := NewUserService(db)
    
    return &UserServiceTest{
        service: service,
        db:      db,
    }
}

func TestUserService(t *testing.T) {
    test := newUserServiceTest(t)
    
    user, err := test.service.Create("Alice", "alice@example.com")
    if err != nil {
        t.Fatal(err)
    }
    
    if user.Name != "Alice" {
        t.Errorf("got name %s, want Alice", user.Name)
    }
}
```

## 🎭 Mocking

### Interface-Based Mocking

```go
// Interface
type UserStore interface {
    Get(id int) (*User, error)
    Save(user *User) error
}

// Real implementation
type DatabaseStore struct {
    db *sql.DB
}

func (s *DatabaseStore) Get(id int) (*User, error) {
    // Real database query
    return nil, nil
}

// Mock implementation
type MockStore struct {
    users map[int]*User
}

func NewMockStore() *MockStore {
    return &MockStore{
        users: make(map[int]*User),
    }
}

func (m *MockStore) Get(id int) (*User, error) {
    if user, ok := m.users[id]; ok {
        return user, nil
    }
    return nil, errors.New("not found")
}

func (m *MockStore) Save(user *User) error {
    m.users[user.ID] = user
    return nil
}

// Service
type UserService struct {
    store UserStore
}

func NewUserService(store UserStore) *UserService {
    return &UserService{store: store}
}

func (s *UserService) GetUser(id int) (*User, error) {
    return s.store.Get(id)
}

// Test
func TestUserService_GetUser(t *testing.T) {
    mockStore := NewMockStore()
    mockStore.Save(&User{ID: 1, Name: "Alice"})
    
    service := NewUserService(mockStore)
    
    user, err := service.GetUser(1)
    if err != nil {
        t.Fatal(err)
    }
    
    if user.Name != "Alice" {
        t.Errorf("got %s, want Alice", user.Name)
    }
}
```

### HTTP Mocking

```go
func TestHTTPClient(t *testing.T) {
    // Create mock server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path == "/users/1" {
            w.WriteHeader(http.StatusOK)
            json.NewEncoder(w).Encode(User{ID: 1, Name: "Alice"})
        } else {
            w.WriteHeader(http.StatusNotFound)
        }
    }))
    defer server.Close()
    
    // Test with mock server
    resp, err := http.Get(server.URL + "/users/1")
    if err != nil {
        t.Fatal(err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        t.Errorf("got status %d, want %d", resp.StatusCode, http.StatusOK)
    }
    
    var user User
    json.NewDecoder(resp.Body).Decode(&user)
    
    if user.Name != "Alice" {
        t.Errorf("got name %s, want Alice", user.Name)
    }
}
```

## 📊 Benchmarking

### Basic Benchmarks

```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(2, 3)
    }
}

func BenchmarkMultiply(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Multiply(3, 4)
    }
}
```

```bash
# Run benchmarks
go test -bench=.

# With memory stats
go test -bench=. -benchmem

# Run specific benchmark
go test -bench=BenchmarkAdd
```

### Table-Driven Benchmarks

```go
func BenchmarkFibonacci(b *testing.B) {
    benchmarks := []struct {
        name  string
        input int
    }{
        {"fib 10", 10},
        {"fib 20", 20},
        {"fib 30", 30},
    }
    
    for _, bm := range benchmarks {
        b.Run(bm.name, func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                Fibonacci(bm.input)
            }
        })
    }
}
```

### Benchmark with Setup

```go
func BenchmarkMapAccess(b *testing.B) {
    // Setup (not timed)
    m := make(map[int]int)
    for i := 0; i < 1000; i++ {
        m[i] = i
    }
    
    // Reset timer to exclude setup
    b.ResetTimer()
    
    // Benchmark
    for i := 0; i < b.N; i++ {
        _ = m[500]
    }
}
```

## 📈 Coverage

### Generating Coverage Reports

```bash
# Run tests with coverage
go test -cover

# Generate coverage profile
go test -coverprofile=coverage.out

# View coverage in terminal
go tool cover -func=coverage.out

# Generate HTML report
go tool cover -html=coverage.out

# Coverage for all packages
go test -cover ./...
```

### Coverage Example

```go
// calculator.go
package calculator

func Add(a, b int) int {
    return a + b
}

func Subtract(a, b int) int {
    return a - b
}

func Divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// calculator_test.go
func TestCalculator(t *testing.T) {
    // Test Add
    if Add(2, 3) != 5 {
        t.Error("Add failed")
    }
    
    // Test Divide
    result, err := Divide(10, 2)
    if err != nil || result != 5 {
        t.Error("Divide failed")
    }
    
    // Test Divide error case
    _, err = Divide(10, 0)
    if err == nil {
        t.Error("Expected error")
    }
}

// Coverage will show:
// - Add: 100%
// - Subtract: 0% (not tested)
// - Divide: 100%
```

## 🎨 Testing Patterns

### Parallel Tests

```go
func TestParallel(t *testing.T) {
    tests := []struct {
        name string
        id   int
    }{
        {"test 1", 1},
        {"test 2", 2},
        {"test 3", 3},
    }
    
    for _, tt := range tests {
        tt := tt  // Capture range variable
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()  // Run in parallel
            
            // Test logic
            user := fetchUser(tt.id)
            if user == nil {
                t.Error("user not found")
            }
        })
    }
}
```

### Test Timeouts

```go
func TestWithTimeout(t *testing.T) {
    timeout := time.After(5 * time.Second)
    done := make(chan bool)
    
    go func() {
        // Long running operation
        result := performOperation()
        done <- result
    }()
    
    select {
    case result := <-done:
        if !result {
            t.Error("operation failed")
        }
    case <-timeout:
        t.Fatal("test timed out")
    }
}
```

### Golden Files

```go
func TestOutput(t *testing.T) {
    output := generateReport()
    
    goldenFile := "testdata/report.golden"
    
    // Update golden file with -update flag
    if *update {
        os.WriteFile(goldenFile, []byte(output), 0644)
    }
    
    // Compare with golden file
    golden, err := os.ReadFile(goldenFile)
    if err != nil {
        t.Fatal(err)
    }
    
    if output != string(golden) {
        t.Errorf("output doesn't match golden file")
    }
}

var update = flag.Bool("update", false, "update golden files")
```

## 💼 Complete Example: Testing a Service

```go
// user_service.go
package user

import (
    "errors"
    "regexp"
)

var emailRegex = regexp.MustCompile(`^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$`)

type User struct {
    ID    int
    Name  string
    Email string
    Age   int
}

type UserStore interface {
    Save(user *User) error
    Find(id int) (*User, error)
}

type UserService struct {
    store UserStore
}

func NewUserService(store UserStore) *UserService {
    return &UserService{store: store}
}

func (s *UserService) CreateUser(name, email string, age int) (*User, error) {
    if name == "" {
        return nil, errors.New("name is required")
    }
    
    if !emailRegex.MatchString(email) {
        return nil, errors.New("invalid email")
    }
    
    if age < 0 || age > 150 {
        return nil, errors.New("invalid age")
    }
    
    user := &User{
        Name:  name,
        Email: email,
        Age:   age,
    }
    
    if err := s.store.Save(user); err != nil {
        return nil, err
    }
    
    return user, nil
}

// user_service_test.go
package user

import (
    "errors"
    "testing"
)

type MockUserStore struct {
    users   map[int]*User
    saveErr error
}

func NewMockUserStore() *MockUserStore {
    return &MockUserStore{
        users: make(map[int]*User),
    }
}

func (m *MockUserStore) Save(user *User) error {
    if m.saveErr != nil {
        return m.saveErr
    }
    user.ID = len(m.users) + 1
    m.users[user.ID] = user
    return nil
}

func (m *MockUserStore) Find(id int) (*User, error) {
    if user, ok := m.users[id]; ok {
        return user, nil
    }
    return nil, errors.New("not found")
}

func TestUserService_CreateUser(t *testing.T) {
    tests := []struct {
        name        string
        userName    string
        email       string
        age         int
        storeError  error
        expectError bool
        errorMsg    string
    }{
        {
            name:        "valid user",
            userName:    "Alice",
            email:       "alice@example.com",
            age:         25,
            expectError: false,
        },
        {
            name:        "empty name",
            userName:    "",
            email:       "alice@example.com",
            age:         25,
            expectError: true,
            errorMsg:    "name is required",
        },
        {
            name:        "invalid email",
            userName:    "Alice",
            email:       "invalid-email",
            age:         25,
            expectError: true,
            errorMsg:    "invalid email",
        },
        {
            name:        "negative age",
            userName:    "Alice",
            email:       "alice@example.com",
            age:         -1,
            expectError: true,
            errorMsg:    "invalid age",
        },
        {
            name:        "store error",
            userName:    "Alice",
            email:       "alice@example.com",
            age:         25,
            storeError:  errors.New("database error"),
            expectError: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            store := NewMockUserStore()
            store.saveErr = tt.storeError
            
            service := NewUserService(store)
            
            user, err := service.CreateUser(tt.userName, tt.email, tt.age)
            
            if tt.expectError {
                if err == nil {
                    t.Error("expected error, got nil")
                }
                if tt.errorMsg != "" && err.Error() != tt.errorMsg {
                    t.Errorf("got error %q, want %q", err.Error(), tt.errorMsg)
                }
                return
            }
            
            if err != nil {
                t.Errorf("unexpected error: %v", err)
            }
            
            if user.Name != tt.userName {
                t.Errorf("got name %s, want %s", user.Name, tt.userName)
            }
            
            if user.Email != tt.email {
                t.Errorf("got email %s, want %s", user.Email, tt.email)
            }
            
            if user.Age != tt.age {
                t.Errorf("got age %d, want %d", user.Age, tt.age)
            }
        })
    }
}

func BenchmarkCreateUser(b *testing.B) {
    store := NewMockUserStore()
    service := NewUserService(store)
    
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        service.CreateUser("Alice", "alice@example.com", 25)
    }
}
```

## 🎯 Exercises

### Exercise 1: String Utils
Write tests for string utility functions (reverse, palindrome, etc.)

### Exercise 2: HTTP Handler
Test an HTTP handler with different request scenarios.

### Exercise 3: Concurrent Code
Test concurrent code with race detection.

### Solutions

```go
// Exercise 1
func TestReverse(t *testing.T) {
    tests := []struct {
        input    string
        expected string
    }{
        {"hello", "olleh"},
        {"Go", "oG"},
        {"", ""},
        {"a", "a"},
    }
    
    for _, tt := range tests {
        result := Reverse(tt.input)
        if result != tt.expected {
            t.Errorf("Reverse(%q) = %q; want %q",
                tt.input, result, tt.expected)
        }
    }
}

// Run with race detector
// go test -race
```

## 🔑 Key Takeaways

- Tests go in `*_test.go` files
- Use table-driven tests for multiple scenarios
- Mock dependencies using interfaces
- Use `t.Helper()` for helper functions
- Benchmark with `-bench` flag
- Check coverage with `-cover` flag
- Run parallel tests with `t.Parallel()`
- Use `httptest` for testing HTTP handlers

## 📖 Next Steps

Continue to [Chapter 12: Packages](12-packages.md) to learn about organizing code into packages.

