# Chapter 12: Packages

## 📦 Package Basics

Every Go file belongs to a package. Packages help organize and reuse code.

### Package Declaration

```go
// File: math/calculator.go
package math

func Add(a, b int) int {
    return a + b
}
```

### Importing Packages

```go
package main

import (
    "fmt"           // Standard library
    "math/rand"     // Nested standard library
    
    "github.com/user/project/math"  // External package
)

func main() {
    fmt.Println(math.Add(2, 3))
}
```

### Import Variations

```go
// Standard import
import "fmt"

// Multiple imports
import (
    "fmt"
    "os"
)

// Aliased import
import f "fmt"

// Dot import (not recommended)
import . "fmt"
// Println("hello")  // Can use without package name

// Blank import (for side effects only)
import _ "github.com/lib/pq"
```

## 🔍 Visibility Rules

### Exported vs Unexported

```go
package user

// Exported (starts with capital letter)
type User struct {
    Name  string  // Exported field
    Email string  // Exported field
    age   int     // unexported field (private)
}

// Exported function
func NewUser(name string) *User {
    return &User{Name: name}
}

// unexported function (private)
func validateEmail(email string) bool {
    return strings.Contains(email, "@")
}
```

### Using Exported Items

```go
package main

import "myapp/user"

func main() {
    // Can use exported items
    u := user.NewUser("Alice")
    fmt.Println(u.Name)  // OK
    
    // Cannot use unexported items
    // fmt.Println(u.age)  // ERROR: unexported field
    // user.validateEmail("test")  // ERROR: unexported function
}
```

## 📂 Package Organization

### Single Package

```
calculator/
├── add.go
├── subtract.go
├── multiply.go
└── divide.go
```

```go
// All files in same package
package calculator

// add.go
func Add(a, b int) int {
    return a + b
}

// subtract.go
func Subtract(a, b int) int {
    return a - b
}
```

### Multiple Packages

```
myapp/
├── main.go
├── user/
│   ├── user.go
│   └── validation.go
├── database/
│   └── db.go
└── api/
    ├── handler.go
    └── middleware.go
```

### Internal Packages

```
myapp/
├── main.go
├── internal/
│   └── helper/
│       └── helper.go
└── pkg/
    └── util/
        └── util.go
```

- `internal/` packages only accessible within parent tree
- `pkg/` packages intended for external use

## 🎯 Package Patterns

### Constructor Pattern

```go
package user

type User struct {
    id    int
    name  string
    email string
}

// Constructor
func NewUser(name, email string) *User {
    return &User{
        id:    generateID(),
        name:  name,
        email: email,
    }
}

// Constructor with validation
func NewUserSafe(name, email string) (*User, error) {
    if name == "" {
        return nil, errors.New("name required")
    }
    if !strings.Contains(email, "@") {
        return nil, errors.New("invalid email")
    }
    return &User{
        id:    generateID(),
        name:  name,
        email: email,
    }, nil
}
```

### Package Initialization

```go
package config

import "os"

var (
    DatabaseURL string
    Port        string
)

// init runs automatically before main
func init() {
    DatabaseURL = os.Getenv("DATABASE_URL")
    if DatabaseURL == "" {
        DatabaseURL = "localhost:5432"
    }
    
    Port = os.Getenv("PORT")
    if Port == "" {
        Port = "8080"
    }
}
```

Multiple init functions execute in order:

```go
func init() {
    fmt.Println("First init")
}

func init() {
    fmt.Println("Second init")
}

// Execution order:
// 1. Package-level variables
// 2. init functions (in order)
// 3. main function
```

### Singleton Pattern

```go
package database

import "sync"

var (
    instance *Database
    once     sync.Once
)

type Database struct {
    connection string
}

func GetInstance() *Database {
    once.Do(func() {
        instance = &Database{
            connection: "db://localhost:5432",
        }
    })
    return instance
}
```

## 📦 Package Documentation

### Documenting Packages

```go
// Package user provides user management functionality.
// It includes user creation, validation, and storage.
//
// Example usage:
//
//     user := user.NewUser("Alice", "alice@example.com")
//     if err := user.Validate(); err != nil {
//         log.Fatal(err)
//     }
//
package user

// User represents a user in the system.
// It contains personal information and authentication data.
type User struct {
    // ID is the unique identifier for the user
    ID int
    
    // Name is the user's full name
    Name string
    
    // Email is the user's email address
    Email string
}

// NewUser creates a new user with the given name and email.
// It automatically generates a unique ID.
//
// Example:
//
//     user := NewUser("Alice", "alice@example.com")
//
func NewUser(name, email string) *User {
    return &User{
        ID:    generateID(),
        Name:  name,
        Email: email,
    }
}
```

### Viewing Documentation

```bash
# View package documentation
go doc user

# View specific function
go doc user.NewUser

# View in browser
go doc -http=:6060
# Visit http://localhost:6060
```

## 🎨 Advanced Package Patterns

### Facade Pattern

```go
// Package api provides a simple facade for complex operations
package api

import (
    "myapp/database"
    "myapp/cache"
    "myapp/logging"
)

type API struct {
    db     *database.DB
    cache  *cache.Cache
    logger *logging.Logger
}

func New() *API {
    return &API{
        db:     database.Connect(),
        cache:  cache.New(),
        logger: logging.New(),
    }
}

// Simple API hides complexity
func (a *API) GetUser(id int) (*User, error) {
    // Check cache
    if user := a.cache.Get(id); user != nil {
        return user, nil
    }
    
    // Query database
    user, err := a.db.FindUser(id)
    if err != nil {
        a.logger.Error(err)
        return nil, err
    }
    
    // Update cache
    a.cache.Set(id, user)
    
    return user, nil
}
```

### Package with Options

```go
package server

type Server struct {
    host    string
    port    int
    timeout time.Duration
}

type Option func(*Server)

func WithHost(host string) Option {
    return func(s *Server) {
        s.host = host
    }
}

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        host:    "localhost",
        port:    8080,
        timeout: 30 * time.Second,
    }
    
    for _, opt := range opts {
        opt(s)
    }
    
    return s
}

// Usage
server := server.NewServer(
    server.WithHost("0.0.0.0"),
    server.WithPort(3000),
)
```

## 💼 Complete Example: Logger Package

```go
// Package logger provides structured logging functionality
package logger

import (
    "fmt"
    "io"
    "os"
    "time"
)

// Level represents log severity
type Level int

const (
    DEBUG Level = iota
    INFO
    WARN
    ERROR
)

func (l Level) String() string {
    switch l {
    case DEBUG:
        return "DEBUG"
    case INFO:
        return "INFO"
    case WARN:
        return "WARN"
    case ERROR:
        return "ERROR"
    default:
        return "UNKNOWN"
    }
}

// Logger handles logging operations
type Logger struct {
    output io.Writer
    level  Level
    prefix string
}

// Option configures Logger
type Option func(*Logger)

// WithLevel sets minimum log level
func WithLevel(level Level) Option {
    return func(l *Logger) {
        l.level = level
    }
}

// WithPrefix sets log prefix
func WithPrefix(prefix string) Option {
    return func(l *Logger) {
        l.prefix = prefix
    }
}

// WithOutput sets output writer
func WithOutput(w io.Writer) Option {
    return func(l *Logger) {
        l.output = w
    }
}

// New creates a new Logger
func New(opts ...Option) *Logger {
    l := &Logger{
        output: os.Stdout,
        level:  INFO,
        prefix: "",
    }
    
    for _, opt := range opts {
        opt(l)
    }
    
    return l
}

// log writes log message
func (l *Logger) log(level Level, format string, args ...interface{}) {
    if level < l.level {
        return
    }
    
    timestamp := time.Now().Format("2006-01-02 15:04:05")
    message := fmt.Sprintf(format, args...)
    
    logLine := fmt.Sprintf("[%s] %s: %s%s\n",
        timestamp, level, l.prefix, message)
    
    l.output.Write([]byte(logLine))
}

// Debug logs debug message
func (l *Logger) Debug(format string, args ...interface{}) {
    l.log(DEBUG, format, args...)
}

// Info logs info message
func (l *Logger) Info(format string, args ...interface{}) {
    l.log(INFO, format, args...)
}

// Warn logs warning message
func (l *Logger) Warn(format string, args ...interface{}) {
    l.log(WARN, format, args...)
}

// Error logs error message
func (l *Logger) Error(format string, args ...interface{}) {
    l.log(ERROR, format, args...)
}

// Usage example
package main

import "myapp/logger"

func main() {
    log := logger.New(
        logger.WithLevel(logger.DEBUG),
        logger.WithPrefix("[MyApp] "),
    )
    
    log.Debug("Debug message")
    log.Info("Info message")
    log.Warn("Warning message")
    log.Error("Error message")
}
```

## 🎯 Exercises

### Exercise 1: String Utils Package
Create a package with string utility functions.

### Exercise 2: Config Package
Build a configuration package that reads from files and environment.

### Exercise 3: Validation Package
Create a reusable validation package.

### Solutions

```go
// Exercise 1: String Utils
package stringutil

import "strings"

// Reverse reverses a string
func Reverse(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}

// IsPalindrome checks if string is palindrome
func IsPalindrome(s string) bool {
    s = strings.ToLower(s)
    return s == Reverse(s)
}

// Exercise 2: Config
package config

import (
    "encoding/json"
    "os"
)

type Config struct {
    Database struct {
        Host string `json:"host"`
        Port int    `json:"port"`
    } `json:"database"`
}

func Load(path string) (*Config, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    var cfg Config
    if err := json.NewDecoder(file).Decode(&cfg); err != nil {
        return nil, err
    }
    
    return &cfg, nil
}
```

## 🔑 Key Takeaways

- Package name should be lowercase, short, and descriptive
- Exported names start with uppercase letter
- Use `internal/` for packages not meant to be imported externally
- Document packages and exported items
- Use `init()` for package initialization
- Follow standard project layout
- Keep packages focused and cohesive

## 📖 Next Steps

Continue to [Chapter 13: Modules and Dependencies](13-modules.md) to learn about Go modules and dependency management.

