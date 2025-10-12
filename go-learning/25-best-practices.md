# Chapter 25: Best Practices

## 📝 Code Style

### Naming Conventions

```go
// Good
var userCount int
func getUserByID(id int) *User {}
type HTTPServer struct {}

// Bad
var user_count int
func get_user_by_id(id int) *User {}
type Http_Server struct {}
```

### Package Names

```go
// Good
package user
package http

// Bad
package userPackage
package HTTPPackage
```

## 🎯 Error Handling

```go
// Always handle errors
file, err := os.Open("file.txt")
if err != nil {
    return fmt.Errorf("opening file: %w", err)
}
defer file.Close()

// Don't ignore errors
_ = someFunction()  // Bad

// Use errors.Is and errors.As
if errors.Is(err, ErrNotFound) {
    // Handle not found
}
```

## 🔒 Concurrency

```go
// Use sync.WaitGroup for goroutines
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        // Work
    }(i)
}
wg.Wait()

// Use context for cancellation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// Protect shared state with mutex
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}
```

## 🧪 Testing

```go
// Use table-driven tests
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive", 2, 3, 5},
        {"negative", -2, -3, -5},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("got %d, want %d", result, tt.expected)
            }
        })
    }
}
```

## 📦 Project Structure

```
myproject/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── handler/
│   ├── service/
│   └── repository/
├── pkg/
│   └── utils/
├── api/
├── configs/
├── scripts/
├── go.mod
├── go.sum
└── README.md
```

## 🔑 Key Takeaways

- Follow Go conventions
- Use `gofmt` and `goimports`
- Handle all errors
- Write tests
- Use meaningful names
- Keep functions small
- Document exported items

## 📖 Next Steps

Continue to [Chapter 26: Performance](26-performance.md).

