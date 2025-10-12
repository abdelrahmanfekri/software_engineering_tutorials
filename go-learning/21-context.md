# Chapter 21: Context Package

## 📋 What is Context?

Context carries deadlines, cancellation signals, and request-scoped values across API boundaries and between processes.

```go
import "context"

type Context interface {
    Deadline() (deadline time.Time, ok bool)
    Done() <-chan struct{}
    Err() error
    Value(key interface{}) interface{}
}
```

## 🎯 Creating Contexts

### Background and TODO

```go
// Background returns empty context (root context)
ctx := context.Background()

// TODO returns empty context for placeholder
ctx := context.TODO()
```

### WithCancel

```go
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    go func() {
        time.Sleep(2 * time.Second)
        cancel()  // Cancel after 2 seconds
    }()
    
    doWork(ctx)
}

func doWork(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Work cancelled:", ctx.Err())
            return
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}
```

### WithTimeout

```go
func fetchData(url string) ([]byte, error) {
    // 5 second timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    return io.ReadAll(resp.Body)
}
```

### WithDeadline

```go
func processUntil(deadline time.Time) {
    ctx, cancel := context.WithDeadline(context.Background(), deadline)
    defer cancel()
    
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Deadline exceeded:", ctx.Err())
            return
        default:
            // Do work
            time.Sleep(100 * time.Millisecond)
        }
    }
}

// Usage
deadline := time.Now().Add(3 * time.Second)
processUntil(deadline)
```

### WithValue

```go
type key string

const userIDKey key = "userID"
const requestIDKey key = "requestID"

func main() {
    ctx := context.Background()
    ctx = context.WithValue(ctx, userIDKey, 123)
    ctx = context.WithValue(ctx, requestIDKey, "abc-123")
    
    handleRequest(ctx)
}

func handleRequest(ctx context.Context) {
    userID := ctx.Value(userIDKey).(int)
    requestID := ctx.Value(requestIDKey).(string)
    
    fmt.Printf("Processing request %s for user %d\n", requestID, userID)
}
```

## 🔄 Context Patterns

### HTTP Server with Context

```go
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    // Add timeout
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    // Add values
    ctx = context.WithValue(ctx, requestIDKey, generateID())
    
    result, err := processRequest(ctx)
    if err != nil {
        if err == context.DeadlineExceeded {
            http.Error(w, "Request timeout", http.StatusRequestTimeout)
            return
        }
        if err == context.Canceled {
            http.Error(w, "Request cancelled", http.StatusBadRequest)
            return
        }
        http.Error(w, "Internal error", http.StatusInternalServerError)
        return
    }
    
    json.NewEncoder(w).Encode(result)
}

func processRequest(ctx context.Context) (interface{}, error) {
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    case result := <-doSomeWork(ctx):
        return result, nil
    }
}
```

### Database Queries with Context

```go
func getUserByID(ctx context.Context, db *sql.DB, id int) (*User, error) {
    query := `SELECT id, name, email FROM users WHERE id = $1`
    
    var user User
    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    
    return &user, nil
}

// Usage with timeout
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

user, err := getUserByID(ctx, db, 123)
```

### Goroutines with Context

```go
func worker(ctx context.Context, id int, jobs <-chan int, results chan<- int) {
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("Worker %d stopping: %v\n", id, ctx.Err())
            return
        case job := <-jobs:
            // Process job
            result := job * 2
            
            select {
            case results <- result:
            case <-ctx.Done():
                return
            }
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    jobs := make(chan int, 100)
    results := make(chan int, 100)
    
    // Start workers
    for w := 1; w <= 3; w++ {
        go worker(ctx, w, jobs, results)
    }
    
    // Send jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for a := 1; a <= 5; a++ {
        <-results
    }
    
    cancel()  // Cancel workers
    time.Sleep(time.Second)
}
```

## 🎨 Advanced Patterns

### Context Middleware (HTTP)

```go
func withRequestID(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        requestID := generateRequestID()
        ctx := context.WithValue(r.Context(), requestIDKey, requestID)
        next(w, r.WithContext(ctx))
    }
}

func withUserAuth(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        
        user, err := validateToken(token)
        if err != nil {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        
        ctx := context.WithValue(r.Context(), userKey, user)
        next(w, r.WithContext(ctx))
    }
}

// Usage
http.HandleFunc("/api/data", withRequestID(withUserAuth(dataHandler)))
```

### Fan-out Pattern with Context

```go
func search(ctx context.Context, query string) []Result {
    results := make(chan Result)
    
    // Search multiple sources concurrently
    go func() { results <- searchDatabase(ctx, query) }()
    go func() { results <- searchCache(ctx, query) }()
    go func() { results <- searchAPI(ctx, query) }()
    
    var allResults []Result
    
    for i := 0; i < 3; i++ {
        select {
        case r := <-results:
            allResults = append(allResults, r)
        case <-ctx.Done():
            return allResults
        }
    }
    
    return allResults
}
```

### Pipeline with Context

```go
func generator(ctx context.Context, nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            select {
            case out <- n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func square(ctx context.Context, in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            select {
            case out <- n * n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    in := generator(ctx, 1, 2, 3, 4, 5)
    out := square(ctx, in)
    
    for n := range out {
        fmt.Println(n)
    }
}
```

## 💼 Complete Example: API Client with Context

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type APIClient struct {
    baseURL    string
    httpClient *http.Client
}

func NewAPIClient(baseURL string) *APIClient {
    return &APIClient{
        baseURL: baseURL,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *APIClient) Get(ctx context.Context, path string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+path, nil)
    if err != nil {
        return nil, err
    }
    
    // Add request ID from context
    if requestID, ok := ctx.Value(requestIDKey).(string); ok {
        req.Header.Set("X-Request-ID", requestID)
    }
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API error: %d", resp.StatusCode)
    }
    
    return io.ReadAll(resp.Body)
}

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func (c *APIClient) GetUser(ctx context.Context, id int) (*User, error) {
    data, err := c.Get(ctx, fmt.Sprintf("/users/%d", id))
    if err != nil {
        return nil, err
    }
    
    var user User
    if err := json.Unmarshal(data, &user); err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (c *APIClient) GetUsers(ctx context.Context) ([]User, error) {
    data, err := c.Get(ctx, "/users")
    if err != nil {
        return nil, err
    }
    
    var users []User
    if err := json.Unmarshal(data, &users); err != nil {
        return nil, err
    }
    
    return users, nil
}

func main() {
    client := NewAPIClient("https://api.example.com")
    
    // Request with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    // Add request ID
    ctx = context.WithValue(ctx, requestIDKey, "req-123")
    
    users, err := client.GetUsers(ctx)
    if err != nil {
        if err == context.DeadlineExceeded {
            fmt.Println("Request timed out")
        } else {
            fmt.Println("Error:", err)
        }
        return
    }
    
    for _, user := range users {
        fmt.Printf("User: %s (%s)\n", user.Name, user.Email)
    }
}
```

## 🔑 Key Takeaways

- Context carries cancellation signals and deadlines
- Always pass context as first parameter
- Use `context.Background()` as root context
- Use `WithTimeout` for time-limited operations
- Use `WithCancel` for manual cancellation
- Use `WithValue` sparingly (only for request-scoped data)
- Always check `ctx.Done()` in long-running operations
- Don't store contexts in structs

## 📖 Next Steps

Continue to [Chapter 22: Reflection](22-reflection.md) to learn about runtime type inspection in Go.

