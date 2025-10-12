# Chapter 8: Goroutines and Channels

## 🚀 Goroutines

Goroutines are lightweight threads managed by the Go runtime.

### Basic Goroutine

```go
package main

import (
    "fmt"
    "time"
)

func say(s string) {
    for i := 0; i < 3; i++ {
        fmt.Println(s)
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    // Regular function call (synchronous)
    // say("hello")
    
    // Goroutine (asynchronous)
    go say("hello")
    say("world")
    
    // Output is interleaved:
    // world
    // hello
    // world
    // hello
    // world
    // hello
}
```

### Anonymous Goroutines

```go
func main() {
    // Anonymous goroutine
    go func() {
        fmt.Println("Running in goroutine")
    }()
    
    // With parameters
    message := "Hello"
    go func(msg string) {
        fmt.Println(msg)
    }(message)
    
    time.Sleep(time.Second)  // Wait for goroutines
}
```

### Multiple Goroutines

```go
func main() {
    // Launch 10 goroutines
    for i := 0; i < 10; i++ {
        go func(id int) {
            fmt.Printf("Goroutine %d\n", id)
        }(i)  // Pass i as parameter!
    }
    
    time.Sleep(time.Second)
}
```

## 📬 Channels

Channels allow goroutines to communicate safely.

### Basic Channel Operations

```go
// Create channel
ch := make(chan int)

// Send to channel (blocks until received)
ch <- 42

// Receive from channel (blocks until sent)
value := <-ch

// Send in goroutine
go func() {
    ch <- 42
}()

// Receive in main
value := <-ch
fmt.Println(value)  // 42
```

### Buffered Channels

```go
// Unbuffered channel (capacity 0)
ch1 := make(chan int)

// Buffered channel (capacity 3)
ch2 := make(chan int, 3)

// Can send without blocking until buffer is full
ch2 <- 1
ch2 <- 2
ch2 <- 3
// ch2 <- 4  // Would block!

fmt.Println(<-ch2)  // 1
fmt.Println(<-ch2)  // 2

// Now can send again
ch2 <- 4
```

### Channel Direction

```go
// Send-only channel
func send(ch chan<- int) {
    ch <- 42
    // val := <-ch  // ERROR: cannot receive
}

// Receive-only channel
func receive(ch <-chan int) {
    val := <-ch
    fmt.Println(val)
    // ch <- 42  // ERROR: cannot send
}

func main() {
    ch := make(chan int)
    go send(ch)
    receive(ch)
}
```

### Closing Channels

```go
func main() {
    ch := make(chan int, 3)
    
    // Send values
    ch <- 1
    ch <- 2
    ch <- 3
    
    // Close channel (no more sends)
    close(ch)
    
    // Can still receive from closed channel
    fmt.Println(<-ch)  // 1
    fmt.Println(<-ch)  // 2
    fmt.Println(<-ch)  // 3
    fmt.Println(<-ch)  // 0 (zero value)
    
    // Check if channel is closed
    val, ok := <-ch
    if !ok {
        fmt.Println("Channel closed")
    }
}
```

### Range over Channel

```go
func main() {
    ch := make(chan int, 5)
    
    // Send values in goroutine
    go func() {
        for i := 0; i < 5; i++ {
            ch <- i
        }
        close(ch)  // Important: close when done
    }()
    
    // Range automatically stops when channel is closed
    for val := range ch {
        fmt.Println(val)
    }
}
```

## 🎯 Select Statement

Select lets you wait on multiple channel operations.

### Basic Select

```go
func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "one"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "two"
    }()
    
    // Wait for both
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received:", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received:", msg2)
        }
    }
}
```

### Select with Default

```go
func main() {
    ch := make(chan int, 1)
    
    select {
    case ch <- 1:
        fmt.Println("Sent")
    default:
        fmt.Println("Buffer full, not sent")
    }
    
    select {
    case val := <-ch:
        fmt.Println("Received:", val)
    default:
        fmt.Println("No data available")
    }
}
```

### Timeout Pattern

```go
func main() {
    ch := make(chan string)
    
    go func() {
        time.Sleep(2 * time.Second)
        ch <- "result"
    }()
    
    select {
    case res := <-ch:
        fmt.Println(res)
    case <-time.After(1 * time.Second):
        fmt.Println("Timeout!")
    }
}
```

## 🔒 Synchronization

### WaitGroup

```go
import "sync"

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()  // Decrement counter when done
    
    fmt.Printf("Worker %d starting\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup
    
    for i := 1; i <= 5; i++ {
        wg.Add(1)  // Increment counter
        go worker(i, &wg)
    }
    
    wg.Wait()  // Wait for all goroutines
    fmt.Println("All workers done")
}
```

### Mutex (Mutual Exclusion)

```go
import "sync"

type SafeCounter struct {
    mu    sync.Mutex
    value int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}

func main() {
    counter := &SafeCounter{}
    var wg sync.WaitGroup
    
    // 1000 goroutines incrementing
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }
    
    wg.Wait()
    fmt.Println("Final count:", counter.Value())  // 1000
}
```

### RWMutex (Read-Write Mutex)

```go
import "sync"

type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func NewCache() *Cache {
    return &Cache{
        data: make(map[string]string),
    }
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()  // Exclusive lock
    defer c.mu.Unlock()
    c.data[key] = value
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()  // Read lock (multiple readers allowed)
    defer c.mu.RUnlock()
    val, ok := c.data[key]
    return val, ok
}
```

### Once (Execute Only Once)

```go
import "sync"

var (
    instance *Database
    once     sync.Once
)

type Database struct {
    conn string
}

func GetDatabase() *Database {
    once.Do(func() {
        fmt.Println("Creating database instance")
        instance = &Database{conn: "db://localhost"}
    })
    return instance
}

func main() {
    // Only creates once, even with multiple goroutines
    for i := 0; i < 10; i++ {
        go func() {
            db := GetDatabase()
            fmt.Println(db)
        }()
    }
    
    time.Sleep(time.Second)
}
```

## 🎨 Common Patterns

### Worker Pool

```go
func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job)
        time.Sleep(time.Second)
        results <- job * 2
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)
    
    // Start 3 workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }
    
    // Send 9 jobs
    for j := 1; j <= 9; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for a := 1; a <= 9; a++ {
        <-results
    }
}
```

### Pipeline

```go
func generator(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

func main() {
    // Set up pipeline
    c := generator(2, 3, 4)
    out := square(c)
    
    // Consume output
    for result := range out {
        fmt.Println(result)  // 4, 9, 16
    }
}
```

### Fan-out, Fan-in

```go
func fanOut(in <-chan int, n int) []<-chan int {
    channels := make([]<-chan int, n)
    for i := 0; i < n; i++ {
        channels[i] = square(in)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for n := range c {
                out <- n
            }
        }(ch)
    }
    
    go func() {
        wg.Wait()
        close(out)
    }()
    
    return out
}

func main() {
    in := generator(1, 2, 3, 4, 5)
    
    // Fan out to 3 workers
    channels := fanOut(in, 3)
    
    // Fan in to single channel
    result := fanIn(channels...)
    
    for n := range result {
        fmt.Println(n)
    }
}
```

### Context for Cancellation

```go
import "context"

func worker(ctx context.Context, id int) {
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("Worker %d cancelled\n", id)
            return
        default:
            fmt.Printf("Worker %d working\n", id)
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    for i := 1; i <= 3; i++ {
        go worker(ctx, i)
    }
    
    time.Sleep(2 * time.Second)
    cancel()  // Cancel all workers
    time.Sleep(time.Second)
}
```

## 💼 Complete Example: URL Fetcher

```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "sync"
    "time"
)

type Result struct {
    URL    string
    Status int
    Length int
    Err    error
}

func fetch(url string) Result {
    start := time.Now()
    resp, err := http.Get(url)
    if err != nil {
        return Result{URL: url, Err: err}
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return Result{URL: url, Status: resp.StatusCode, Err: err}
    }
    
    duration := time.Since(start)
    fmt.Printf("%s: %d bytes in %.2fs\n", url, len(body), duration.Seconds())
    
    return Result{
        URL:    url,
        Status: resp.StatusCode,
        Length: len(body),
    }
}

func fetchAll(urls []string) []Result {
    results := make([]Result, len(urls))
    var wg sync.WaitGroup
    
    for i, url := range urls {
        wg.Add(1)
        go func(index int, u string) {
            defer wg.Done()
            results[index] = fetch(u)
        }(i, url)
    }
    
    wg.Wait()
    return results
}

func main() {
    urls := []string{
        "https://golang.org",
        "https://google.com",
        "https://github.com",
    }
    
    start := time.Now()
    results := fetchAll(urls)
    duration := time.Since(start)
    
    fmt.Printf("\nFetched %d URLs in %.2fs\n", len(urls), duration.Seconds())
    
    for _, result := range results {
        if result.Err != nil {
            fmt.Printf("Error fetching %s: %v\n", result.URL, result.Err)
        } else {
            fmt.Printf("%s: Status %d, %d bytes\n",
                result.URL, result.Status, result.Length)
        }
    }
}
```

## 🎯 Exercises

### Exercise 1: Concurrent Counter
Create a thread-safe counter with increment, decrement, and value methods.

### Exercise 2: Rate Limiter
Implement a rate limiter that allows N requests per second.

### Exercise 3: Concurrent Map
Build a thread-safe map with Get, Set, and Delete operations.

### Solutions

```go
// Exercise 1: Concurrent Counter
type Counter struct {
    mu    sync.Mutex
    value int
}

func (c *Counter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

func (c *Counter) Decrement() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value--
}

func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}

// Exercise 2: Rate Limiter
type RateLimiter struct {
    tokens chan struct{}
}

func NewRateLimiter(rate int) *RateLimiter {
    rl := &RateLimiter{
        tokens: make(chan struct{}, rate),
    }
    
    // Refill tokens
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        
        for range ticker.C {
            for i := 0; i < rate; i++ {
                select {
                case rl.tokens <- struct{}{}:
                default:
                }
            }
        }
    }()
    
    return rl
}

func (rl *RateLimiter) Wait() {
    <-rl.tokens
}

// Exercise 3: Concurrent Map
type ConcurrentMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func NewConcurrentMap() *ConcurrentMap {
    return &ConcurrentMap{
        data: make(map[string]interface{}),
    }
}

func (cm *ConcurrentMap) Set(key string, value interface{}) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    cm.data[key] = value
}

func (cm *ConcurrentMap) Get(key string) (interface{}, bool) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    val, ok := cm.data[key]
    return val, ok
}

func (cm *ConcurrentMap) Delete(key string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    delete(cm.data, key)
}
```

## 🔑 Key Takeaways

- Goroutines are lightweight (thousands can run concurrently)
- Channels enable safe communication between goroutines
- Use `sync.WaitGroup` to wait for goroutines to finish
- Use `sync.Mutex` to protect shared state
- `select` handles multiple channel operations
- Prefer communication over shared memory ("Don't communicate by sharing memory, share memory by communicating")

## 📖 Next Steps

Continue to [Chapter 9: Concurrency Patterns](09-concurrency-patterns.md) for advanced concurrency patterns and best practices.

