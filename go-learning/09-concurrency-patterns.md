# Chapter 9: Concurrency Patterns

## 🎯 Advanced Channel Patterns

### Done Channel Pattern

```go
func worker(done <-chan bool) {
    for {
        select {
        case <-done:
            fmt.Println("Worker stopping")
            return
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    done := make(chan bool)
    
    go worker(done)
    
    time.Sleep(2 * time.Second)
    close(done)  // Signal done
    time.Sleep(time.Second)
}
```

### Error Group Pattern

```go
type Result struct {
    Value int
    Err   error
}

func fetchData(id int) (int, error) {
    time.Sleep(time.Duration(id) * 100 * time.Millisecond)
    if id == 3 {
        return 0, fmt.Errorf("error fetching %d", id)
    }
    return id * 10, nil
}

func fetchAll(ids []int) ([]int, error) {
    results := make(chan Result, len(ids))
    
    for _, id := range ids {
        go func(i int) {
            value, err := fetchData(i)
            results <- Result{Value: value, Err: err}
        }(id)
    }
    
    values := make([]int, 0, len(ids))
    for i := 0; i < len(ids); i++ {
        result := <-results
        if result.Err != nil {
            return nil, result.Err
        }
        values = append(values, result.Value)
    }
    
    return values, nil
}
```

### Semaphore Pattern (Limiting Concurrency)

```go
type Semaphore chan struct{}

func NewSemaphore(max int) Semaphore {
    return make(Semaphore, max)
}

func (s Semaphore) Acquire() {
    s <- struct{}{}
}

func (s Semaphore) Release() {
    <-s
}

func main() {
    sem := NewSemaphore(3)  // Max 3 concurrent operations
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            
            sem.Acquire()
            defer sem.Release()
            
            fmt.Printf("Task %d running\n", id)
            time.Sleep(time.Second)
            fmt.Printf("Task %d done\n", id)
        }(i)
    }
    
    wg.Wait()
}
```

## 🔄 Pipeline Patterns

### Generator Pattern

```go
func fibonacci() <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        a, b := 0, 1
        for i := 0; i < 10; i++ {
            ch <- a
            a, b = b, a+b
        }
    }()
    return ch
}

func main() {
    for num := range fibonacci() {
        fmt.Println(num)
    }
}
```

### Filter Pattern

```go
func filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
    }()
    return out
}

func main() {
    // Generate numbers
    nums := generator(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    // Filter even numbers
    evens := filter(nums, func(n int) bool { return n%2 == 0 })
    
    // Consume
    for n := range evens {
        fmt.Println(n)  // 2, 4, 6, 8, 10
    }
}
```

### Merge Pattern

```go
func merge(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    
    output := func(c <-chan int) {
        defer wg.Done()
        for n := range c {
            out <- n
        }
    }
    
    wg.Add(len(channels))
    for _, c := range channels {
        go output(c)
    }
    
    go func() {
        wg.Wait()
        close(out)
    }()
    
    return out
}

func main() {
    ch1 := generator(1, 2, 3)
    ch2 := generator(4, 5, 6)
    ch3 := generator(7, 8, 9)
    
    merged := merge(ch1, ch2, ch3)
    
    for n := range merged {
        fmt.Println(n)
    }
}
```

### Tee Pattern (Split Stream)

```go
func tee(in <-chan int) (<-chan int, <-chan int) {
    out1 := make(chan int)
    out2 := make(chan int)
    
    go func() {
        defer close(out1)
        defer close(out2)
        
        for n := range in {
            out1, out2 := out1, out2  // Local copies for goroutines
            
            for i := 0; i < 2; i++ {
                select {
                case out1 <- n:
                    out1 = nil  // Prevent sending again
                case out2 <- n:
                    out2 = nil
                }
            }
        }
    }()
    
    return out1, out2
}
```

## 🏭 Worker Pool Patterns

### Dynamic Worker Pool

```go
type Job struct {
    ID   int
    Data interface{}
}

type Result struct {
    Job    Job
    Result interface{}
    Err    error
}

type WorkerPool struct {
    jobs    chan Job
    results chan Result
    workers int
}

func NewWorkerPool(workers, bufferSize int) *WorkerPool {
    return &WorkerPool{
        jobs:    make(chan Job, bufferSize),
        results: make(chan Result, bufferSize),
        workers: workers,
    }
}

func (wp *WorkerPool) Start(process func(Job) (interface{}, error)) {
    for i := 0; i < wp.workers; i++ {
        go func(workerID int) {
            for job := range wp.jobs {
                result, err := process(job)
                wp.results <- Result{
                    Job:    job,
                    Result: result,
                    Err:    err,
                }
            }
        }(i)
    }
}

func (wp *WorkerPool) Submit(job Job) {
    wp.jobs <- job
}

func (wp *WorkerPool) Results() <-chan Result {
    return wp.results
}

func (wp *WorkerPool) Close() {
    close(wp.jobs)
}

// Usage
func main() {
    pool := NewWorkerPool(5, 10)
    
    pool.Start(func(job Job) (interface{}, error) {
        time.Sleep(100 * time.Millisecond)
        return job.Data.(int) * 2, nil
    })
    
    // Submit jobs
    go func() {
        for i := 0; i < 20; i++ {
            pool.Submit(Job{ID: i, Data: i})
        }
        pool.Close()
    }()
    
    // Collect results
    count := 0
    for result := range pool.Results() {
        fmt.Printf("Job %d result: %v\n", result.Job.ID, result.Result)
        count++
        if count == 20 {
            break
        }
    }
}
```

## 🔔 Pub/Sub Pattern

```go
type Event struct {
    Topic string
    Data  interface{}
}

type Subscriber struct {
    ID     string
    Events chan Event
}

type PubSub struct {
    mu          sync.RWMutex
    subscribers map[string]map[string]*Subscriber
}

func NewPubSub() *PubSub {
    return &PubSub{
        subscribers: make(map[string]map[string]*Subscriber),
    }
}

func (ps *PubSub) Subscribe(topic string, subscriberID string) *Subscriber {
    ps.mu.Lock()
    defer ps.mu.Unlock()
    
    if ps.subscribers[topic] == nil {
        ps.subscribers[topic] = make(map[string]*Subscriber)
    }
    
    sub := &Subscriber{
        ID:     subscriberID,
        Events: make(chan Event, 10),
    }
    
    ps.subscribers[topic][subscriberID] = sub
    return sub
}

func (ps *PubSub) Unsubscribe(topic string, subscriberID string) {
    ps.mu.Lock()
    defer ps.mu.Unlock()
    
    if subs, ok := ps.subscribers[topic]; ok {
        if sub, ok := subs[subscriberID]; ok {
            close(sub.Events)
            delete(subs, subscriberID)
        }
    }
}

func (ps *PubSub) Publish(topic string, data interface{}) {
    ps.mu.RLock()
    defer ps.mu.RUnlock()
    
    if subs, ok := ps.subscribers[topic]; ok {
        event := Event{Topic: topic, Data: data}
        for _, sub := range subs {
            select {
            case sub.Events <- event:
            default:
                // Drop if buffer full
            }
        }
    }
}

// Usage
func main() {
    ps := NewPubSub()
    
    // Create subscribers
    sub1 := ps.Subscribe("news", "subscriber-1")
    sub2 := ps.Subscribe("news", "subscriber-2")
    
    // Listen for events
    go func() {
        for event := range sub1.Events {
            fmt.Printf("Sub1 received: %v\n", event.Data)
        }
    }()
    
    go func() {
        for event := range sub2.Events {
            fmt.Printf("Sub2 received: %v\n", event.Data)
        }
    }()
    
    // Publish events
    ps.Publish("news", "Breaking news 1")
    ps.Publish("news", "Breaking news 2")
    
    time.Sleep(time.Second)
}
```

## ⏰ Timeout and Cancellation

### Timeout with Context

```go
func doWork(ctx context.Context) error {
    select {
    case <-time.After(2 * time.Second):
        fmt.Println("Work completed")
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()
    
    if err := doWork(ctx); err != nil {
        fmt.Println("Error:", err)  // context deadline exceeded
    }
}
```

### Cancellation with Context

```go
func worker(ctx context.Context, id int) {
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("Worker %d cancelled: %v\n", id, ctx.Err())
            return
        default:
            fmt.Printf("Worker %d working\n", id)
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    // Start workers
    for i := 1; i <= 3; i++ {
        go worker(ctx, i)
    }
    
    time.Sleep(2 * time.Second)
    cancel()  // Cancel all workers
    time.Sleep(time.Second)
}
```

## 🎨 Advanced Patterns

### Futures/Promises

```go
type Future struct {
    result chan interface{}
}

func NewFuture(fn func() interface{}) *Future {
    f := &Future{result: make(chan interface{}, 1)}
    
    go func() {
        f.result <- fn()
    }()
    
    return f
}

func (f *Future) Get() interface{} {
    return <-f.result
}

// Usage
func main() {
    future := NewFuture(func() interface{} {
        time.Sleep(2 * time.Second)
        return "Result from expensive computation"
    })
    
    fmt.Println("Doing other work...")
    time.Sleep(1 * time.Second)
    
    fmt.Println("Getting result...")
    result := future.Get()
    fmt.Println(result)
}
```

### Circuit Breaker

```go
type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    maxFailures  int
    resetTimeout time.Duration
    
    mu           sync.Mutex
    state        State
    failures     int
    lastFailTime time.Time
}

func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        maxFailures:  maxFailures,
        resetTimeout: resetTimeout,
        state:        StateClosed,
    }
}

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mu.Lock()
    
    // Check if we should transition from Open to HalfOpen
    if cb.state == StateOpen {
        if time.Since(cb.lastFailTime) > cb.resetTimeout {
            cb.state = StateHalfOpen
        } else {
            cb.mu.Unlock()
            return fmt.Errorf("circuit breaker is open")
        }
    }
    
    cb.mu.Unlock()
    
    // Execute function
    err := fn()
    
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if err != nil {
        cb.failures++
        cb.lastFailTime = time.Now()
        
        if cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
        
        return err
    }
    
    // Success - reset
    if cb.state == StateHalfOpen {
        cb.state = StateClosed
    }
    cb.failures = 0
    
    return nil
}
```

## 💼 Complete Example: Web Crawler

```go
package main

import (
    "fmt"
    "sync"
)

type Fetcher interface {
    Fetch(url string) (urls []string, err error)
}

type SafeURLSet struct {
    mu      sync.Mutex
    visited map[string]bool
}

func (s *SafeURLSet) Add(url string) bool {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if s.visited[url] {
        return false
    }
    s.visited[url] = true
    return true
}

func Crawl(url string, depth int, fetcher Fetcher, wg *sync.WaitGroup, visited *SafeURLSet) {
    defer wg.Done()
    
    if depth <= 0 {
        return
    }
    
    if !visited.Add(url) {
        return
    }
    
    urls, err := fetcher.Fetch(url)
    if err != nil {
        fmt.Println(err)
        return
    }
    
    fmt.Printf("Found: %s\n", url)
    
    for _, u := range urls {
        wg.Add(1)
        go Crawl(u, depth-1, fetcher, wg, visited)
    }
}

func main() {
    visited := &SafeURLSet{visited: make(map[string]bool)}
    var wg sync.WaitGroup
    
    wg.Add(1)
    go Crawl("https://golang.org/", 4, fetcher, &wg, visited)
    
    wg.Wait()
}

// Mock fetcher
type fakeFetcher map[string]*fakeResult

type fakeResult struct {
    body string
    urls []string
}

func (f fakeFetcher) Fetch(url string) ([]string, error) {
    if res, ok := f[url]; ok {
        return res.urls, nil
    }
    return nil, fmt.Errorf("not found: %s", url)
}

var fetcher = fakeFetcher{
    "https://golang.org/": &fakeResult{
        "The Go Programming Language",
        []string{
            "https://golang.org/pkg/",
            "https://golang.org/cmd/",
        },
    },
    "https://golang.org/pkg/": &fakeResult{
        "Packages",
        []string{
            "https://golang.org/",
            "https://golang.org/cmd/",
        },
    },
}
```

## 🔑 Key Takeaways

- Use done channels for cancellation
- Semaphores limit concurrent operations
- Pipeline patterns enable stream processing
- Worker pools manage concurrent tasks efficiently
- Pub/Sub enables event-driven architectures
- Context provides cancellation and timeouts
- Circuit breakers prevent cascading failures

## 📖 Next Steps

Continue to [Chapter 10: Error Handling](10-error-handling.md) to learn about robust error handling in Go.

