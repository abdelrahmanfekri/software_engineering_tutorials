# Chapter 26: Performance Optimization

## 📊 Profiling

### CPU Profiling

```go
import "runtime/pprof"

func main() {
    f, _ := os.Create("cpu.prof")
    defer f.Close()
    
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()
    
    // Code to profile
}

// Analyze: go tool pprof cpu.prof
```

### Memory Profiling

```go
import "runtime"

func main() {
    defer func() {
        f, _ := os.Create("mem.prof")
        defer f.Close()
        runtime.GC()
        pprof.WriteHeapProfile(f)
    }()
    
    // Code to profile
}
```

### Benchmarking

```go
func BenchmarkSum(b *testing.B) {
    data := make([]int, 1000)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = sum(data)
    }
}

// Run: go test -bench=. -benchmem
```

## ⚡ Optimization Tips

### Use Pointers for Large Structs

```go
// Bad: Copies entire struct
func process(user User) {}

// Good: Passes pointer
func process(user *User) {}
```

### Preallocate Slices

```go
// Bad
var items []int
for i := 0; i < 1000; i++ {
    items = append(items, i)  // Multiple allocations
}

// Good
items := make([]int, 0, 1000)  // Preallocate capacity
for i := 0; i < 1000; i++ {
    items = append(items, i)  // One allocation
}
```

### Use sync.Pool for Temporary Objects

```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func process() {
    buf := bufferPool.Get().(*bytes.Buffer)
    defer bufferPool.Put(buf)
    
    buf.Reset()
    // Use buf
}
```

### String Builder

```go
// Bad
var s string
for i := 0; i < 1000; i++ {
    s += "text"  // Creates new string each time
}

// Good
var sb strings.Builder
for i := 0; i < 1000; i++ {
    sb.WriteString("text")
}
s := sb.String()
```

## 🔑 Key Takeaways

- Profile before optimizing
- Preallocate when size is known
- Use pointers for large structs
- Reuse objects with sync.Pool
- Avoid premature optimization

## 📖 Next Steps

Continue to [Chapter 27: Project Structure](27-project-structure.md).

