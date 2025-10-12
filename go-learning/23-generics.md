# Chapter 23: Generics (Go 1.18+)

Generics allow you to write functions and types that work with multiple types.

## 🎯 Generic Functions

```go
// Basic generic function
func Print[T any](value T) {
    fmt.Println(value)
}

// Usage
Print[int](42)
Print[string]("hello")
Print(42)  // Type inference

// Generic function with constraint
func Max[T constraints.Ordered](a, b T) T {
    if a > b {
        return a
    }
    return b
}

result := Max(10, 20)        // 20
result := Max(3.14, 2.71)    // 3.14
result := Max("apple", "banana")  // "banana"
```

## 📦 Generic Types

```go
// Generic stack
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// Usage
stack := Stack[int]{}
stack.Push(1)
stack.Push(2)
value, ok := stack.Pop()  // 2, true
```

## 🔧 Type Constraints

```go
// Built-in constraints
import "golang.org/x/exp/constraints"

func Sum[T constraints.Integer](numbers []T) T {
    var total T
    for _, n := range numbers {
        total += n
    }
    return total
}

// Custom constraint
type Number interface {
    int | int64 | float64
}

func Add[T Number](a, b T) T {
    return a + b
}

// Complex constraint
type Stringer interface {
    String() string
}

func PrintAll[T Stringer](items []T) {
    for _, item := range items {
        fmt.Println(item.String())
    }
}
```

## 💼 Practical Examples

```go
// Generic map function
func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

nums := []int{1, 2, 3}
doubled := Map(nums, func(n int) int { return n * 2 })

// Generic filter
func Filter[T any](slice []T, predicate func(T) bool) []T {
    result := []T{}
    for _, v := range slice {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

evens := Filter(nums, func(n int) bool { return n%2 == 0 })
```

## 🔑 Key Takeaways

- Generics reduce code duplication
- Use for data structures and algorithms
- Don't overuse - simple code is better
- Type inference works most of the time

## 📖 Next Steps

Continue to [Chapter 24: Advanced Patterns](24-advanced-patterns.md).

