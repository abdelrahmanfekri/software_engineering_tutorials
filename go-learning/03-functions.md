# Chapter 3: Functions

## 📚 Basic Functions

### Function Declaration

```go
// Basic function
func greet() {
    fmt.Println("Hello!")
}

// Function with parameters
func greetPerson(name string) {
    fmt.Println("Hello,", name)
}

// Function with return value
func add(a int, b int) int {
    return a + b
}

// Shorthand for same-type parameters
func multiply(a, b, c int) int {
    return a * b * c
}

// Function with multiple return values
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Named return values
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return  // Naked return (returns x and y)
}
```

### Calling Functions

```go
func main() {
    greet()
    greetPerson("Alice")
    
    result := add(5, 3)
    fmt.Println("5 + 3 =", result)
    
    // Multiple return values
    quotient, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", quotient)
    }
    
    // Ignore return value with _
    _, err = divide(10, 0)
    if err != nil {
        fmt.Println("Error:", err)
    }
    
    x, y := split(17)
    fmt.Println(x, y)
}
```

## 🎯 Variadic Functions

Functions that accept variable number of arguments:

```go
// Variadic function
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

func printAll(prefix string, values ...interface{}) {
    fmt.Print(prefix, ": ")
    for _, v := range values {
        fmt.Print(v, " ")
    }
    fmt.Println()
}

func main() {
    // Call with any number of arguments
    fmt.Println(sum())           // 0
    fmt.Println(sum(1))          // 1
    fmt.Println(sum(1, 2, 3))    // 6
    
    // Spread slice into variadic function
    numbers := []int{1, 2, 3, 4, 5}
    fmt.Println(sum(numbers...))  // 15
    
    printAll("Values", 1, "hello", 3.14, true)
}
```

## 🔄 Higher-Order Functions

Functions that accept or return functions:

```go
// Function as parameter
func apply(nums []int, fn func(int) int) []int {
    result := make([]int, len(nums))
    for i, num := range nums {
        result[i] = fn(num)
    }
    return result
}

// Function as return value
func multiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}

func main() {
    numbers := []int{1, 2, 3, 4, 5}
    
    // Pass function as argument
    squared := apply(numbers, func(x int) int {
        return x * x
    })
    fmt.Println(squared)  // [1 4 9 16 25]
    
    // Use returned function
    double := multiplier(2)
    triple := multiplier(3)
    
    fmt.Println(double(5))  // 10
    fmt.Println(triple(5))  // 15
}
```

## 📦 Closures

Functions that capture variables from their surrounding scope:

```go
// Basic closure
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

// Closure with parameters
func adder(initial int) func(int) int {
    sum := initial
    return func(x int) int {
        sum += x
        return sum
    }
}

func main() {
    // Each closure has its own state
    c1 := counter()
    c2 := counter()
    
    fmt.Println(c1())  // 1
    fmt.Println(c1())  // 2
    fmt.Println(c2())  // 1
    fmt.Println(c1())  // 3
    
    // Closure with state
    add := adder(10)
    fmt.Println(add(5))   // 15
    fmt.Println(add(10))  // 25
    fmt.Println(add(3))   // 28
}
```

## 🎭 Anonymous Functions

Functions without names:

```go
func main() {
    // Define and call immediately
    func() {
        fmt.Println("Anonymous function")
    }()
    
    // Assign to variable
    greet := func(name string) {
        fmt.Println("Hello,", name)
    }
    greet("Alice")
    
    // Use in expressions
    numbers := []int{1, 2, 3, 4, 5}
    
    // Filter even numbers
    var evens []int
    for _, n := range numbers {
        if func(x int) bool { return x%2 == 0 }(n) {
            evens = append(evens, n)
        }
    }
    fmt.Println(evens)  // [2 4]
}
```

## 🔁 Recursion

Functions that call themselves:

```go
// Factorial
func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

// Fibonacci
func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

// Optimized Fibonacci with memoization
func fibMemo() func(int) int {
    cache := make(map[int]int)
    var fib func(int) int
    
    fib = func(n int) int {
        if n <= 1 {
            return n
        }
        if val, ok := cache[n]; ok {
            return val
        }
        result := fib(n-1) + fib(n-2)
        cache[n] = result
        return result
    }
    
    return fib
}

func main() {
    fmt.Println(factorial(5))   // 120
    fmt.Println(fibonacci(10))  // 55
    
    fib := fibMemo()
    fmt.Println(fib(50))  // Fast with memoization
}
```

## ⚡ Defer Statement

Defer postpones function execution until surrounding function returns:

```go
func main() {
    // Deferred calls are executed in LIFO order
    defer fmt.Println("World")
    defer fmt.Println("Hello")
    fmt.Println("!")
}
// Output: !, Hello, World

// Common use: Resource cleanup
func readFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()  // Ensures file is closed
    
    // Read file...
    // Even if error occurs, file.Close() will be called
    
    return nil
}

// Defer with parameters (evaluated immediately)
func deferExample() {
    i := 0
    defer fmt.Println(i)  // Prints 0 (value when defer was called)
    i++
    fmt.Println(i)  // Prints 1
}

// Defer in loops (careful!)
func deferLoop() {
    for i := 0; i < 5; i++ {
        defer fmt.Println(i)
    }
    // Prints: 4 3 2 1 0 (LIFO order)
}
```

## 💥 Panic and Recover

Panic stops normal execution, recover regains control:

```go
// Panic
func mustBePositive(n int) {
    if n < 0 {
        panic("number must be positive")
    }
    fmt.Println("Number is:", n)
}

// Recover from panic
func safeFunction() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from:", r)
        }
    }()
    
    mustBePositive(-1)  // This will panic
    fmt.Println("This won't print")
}

// Practical example: HTTP server
func handler(w http.ResponseWriter, r *http.Request) {
    defer func() {
        if err := recover(); err != nil {
            log.Printf("Panic: %v", err)
            http.Error(w, "Internal Server Error", 500)
        }
    }()
    
    // Handle request (might panic)
}

func main() {
    safeFunction()
    fmt.Println("Program continues")
}
```

## 🎨 Function Examples

### Map Function

```go
func mapInts(arr []int, fn func(int) int) []int {
    result := make([]int, len(arr))
    for i, v := range arr {
        result[i] = fn(v)
    }
    return result
}

nums := []int{1, 2, 3, 4, 5}
squared := mapInts(nums, func(x int) int { return x * x })
// [1, 4, 9, 16, 25]
```

### Filter Function

```go
func filter(arr []int, predicate func(int) bool) []int {
    result := []int{}
    for _, v := range arr {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

nums := []int{1, 2, 3, 4, 5, 6}
evens := filter(nums, func(x int) bool { return x%2 == 0 })
// [2, 4, 6]
```

### Reduce Function

```go
func reduce(arr []int, initial int, fn func(int, int) int) int {
    result := initial
    for _, v := range arr {
        result = fn(result, v)
    }
    return result
}

nums := []int{1, 2, 3, 4, 5}
sum := reduce(nums, 0, func(acc, x int) int { return acc + x })
// 15
```

## 🎯 Complete Example: Calculator

```go
package main

import (
    "fmt"
    "math"
)

type Operation func(float64, float64) float64

func calculate(a, b float64, op Operation) float64 {
    return op(a, b)
}

func getOperation(operator string) (Operation, error) {
    operations := map[string]Operation{
        "+": func(a, b float64) float64 { return a + b },
        "-": func(a, b float64) float64 { return a - b },
        "*": func(a, b float64) float64 { return a * b },
        "/": func(a, b float64) float64 { return a / b },
        "^": func(a, b float64) float64 { return math.Pow(a, b) },
    }
    
    if op, exists := operations[operator]; exists {
        return op, nil
    }
    return nil, fmt.Errorf("unknown operator: %s", operator)
}

func main() {
    a, b := 10.0, 3.0
    
    operators := []string{"+", "-", "*", "/", "^"}
    
    for _, op := range operators {
        operation, err := getOperation(op)
        if err != nil {
            fmt.Println(err)
            continue
        }
        
        result := calculate(a, b, operation)
        fmt.Printf("%.1f %s %.1f = %.2f\n", a, op, b, result)
    }
}
```

## 🎯 Exercises

### Exercise 1: Prime Checker
Write a function that checks if a number is prime.

### Exercise 2: Function Composer
Create a function that composes two functions: `compose(f, g)(x)` = `f(g(x))`

### Exercise 3: Retry Logic
Write a function that retries an operation up to N times with exponential backoff.

### Solutions

```go
// Exercise 1: Prime Checker
func isPrime(n int) bool {
    if n <= 1 {
        return false
    }
    if n <= 3 {
        return true
    }
    if n%2 == 0 || n%3 == 0 {
        return false
    }
    
    for i := 5; i*i <= n; i += 6 {
        if n%i == 0 || n%(i+2) == 0 {
            return false
        }
    }
    return true
}

// Exercise 2: Function Composer
func compose(f, g func(int) int) func(int) int {
    return func(x int) int {
        return f(g(x))
    }
}

// Usage
double := func(x int) int { return x * 2 }
square := func(x int) int { return x * x }
doubleSquare := compose(double, square)
fmt.Println(doubleSquare(3))  // 18 (2 * 3^2)

// Exercise 3: Retry Logic
func retry(attempts int, sleep time.Duration, fn func() error) error {
    for i := 0; i < attempts; i++ {
        if err := fn(); err != nil {
            if i < attempts-1 {
                time.Sleep(sleep)
                sleep *= 2  // Exponential backoff
                continue
            }
            return err
        }
        return nil
    }
    return nil
}

// Usage
err := retry(3, time.Second, func() error {
    // Some operation that might fail
    return doSomething()
})
```

## 🔑 Key Takeaways

- Functions are first-class citizens in Go
- Multiple return values are idiomatic (especially for errors)
- Use named return values for clarity
- Closures capture variables from surrounding scope
- Defer ensures cleanup code runs
- Panic/recover is for exceptional situations (not regular error handling)
- Variadic functions accept variable arguments

## 📖 Next Steps

Continue to [Chapter 4: Data Structures](04-data-structures.md) to learn about arrays, slices, and maps.

