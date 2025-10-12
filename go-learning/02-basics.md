# Chapter 2: Basics - Variables, Types, and Control Flow

## 📦 Package and Imports

Every Go file starts with a package declaration:

```go
package main  // Executable package

import (
    "fmt"      // Single import
    "strings"
)

// Or single line
import "fmt"
```

## 🔤 Variables

### Declaration Methods

```go
package main

import "fmt"

func main() {
    // Method 1: var with type
    var name string = "Go"
    
    // Method 2: var with type inference
    var age = 15
    
    // Method 3: Short declaration (most common)
    language := "Go"
    
    // Multiple declarations
    var x, y int = 1, 2
    a, b := 3, 4
    
    // Zero values (default values)
    var empty string  // ""
    var num int       // 0
    var truth bool    // false
    var ptr *int      // nil
    
    fmt.Println(name, age, language, x, y, a, b)
}
```

### Constants

```go
// Single constant
const Pi = 3.14159

// Multiple constants
const (
    StatusOK = 200
    StatusNotFound = 404
    StatusError = 500
)

// Typed constants
const MaxSize int = 100

// Untyped constants (flexible)
const Timeout = 30

// iota for enumerations
const (
    Sunday = iota     // 0
    Monday            // 1
    Tuesday           // 2
    Wednesday         // 3
    Thursday          // 4
    Friday            // 5
    Saturday          // 6
)

// iota with expressions
const (
    _  = iota             // Skip 0
    KB = 1 << (10 * iota) // 1024
    MB                     // 1048576
    GB                     // 1073741824
)
```

## 🎯 Basic Data Types

### Numeric Types

```go
// Integers
var i8 int8 = 127          // -128 to 127
var i16 int16 = 32767      // -32768 to 32767
var i32 int32 = 2147483647
var i64 int64 = 9223372036854775807

// Unsigned integers
var ui8 uint8 = 255        // 0 to 255 (byte)
var ui16 uint16 = 65535
var ui32 uint32 = 4294967295
var ui64 uint64 = 18446744073709551615

// Platform-dependent
var i int   // 32 or 64 bit depending on system
var ui uint

// Floating point
var f32 float32 = 3.14
var f64 float64 = 3.14159265359

// Complex numbers
var c64 complex64 = 1 + 2i
var c128 complex128 = complex(1, 2)

// Type aliases
var b byte = 255        // alias for uint8
var r rune = '世'        // alias for int32 (Unicode code point)
```

### String Type

```go
// String basics
var str string = "Hello, Go!"

// Raw string (ignore escape sequences)
var raw string = `Line 1
Line 2
Line 3`

// String operations
s1 := "Hello"
s2 := "World"
combined := s1 + " " + s2  // Concatenation

// String properties
length := len(combined)     // Length in bytes
firstChar := combined[0]    // Byte at index

// Strings are immutable
// combined[0] = 'h'  // ERROR!

// Iterate over string
for i, ch := range "Hello" {
    fmt.Printf("Index: %d, Character: %c, Value: %d\n", i, ch, ch)
}
```

### Boolean Type

```go
var isActive bool = true
var isValid bool = false

// Comparison results in bool
result := 5 > 3  // true
equal := "a" == "a"  // true
```

## ➕ Operators

### Arithmetic Operators

```go
a, b := 10, 3

sum := a + b       // 13
diff := a - b      // 7
product := a * b   // 30
quotient := a / b  // 3 (integer division)
remainder := a % b // 1

// Increment/Decrement
a++  // a = 11
b--  // b = 2

// No pre-increment in Go
// ++a  // ERROR!
```

### Comparison Operators

```go
a, b := 5, 10

equal := a == b        // false
notEqual := a != b     // true
less := a < b          // true
lessOrEqual := a <= b  // true
greater := a > b       // false
greaterOrEqual := a >= b // false
```

### Logical Operators

```go
true && false  // false (AND)
true || false  // true (OR)
!true          // false (NOT)

// Short-circuit evaluation
result := false && someExpensiveFunction()  // someExpensiveFunction() not called
```

### Bitwise Operators

```go
a, b := 12, 25  // 1100, 11001 in binary

and := a & b     // 8 (1000)
or := a | b      // 29 (11101)
xor := a ^ b     // 21 (10101)
leftShift := a << 2   // 48 (110000)
rightShift := b >> 2  // 6 (110)
```

## 🔄 Control Flow

### If Statement

```go
// Basic if
if x > 0 {
    fmt.Println("Positive")
}

// If-else
if x > 0 {
    fmt.Println("Positive")
} else {
    fmt.Println("Non-positive")
}

// If-else if-else
if x > 0 {
    fmt.Println("Positive")
} else if x < 0 {
    fmt.Println("Negative")
} else {
    fmt.Println("Zero")
}

// If with initialization
if num := 9; num < 10 {
    fmt.Println(num, "is less than 10")
    // num is only accessible here
}
// num is not accessible here
```

### Switch Statement

```go
// Basic switch
day := "Monday"
switch day {
case "Monday":
    fmt.Println("Start of work week")
case "Friday":
    fmt.Println("TGIF!")
case "Saturday", "Sunday":
    fmt.Println("Weekend!")
default:
    fmt.Println("Midweek")
}

// Switch with initialization
switch hour := time.Now().Hour(); {
case hour < 12:
    fmt.Println("Good morning!")
case hour < 18:
    fmt.Println("Good afternoon!")
default:
    fmt.Println("Good evening!")
}

// Type switch (more on this later)
var i interface{} = 42
switch v := i.(type) {
case int:
    fmt.Printf("Integer: %d\n", v)
case string:
    fmt.Printf("String: %s\n", v)
default:
    fmt.Printf("Unknown type\n")
}

// Fallthrough (rare in Go)
num := 2
switch num {
case 1:
    fmt.Println("One")
    fallthrough
case 2:
    fmt.Println("Two")
    fallthrough
case 3:
    fmt.Println("Three")
}
// Output: Two, Three
```

### For Loop (The Only Loop in Go!)

```go
// Standard for loop
for i := 0; i < 5; i++ {
    fmt.Println(i)
}

// While-style loop
i := 0
for i < 5 {
    fmt.Println(i)
    i++
}

// Infinite loop
for {
    fmt.Println("Forever")
    break  // Use break to exit
}

// Range loop (for arrays, slices, maps, strings)
numbers := []int{1, 2, 3, 4, 5}
for index, value := range numbers {
    fmt.Printf("Index: %d, Value: %d\n", index, value)
}

// Ignore index with _
for _, value := range numbers {
    fmt.Println(value)
}

// Only index
for index := range numbers {
    fmt.Println(index)
}

// Range over string (iterates over runes)
for index, char := range "Hello" {
    fmt.Printf("%d: %c\n", index, char)
}
```

### Break and Continue

```go
// Break exits the loop
for i := 0; i < 10; i++ {
    if i == 5 {
        break
    }
    fmt.Println(i)
}

// Continue skips to next iteration
for i := 0; i < 10; i++ {
    if i%2 == 0 {
        continue
    }
    fmt.Println(i)  // Only odd numbers
}

// Labeled break (for nested loops)
outer:
for i := 0; i < 3; i++ {
    for j := 0; j < 3; j++ {
        if i == 1 && j == 1 {
            break outer
        }
        fmt.Printf("i=%d, j=%d\n", i, j)
    }
}
```

## 🎯 Type Conversion

Go requires explicit type conversion:

```go
// Integer conversions
var i int = 42
var f float64 = float64(i)
var u uint = uint(i)

// Float to int (truncates)
var pi float64 = 3.14
var rounded int = int(pi)  // 3

// String conversions
import "strconv"

// Int to string
num := 42
str := strconv.Itoa(num)  // "42"

// String to int
str2 := "123"
num2, err := strconv.Atoi(str2)
if err != nil {
    fmt.Println("Conversion error:", err)
}

// Advanced conversions
f64 := 3.14159
s := strconv.FormatFloat(f64, 'f', 2, 64)  // "3.14"

// Rune to string
r := 'A'
str3 := string(r)  // "A"

// Byte slice to string
bytes := []byte{'H', 'e', 'l', 'l', 'o'}
str4 := string(bytes)  // "Hello"
```

## 📊 Complete Example

```go
package main

import (
    "fmt"
    "strconv"
)

func main() {
    // Variable declarations
    name := "Go Programming"
    version := 1.21
    isModern := true
    
    fmt.Printf("Learning: %s\n", name)
    fmt.Printf("Version: %.2f\n", version)
    fmt.Printf("Is modern: %v\n", isModern)
    
    // Constants
    const MaxAttempts = 3
    
    // Control flow
    for attempt := 1; attempt <= MaxAttempts; attempt++ {
        fmt.Printf("Attempt %d of %d\n", attempt, MaxAttempts)
        
        if attempt == MaxAttempts {
            fmt.Println("Final attempt!")
        }
    }
    
    // Switch statement
    day := 3
    switch day {
    case 1:
        fmt.Println("Monday")
    case 2:
        fmt.Println("Tuesday")
    case 3:
        fmt.Println("Wednesday")
    default:
        fmt.Println("Other day")
    }
    
    // Type conversion
    score := 95
    percentage := float64(score) / 100.0
    grade := "Grade: " + strconv.Itoa(score)
    
    fmt.Printf("Percentage: %.2f\n", percentage)
    fmt.Println(grade)
}
```

## 🎯 Exercises

### Exercise 1: FizzBuzz
Write a program that prints numbers 1 to 100, but:
- For multiples of 3, print "Fizz"
- For multiples of 5, print "Buzz"
- For multiples of both, print "FizzBuzz"

### Exercise 2: Temperature Converter
Create a program that converts temperatures between Celsius and Fahrenheit.

### Exercise 3: Number Classifier
Write a program that takes a number and prints whether it's:
- Positive, negative, or zero
- Even or odd
- Single digit, double digit, or more

### Solutions

```go
// Exercise 1: FizzBuzz
package main

import "fmt"

func main() {
    for i := 1; i <= 100; i++ {
        if i%15 == 0 {
            fmt.Println("FizzBuzz")
        } else if i%3 == 0 {
            fmt.Println("Fizz")
        } else if i%5 == 0 {
            fmt.Println("Buzz")
        } else {
            fmt.Println(i)
        }
    }
}

// Exercise 2: Temperature Converter
package main

import "fmt"

func main() {
    celsius := 25.0
    fahrenheit := celsius*9/5 + 32
    fmt.Printf("%.1f°C = %.1f°F\n", celsius, fahrenheit)
    
    fahrenheit = 77.0
    celsius = (fahrenheit - 32) * 5 / 9
    fmt.Printf("%.1f°F = %.1f°C\n", fahrenheit, celsius)
}

// Exercise 3: Number Classifier
package main

import "fmt"

func main() {
    num := 42
    
    // Sign
    if num > 0 {
        fmt.Println("Positive")
    } else if num < 0 {
        fmt.Println("Negative")
    } else {
        fmt.Println("Zero")
    }
    
    // Even/Odd
    if num%2 == 0 {
        fmt.Println("Even")
    } else {
        fmt.Println("Odd")
    }
    
    // Digits
    absNum := num
    if absNum < 0 {
        absNum = -absNum
    }
    
    switch {
    case absNum < 10:
        fmt.Println("Single digit")
    case absNum < 100:
        fmt.Println("Double digit")
    default:
        fmt.Println("Three or more digits")
    }
}
```

## 🔑 Key Takeaways

- Use `:=` for short variable declarations (most common)
- Go has strong typing with explicit conversions
- `for` is the only loop construct (but very flexible)
- `switch` doesn't need `break` (no fallthrough by default)
- Use `const` and `iota` for enumerations
- Zero values: `0` for numbers, `""` for strings, `false` for bool, `nil` for pointers

## 📖 Next Steps

Continue to [Chapter 3: Functions](03-functions.md) to learn about functions, methods, and closures.

