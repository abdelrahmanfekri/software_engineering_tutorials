# Chapter 4: Data Structures - Arrays, Slices, and Maps

## 📊 Arrays

Fixed-size sequences of elements of the same type.

### Array Declaration

```go
// Declaration with size
var arr [5]int  // Array of 5 integers, zero-valued

// Declaration with initialization
var arr2 = [5]int{1, 2, 3, 4, 5}

// Short declaration
arr3 := [3]string{"Go", "Rust", "Python"}

// Compiler infers size
arr4 := [...]int{1, 2, 3, 4}  // Size is 4

// Partial initialization
arr5 := [5]int{1, 2}  // [1, 2, 0, 0, 0]

// Initialize specific indices
arr6 := [5]int{0: 10, 2: 30, 4: 50}  // [10, 0, 30, 0, 50]
```

### Array Operations

```go
// Access elements
arr := [5]int{1, 2, 3, 4, 5}
first := arr[0]   // 1
last := arr[4]    // 5

// Modify elements
arr[2] = 10

// Length
length := len(arr)  // 5

// Iterate
for i := 0; i < len(arr); i++ {
    fmt.Println(arr[i])
}

// Range loop
for index, value := range arr {
    fmt.Printf("arr[%d] = %d\n", index, value)
}

// Arrays are values (copied on assignment)
arr2 := arr  // arr2 is a copy, not a reference
arr2[0] = 100
fmt.Println(arr[0])   // Still 1
fmt.Println(arr2[0])  // 100

// Comparing arrays
a := [3]int{1, 2, 3}
b := [3]int{1, 2, 3}
fmt.Println(a == b)  // true
```

### Multidimensional Arrays

```go
// 2D array
var matrix [3][4]int

// Initialize
matrix2 := [2][3]int{
    {1, 2, 3},
    {4, 5, 6},
}

// Access
matrix2[0][1] = 10

// Iterate
for i := 0; i < len(matrix2); i++ {
    for j := 0; j < len(matrix2[i]); j++ {
        fmt.Printf("%d ", matrix2[i][j])
    }
    fmt.Println()
}
```

## 🎯 Slices

Dynamic, flexible view into arrays. Most commonly used in Go.

### Slice Creation

```go
// Create from array
arr := [5]int{1, 2, 3, 4, 5}
slice1 := arr[1:4]  // [2, 3, 4]
slice2 := arr[:3]   // [1, 2, 3] (from start)
slice3 := arr[2:]   // [3, 4, 5] (to end)
slice4 := arr[:]    // [1, 2, 3, 4, 5] (entire array)

// Literal notation (most common)
slice := []int{1, 2, 3, 4, 5}

// Using make
slice5 := make([]int, 5)       // length 5, capacity 5
slice6 := make([]int, 3, 10)   // length 3, capacity 10

// Empty slice
var slice7 []int     // nil slice
slice8 := []int{}    // empty but non-nil
slice9 := make([]int, 0)  // empty but non-nil
```

### Slice Properties

```go
slice := make([]int, 3, 5)

// Length: number of elements
len(slice)  // 3

// Capacity: size of underlying array
cap(slice)  // 5

// Nil check
var nilSlice []int
if nilSlice == nil {
    fmt.Println("Slice is nil")
}
```

### Slice Operations

```go
// Append (most important operation!)
slice := []int{1, 2, 3}
slice = append(slice, 4)        // [1, 2, 3, 4]
slice = append(slice, 5, 6, 7)  // [1, 2, 3, 4, 5, 6, 7]

// Append another slice
slice2 := []int{8, 9, 10}
slice = append(slice, slice2...)  // Spread operator

// Copy
source := []int{1, 2, 3, 4, 5}
dest := make([]int, len(source))
n := copy(dest, source)  // Returns number of elements copied
fmt.Println(n, dest)     // 5 [1 2 3 4 5]

// Delete element at index
slice = []int{1, 2, 3, 4, 5}
index := 2
slice = append(slice[:index], slice[index+1:]...)
// [1, 2, 4, 5]

// Insert element at index
slice = []int{1, 2, 4, 5}
index = 2
value := 3
slice = append(slice[:index], append([]int{value}, slice[index:]...)...)
// [1, 2, 3, 4, 5]

// Slicing creates view (shares underlying array)
original := []int{1, 2, 3, 4, 5}
view := original[1:4]  // [2, 3, 4]
view[0] = 100
fmt.Println(original)  // [1, 100, 3, 4, 5] - modified!

// Full slice expression (control capacity)
slice = original[1:3:4]  // [low:high:max]
// Elements from index 1 to 2, capacity from 1 to 3
```

### Slice Tricks

```go
// Reverse a slice
func reverse(s []int) {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}

// Filter (remove elements)
func filter(s []int, predicate func(int) bool) []int {
    result := s[:0]  // Reuse underlying array
    for _, v := range s {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

// Remove duplicates (preserving order)
func unique(s []int) []int {
    seen := make(map[int]bool)
    result := []int{}
    for _, v := range s {
        if !seen[v] {
            seen[v] = true
            result = append(result, v)
        }
    }
    return result
}
```

## 🗺️ Maps

Key-value pairs (hash tables/dictionaries).

### Map Creation

```go
// Declare and initialize
ages := map[string]int{
    "Alice": 25,
    "Bob":   30,
    "Carol": 28,
}

// Make (empty map)
scores := make(map[string]int)

// Var declaration (nil map - can't add elements!)
var nilMap map[string]int  // Don't use!

// Empty map (preferred over nil map)
emptyMap := map[string]int{}
```

### Map Operations

```go
// Add/Update
ages := make(map[string]int)
ages["Alice"] = 25
ages["Bob"] = 30

// Get value
age := ages["Alice"]  // 25

// Get with existence check (important!)
age, exists := ages["Alice"]
if exists {
    fmt.Println("Alice's age:", age)
}

// Zero value if key doesn't exist
missing := ages["Unknown"]  // 0 (zero value for int)

// Delete
delete(ages, "Bob")

// Length
count := len(ages)

// Check if key exists
if _, exists := ages["Alice"]; exists {
    fmt.Println("Alice exists")
}
```

### Iterating Maps

```go
ages := map[string]int{
    "Alice": 25,
    "Bob":   30,
    "Carol": 28,
}

// Iterate over keys and values
for name, age := range ages {
    fmt.Printf("%s is %d years old\n", name, age)
}

// Iterate over keys only
for name := range ages {
    fmt.Println(name)
}

// Note: Map iteration order is RANDOM!
// If you need sorted iteration, extract keys and sort
```

### Advanced Map Operations

```go
// Count word frequencies
text := "go go go is awesome go"
words := strings.Fields(text)
frequency := make(map[string]int)

for _, word := range words {
    frequency[word]++
}
fmt.Println(frequency)  // map[go:4 is:1 awesome:1]

// Group by key
type Person struct {
    Name string
    City string
}

people := []Person{
    {"Alice", "NYC"},
    {"Bob", "LA"},
    {"Carol", "NYC"},
}

byCity := make(map[string][]Person)
for _, person := range people {
    byCity[person.City] = append(byCity[person.City], person)
}

// Invert map
ages := map[string]int{
    "Alice": 25,
    "Bob":   30,
}

ageToName := make(map[int][]string)
for name, age := range ages {
    ageToName[age] = append(ageToName[age], name)
}
```

### Maps with Complex Keys

```go
// Struct as key (must be comparable)
type Point struct {
    X, Y int
}

distances := map[Point]float64{
    {0, 0}: 0.0,
    {1, 1}: 1.414,
    {2, 2}: 2.828,
}

// Array as key (arrays are comparable)
grid := map[[2]int]string{
    {0, 0}: "origin",
    {1, 0}: "right",
    {0, 1}: "up",
}

// Slices CANNOT be map keys (not comparable)
// map[[]int]string  // ERROR!
// Use string representation or convert to array
```

## 📦 Combining Data Structures

### Slice of Slices (2D Slice)

```go
// Dynamic 2D slice
matrix := [][]int{
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
}

// Create with make
rows, cols := 3, 4
matrix2 := make([][]int, rows)
for i := range matrix2 {
    matrix2[i] = make([]int, cols)
}
```

### Map of Slices

```go
// Group items
type Student struct {
    Name  string
    Grade string
}

students := []Student{
    {"Alice", "A"},
    {"Bob", "B"},
    {"Carol", "A"},
}

byGrade := make(map[string][]Student)
for _, student := range students {
    byGrade[student.Grade] = append(byGrade[student.Grade], student)
}
```

### Slice of Maps

```go
users := []map[string]string{
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
}
```

## 🎯 Complete Example: Phonebook

```go
package main

import (
    "fmt"
    "strings"
)

type Contact struct {
    Name  string
    Phone string
    Email string
}

type PhoneBook struct {
    contacts map[string]Contact
}

func NewPhoneBook() *PhoneBook {
    return &PhoneBook{
        contacts: make(map[string]Contact),
    }
}

func (pb *PhoneBook) Add(contact Contact) {
    key := strings.ToLower(contact.Name)
    pb.contacts[key] = contact
}

func (pb *PhoneBook) Find(name string) (Contact, bool) {
    key := strings.ToLower(name)
    contact, exists := pb.contacts[key]
    return contact, exists
}

func (pb *PhoneBook) Delete(name string) bool {
    key := strings.ToLower(name)
    if _, exists := pb.contacts[key]; exists {
        delete(pb.contacts, key)
        return true
    }
    return false
}

func (pb *PhoneBook) List() []Contact {
    contacts := make([]Contact, 0, len(pb.contacts))
    for _, contact := range pb.contacts {
        contacts = append(contacts, contact)
    }
    return contacts
}

func main() {
    pb := NewPhoneBook()
    
    // Add contacts
    pb.Add(Contact{"Alice", "123-456-7890", "alice@example.com"})
    pb.Add(Contact{"Bob", "234-567-8901", "bob@example.com"})
    
    // Find contact
    if contact, found := pb.Find("alice"); found {
        fmt.Printf("Found: %+v\n", contact)
    }
    
    // List all
    fmt.Println("All contacts:")
    for _, contact := range pb.List() {
        fmt.Printf("- %s: %s\n", contact.Name, contact.Phone)
    }
    
    // Delete
    if pb.Delete("bob") {
        fmt.Println("Deleted Bob")
    }
}
```

## 🎯 Exercises

### Exercise 1: Implement Set
Create a set data structure using a map.

### Exercise 2: Two Sum
Given an array and target, find two numbers that sum to target.

### Exercise 3: Merge Intervals
Given overlapping intervals, merge them.

### Solutions

```go
// Exercise 1: Set Implementation
type Set struct {
    data map[int]bool
}

func NewSet() *Set {
    return &Set{data: make(map[int]bool)}
}

func (s *Set) Add(value int) {
    s.data[value] = true
}

func (s *Set) Remove(value int) {
    delete(s.data, value)
}

func (s *Set) Contains(value int) bool {
    return s.data[value]
}

func (s *Set) Size() int {
    return len(s.data)
}

func (s *Set) Values() []int {
    values := make([]int, 0, len(s.data))
    for v := range s.data {
        values = append(values, v)
    }
    return values
}

// Exercise 2: Two Sum
func twoSum(nums []int, target int) []int {
    seen := make(map[int]int)  // value -> index
    
    for i, num := range nums {
        complement := target - num
        if j, found := seen[complement]; found {
            return []int{j, i}
        }
        seen[num] = i
    }
    
    return nil
}

// Usage
nums := []int{2, 7, 11, 15}
result := twoSum(nums, 9)
fmt.Println(result)  // [0, 1]

// Exercise 3: Merge Intervals
type Interval struct {
    Start, End int
}

func merge(intervals []Interval) []Interval {
    if len(intervals) == 0 {
        return nil
    }
    
    // Sort by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i].Start < intervals[j].Start
    })
    
    result := []Interval{intervals[0]}
    
    for i := 1; i < len(intervals); i++ {
        current := intervals[i]
        last := &result[len(result)-1]
        
        if current.Start <= last.End {
            // Merge
            if current.End > last.End {
                last.End = current.End
            }
        } else {
            // Add new interval
            result = append(result, current)
        }
    }
    
    return result
}
```

## 🔑 Key Takeaways

- **Arrays**: Fixed size, value type, rarely used directly
- **Slices**: Dynamic, reference type, most common
- **Maps**: Key-value pairs, unordered, fast lookups
- Always check map existence: `value, exists := map[key]`
- Slices share underlying array - be careful!
- Use `append` for slices, `make` for initialization
- Maps require initialization (use `make` or literal)

## 📖 Next Steps

Continue to [Chapter 5: Structs and Methods](05-structs-methods.md) to learn about custom types and object-oriented programming in Go.

