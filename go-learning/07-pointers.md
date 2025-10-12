# Chapter 7: Pointers

## 📍 What are Pointers?

Pointers store memory addresses of values.

### Basic Pointer Operations

```go
// Create a variable
x := 42

// Get pointer to x (address-of operator &)
ptr := &x

// Get value from pointer (dereference operator *)
value := *ptr

fmt.Println("Value:", x)        // 42
fmt.Println("Address:", ptr)    // 0xc0000140b0 (example)
fmt.Println("Dereferenced:", value)  // 42

// Modify through pointer
*ptr = 100
fmt.Println("x is now:", x)  // 100
```

### Pointer Types

```go
var p *int      // Pointer to int
var p2 *string  // Pointer to string
var p3 *Person  // Pointer to struct

// Zero value of pointer is nil
var nilPtr *int
fmt.Println(nilPtr == nil)  // true

// Dereferencing nil causes panic!
// fmt.Println(*nilPtr)  // PANIC!

// Safe check
if nilPtr != nil {
    fmt.Println(*nilPtr)
}
```

## 🔄 Pointers vs Values

### Value Semantics

```go
func double(x int) {
    x = x * 2
    // Only modifies local copy
}

func main() {
    num := 10
    double(num)
    fmt.Println(num)  // Still 10 (not modified)
}
```

### Pointer Semantics

```go
func double(x *int) {
    *x = *x * 2
    // Modifies original value
}

func main() {
    num := 10
    double(&num)
    fmt.Println(num)  // 20 (modified!)
}
```

## 🏗️ Pointers with Structs

### Struct Pointers

```go
type Person struct {
    Name string
    Age  int
}

// Without pointer (value copy)
func celebrateBirthday(p Person) {
    p.Age++
    // Only modifies copy
}

// With pointer (modifies original)
func celebrateBirthdayPtr(p *Person) {
    p.Age++  // Automatic dereferencing!
    // Same as: (*p).Age++
}

func main() {
    person := Person{Name: "Alice", Age: 25}
    
    celebrateBirthday(person)
    fmt.Println(person.Age)  // Still 25
    
    celebrateBirthdayPtr(&person)
    fmt.Println(person.Age)  // 26
}
```

### Creating Struct Pointers

```go
// Method 1: Address-of operator
p1 := Person{Name: "Alice", Age: 25}
ptr1 := &p1

// Method 2: New (returns pointer)
ptr2 := new(Person)
ptr2.Name = "Bob"
ptr2.Age = 30

// Method 3: Composite literal
ptr3 := &Person{
    Name: "Carol",
    Age:  28,
}

// All three are equivalent
fmt.Printf("%T\n", ptr1)  // *main.Person
fmt.Printf("%T\n", ptr2)  // *main.Person
fmt.Printf("%T\n", ptr3)  // *main.Person
```

## 🎯 When to Use Pointers

### Use Pointers When:

```go
// 1. Need to modify the value
func increment(n *int) {
    *n++
}

// 2. Large struct (avoid copying)
type LargeStruct struct {
    Data [1000000]int
}

func process(ls *LargeStruct) {
    // Efficient: pass by reference
}

// 3. Optional/nullable value
func findUser(id int) *User {
    // Can return nil if not found
    if user, found := users[id]; found {
        return &user
    }
    return nil
}

// 4. Shared state
type Counter struct {
    value int
}

func (c *Counter) Increment() {
    c.value++
}
```

### Use Values When:

```go
// 1. Small data types
func add(a, b int) int {
    return a + b
}

// 2. Immutable operations
type Point struct {
    X, Y int
}

func (p Point) Distance() float64 {
    return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}

// 3. Want a copy
func processData(data []int) []int {
    // Work with copy, don't affect original
    result := make([]int, len(data))
    copy(result, data)
    return result
}
```

## 🔍 Pointer Indirection

### Multiple Levels of Indirection

```go
x := 42
ptr := &x       // *int
pptr := &ptr    // **int
ppptr := &pptr  // ***int

fmt.Println(***ppptr)  // 42

// Modify through multiple indirection
***ppptr = 100
fmt.Println(x)  // 100
```

### Practical Example

```go
// Linked list node
type Node struct {
    Value int
    Next  *Node
}

func main() {
    // Create linked list: 1 -> 2 -> 3
    head := &Node{Value: 1}
    head.Next = &Node{Value: 2}
    head.Next.Next = &Node{Value: 3}
    
    // Traverse
    current := head
    for current != nil {
        fmt.Println(current.Value)
        current = current.Next
    }
}
```

## 🚫 Common Pointer Mistakes

### Mistake 1: Nil Pointer Dereference

```go
var ptr *int
// *ptr = 10  // PANIC: nil pointer dereference

// Fix: Check for nil
if ptr != nil {
    *ptr = 10
}

// Or initialize
ptr = new(int)
*ptr = 10
```

### Mistake 2: Pointer to Loop Variable

```go
// Wrong!
var ptrs []*int
for i := 0; i < 3; i++ {
    ptrs = append(ptrs, &i)  // All point to same variable!
}
for _, p := range ptrs {
    fmt.Println(*p)  // Prints: 3, 3, 3
}

// Correct
var ptrs []*int
for i := 0; i < 3; i++ {
    num := i  // Create new variable
    ptrs = append(ptrs, &num)
}
```

### Mistake 3: Returning Pointer to Local Variable

```go
// This is actually SAFE in Go (unlike C)
func createInt() *int {
    x := 42
    return &x  // OK! Escapes to heap
}

// Go automatically moves x to heap
// This is called "escape analysis"
```

## 🎨 Pointer Patterns

### Optional Parameters

```go
type Config struct {
    Timeout *int
    MaxConn *int
    Debug   *bool
}

func NewConfig() *Config {
    return &Config{}
}

func (c *Config) SetTimeout(timeout int) *Config {
    c.Timeout = &timeout
    return c
}

func (c *Config) SetMaxConn(max int) *Config {
    c.MaxConn = &max
    return c
}

// Usage
config := NewConfig().
    SetTimeout(30).
    SetMaxConn(100)

// Check if value was set
if config.Timeout != nil {
    fmt.Println("Timeout:", *config.Timeout)
}
```

### Modifying Map Values (Struct)

```go
// Problem: Can't modify struct in map directly
type Person struct {
    Name string
    Age  int
}

people := map[string]Person{
    "alice": {Name: "Alice", Age: 25},
}

// people["alice"].Age++  // ERROR: cannot assign

// Solution 1: Reassign
person := people["alice"]
person.Age++
people["alice"] = person

// Solution 2: Use pointer
people2 := map[string]*Person{
    "alice": {Name: "Alice", Age: 25},
}

people2["alice"].Age++  // OK!
```

### Circular References

```go
type Node struct {
    Value int
    Next  *Node
    Prev  *Node
}

func createDoublyLinkedList() *Node {
    node1 := &Node{Value: 1}
    node2 := &Node{Value: 2}
    node3 := &Node{Value: 3}
    
    node1.Next = node2
    node2.Prev = node1
    node2.Next = node3
    node3.Prev = node2
    
    return node1
}
```

## 💼 Complete Example: Binary Tree

```go
package main

import "fmt"

type TreeNode struct {
    Value int
    Left  *TreeNode
    Right *TreeNode
}

func NewNode(value int) *TreeNode {
    return &TreeNode{Value: value}
}

func (n *TreeNode) Insert(value int) {
    if value < n.Value {
        if n.Left == nil {
            n.Left = NewNode(value)
        } else {
            n.Left.Insert(value)
        }
    } else {
        if n.Right == nil {
            n.Right = NewNode(value)
        } else {
            n.Right.Insert(value)
        }
    }
}

func (n *TreeNode) Search(value int) bool {
    if n == nil {
        return false
    }
    
    if value == n.Value {
        return true
    } else if value < n.Value {
        return n.Left.Search(value)
    } else {
        return n.Right.Search(value)
    }
}

func (n *TreeNode) InOrder(visit func(int)) {
    if n == nil {
        return
    }
    
    n.Left.InOrder(visit)
    visit(n.Value)
    n.Right.InOrder(visit)
}

func main() {
    root := NewNode(10)
    root.Insert(5)
    root.Insert(15)
    root.Insert(3)
    root.Insert(7)
    root.Insert(12)
    root.Insert(17)
    
    fmt.Println("In-order traversal:")
    root.InOrder(func(val int) {
        fmt.Printf("%d ", val)
    })
    fmt.Println()
    
    fmt.Println("Search 7:", root.Search(7))   // true
    fmt.Println("Search 20:", root.Search(20)) // false
}
```

## 🎯 Exercises

### Exercise 1: Swap Function
Write a function that swaps two integers using pointers.

### Exercise 2: Linked List
Implement a singly linked list with Add, Remove, and Find operations.

### Exercise 3: Reference Counter
Create a reference counter that tracks object usage.

### Solutions

```go
// Exercise 1: Swap
func swap(a, b *int) {
    *a, *b = *b, *a
}

func main() {
    x, y := 10, 20
    swap(&x, &y)
    fmt.Println(x, y)  // 20 10
}

// Exercise 2: Linked List
type LinkedList struct {
    head *Node
}

type Node struct {
    data int
    next *Node
}

func (ll *LinkedList) Add(data int) {
    newNode := &Node{data: data}
    
    if ll.head == nil {
        ll.head = newNode
        return
    }
    
    current := ll.head
    for current.next != nil {
        current = current.next
    }
    current.next = newNode
}

func (ll *LinkedList) Remove(data int) bool {
    if ll.head == nil {
        return false
    }
    
    if ll.head.data == data {
        ll.head = ll.head.next
        return true
    }
    
    current := ll.head
    for current.next != nil {
        if current.next.data == data {
            current.next = current.next.next
            return true
        }
        current = current.next
    }
    
    return false
}

func (ll *LinkedList) Find(data int) bool {
    current := ll.head
    for current != nil {
        if current.data == data {
            return true
        }
        current = current.next
    }
    return false
}

// Exercise 3: Reference Counter
type RefCounter struct {
    count *int
}

func NewRefCounter() *RefCounter {
    count := 0
    return &RefCounter{count: &count}
}

func (rc *RefCounter) Increment() {
    *rc.count++
}

func (rc *RefCounter) Decrement() {
    *rc.count--
}

func (rc *RefCounter) Count() int {
    return *rc.count
}
```

## 🔑 Key Takeaways

- `&` gets address, `*` dereferences pointer
- Pointers enable modification of values
- Use pointers for large structs to avoid copying
- Go automatically handles escape analysis (heap vs stack)
- Nil pointers cause panics when dereferenced
- Struct fields accessed through pointers use `.` (automatic dereferencing)
- Return pointers for optional/nullable values

## 📖 Next Steps

Continue to [Chapter 8: Goroutines and Channels](08-concurrency.md) to learn about Go's powerful concurrency model.

