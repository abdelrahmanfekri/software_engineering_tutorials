# Chapter 5: Structs and Methods

## 📦 Structs

Structs are typed collections of fields. They're Go's way of creating custom data types.

### Basic Struct Definition

```go
// Define a struct
type Person struct {
    Name string
    Age  int
    City string
}

// Create instances
func main() {
    // Method 1: Field names
    p1 := Person{
        Name: "Alice",
        Age:  25,
        City: "NYC",
    }
    
    // Method 2: Positional (not recommended)
    p2 := Person{"Bob", 30, "LA"}
    
    // Method 3: Zero values
    var p3 Person  // Name: "", Age: 0, City: ""
    
    // Method 4: Partial initialization
    p4 := Person{Name: "Carol"}  // Age: 0, City: ""
    
    // Access fields
    fmt.Println(p1.Name)  // Alice
    p1.Age = 26           // Modify field
}
```

### Anonymous Structs

```go
// Useful for one-time use
person := struct {
    Name string
    Age  int
}{
    Name: "Alice",
    Age:  25,
}

// Common in table-driven tests
tests := []struct {
    input    int
    expected int
}{
    {2, 4},
    {3, 9},
    {4, 16},
}
```

### Nested Structs

```go
type Address struct {
    Street  string
    City    string
    ZipCode string
}

type Person struct {
    Name    string
    Age     int
    Address Address  // Nested struct
}

func main() {
    p := Person{
        Name: "Alice",
        Age:  25,
        Address: Address{
            Street:  "123 Main St",
            City:    "NYC",
            ZipCode: "10001",
        },
    }
    
    fmt.Println(p.Address.City)  // NYC
}
```

### Embedded Structs (Composition)

```go
// Embedding provides composition (like inheritance)
type User struct {
    Email    string
    Password string
}

type Admin struct {
    User             // Embedded (promoted fields)
    AccessLevel int
}

func main() {
    admin := Admin{
        User: User{
            Email:    "admin@example.com",
            Password: "secret",
        },
        AccessLevel: 10,
    }
    
    // Can access User fields directly (promoted)
    fmt.Println(admin.Email)  // admin@example.com
    
    // Or explicitly
    fmt.Println(admin.User.Email)
}
```

### Struct Tags

Used for metadata (encoding/decoding, validation, etc.)

```go
import "encoding/json"

type User struct {
    ID        int    `json:"id"`
    Name      string `json:"name"`
    Email     string `json:"email,omitempty"`
    Password  string `json:"-"`  // Never serialize
    CreatedAt time.Time `json:"created_at"`
}

func main() {
    user := User{
        ID:       1,
        Name:     "Alice",
        Email:    "alice@example.com",
        Password: "secret",
    }
    
    // Marshal to JSON
    data, _ := json.Marshal(user)
    fmt.Println(string(data))
    // {"id":1,"name":"Alice","email":"alice@example.com","created_at":"0001-01-01T00:00:00Z"}
    
    // Unmarshal from JSON
    jsonStr := `{"id":2,"name":"Bob"}`
    var user2 User
    json.Unmarshal([]byte(jsonStr), &user2)
}
```

### Struct Comparison

```go
type Point struct {
    X, Y int
}

p1 := Point{1, 2}
p2 := Point{1, 2}
p3 := Point{2, 3}

fmt.Println(p1 == p2)  // true
fmt.Println(p1 == p3)  // false

// Structs with slices/maps are NOT comparable
type Container struct {
    Items []int
}

c1 := Container{[]int{1, 2, 3}}
c2 := Container{[]int{1, 2, 3}}
// fmt.Println(c1 == c2)  // ERROR: invalid operation
```

## 🎯 Methods

Functions with a receiver (associated with a type).

### Value Receivers

```go
type Rectangle struct {
    Width, Height float64
}

// Method with value receiver
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func main() {
    rect := Rectangle{Width: 10, Height: 5}
    
    fmt.Println(rect.Area())       // 50
    fmt.Println(rect.Perimeter())  // 30
}
```

### Pointer Receivers

Use pointer receivers to modify the receiver or avoid copying.

```go
type Counter struct {
    Value int
}

// Pointer receiver (can modify)
func (c *Counter) Increment() {
    c.Value++
}

func (c *Counter) Decrement() {
    c.Value--
}

// Value receiver (read-only)
func (c Counter) GetValue() int {
    return c.Value
}

func main() {
    counter := Counter{Value: 0}
    
    counter.Increment()
    counter.Increment()
    fmt.Println(counter.GetValue())  // 2
    
    counter.Decrement()
    fmt.Println(counter.GetValue())  // 1
}
```

### When to Use Pointer vs Value Receivers

```go
// Use POINTER receivers when:
// 1. Method modifies the receiver
// 2. Receiver is large (avoid copying)
// 3. Consistency (if some methods use *, all should)

type LargeStruct struct {
    Data [1000000]int
}

func (l *LargeStruct) Process() {
    // Avoid copying 1M ints
}

// Use VALUE receivers when:
// 1. Receiver is small
// 2. Receiver is immutable
// 3. Receiver is built-in type or basic struct

type Point struct {
    X, Y int
}

func (p Point) Distance() float64 {
    return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}
```

### Methods on Non-Struct Types

```go
// Can define methods on any type in same package
type MyInt int

func (m MyInt) IsEven() bool {
    return m%2 == 0
}

func (m MyInt) Double() MyInt {
    return m * 2
}

func main() {
    var num MyInt = 5
    fmt.Println(num.IsEven())  // false
    fmt.Println(num.Double())  // 10
}

// Custom string type with methods
type EmailAddress string

func (e EmailAddress) IsValid() bool {
    return strings.Contains(string(e), "@")
}

func (e EmailAddress) Domain() string {
    parts := strings.Split(string(e), "@")
    if len(parts) == 2 {
        return parts[1]
    }
    return ""
}
```

### Method Chaining

```go
type StringBuilder struct {
    data strings.Builder
}

func (sb *StringBuilder) Append(s string) *StringBuilder {
    sb.data.WriteString(s)
    return sb
}

func (sb *StringBuilder) AppendLine(s string) *StringBuilder {
    sb.data.WriteString(s)
    sb.data.WriteString("\n")
    return sb
}

func (sb *StringBuilder) String() string {
    return sb.data.String()
}

func main() {
    result := new(StringBuilder).
        Append("Hello").
        Append(" ").
        AppendLine("World").
        Append("Go is ").
        Append("awesome!").
        String()
    
    fmt.Println(result)
}
```

## 🏗️ Constructor Functions

Go doesn't have constructors, but convention is to use `New` functions.

```go
type User struct {
    id       int
    username string
    email    string
}

// Constructor function
func NewUser(username, email string) *User {
    return &User{
        id:       generateID(),
        username: username,
        email:    email,
    }
}

// Constructor with validation
func NewUserSafe(username, email string) (*User, error) {
    if username == "" {
        return nil, errors.New("username cannot be empty")
    }
    if !strings.Contains(email, "@") {
        return nil, errors.New("invalid email")
    }
    
    return &User{
        id:       generateID(),
        username: username,
        email:    email,
    }, nil
}

// Factory with options
type UserOptions struct {
    Username string
    Email    string
    Role     string
}

func NewUserWithOptions(opts UserOptions) *User {
    user := &User{
        id:       generateID(),
        username: opts.Username,
        email:    opts.Email,
    }
    
    if opts.Role == "" {
        opts.Role = "user"
    }
    
    return user
}
```

## 🎨 Functional Options Pattern

Elegant way to handle optional parameters.

```go
type Server struct {
    host    string
    port    int
    timeout time.Duration
    maxConn int
}

// Option function type
type Option func(*Server)

// Option functions
func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func WithMaxConnections(max int) Option {
    return func(s *Server) {
        s.maxConn = max
    }
}

// Constructor with options
func NewServer(host string, opts ...Option) *Server {
    // Default values
    server := &Server{
        host:    host,
        port:    8080,
        timeout: 30 * time.Second,
        maxConn: 100,
    }
    
    // Apply options
    for _, opt := range opts {
        opt(server)
    }
    
    return server
}

func main() {
    // Use with various combinations
    s1 := NewServer("localhost")
    
    s2 := NewServer("localhost",
        WithPort(9000),
        WithTimeout(60*time.Second),
    )
    
    s3 := NewServer("localhost",
        WithPort(3000),
        WithMaxConnections(500),
    )
}
```

## 💼 Complete Example: Bank Account

```go
package main

import (
    "errors"
    "fmt"
    "time"
)

type Transaction struct {
    Type      string
    Amount    float64
    Balance   float64
    Timestamp time.Time
}

type Account struct {
    id           string
    owner        string
    balance      float64
    transactions []Transaction
}

func NewAccount(owner string, initialBalance float64) *Account {
    return &Account{
        id:           generateID(),
        owner:        owner,
        balance:      initialBalance,
        transactions: []Transaction{},
    }
}

func (a *Account) Deposit(amount float64) error {
    if amount <= 0 {
        return errors.New("deposit amount must be positive")
    }
    
    a.balance += amount
    a.addTransaction("deposit", amount)
    return nil
}

func (a *Account) Withdraw(amount float64) error {
    if amount <= 0 {
        return errors.New("withdrawal amount must be positive")
    }
    
    if amount > a.balance {
        return errors.New("insufficient funds")
    }
    
    a.balance -= amount
    a.addTransaction("withdrawal", amount)
    return nil
}

func (a *Account) Transfer(to *Account, amount float64) error {
    if err := a.Withdraw(amount); err != nil {
        return err
    }
    
    if err := to.Deposit(amount); err != nil {
        // Rollback
        a.Deposit(amount)
        return err
    }
    
    return nil
}

func (a *Account) GetBalance() float64 {
    return a.balance
}

func (a *Account) GetTransactions() []Transaction {
    return a.transactions
}

func (a *Account) addTransaction(txType string, amount float64) {
    tx := Transaction{
        Type:      txType,
        Amount:    amount,
        Balance:   a.balance,
        Timestamp: time.Now(),
    }
    a.transactions = append(a.transactions, tx)
}

func main() {
    alice := NewAccount("Alice", 1000.0)
    bob := NewAccount("Bob", 500.0)
    
    // Deposit
    alice.Deposit(500.0)
    fmt.Printf("Alice's balance: $%.2f\n", alice.GetBalance())
    
    // Withdraw
    alice.Withdraw(200.0)
    fmt.Printf("Alice's balance: $%.2f\n", alice.GetBalance())
    
    // Transfer
    alice.Transfer(bob, 300.0)
    fmt.Printf("Alice's balance: $%.2f\n", alice.GetBalance())
    fmt.Printf("Bob's balance: $%.2f\n", bob.GetBalance())
    
    // Show transactions
    fmt.Println("\nAlice's transactions:")
    for _, tx := range alice.GetTransactions() {
        fmt.Printf("%s: $%.2f (Balance: $%.2f) at %s\n",
            tx.Type, tx.Amount, tx.Balance, tx.Timestamp.Format("15:04:05"))
    }
}
```

## 🎯 Exercises

### Exercise 1: Rectangle Operations
Create a `Rectangle` struct with methods for area, perimeter, and resizing.

### Exercise 2: Stack Implementation
Implement a stack data structure with Push, Pop, and Peek methods.

### Exercise 3: Shopping Cart
Create a shopping cart system with items, quantities, and total calculation.

### Solutions

```go
// Exercise 1: Rectangle
type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

// Exercise 2: Stack
type Stack struct {
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (int, error) {
    if s.IsEmpty() {
        return 0, errors.New("stack is empty")
    }
    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]
    return item, nil
}

func (s *Stack) Peek() (int, error) {
    if s.IsEmpty() {
        return 0, errors.New("stack is empty")
    }
    return s.items[len(s.items)-1], nil
}

func (s *Stack) IsEmpty() bool {
    return len(s.items) == 0
}

// Exercise 3: Shopping Cart
type Item struct {
    Name     string
    Price    float64
    Quantity int
}

type Cart struct {
    items []Item
}

func (c *Cart) AddItem(item Item) {
    for i := range c.items {
        if c.items[i].Name == item.Name {
            c.items[i].Quantity += item.Quantity
            return
        }
    }
    c.items = append(c.items, item)
}

func (c *Cart) Total() float64 {
    total := 0.0
    for _, item := range c.items {
        total += item.Price * float64(item.Quantity)
    }
    return total
}
```

## 🔑 Key Takeaways

- Structs group related data together
- Methods add behavior to types
- Use pointer receivers to modify state or avoid copying
- Constructor functions (`New*`) are convention for initialization
- Embedding provides composition (not inheritance)
- Struct tags enable metadata for reflection
- Functional options pattern for flexible APIs

## 📖 Next Steps

Continue to [Chapter 6: Interfaces](06-interfaces.md) to learn about Go's powerful interface system and polymorphism.

