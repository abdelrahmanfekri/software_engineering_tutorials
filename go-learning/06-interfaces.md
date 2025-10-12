# Chapter 6: Interfaces

## 🎭 What are Interfaces?

Interfaces define behavior (methods) without implementation. They enable polymorphism in Go.

### Basic Interface

```go
// Define interface
type Writer interface {
    Write([]byte) (int, error)
}

// Any type with Write method implements Writer
type ConsoleWriter struct{}

func (cw ConsoleWriter) Write(data []byte) (int, error) {
    n, err := fmt.Println(string(data))
    return n, err
}

func main() {
    var w Writer = ConsoleWriter{}
    w.Write([]byte("Hello, Interface!"))
}
```

### Interface Implementation

Interfaces are **implicitly** implemented (no `implements` keyword).

```go
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Rectangle implements Shape
type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// Circle implements Shape
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

// Function that works with any Shape
func PrintShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f\n", s.Area())
    fmt.Printf("Perimeter: %.2f\n", s.Perimeter())
}

func main() {
    rect := Rectangle{Width: 10, Height: 5}
    circle := Circle{Radius: 7}
    
    PrintShapeInfo(rect)
    PrintShapeInfo(circle)
}
```

## 🔍 Empty Interface

`interface{}` (or `any` in Go 1.18+) accepts any type.

```go
// Can hold any value
var anything interface{}

anything = 42
anything = "hello"
anything = []int{1, 2, 3}
anything = struct{ Name string }{"Alice"}

// Using any (Go 1.18+)
var value any = "hello"

// Common use: generic data structures
func Print(v interface{}) {
    fmt.Println(v)
}

Print(42)
Print("hello")
Print([]int{1, 2, 3})
```

## 🎯 Type Assertions

Extract concrete type from interface.

```go
var i interface{} = "hello"

// Type assertion
s := i.(string)
fmt.Println(s)  // "hello"

// Safe type assertion (with check)
s, ok := i.(string)
if ok {
    fmt.Println("String:", s)
} else {
    fmt.Println("Not a string")
}

// Unsafe assertion (panics if wrong)
// n := i.(int)  // PANIC!

// Type switch
func describe(i interface{}) {
    switch v := i.(type) {
    case int:
        fmt.Printf("Integer: %d\n", v)
    case string:
        fmt.Printf("String: %s\n", v)
    case bool:
        fmt.Printf("Boolean: %t\n", v)
    case []int:
        fmt.Printf("Slice of ints: %v\n", v)
    default:
        fmt.Printf("Unknown type: %T\n", v)
    }
}

describe(42)
describe("hello")
describe(true)
describe([]int{1, 2, 3})
```

## 📚 Standard Library Interfaces

### io.Reader and io.Writer

```go
// Reader interface
type Reader interface {
    Read(p []byte) (n int, err error)
}

// Writer interface
type Writer interface {
    Write(p []byte) (n int, err error)
}

// Example: Reading from string
func main() {
    reader := strings.NewReader("Hello, Go!")
    
    buf := make([]byte, 8)
    for {
        n, err := reader.Read(buf)
        if err == io.EOF {
            break
        }
        fmt.Print(string(buf[:n]))
    }
}

// Example: Writing to buffer
func main() {
    var buf bytes.Buffer
    
    buf.Write([]byte("Hello "))
    buf.WriteString("World!")
    
    fmt.Println(buf.String())  // "Hello World!"
}
```

### Stringer Interface

```go
// Stringer for custom string representation
type Stringer interface {
    String() string
}

type Person struct {
    Name string
    Age  int
}

func (p Person) String() string {
    return fmt.Sprintf("%s (%d years old)", p.Name, p.Age)
}

func main() {
    p := Person{"Alice", 25}
    fmt.Println(p)  // Calls String() method
    // Output: Alice (25 years old)
}
```

### error Interface

```go
// Built-in error interface
type error interface {
    Error() string
}

// Custom error
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

func validateAge(age int) error {
    if age < 0 {
        return ValidationError{
            Field:   "age",
            Message: "must be non-negative",
        }
    }
    if age > 150 {
        return ValidationError{
            Field:   "age",
            Message: "must be realistic",
        }
    }
    return nil
}

func main() {
    if err := validateAge(-5); err != nil {
        fmt.Println(err)  // age: must be non-negative
    }
}
```

### sort.Interface

```go
type Interface interface {
    Len() int
    Less(i, j int) bool
    Swap(i, j int)
}

// Example: Sort custom type
type Person struct {
    Name string
    Age  int
}

type ByAge []Person

func (a ByAge) Len() int           { return len(a) }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func main() {
    people := []Person{
        {"Bob", 31},
        {"Alice", 25},
        {"Carol", 28},
    }
    
    sort.Sort(ByAge(people))
    fmt.Println(people)
    // [{Alice 25} {Carol 28} {Bob 31}]
    
    // Or use sort.Slice (easier)
    sort.Slice(people, func(i, j int) bool {
        return people[i].Name < people[j].Name
    })
}
```

## 🏗️ Interface Composition

Interfaces can embed other interfaces.

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

// Composed interfaces
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// io.ReadWriteCloser from standard library
// Used by files, network connections, etc.
```

## 🎨 Design Patterns with Interfaces

### Strategy Pattern

```go
// Strategy interface
type PaymentMethod interface {
    Pay(amount float64) error
}

// Concrete strategies
type CreditCard struct {
    Number string
}

func (cc CreditCard) Pay(amount float64) error {
    fmt.Printf("Paid $%.2f with Credit Card ending in %s\n",
        amount, cc.Number[len(cc.Number)-4:])
    return nil
}

type PayPal struct {
    Email string
}

func (pp PayPal) Pay(amount float64) error {
    fmt.Printf("Paid $%.2f via PayPal account %s\n", amount, pp.Email)
    return nil
}

// Context
type ShoppingCart struct {
    total float64
}

func (sc *ShoppingCart) Checkout(method PaymentMethod) error {
    return method.Pay(sc.total)
}

func main() {
    cart := &ShoppingCart{total: 99.99}
    
    // Different payment methods
    cart.Checkout(CreditCard{Number: "1234567890123456"})
    cart.Checkout(PayPal{Email: "user@example.com"})
}
```

### Dependency Injection

```go
// Interface for storage
type UserStore interface {
    Save(user User) error
    Find(id int) (User, error)
}

// Service depends on interface
type UserService struct {
    store UserStore
}

func NewUserService(store UserStore) *UserService {
    return &UserService{store: store}
}

func (s *UserService) CreateUser(name string) error {
    user := User{Name: name}
    return s.store.Save(user)
}

// Different implementations
type MemoryStore struct {
    users map[int]User
}

func (m *MemoryStore) Save(user User) error {
    m.users[user.ID] = user
    return nil
}

type DatabaseStore struct {
    db *sql.DB
}

func (d *DatabaseStore) Save(user User) error {
    // Save to database
    return nil
}

// Use different implementations
func main() {
    // In-memory for testing
    memStore := &MemoryStore{users: make(map[int]User)}
    service := NewUserService(memStore)
    
    // Database for production
    dbStore := &DatabaseStore{db: getDB()}
    productionService := NewUserService(dbStore)
}
```

### Adapter Pattern

```go
// Target interface
type Logger interface {
    Log(message string)
}

// Adaptee (third-party logger)
type ThirdPartyLogger struct{}

func (tpl ThirdPartyLogger) WriteLog(level, msg string) {
    fmt.Printf("[%s] %s\n", level, msg)
}

// Adapter
type LoggerAdapter struct {
    logger ThirdPartyLogger
}

func (la LoggerAdapter) Log(message string) {
    la.logger.WriteLog("INFO", message)
}

// Usage
func main() {
    var logger Logger = LoggerAdapter{logger: ThirdPartyLogger{}}
    logger.Log("Application started")
}
```

## 🔄 Interface Best Practices

### Accept Interfaces, Return Structs

```go
// Good: Accept interface (flexible)
func ProcessData(r io.Reader) error {
    // Can work with files, buffers, network, etc.
    data, err := io.ReadAll(r)
    // Process data...
    return err
}

// Good: Return concrete type (clear)
func OpenFile(path string) (*os.File, error) {
    return os.Open(path)
}

// Bad: Return interface (unclear, limits options)
func OpenFile(path string) (io.Reader, error) {
    return os.Open(path)  // Lose file-specific methods
}
```

### Small Interfaces

```go
// Good: Small, focused interfaces
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// Bad: Large interface (hard to implement)
type DataStore interface {
    Save(data interface{}) error
    Load(id int) (interface{}, error)
    Update(id int, data interface{}) error
    Delete(id int) error
    List() ([]interface{}, error)
    Count() (int, error)
    Backup() error
    Restore(backup string) error
}

// Better: Split into smaller interfaces
type Saver interface {
    Save(data interface{}) error
}

type Loader interface {
    Load(id int) (interface{}, error)
}
```

## 💼 Complete Example: Plugin System

```go
package main

import (
    "fmt"
)

// Plugin interface
type Plugin interface {
    Name() string
    Execute(args map[string]interface{}) (interface{}, error)
}

// Registry
type PluginRegistry struct {
    plugins map[string]Plugin
}

func NewPluginRegistry() *PluginRegistry {
    return &PluginRegistry{
        plugins: make(map[string]Plugin),
    }
}

func (pr *PluginRegistry) Register(plugin Plugin) {
    pr.plugins[plugin.Name()] = plugin
}

func (pr *PluginRegistry) Execute(name string, args map[string]interface{}) (interface{}, error) {
    plugin, exists := pr.plugins[name]
    if !exists {
        return nil, fmt.Errorf("plugin not found: %s", name)
    }
    return plugin.Execute(args)
}

// Concrete plugins
type GreetPlugin struct{}

func (g GreetPlugin) Name() string {
    return "greet"
}

func (g GreetPlugin) Execute(args map[string]interface{}) (interface{}, error) {
    name, ok := args["name"].(string)
    if !ok {
        name = "World"
    }
    return fmt.Sprintf("Hello, %s!", name), nil
}

type MathPlugin struct{}

func (m MathPlugin) Name() string {
    return "math"
}

func (m MathPlugin) Execute(args map[string]interface{}) (interface{}, error) {
    a, _ := args["a"].(float64)
    b, _ := args["b"].(float64)
    op, _ := args["op"].(string)
    
    switch op {
    case "add":
        return a + b, nil
    case "subtract":
        return a - b, nil
    case "multiply":
        return a * b, nil
    case "divide":
        if b == 0 {
            return nil, fmt.Errorf("division by zero")
        }
        return a / b, nil
    default:
        return nil, fmt.Errorf("unknown operation: %s", op)
    }
}

func main() {
    registry := NewPluginRegistry()
    
    // Register plugins
    registry.Register(GreetPlugin{})
    registry.Register(MathPlugin{})
    
    // Execute plugins
    result, _ := registry.Execute("greet", map[string]interface{}{
        "name": "Alice",
    })
    fmt.Println(result)
    
    result, _ = registry.Execute("math", map[string]interface{}{
        "a":  10.0,
        "b":  5.0,
        "op": "multiply",
    })
    fmt.Println(result)
}
```

## 🎯 Exercises

### Exercise 1: Animal Interface
Create an `Animal` interface with `Speak()` and `Move()` methods. Implement for Dog, Cat, and Bird.

### Exercise 2: Database Interface
Design a database interface with CRUD operations and implement in-memory and mock versions.

### Exercise 3: Notification System
Create a notification system that can send emails, SMS, and push notifications.

### Solutions

```go
// Exercise 1: Animals
type Animal interface {
    Speak() string
    Move() string
}

type Dog struct {
    Name string
}

func (d Dog) Speak() string { return "Woof!" }
func (d Dog) Move() string  { return "Running" }

type Cat struct {
    Name string
}

func (c Cat) Speak() string { return "Meow!" }
func (c Cat) Move() string  { return "Walking" }

type Bird struct {
    Name string
}

func (b Bird) Speak() string { return "Tweet!" }
func (b Bird) Move() string  { return "Flying" }

func AnimalBehavior(a Animal) {
    fmt.Printf("%s and %s\n", a.Speak(), a.Move())
}

// Exercise 2: Database
type Database interface {
    Create(item interface{}) error
    Read(id int) (interface{}, error)
    Update(id int, item interface{}) error
    Delete(id int) error
}

type InMemoryDB struct {
    data map[int]interface{}
}

func NewInMemoryDB() *InMemoryDB {
    return &InMemoryDB{data: make(map[int]interface{})}
}

func (db *InMemoryDB) Create(item interface{}) error {
    id := len(db.data) + 1
    db.data[id] = item
    return nil
}

func (db *InMemoryDB) Read(id int) (interface{}, error) {
    item, exists := db.data[id]
    if !exists {
        return nil, fmt.Errorf("not found")
    }
    return item, nil
}

// Exercise 3: Notifications
type Notifier interface {
    Send(to, message string) error
}

type EmailNotifier struct{}

func (e EmailNotifier) Send(to, message string) error {
    fmt.Printf("Email to %s: %s\n", to, message)
    return nil
}

type SMSNotifier struct{}

func (s SMSNotifier) Send(to, message string) error {
    fmt.Printf("SMS to %s: %s\n", to, message)
    return nil
}

type NotificationService struct {
    notifiers []Notifier
}

func (ns *NotificationService) Notify(to, message string) {
    for _, notifier := range ns.notifiers {
        notifier.Send(to, message)
    }
}
```

## 🔑 Key Takeaways

- Interfaces are implicitly implemented
- Small interfaces are better (Go proverb: "The bigger the interface, the weaker the abstraction")
- Accept interfaces, return concrete types
- Empty interface (`interface{}` or `any`) accepts any type
- Type assertions extract concrete types from interfaces
- Standard library has many useful interfaces (io.Reader, io.Writer, error, etc.)

## 📖 Next Steps

Continue to [Chapter 7: Pointers](07-pointers.md) to understand memory management and pointer usage in Go.

