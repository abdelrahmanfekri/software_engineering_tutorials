# Chapter 22: Reflection

Reflection allows programs to inspect and manipulate values at runtime.

```go
import "reflect"
```

## 🔍 Basic Reflection

```go
// Get type
var x int = 42
t := reflect.TypeOf(x)
fmt.Println(t)  // int

// Get value
v := reflect.ValueOf(x)
fmt.Println(v)  // 42

// Get kind
fmt.Println(t.Kind())  // int
```

## 🎯 Type and Value

```go
type User struct {
    Name string
    Age  int
}

user := User{Name: "Alice", Age: 25}

t := reflect.TypeOf(user)
v := reflect.ValueOf(user)

fmt.Println("Type:", t.Name())        // User
fmt.Println("Package:", t.PkgPath())  // main
fmt.Println("Kind:", t.Kind())        // struct
fmt.Println("NumField:", t.NumField()) // 2

// Iterate fields
for i := 0; i < t.NumField(); i++ {
    field := t.Field(i)
    value := v.Field(i)
    fmt.Printf("%s: %v\n", field.Name, value)
}
```

## 📝 Modifying Values

```go
x := 42
v := reflect.ValueOf(&x)  // Must pass pointer
v = v.Elem()              // Dereference

if v.CanSet() {
    v.SetInt(100)
}
fmt.Println(x)  // 100
```

## 🔑 Key Takeaways

- Use sparingly (performance cost)
- Useful for serialization, ORMs, and generic code
- Check `CanSet()` before modifying
- Prefer generics over reflection when possible

## 📖 Next Steps

Continue to [Chapter 23: Generics](23-generics.md).

