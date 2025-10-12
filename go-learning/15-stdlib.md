# Chapter 15: Standard Library Essentials

## 📝 strings Package

### Common String Operations

```go
import "strings"

// Contains
strings.Contains("hello world", "world")  // true

// Split
parts := strings.Split("a,b,c", ",")  // ["a", "b", "c"]

// Join
joined := strings.Join([]string{"a", "b", "c"}, "-")  // "a-b-c"

// Replace
replaced := strings.Replace("hello", "l", "L", 1)  // "heLlo"
replacedAll := strings.ReplaceAll("hello", "l", "L")  // "heLLo"

// Trim
trimmed := strings.Trim("  hello  ", " ")  // "hello"
strings.TrimSpace("  hello  ")  // "hello"
strings.TrimPrefix("hello world", "hello ")  // "world"
strings.TrimSuffix("hello world", " world")  // "hello"

// Case conversion
strings.ToUpper("hello")  // "HELLO"
strings.ToLower("HELLO")  // "hello"
strings.Title("hello world")  // "Hello World"

// Starts/Ends with
strings.HasPrefix("hello", "he")  // true
strings.HasSuffix("hello", "lo")  // true

// Count occurrences
strings.Count("hello", "l")  // 2

// Index
strings.Index("hello", "ll")  // 2
strings.LastIndex("hello", "l")  // 3
```

### String Builder

```go
var builder strings.Builder

builder.WriteString("Hello")
builder.WriteString(" ")
builder.WriteString("World")
builder.WriteByte('!')
builder.WriteRune('🎉')

result := builder.String()  // "Hello World!🎉"

// More efficient than concatenation for loops
```

## 🔢 fmt Package

### Printing

```go
import "fmt"

// Print variants
fmt.Print("Hello")       // No newline
fmt.Println("Hello")     // With newline
fmt.Printf("Name: %s\n", name)  // Formatted

// Sprint variants (return string)
s := fmt.Sprint("Hello")
s := fmt.Sprintf("Age: %d", age)
s := fmt.Sprintln("Hello")

// Fprint variants (write to io.Writer)
fmt.Fprint(os.Stdout, "Hello")
fmt.Fprintf(os.Stdout, "Age: %d\n", age)
```

### Format Verbs

```go
// General
%v    // Default format
%+v   // With field names (structs)
%#v   // Go-syntax representation
%T    // Type
%%    // Literal %

// Boolean
%t    // true or false

// Integer
%d    // Decimal
%b    // Binary
%o    // Octal
%x    // Hexadecimal (lowercase)
%X    // Hexadecimal (uppercase)

// Float
%f    // Decimal point
%e    // Scientific notation
%.2f  // 2 decimal places

// String
%s    // String
%q    // Quoted string
%x    // Hex dump

// Pointer
%p    // Pointer address

// Examples
fmt.Printf("%v\n", user)           // {Alice 25}
fmt.Printf("%+v\n", user)          // {Name:Alice Age:25}
fmt.Printf("%#v\n", user)          // main.User{Name:"Alice", Age:25}
fmt.Printf("%T\n", user)           // main.User
fmt.Printf("%t\n", true)           // true
fmt.Printf("%d\n", 42)             // 42
fmt.Printf("%b\n", 42)             // 101010
fmt.Printf("%x\n", 255)            // ff
fmt.Printf("%.2f\n", 3.14159)      // 3.14
fmt.Printf("%q\n", "hello")        // "hello"
fmt.Printf("%p\n", &user)          // 0xc000010230
```

## ⏰ time Package

### Working with Time

```go
import "time"

// Current time
now := time.Now()

// Specific time
t := time.Date(2024, time.January, 1, 12, 0, 0, 0, time.UTC)

// Parsing
layout := "2006-01-02"
t, err := time.Parse(layout, "2024-01-01")

layout2 := "2006-01-02 15:04:05"
t, err := time.Parse(layout2, "2024-01-01 12:30:45")

// Formatting
formatted := now.Format("2006-01-02 15:04:05")
formatted := now.Format(time.RFC3339)

// Components
year := now.Year()
month := now.Month()
day := now.Day()
hour := now.Hour()
minute := now.Minute()
second := now.Second()

// Comparisons
t1.Before(t2)
t1.After(t2)
t1.Equal(t2)

// Arithmetic
future := now.Add(24 * time.Hour)
past := now.Add(-24 * time.Hour)
diff := t2.Sub(t1)  // Duration

// Duration
d := 5 * time.Second
d := 10 * time.Minute
d := 2 * time.Hour
d := time.Duration(100) * time.Millisecond
```

### Timers and Tickers

```go
// Timer (fires once)
timer := time.NewTimer(2 * time.Second)
<-timer.C
fmt.Println("Timer fired!")

// After (simpler)
<-time.After(2 * time.Second)

// Ticker (fires repeatedly)
ticker := time.NewTicker(1 * time.Second)
defer ticker.Stop()

for t := range ticker.C {
    fmt.Println("Tick at", t)
}

// Stop ticker
ticker.Stop()
```

## 🔐 crypto Package

### Hashing

```go
import (
    "crypto/md5"
    "crypto/sha256"
    "encoding/hex"
)

// MD5
data := []byte("hello")
hash := md5.Sum(data)
hashStr := hex.EncodeToString(hash[:])

// SHA256
hash256 := sha256.Sum256(data)
hashStr256 := hex.EncodeToString(hash256[:])

// Streaming hash
hasher := sha256.New()
hasher.Write([]byte("hello"))
hasher.Write([]byte(" world"))
hashBytes := hasher.Sum(nil)
```

### Random Numbers

```go
import (
    "crypto/rand"
    "math/big"
    "math/rand/v2"
)

// Crypto random (secure)
n, err := rand.Int(rand.Reader, big.NewInt(100))

bytes := make([]byte, 32)
rand.Read(bytes)

// Math random (fast, not secure)
rand.IntN(100)           // 0-99
rand.Float64()          // 0.0-1.0
rand.Shuffle(len(slice), func(i, j int) {
    slice[i], slice[j] = slice[j], slice[i]
})
```

## 🌐 net/http Package (Basic)

### HTTP Client

```go
import "net/http"

// GET request
resp, err := http.Get("https://api.example.com/users")
if err != nil {
    return err
}
defer resp.Body.Close()

body, err := io.ReadAll(resp.Body)

// POST request
data := []byte(`{"name":"Alice"}`)
resp, err := http.Post("https://api.example.com/users",
    "application/json", bytes.NewBuffer(data))

// Custom request
req, err := http.NewRequest("PUT", "https://api.example.com/users/1",
    bytes.NewBuffer(data))
req.Header.Set("Content-Type", "application/json")
req.Header.Set("Authorization", "Bearer token")

client := &http.Client{Timeout: 10 * time.Second}
resp, err := client.Do(req)
```

## 📊 encoding Package

### JSON

```go
import "encoding/json"

type User struct {
    Name  string `json:"name"`
    Age   int    `json:"age"`
    Email string `json:"email,omitempty"`
}

// Marshal (struct to JSON)
user := User{Name: "Alice", Age: 25}
jsonData, err := json.Marshal(user)
// {"name":"Alice","age":25}

// Pretty print
jsonData, err := json.MarshalIndent(user, "", "  ")

// Unmarshal (JSON to struct)
jsonStr := `{"name":"Bob","age":30}`
var user User
err := json.Unmarshal([]byte(jsonStr), &user)
```

### Base64

```go
import "encoding/base64"

// Encode
data := []byte("hello world")
encoded := base64.StdEncoding.EncodeToString(data)

// Decode
decoded, err := base64.StdEncoding.DecodeString(encoded)
```

### Hex

```go
import "encoding/hex"

// Encode
data := []byte("hello")
hexStr := hex.EncodeToString(data)  // "68656c6c6f"

// Decode
decoded, err := hex.DecodeString(hexStr)
```

## 🗜️ compress Package

### Gzip

```go
import (
    "compress/gzip"
)

// Compress
func compressGzip(data []byte) ([]byte, error) {
    var buf bytes.Buffer
    writer := gzip.NewWriter(&buf)
    
    _, err := writer.Write(data)
    if err != nil {
        return nil, err
    }
    
    if err := writer.Close(); err != nil {
        return nil, err
    }
    
    return buf.Bytes(), nil
}

// Decompress
func decompressGzip(data []byte) ([]byte, error) {
    reader, err := gzip.NewReader(bytes.NewReader(data))
    if err != nil {
        return nil, err
    }
    defer reader.Close()
    
    return io.ReadAll(reader)
}
```

## 🔍 regexp Package

### Regular Expressions

```go
import "regexp"

// Match
matched, _ := regexp.MatchString(`\d+`, "There are 42 apples")  // true

// Compile (for reuse)
re := regexp.MustCompile(`\d+`)
matched := re.MatchString("There are 42 apples")  // true

// Find
re.FindString("There are 42 apples")  // "42"
re.FindAllString("There are 42 apples and 10 oranges", -1)  // ["42", "10"]

// Replace
re.ReplaceAllString("There are 42 apples", "XX")
// "There are XX apples"

// Submatches
re := regexp.MustCompile(`(\d+) (\w+)`)
matches := re.FindStringSubmatch("There are 42 apples")
// ["42 apples", "42", "apples"]

// Named groups
re := regexp.MustCompile(`(?P<count>\d+) (?P<item>\w+)`)
matches := re.FindStringSubmatch("42 apples")
```

## 🎯 sort Package

```go
import "sort"

// Slice of ints
nums := []int{3, 1, 4, 1, 5, 9}
sort.Ints(nums)
sort.Sort(sort.Reverse(sort.IntSlice(nums)))

// Slice of strings
names := []string{"Charlie", "Alice", "Bob"}
sort.Strings(names)

// Custom sorting
sort.Slice(users, func(i, j int) bool {
    return users[i].Age < users[j].Age
})

// Binary search
index := sort.SearchInts(nums, 5)

// Is sorted?
isSorted := sort.IntsAreSorted(nums)
```

## 💼 Complete Example: Utilities Package

```go
package utils

import (
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "regexp"
    "strings"
    "time"
)

// StringUtils
type StringUtils struct{}

func (StringUtils) Slugify(s string) string {
    s = strings.ToLower(s)
    s = strings.ReplaceAll(s, " ", "-")
    
    re := regexp.MustCompile(`[^a-z0-9\-]`)
    s = re.ReplaceAllString(s, "")
    
    return s
}

func (StringUtils) Truncate(s string, max int) string {
    if len(s) <= max {
        return s
    }
    return s[:max] + "..."
}

// CryptoUtils
type CryptoUtils struct{}

func (CryptoUtils) Hash(data string) string {
    hash := sha256.Sum256([]byte(data))
    return hex.EncodeToString(hash[:])
}

// TimeUtils
type TimeUtils struct{}

func (TimeUtils) FormatRelative(t time.Time) string {
    diff := time.Since(t)
    
    switch {
    case diff < time.Minute:
        return "just now"
    case diff < time.Hour:
        mins := int(diff.Minutes())
        return fmt.Sprintf("%d minute(s) ago", mins)
    case diff < 24*time.Hour:
        hours := int(diff.Hours())
        return fmt.Sprintf("%d hour(s) ago", hours)
    default:
        days := int(diff.Hours() / 24)
        return fmt.Sprintf("%d day(s) ago", days)
    }
}

// JSONUtils
type JSONUtils struct{}

func (JSONUtils) Pretty(data interface{}) (string, error) {
    bytes, err := json.MarshalIndent(data, "", "  ")
    if err != nil {
        return "", err
    }
    return string(bytes), nil
}

func (JSONUtils) Minify(jsonStr string) (string, error) {
    var data interface{}
    if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
        return "", err
    }
    
    bytes, err := json.Marshal(data)
    if err != nil {
        return "", err
    }
    
    return string(bytes), nil
}

// Example usage
func main() {
    str := StringUtils{}
    fmt.Println(str.Slugify("Hello World!"))  // "hello-world"
    
    crypto := CryptoUtils{}
    fmt.Println(crypto.Hash("password"))
    
    timeUtil := TimeUtils{}
    pastTime := time.Now().Add(-2 * time.Hour)
    fmt.Println(timeUtil.FormatRelative(pastTime))  // "2 hour(s) ago"
    
    jsonUtil := JSONUtils{}
    user := map[string]interface{}{"name": "Alice", "age": 25}
    pretty, _ := jsonUtil.Pretty(user)
    fmt.Println(pretty)
}
```

## 🎯 Exercises

### Exercise 1: String Processor
Build a string processing utility with various transformations.

### Exercise 2: Date Calculator
Create a date utility that calculates business days, age, etc.

### Exercise 3: File Hash Calculator
Build a tool to calculate file checksums.

## 🔑 Key Takeaways

- Go has a rich standard library covering most common needs
- `strings` package for string operations
- `time` package for date/time handling
- `encoding/json` for JSON marshaling/unmarshaling
- `regexp` for regular expressions
- `crypto` packages for security operations
- Always check documentation: https://pkg.go.dev/std

## 📖 Next Steps

Continue to [Chapter 16: HTTP and Web Servers](16-web-http.md) to build web applications.

