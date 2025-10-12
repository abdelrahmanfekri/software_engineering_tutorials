# Chapter 14: I/O and Files

## 📄 Reading Files

### Reading Entire File

```go
import (
    "io"
    "os"
)

// Method 1: ReadFile (simplest)
func readFile1(path string) ([]byte, error) {
    return os.ReadFile(path)
}

// Method 2: Open and Read
func readFile2(path string) ([]byte, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    return io.ReadAll(file)
}

// Method 3: With buffer
func readFile3(path string) (string, error) {
    file, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer file.Close()
    
    var content strings.Builder
    buf := make([]byte, 1024)
    
    for {
        n, err := file.Read(buf)
        if err == io.EOF {
            break
        }
        if err != nil {
            return "", err
        }
        content.Write(buf[:n])
    }
    
    return content.String(), nil
}
```

### Reading Line by Line

```go
import "bufio"

func readLines(path string) ([]string, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    var lines []string
    scanner := bufio.NewScanner(file)
    
    for scanner.Scan() {
        lines = append(lines, scanner.Text())
    }
    
    if err := scanner.Err(); err != nil {
        return nil, err
    }
    
    return lines, nil
}

// Process large files efficiently
func processLargeFile(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    lineNum := 0
    
    for scanner.Scan() {
        lineNum++
        line := scanner.Text()
        // Process line
        fmt.Printf("Line %d: %s\n", lineNum, line)
    }
    
    return scanner.Err()
}
```

## ✍️ Writing Files

### Writing to File

```go
// Method 1: WriteFile (simplest)
func writeFile1(path string, data []byte) error {
    return os.WriteFile(path, data, 0644)
}

// Method 2: Create and Write
func writeFile2(path string, content string) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.WriteString(content)
    return err
}

// Method 3: With buffered writer
func writeFile3(path string, lines []string) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := bufio.NewWriter(file)
    defer writer.Flush()
    
    for _, line := range lines {
        _, err := writer.WriteString(line + "\n")
        if err != nil {
            return err
        }
    }
    
    return nil
}
```

### Appending to File

```go
func appendToFile(path string, content string) error {
    file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.WriteString(content)
    return err
}
```

## 📁 File Operations

### File Info

```go
func getFileInfo(path string) error {
    info, err := os.Stat(path)
    if err != nil {
        return err
    }
    
    fmt.Println("Name:", info.Name())
    fmt.Println("Size:", info.Size())
    fmt.Println("Mode:", info.Mode())
    fmt.Println("Modified:", info.ModTime())
    fmt.Println("Is Dir:", info.IsDir())
    
    return nil
}

// Check if file exists
func fileExists(path string) bool {
    _, err := os.Stat(path)
    return !os.IsNotExist(err)
}
```

### Directory Operations

```go
// Create directory
func createDir(path string) error {
    return os.Mkdir(path, 0755)
}

// Create directory and parents
func createDirAll(path string) error {
    return os.MkdirAll(path, 0755)
}

// List directory contents
func listDir(path string) ([]string, error) {
    entries, err := os.ReadDir(path)
    if err != nil {
        return nil, err
    }
    
    var names []string
    for _, entry := range entries {
        names = append(names, entry.Name())
    }
    
    return names, nil
}

// Walk directory tree
import "path/filepath"

func walkDir(root string) error {
    return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        
        if info.IsDir() {
            fmt.Println("Directory:", path)
        } else {
            fmt.Println("File:", path, "Size:", info.Size())
        }
        
        return nil
    })
}
```

### Copy, Move, Delete

```go
// Copy file
func copyFile(src, dst string) error {
    source, err := os.Open(src)
    if err != nil {
        return err
    }
    defer source.Close()
    
    destination, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer destination.Close()
    
    _, err = io.Copy(destination, source)
    return err
}

// Move/Rename file
func moveFile(src, dst string) error {
    return os.Rename(src, dst)
}

// Delete file
func deleteFile(path string) error {
    return os.Remove(path)
}

// Delete directory and contents
func deleteDir(path string) error {
    return os.RemoveAll(path)
}
```

## 🔍 Working with Readers and Writers

### io.Reader Interface

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}

// Example: Reading from different sources
func readFromReader(r io.Reader) error {
    buf := make([]byte, 1024)
    
    for {
        n, err := r.Read(buf)
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        
        fmt.Print(string(buf[:n]))
    }
    
    return nil
}

// Usage with different readers
func main() {
    // File reader
    file, _ := os.Open("file.txt")
    readFromReader(file)
    
    // String reader
    strReader := strings.NewReader("Hello, World!")
    readFromReader(strReader)
    
    // Bytes reader
    bytesReader := bytes.NewReader([]byte("Bytes data"))
    readFromReader(bytesReader)
}
```

### io.Writer Interface

```go
type Writer interface {
    Write(p []byte) (n int, err error)
}

// Example: Writing to different destinations
func writeToWriter(w io.Writer, data string) error {
    _, err := w.Write([]byte(data))
    return err
}

// Usage
func main() {
    // File writer
    file, _ := os.Create("output.txt")
    writeToWriter(file, "Hello, File!")
    
    // Buffer writer
    var buf bytes.Buffer
    writeToWriter(&buf, "Hello, Buffer!")
    fmt.Println(buf.String())
    
    // Stdout writer
    writeToWriter(os.Stdout, "Hello, Console!")
}
```

### Multi-Writer

```go
func writeToMultiple() error {
    file, err := os.Create("output.txt")
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Write to both file and stdout
    multiWriter := io.MultiWriter(file, os.Stdout)
    
    _, err = multiWriter.Write([]byte("This goes to file and console\n"))
    return err
}
```

### Pipe

```go
func usePipe() {
    reader, writer := io.Pipe()
    
    go func() {
        defer writer.Close()
        writer.Write([]byte("Hello through pipe!"))
    }()
    
    data, _ := io.ReadAll(reader)
    fmt.Println(string(data))
}
```

## 📋 Buffered I/O

### Buffered Reader

```go
import "bufio"

func bufferedRead(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    reader := bufio.NewReader(file)
    
    // Read line
    line, err := reader.ReadString('\n')
    if err != nil {
        return err
    }
    fmt.Println(line)
    
    // Read bytes
    buf := make([]byte, 100)
    n, err := reader.Read(buf)
    if err != nil {
        return err
    }
    fmt.Println(string(buf[:n]))
    
    return nil
}
```

### Buffered Writer

```go
func bufferedWrite(path string) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := bufio.NewWriter(file)
    defer writer.Flush()  // Important!
    
    for i := 0; i < 100; i++ {
        writer.WriteString(fmt.Sprintf("Line %d\n", i))
    }
    
    return nil
}
```

## 🎯 JSON File Operations

```go
import "encoding/json"

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

// Write JSON to file
func writeJSON(path string, users []User) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")  // Pretty print
    
    return encoder.Encode(users)
}

// Read JSON from file
func readJSON(path string) ([]User, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    var users []User
    decoder := json.NewDecoder(file)
    
    if err := decoder.Decode(&users); err != nil {
        return nil, err
    }
    
    return users, nil
}
```

## 💼 Complete Example: Log File Manager

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "path/filepath"
    "time"
)

type LogManager struct {
    logDir  string
    maxSize int64  // bytes
}

func NewLogManager(dir string, maxSize int64) *LogManager {
    os.MkdirAll(dir, 0755)
    return &LogManager{
        logDir:  dir,
        maxSize: maxSize,
    }
}

func (lm *LogManager) getCurrentLogPath() string {
    date := time.Now().Format("2006-01-02")
    return filepath.Join(lm.logDir, fmt.Sprintf("app-%s.log", date))
}

func (lm *LogManager) Log(message string) error {
    logPath := lm.getCurrentLogPath()
    
    // Check if rotation needed
    if info, err := os.Stat(logPath); err == nil {
        if info.Size() >= lm.maxSize {
            if err := lm.rotateLog(logPath); err != nil {
                return err
            }
        }
    }
    
    // Append to log
    file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        return err
    }
    defer file.Close()
    
    timestamp := time.Now().Format("2006-01-02 15:04:05")
    logLine := fmt.Sprintf("[%s] %s\n", timestamp, message)
    
    _, err = file.WriteString(logLine)
    return err
}

func (lm *LogManager) rotateLog(logPath string) error {
    timestamp := time.Now().Format("150405")
    rotatedPath := fmt.Sprintf("%s.%s", logPath, timestamp)
    return os.Rename(logPath, rotatedPath)
}

func (lm *LogManager) ReadLogs(date string) ([]string, error) {
    logPath := filepath.Join(lm.logDir, fmt.Sprintf("app-%s.log", date))
    
    file, err := os.Open(logPath)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    var lines []string
    scanner := bufio.NewScanner(file)
    
    for scanner.Scan() {
        lines = append(lines, scanner.Text())
    }
    
    return lines, scanner.Err()
}

func (lm *LogManager) CleanOldLogs(days int) error {
    cutoff := time.Now().AddDate(0, 0, -days)
    
    return filepath.Walk(lm.logDir, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        
        if !info.IsDir() && info.ModTime().Before(cutoff) {
            fmt.Printf("Deleting old log: %s\n", path)
            return os.Remove(path)
        }
        
        return nil
    })
}

func main() {
    logManager := NewLogManager("logs", 1024*1024)  // 1MB max
    
    // Write logs
    logManager.Log("Application started")
    logManager.Log("User logged in")
    logManager.Log("Processing request")
    
    // Read logs
    today := time.Now().Format("2006-01-02")
    logs, _ := logManager.ReadLogs(today)
    
    fmt.Println("Today's logs:")
    for _, log := range logs {
        fmt.Println(log)
    }
    
    // Clean old logs
    logManager.CleanOldLogs(30)
}
```

## 🎯 Exercises

### Exercise 1: File Search
Implement a file search utility that finds files by pattern.

### Exercise 2: CSV Reader/Writer
Create functions to read and write CSV files.

### Exercise 3: File Backup
Build a file backup utility with compression.

### Solutions

```go
// Exercise 1: File Search
func findFiles(root, pattern string) ([]string, error) {
    var matches []string
    
    err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        
        if !info.IsDir() {
            matched, _ := filepath.Match(pattern, filepath.Base(path))
            if matched {
                matches = append(matches, path)
            }
        }
        
        return nil
    })
    
    return matches, err
}

// Exercise 2: CSV
import "encoding/csv"

func writeCSV(path string, records [][]string) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := csv.NewWriter(file)
    defer writer.Flush()
    
    return writer.WriteAll(records)
}

func readCSV(path string) ([][]string, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    reader := csv.NewReader(file)
    return reader.ReadAll()
}
```

## 🔑 Key Takeaways

- Use `os.ReadFile` for small files, buffered I/O for large files
- Always `defer file.Close()`
- Use `io.Reader` and `io.Writer` interfaces for flexibility
- Buffered I/O improves performance
- Check errors for all I/O operations
- Use `filepath` package for path operations

## 📖 Next Steps

Continue to [Chapter 15: Standard Library Essentials](15-stdlib.md) to explore Go's rich standard library.

