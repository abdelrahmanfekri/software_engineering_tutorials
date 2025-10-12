# Chapter 1: Setup and Installation

## 📦 Installing Go

### macOS
```bash
# Using Homebrew
brew install go

# Or download from golang.org
# https://golang.org/dl/
```

### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install golang-go

# Or download and install manually
wget https://golang.org/dl/go1.21.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
```

### Windows
1. Download installer from [golang.org/dl](https://golang.org/dl/)
2. Run the MSI installer
3. Follow installation wizard

## 🔧 Environment Setup

### Set GOPATH (Optional in modern Go)
```bash
# Add to ~/.bashrc, ~/.zshrc, or equivalent
export GOPATH=$HOME/go
export PATH=$PATH:/usr/local/go/bin:$GOPATH/bin
```

### Verify Installation
```bash
go version
# Output: go version go1.21.5 darwin/amd64

go env
# Shows all Go environment variables
```

## 🏗️ Project Structure

Modern Go uses modules (Go 1.11+). No GOPATH required!

```
my-project/
├── go.mod              # Module definition
├── go.sum              # Dependency checksums
├── main.go             # Entry point
├── internal/           # Private packages
│   └── helper/
│       └── helper.go
├── pkg/                # Public packages
│   └── util/
│       └── util.go
└── cmd/                # Multiple executables
    ├── app1/
    │   └── main.go
    └── app2/
        └── main.go
```

## 🎯 Your First Go Program

### Step 1: Create a directory
```bash
mkdir hello-world
cd hello-world
```

### Step 2: Initialize a module
```bash
go mod init example.com/hello
```

This creates `go.mod`:
```
module example.com/hello

go 1.21
```

### Step 3: Create main.go
```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

### Step 4: Run the program
```bash
# Run directly
go run main.go

# Or build and execute
go build
./hello  # On Unix/Mac
hello.exe  # On Windows
```

## 🛠️ Essential Go Commands

```bash
# Run a Go file
go run main.go

# Build an executable
go build
go build -o myapp  # Custom name

# Install to $GOPATH/bin
go install

# Format your code (always do this!)
go fmt ./...

# Download dependencies
go get github.com/pkg/errors
go get -u  # Update dependencies

# List modules
go list -m all

# Clean module cache
go clean -modcache

# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Vet (static analysis)
go vet ./...

# Show documentation
go doc fmt.Println
```

## 📝 IDE Setup

### Visual Studio Code (Recommended)
1. Install VS Code
2. Install "Go" extension by Google
3. Install tools when prompted: `gopls`, `dlv`, `staticcheck`

Features:
- IntelliSense
- Auto-completion
- Debugging
- Testing integration
- Auto-formatting on save

### GoLand (JetBrains)
Professional IDE with excellent Go support (paid).

### Vim/Neovim
- Install `vim-go` plugin
- Or use LSP with `gopls`

## 🎨 Code Formatting

Go has **official** formatting standards enforced by `gofmt`.

```bash
# Format a single file
gofmt -w main.go

# Format all Go files recursively
go fmt ./...

# Check formatting without modifying
gofmt -l .
```

**Best Practice**: Configure your editor to run `gofmt` on save.

## 🔍 Linting and Static Analysis

```bash
# Install staticcheck
go install honnef.co/go/tools/cmd/staticcheck@latest

# Run staticcheck
staticcheck ./...

# Built-in vet
go vet ./...

# Install golangci-lint (comprehensive linter)
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

# Run golangci-lint
golangci-lint run
```

## 📚 Understanding Go Modules

### Create a new module
```bash
go mod init example.com/myproject
```

### Add a dependency
```bash
# Automatically added when you import and run/build
go get github.com/gin-gonic/gin

# Or specific version
go get github.com/gin-gonic/gin@v1.9.0
```

### go.mod example
```go
module example.com/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/pkg/errors v0.9.1
)
```

### Update dependencies
```bash
# Update all to latest
go get -u ./...

# Update specific package
go get -u github.com/gin-gonic/gin

# Tidy up (remove unused dependencies)
go mod tidy
```

## 🏃 Complete Example: Hello CLI

Create a simple CLI application:

```go
// main.go
package main

import (
    "fmt"
    "os"
)

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: hello <name>")
        os.Exit(1)
    }

    name := os.Args[1]
    fmt.Printf("Hello, %s!\n", name)
}
```

Build and run:
```bash
go build -o hello
./hello World
# Output: Hello, World!
```

## 🎯 Exercise

1. Install Go on your system
2. Create a new module called `greeting`
3. Write a program that:
   - Takes a name as command-line argument
   - Prints a personalized greeting
   - Shows the current time
4. Format your code with `gofmt`
5. Build an executable

### Solution

```go
package main

import (
    "fmt"
    "os"
    "time"
)

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Please provide your name")
        return
    }

    name := os.Args[1]
    currentTime := time.Now().Format("15:04:05")
    
    fmt.Printf("Hello, %s!\n", name)
    fmt.Printf("Current time: %s\n", currentTime)
}
```

## 🔑 Key Takeaways

- Go uses modules for dependency management (no GOPATH needed)
- `go run` for quick testing, `go build` for executables
- **Always** run `go fmt` before committing code
- Use an IDE with Go support for better productivity
- `go mod tidy` keeps dependencies clean

## 📖 Next Steps

Continue to [Chapter 2: Basics](02-basics.md) to learn about variables, types, and control flow.

