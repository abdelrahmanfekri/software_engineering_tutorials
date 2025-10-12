# Chapter 13: Modules and Dependencies

## 📦 Go Modules

Go modules are the standard way to manage dependencies (since Go 1.11).

### Creating a Module

```bash
# Initialize a new module
go mod init github.com/username/project

# Creates go.mod file
```

### go.mod File

```go
module github.com/username/myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

require (
    // Indirect dependencies
    github.com/bytedance/sonic v1.9.1 // indirect
    github.com/chenzhuoyu/base64x v0.0.0-20221115062448-fe3a3abad311 // indirect
)

replace github.com/old/package => github.com/new/package v1.2.3
```

### go.sum File

Contains cryptographic hashes of dependencies:

```
github.com/gin-gonic/gin v1.9.1 h1:4idEAncQnU5cB7BeOkPtxjfCSye0AAm1R0RVIqJ+Jmg=
github.com/gin-gonic/gin v1.9.1/go.mod h1:hPrL7YrpYKXt5YId3A/Tnip5kqbEAP+KLuI3SUcPTeU=
```

## 🔧 Managing Dependencies

### Adding Dependencies

```bash
# Add dependency automatically (imports in code)
go get github.com/gin-gonic/gin

# Add specific version
go get github.com/gin-gonic/gin@v1.9.0

# Add latest version
go get github.com/gin-gonic/gin@latest

# Add commit/branch
go get github.com/user/repo@commit-hash
go get github.com/user/repo@branch-name
```

### Updating Dependencies

```bash
# Update all dependencies
go get -u ./...

# Update specific dependency
go get -u github.com/gin-gonic/gin

# Update to specific version
go get github.com/gin-gonic/gin@v1.9.1

# Tidy up (remove unused dependencies)
go mod tidy
```

### Removing Dependencies

```bash
# Remove dependency from code, then:
go mod tidy
```

### Listing Dependencies

```bash
# List all dependencies
go list -m all

# List available versions
go list -m -versions github.com/gin-gonic/gin

# Why is this dependency here?
go mod why github.com/some/package

# Show dependency graph
go mod graph
```

## 📂 Module Structure

### Basic Module

```
myproject/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   └── helper/
│       └── helper.go
├── pkg/
│   └── util/
│       └── util.go
└── cmd/
    ├── server/
    │   └── main.go
    └── cli/
        └── main.go
```

### Multi-Module Repository

```
monorepo/
├── api/
│   ├── go.mod
│   └── main.go
├── worker/
│   ├── go.mod
│   └── main.go
└── shared/
    ├── go.mod
    └── common.go
```

## 🔄 Versioning

### Semantic Versioning

```
v1.2.3
│ │ │
│ │ └─ Patch: Bug fixes
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

### Tagging Versions

```bash
# Tag a version
git tag v1.0.0
git push origin v1.0.0

# Tag with message
git tag -a v1.0.0 -m "First release"
git push origin v1.0.0

# Major version (breaking changes)
git tag v2.0.0
```

### Major Version Suffixes

```go
// v0 or v1 (no suffix needed)
module github.com/user/project

import "github.com/user/project/pkg"

// v2+ (requires suffix)
module github.com/user/project/v2

import "github.com/user/project/v2/pkg"
```

## 🎯 Working with Local Modules

### Replace Directive

```go
// go.mod
module myapp

require github.com/user/library v1.0.0

// Use local version for development
replace github.com/user/library => ../library
```

```bash
# Or via command
go mod edit -replace github.com/user/library=../library
```

### Vendor Directory

```bash
# Copy dependencies to vendor/
go mod vendor

# Build using vendor
go build -mod=vendor

# Update vendor
go mod vendor
```

## 🔍 Advanced Module Features

### Retract Directive

```go
// go.mod
module github.com/user/project

go 1.21

// Retract broken versions
retract (
    v1.0.0 // Published accidentally
    v1.0.1 // Contains critical bug
    [v1.1.0, v1.2.0] // Range of versions
)
```

### Exclude Directive

```go
// go.mod
module myapp

// Exclude specific versions
exclude github.com/broken/package v1.2.0
```

### Minimal Version Selection

Go uses Minimal Version Selection (MVS):

```
A requires B v1.2
A requires C v1.5
B requires C v1.3
C requires D v1.1

Result:
B v1.2 (specified by A)
C v1.5 (highest specified)
D v1.1 (specified by C)
```

## 🛠️ Module Commands Reference

```bash
# Initialize module
go mod init [module-path]

# Download dependencies
go mod download

# Remove unused dependencies
go mod tidy

# Verify dependencies
go mod verify

# Copy dependencies to vendor/
go mod vendor

# Edit go.mod
go mod edit -require github.com/pkg/errors@v0.9.1
go mod edit -replace old=new
go mod edit -exclude github.com/pkg/errors@v0.8.0
go mod edit -retract v1.0.0

# Show module graph
go mod graph

# Why is dependency needed?
go mod why github.com/pkg/errors
```

## 🎨 Publishing a Module

### 1. Create Module

```go
// github.com/user/stringutil
package stringutil

// Reverse reverses a string
func Reverse(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}
```

### 2. Initialize Module

```bash
go mod init github.com/user/stringutil
```

### 3. Add Documentation

```go
// Package stringutil provides string utility functions.
//
// Example:
//
//     reversed := stringutil.Reverse("hello")
//     fmt.Println(reversed) // "olleh"
//
package stringutil
```

### 4. Tag Version

```bash
git tag v1.0.0
git push origin v1.0.0
```

### 5. Use in Other Projects

```bash
go get github.com/user/stringutil@v1.0.0
```

```go
import "github.com/user/stringutil"

func main() {
    fmt.Println(stringutil.Reverse("hello"))
}
```

## 💼 Complete Example: Multi-Package Module

```
myapp/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   ├── database/
│   │   └── db.go
│   └── config/
│       └── config.go
└── pkg/
    ├── models/
    │   └── user.go
    └── api/
        └── handler.go
```

```go
// go.mod
module github.com/user/myapp

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    gorm.io/gorm v1.25.5
)

// main.go
package main

import (
    "github.com/gin-gonic/gin"
    "github.com/user/myapp/internal/config"
    "github.com/user/myapp/internal/database"
    "github.com/user/myapp/pkg/api"
)

func main() {
    cfg := config.Load()
    db := database.Connect(cfg.DatabaseURL)
    
    router := gin.Default()
    api.RegisterRoutes(router, db)
    
    router.Run(":8080")
}

// pkg/models/user.go
package models

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

// pkg/api/handler.go
package api

import (
    "github.com/gin-gonic/gin"
    "github.com/user/myapp/pkg/models"
    "gorm.io/gorm"
)

func RegisterRoutes(router *gin.Engine, db *gorm.DB) {
    router.GET("/users/:id", getUser(db))
    router.POST("/users", createUser(db))
}

func getUser(db *gorm.DB) gin.HandlerFunc {
    return func(c *gin.Context) {
        var user models.User
        if err := db.First(&user, c.Param("id")).Error; err != nil {
            c.JSON(404, gin.H{"error": "not found"})
            return
        }
        c.JSON(200, user)
    }
}

// internal/database/db.go
package database

import (
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
)

func Connect(dsn string) *gorm.DB {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        panic(err)
    }
    return db
}

// internal/config/config.go
package config

import "os"

type Config struct {
    DatabaseURL string
    Port        string
}

func Load() *Config {
    return &Config{
        DatabaseURL: getEnv("DATABASE_URL", "postgres://localhost/myapp"),
        Port:        getEnv("PORT", "8080"),
    }
}

func getEnv(key, fallback string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return fallback
}
```

## 🎯 Exercises

### Exercise 1: Create a Module
Create and publish a simple utility module.

### Exercise 2: Multi-Module Workspace
Set up a workspace with multiple modules.

### Exercise 3: Dependency Management
Practice updating, replacing, and vendoring dependencies.

## 🔑 Key Takeaways

- Go modules are the standard for dependency management
- Use semantic versioning (v1.2.3)
- `go mod tidy` keeps dependencies clean
- `replace` directive for local development
- Major versions (v2+) require path suffix
- Vendor directory for offline builds
- Minimal Version Selection ensures reproducible builds

## 📖 Next Steps

Continue to [Chapter 14: I/O and Files](14-io-files.md) to learn about file operations and I/O in Go.

