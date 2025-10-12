# Chapter 17: Building REST APIs

## 🚀 Using Gin Framework

Gin is a high-performance web framework for Go.

### Installation

```bash
go get -u github.com/gin-gonic/gin
```

### Basic Gin Server

```go
package main

import "github.com/gin-gonic/gin"

func main() {
    r := gin.Default()
    
    r.GET("/ping", func(c *gin.Context) {
        c.JSON(200, gin.H{
            "message": "pong",
        })
    })
    
    r.Run(":8080")
}
```

### Complete CRUD API with Gin

```go
package main

import (
    "net/http"
    "strconv"
    "github.com/gin-gonic/gin"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name" binding:"required"`
    Email string `json:"email" binding:"required,email"`
    Age   int    `json:"age" binding:"gte=0,lte=150"`
}

var users = []User{
    {ID: 1, Name: "Alice", Email: "alice@example.com", Age: 25},
    {ID: 2, Name: "Bob", Email: "bob@example.com", Age: 30},
}

func main() {
    r := gin.Default()
    
    // Routes
    api := r.Group("/api/v1")
    {
        api.GET("/users", getUsers)
        api.GET("/users/:id", getUser)
        api.POST("/users", createUser)
        api.PUT("/users/:id", updateUser)
        api.DELETE("/users/:id", deleteUser)
    }
    
    r.Run(":8080")
}

func getUsers(c *gin.Context) {
    c.JSON(http.StatusOK, users)
}

func getUser(c *gin.Context) {
    id, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID"})
        return
    }
    
    for _, user := range users {
        if user.ID == id {
            c.JSON(http.StatusOK, user)
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
}

func createUser(c *gin.Context) {
    var newUser User
    
    if err := c.ShouldBindJSON(&newUser); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    newUser.ID = len(users) + 1
    users = append(users, newUser)
    
    c.JSON(http.StatusCreated, newUser)
}

func updateUser(c *gin.Context) {
    id, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID"})
        return
    }
    
    var updatedUser User
    if err := c.ShouldBindJSON(&updatedUser); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    for i, user := range users {
        if user.ID == id {
            updatedUser.ID = id
            users[i] = updatedUser
            c.JSON(http.StatusOK, updatedUser)
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
}

func deleteUser(c *gin.Context) {
    id, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID"})
        return
    }
    
    for i, user := range users {
        if user.ID == id {
            users = append(users[:i], users[i+1:]...)
            c.Status(http.StatusNoContent)
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
}
```

### Middleware in Gin

```go
// Logger middleware
func Logger() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        
        c.Next()
        
        duration := time.Since(start)
        fmt.Printf("[%s] %s %s %v\n",
            c.Request.Method,
            c.Request.URL.Path,
            c.Writer.Status(),
            duration)
    }
}

// Auth middleware
func AuthRequired() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        
        if token == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "No authorization header"})
            c.Abort()
            return
        }
        
        // Validate token here
        if !isValidToken(token) {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
            c.Abort()
            return
        }
        
        c.Next()
    }
}

// Usage
func main() {
    r := gin.Default()
    
    // Global middleware
    r.Use(Logger())
    
    // Public routes
    r.POST("/login", login)
    
    // Protected routes
    protected := r.Group("/api")
    protected.Use(AuthRequired())
    {
        protected.GET("/users", getUsers)
        protected.POST("/users", createUser)
    }
    
    r.Run(":8080")
}
```

## 📊 Query Parameters and Validation

```go
func searchUsers(c *gin.Context) {
    // Query parameters
    name := c.Query("name")
    ageStr := c.DefaultQuery("age", "0")
    
    age, err := strconv.Atoi(ageStr)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid age"})
        return
    }
    
    // Search logic
    results := filterUsers(name, age)
    
    c.JSON(http.StatusOK, results)
}

// Binding query params
type SearchParams struct {
    Name     string `form:"name"`
    MinAge   int    `form:"min_age"`
    MaxAge   int    `form:"max_age"`
    Page     int    `form:"page" binding:"required,gte=1"`
    PageSize int    `form:"page_size" binding:"required,gte=1,lte=100"`
}

func searchUsersWithBinding(c *gin.Context) {
    var params SearchParams
    
    if err := c.ShouldBindQuery(&params); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    // Use params
    results := performSearch(params)
    
    c.JSON(http.StatusOK, gin.H{
        "data": results,
        "page": params.Page,
        "page_size": params.PageSize,
    })
}
```

## 🎯 Error Handling

```go
type AppError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

func errorResponse(c *gin.Context, status int, code, message string) {
    c.JSON(status, AppError{
        Code:    code,
        Message: message,
    })
}

func getUserSafe(c *gin.Context) {
    id, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        errorResponse(c, http.StatusBadRequest, "INVALID_ID", "ID must be a number")
        return
    }
    
    user, err := findUser(id)
    if err != nil {
        if err == ErrNotFound {
            errorResponse(c, http.StatusNotFound, "NOT_FOUND", "User not found")
            return
        }
        errorResponse(c, http.StatusInternalServerError, "INTERNAL_ERROR", "Something went wrong")
        return
    }
    
    c.JSON(http.StatusOK, user)
}
```

## 🔒 Authentication with JWT

```bash
go get -u github.com/golang-jwt/jwt/v5
```

```go
package main

import (
    "net/http"
    "time"
    
    "github.com/gin-gonic/gin"
    "github.com/golang-jwt/jwt/v5"
)

var jwtSecret = []byte("your-secret-key")

type Claims struct {
    UserID int    `json:"user_id"`
    Email  string `json:"email"`
    jwt.RegisteredClaims
}

func generateToken(userID int, email string) (string, error) {
    claims := Claims{
        UserID: userID,
        Email:  email,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(jwtSecret)
}

func validateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return jwtSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }
    
    return nil, jwt.ErrSignatureInvalid
}

func login(c *gin.Context) {
    var credentials struct {
        Email    string `json:"email" binding:"required"`
        Password string `json:"password" binding:"required"`
    }
    
    if err := c.ShouldBindJSON(&credentials); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    // Validate credentials (check database)
    user, err := authenticateUser(credentials.Email, credentials.Password)
    if err != nil {
        c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
        return
    }
    
    // Generate token
    token, err := generateToken(user.ID, user.Email)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not generate token"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "token": token,
        "user":  user,
    })
}

func JWTAuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        tokenString := c.GetHeader("Authorization")
        
        if tokenString == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "No authorization header"})
            c.Abort()
            return
        }
        
        // Remove "Bearer " prefix
        if len(tokenString) > 7 && tokenString[:7] == "Bearer " {
            tokenString = tokenString[7:]
        }
        
        claims, err := validateToken(tokenString)
        if err != nil {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
            c.Abort()
            return
        }
        
        // Add claims to context
        c.Set("user_id", claims.UserID)
        c.Set("email", claims.Email)
        
        c.Next()
    }
}

func main() {
    r := gin.Default()
    
    // Public routes
    r.POST("/login", login)
    r.POST("/register", register)
    
    // Protected routes
    api := r.Group("/api")
    api.Use(JWTAuthMiddleware())
    {
        api.GET("/profile", getProfile)
        api.PUT("/profile", updateProfile)
    }
    
    r.Run(":8080")
}

func getProfile(c *gin.Context) {
    userID := c.GetInt("user_id")
    
    user, err := getUserByID(userID)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
        return
    }
    
    c.JSON(http.StatusOK, user)
}
```

## 📁 File Upload with Gin

```go
func uploadFile(c *gin.Context) {
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
        return
    }
    
    // Validate file type
    if !isAllowedFileType(file.Header.Get("Content-Type")) {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type"})
        return
    }
    
    // Validate file size (max 10MB)
    if file.Size > 10<<20 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "File too large"})
        return
    }
    
    // Generate unique filename
    filename := generateFilename(file.Filename)
    filepath := "./uploads/" + filename
    
    // Save file
    if err := c.SaveUploadedFile(file, filepath); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "message":  "File uploaded successfully",
        "filename": filename,
        "url":      "/uploads/" + filename,
    })
}

// Multiple file upload
func uploadMultiple(c *gin.Context) {
    form, err := c.MultipartForm()
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "No files uploaded"})
        return
    }
    
    files := form.File["files"]
    
    var uploaded []string
    for _, file := range files {
        filename := generateFilename(file.Filename)
        filepath := "./uploads/" + filename
        
        if err := c.SaveUploadedFile(file, filepath); err != nil {
            continue
        }
        
        uploaded = append(uploaded, filename)
    }
    
    c.JSON(http.StatusOK, gin.H{
        "message": "Files uploaded",
        "files":   uploaded,
    })
}
```

## 🎨 Response Formats

```go
// JSON
func jsonResponse(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "message": "Success",
        "data":    data,
    })
}

// XML
func xmlResponse(c *gin.Context) {
    c.XML(http.StatusOK, data)
}

// YAML
func yamlResponse(c *gin.Context) {
    c.YAML(http.StatusOK, data)
}

// HTML
func htmlResponse(c *gin.Context) {
    c.HTML(http.StatusOK, "index.html", gin.H{
        "title": "Home",
    })
}

// Download file
func downloadFile(c *gin.Context) {
    c.File("./files/document.pdf")
}

// Stream data
func streamData(c *gin.Context) {
    c.Stream(func(w io.Writer) bool {
        // Stream data
        return true
    })
}
```

## 💼 Complete Example: Todo API

```go
package main

import (
    "net/http"
    "time"
    
    "github.com/gin-gonic/gin"
)

type Todo struct {
    ID        int       `json:"id"`
    Title     string    `json:"title" binding:"required"`
    Completed bool      `json:"completed"`
    CreatedAt time.Time `json:"created_at"`
}

var todos = []Todo{}
var nextID = 1

func main() {
    r := gin.Default()
    
    v1 := r.Group("/api/v1")
    {
        v1.GET("/todos", getTodos)
        v1.GET("/todos/:id", getTodo)
        v1.POST("/todos", createTodo)
        v1.PUT("/todos/:id", updateTodo)
        v1.DELETE("/todos/:id", deleteTodo)
        v1.PATCH("/todos/:id/complete", completeTodo)
    }
    
    r.Run(":8080")
}

func getTodos(c *gin.Context) {
    c.JSON(http.StatusOK, todos)
}

func getTodo(c *gin.Context) {
    id := c.Param("id")
    
    for _, todo := range todos {
        if strconv.Itoa(todo.ID) == id {
            c.JSON(http.StatusOK, todo)
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "Todo not found"})
}

func createTodo(c *gin.Context) {
    var input struct {
        Title string `json:"title" binding:"required"`
    }
    
    if err := c.ShouldBindJSON(&input); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    todo := Todo{
        ID:        nextID,
        Title:     input.Title,
        Completed: false,
        CreatedAt: time.Now(),
    }
    
    todos = append(todos, todo)
    nextID++
    
    c.JSON(http.StatusCreated, todo)
}

func updateTodo(c *gin.Context) {
    id := c.Param("id")
    
    var input struct {
        Title     string `json:"title"`
        Completed bool   `json:"completed"`
    }
    
    if err := c.ShouldBindJSON(&input); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    for i, todo := range todos {
        if strconv.Itoa(todo.ID) == id {
            todos[i].Title = input.Title
            todos[i].Completed = input.Completed
            c.JSON(http.StatusOK, todos[i])
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "Todo not found"})
}

func deleteTodo(c *gin.Context) {
    id := c.Param("id")
    
    for i, todo := range todos {
        if strconv.Itoa(todo.ID) == id {
            todos = append(todos[:i], todos[i+1:]...)
            c.Status(http.StatusNoContent)
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "Todo not found"})
}

func completeTodo(c *gin.Context) {
    id := c.Param("id")
    
    for i, todo := range todos {
        if strconv.Itoa(todo.ID) == id {
            todos[i].Completed = true
            c.JSON(http.StatusOK, todos[i])
            return
        }
    }
    
    c.JSON(http.StatusNotFound, gin.H{"error": "Todo not found"})
}
```

## 🔑 Key Takeaways

- Gin provides high-performance routing
- Use binding for validation
- Middleware for cross-cutting concerns
- JWT for stateless authentication
- Proper error handling and status codes
- Group routes for versioning
- Content negotiation for different response formats

## 📖 Next Steps

Continue to [Chapter 18: Web Frameworks Comparison](18-web-frameworks.md) to explore other popular Go web frameworks.

