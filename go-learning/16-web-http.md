# Chapter 16: HTTP and Web Servers

## 🌐 Basic HTTP Server

### Hello World Server

```go
package main

import (
    "fmt"
    "net/http"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, World!")
    })
    
    fmt.Println("Server starting on :8080")
    http.ListenAndServe(":8080", nil)
}
```

### Multiple Routes

```go
func main() {
    http.HandleFunc("/", homeHandler)
    http.HandleFunc("/about", aboutHandler)
    http.HandleFunc("/users", usersHandler)
    
    http.ListenAndServe(":8080", nil)
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Home Page")
}

func aboutHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "About Page")
}

func usersHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Users Page")
}
```

## 🎯 Request and Response

### Reading Request Data

```go
func handler(w http.ResponseWriter, r *http.Request) {
    // Method
    method := r.Method  // GET, POST, PUT, DELETE, etc.
    
    // URL
    path := r.URL.Path
    query := r.URL.Query()
    
    // Headers
    userAgent := r.Header.Get("User-Agent")
    contentType := r.Header.Get("Content-Type")
    
    // Query parameters
    id := r.URL.Query().Get("id")
    page := r.URL.Query().Get("page")
    
    // Form data (POST)
    r.ParseForm()
    name := r.FormValue("name")
    email := r.FormValue("email")
    
    // JSON body
    var data map[string]interface{}
    json.NewDecoder(r.Body).Decode(&data)
    defer r.Body.Close()
    
    // Raw body
    body, err := io.ReadAll(r.Body)
}
```

### Writing Response

```go
func handler(w http.ResponseWriter, r *http.Request) {
    // Set status code
    w.WriteHeader(http.StatusOK)  // 200
    
    // Set headers
    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("X-Custom-Header", "value")
    
    // Write response
    fmt.Fprintf(w, "Response text")
    
    // JSON response
    response := map[string]interface{}{
        "status":  "success",
        "message": "Data processed",
    }
    json.NewEncoder(w).Encode(response)
}
```

## 📝 REST API Example

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "sync"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type UserStore struct {
    mu    sync.RWMutex
    users map[int]User
    nextID int
}

func NewUserStore() *UserStore {
    return &UserStore{
        users:  make(map[int]User),
        nextID: 1,
    }
}

func (s *UserStore) Create(name, email string) User {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    user := User{
        ID:    s.nextID,
        Name:  name,
        Email: email,
    }
    s.users[s.nextID] = user
    s.nextID++
    
    return user
}

func (s *UserStore) GetAll() []User {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    users := make([]User, 0, len(s.users))
    for _, user := range s.users {
        users = append(users, user)
    }
    return users
}

func (s *UserStore) Get(id int) (User, bool) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    user, ok := s.users[id]
    return user, ok
}

func (s *UserStore) Update(id int, name, email string) bool {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, ok := s.users[id]; !ok {
        return false
    }
    
    s.users[id] = User{
        ID:    id,
        Name:  name,
        Email: email,
    }
    return true
}

func (s *UserStore) Delete(id int) bool {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, ok := s.users[id]; !ok {
        return false
    }
    
    delete(s.users, id)
    return true
}

func main() {
    store := NewUserStore()
    
    http.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "GET":
            getUsersHandler(w, r, store)
        case "POST":
            createUserHandler(w, r, store)
        default:
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        }
    })
    
    http.HandleFunc("/users/", func(w http.ResponseWriter, r *http.Request) {
        id, err := extractID(r.URL.Path)
        if err != nil {
            http.Error(w, "Invalid ID", http.StatusBadRequest)
            return
        }
        
        switch r.Method {
        case "GET":
            getUserHandler(w, r, store, id)
        case "PUT":
            updateUserHandler(w, r, store, id)
        case "DELETE":
            deleteUserHandler(w, r, store, id)
        default:
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        }
    })
    
    fmt.Println("Server starting on :8080")
    http.ListenAndServe(":8080", nil)
}

func getUsersHandler(w http.ResponseWriter, r *http.Request, store *UserStore) {
    users := store.GetAll()
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(users)
}

func createUserHandler(w http.ResponseWriter, r *http.Request, store *UserStore) {
    var input struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    user := store.Create(input.Name, input.Email)
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}

func getUserHandler(w http.ResponseWriter, r *http.Request, store *UserStore, id int) {
    user, ok := store.Get(id)
    if !ok {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func updateUserHandler(w http.ResponseWriter, r *http.Request, store *UserStore, id int) {
    var input struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    if !store.Update(id, input.Name, input.Email) {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    w.WriteHeader(http.StatusNoContent)
}

func deleteUserHandler(w http.ResponseWriter, r *http.Request, store *UserStore, id int) {
    if !store.Delete(id) {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }
    
    w.WriteHeader(http.StatusNoContent)
}

func extractID(path string) (int, error) {
    parts := strings.Split(path, "/")
    if len(parts) < 3 {
        return 0, fmt.Errorf("invalid path")
    }
    return strconv.Atoi(parts[2])
}
```

## 🔧 Middleware

### Basic Middleware

```go
func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        next(w, r)
        
        duration := time.Since(start)
        fmt.Printf("%s %s %v\n", r.Method, r.URL.Path, duration)
    }
}

// Usage
http.HandleFunc("/", loggingMiddleware(homeHandler))
```

### Multiple Middleware

```go
type Middleware func(http.HandlerFunc) http.HandlerFunc

func Chain(f http.HandlerFunc, middlewares ...Middleware) http.HandlerFunc {
    for i := len(middlewares) - 1; i >= 0; i-- {
        f = middlewares[i](f)
    }
    return f
}

func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next(w, r)
    }
}

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next(w, r)
    }
}

// Usage
handler := Chain(
    usersHandler,
    loggingMiddleware,
    authMiddleware,
    corsMiddleware,
)
http.HandleFunc("/users", handler)
```

## 🎨 Custom ServeMux

```go
type Router struct {
    mux *http.ServeMux
}

func NewRouter() *Router {
    return &Router{
        mux: http.NewServeMux(),
    }
}

func (r *Router) GET(pattern string, handler http.HandlerFunc) {
    r.mux.HandleFunc(pattern, func(w http.ResponseWriter, req *http.Request) {
        if req.Method != "GET" {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        handler(w, req)
    })
}

func (r *Router) POST(pattern string, handler http.HandlerFunc) {
    r.mux.HandleFunc(pattern, func(w http.ResponseWriter, req *http.Request) {
        if req.Method != "POST" {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        handler(w, req)
    })
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    r.mux.ServeHTTP(w, req)
}

// Usage
func main() {
    router := NewRouter()
    
    router.GET("/users", getUsers)
    router.POST("/users", createUser)
    
    http.ListenAndServe(":8080", router)
}
```

## 📁 Static Files

```go
// Serve static files
func main() {
    // Serve files from ./static directory
    fs := http.FileServer(http.Dir("./static"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))
    
    http.HandleFunc("/", homeHandler)
    
    http.ListenAndServe(":8080", nil)
}
```

## 📤 File Upload

```go
func uploadHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != "POST" {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    // Parse multipart form (max 10MB)
    r.ParseMultipartForm(10 << 20)
    
    // Get file
    file, header, err := r.FormFile("file")
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    defer file.Close()
    
    // Create file
    dst, err := os.Create("./uploads/" + header.Filename)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer dst.Close()
    
    // Copy file
    if _, err := io.Copy(dst, file); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    fmt.Fprintf(w, "File uploaded successfully: %s", header.Filename)
}
```

## 🔒 HTTPS Server

```go
func main() {
    http.HandleFunc("/", handler)
    
    // HTTPS server
    http.ListenAndServeTLS(":443", "server.crt", "server.key", nil)
}

// Generate self-signed certificate for testing:
// openssl req -newkey rsa:2048 -nodes -keyout server.key -x509 -days 365 -out server.crt
```

## 🎯 Graceful Shutdown

```go
func main() {
    srv := &http.Server{
        Addr:         ":8080",
        Handler:      router,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    // Start server in goroutine
    go func() {
        fmt.Println("Server starting on :8080")
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatal(err)
        }
    }()
    
    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, os.Interrupt)
    <-quit
    
    fmt.Println("Shutting down server...")
    
    // Graceful shutdown with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        log.Fatal("Server forced to shutdown:", err)
    }
    
    fmt.Println("Server exited")
}
```

## 💼 Complete Example: Blog API

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type Post struct {
    ID        int       `json:"id"`
    Title     string    `json:"title"`
    Content   string    `json:"content"`
    Author    string    `json:"author"`
    CreatedAt time.Time `json:"created_at"`
}

type BlogAPI struct {
    mu    sync.RWMutex
    posts map[int]Post
    nextID int
}

func NewBlogAPI() *BlogAPI {
    return &BlogAPI{
        posts:  make(map[int]Post),
        nextID: 1,
    }
}

func (api *BlogAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    switch {
    case r.URL.Path == "/posts" && r.Method == "GET":
        api.listPosts(w, r)
    case r.URL.Path == "/posts" && r.Method == "POST":
        api.createPost(w, r)
    default:
        http.NotFound(w, r)
    }
}

func (api *BlogAPI) listPosts(w http.ResponseWriter, r *http.Request) {
    api.mu.RLock()
    defer api.mu.RUnlock()
    
    posts := make([]Post, 0, len(api.posts))
    for _, post := range api.posts {
        posts = append(posts, post)
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(posts)
}

func (api *BlogAPI) createPost(w http.ResponseWriter, r *http.Request) {
    var input struct {
        Title   string `json:"title"`
        Content string `json:"content"`
        Author  string `json:"author"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    api.mu.Lock()
    defer api.mu.Unlock()
    
    post := Post{
        ID:        api.nextID,
        Title:     input.Title,
        Content:   input.Content,
        Author:    input.Author,
        CreatedAt: time.Now(),
    }
    
    api.posts[api.nextID] = post
    api.nextID++
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(post)
}

func main() {
    api := NewBlogAPI()
    
    fmt.Println("Blog API starting on :8080")
    http.ListenAndServe(":8080", api)
}
```

## 🔑 Key Takeaways

- `http.HandleFunc` for simple routing
- `http.Handler` interface for custom handlers
- Middleware for cross-cutting concerns
- Use proper HTTP status codes
- Always set `Content-Type` header
- Graceful shutdown for production
- `http.ServeMux` or third-party routers for complex routing

## 📖 Next Steps

Continue to [Chapter 17: Web Frameworks](17-rest-apis.md) to learn about popular Go web frameworks like Gin and Echo.

