# Chapter 18: Web Frameworks Comparison

## 🎯 Framework Overview

| Framework | Performance | Learning Curve | Community | Use Case |
|-----------|-------------|----------------|-----------|----------|
| **net/http** | High | Easy | Large | Simple apps, learning |
| **Gin** | Very High | Easy | Very Large | REST APIs, microservices |
| **Echo** | Very High | Easy | Large | REST APIs, microservices |
| **Fiber** | Highest | Medium | Growing | High-performance APIs |
| **Chi** | High | Easy | Medium | RESTful services |

## 🚀 Echo Framework

Similar to Gin but with slight differences.

```bash
go get github.com/labstack/echo/v4
```

### Basic Echo Server

```go
package main

import (
    "net/http"
    "github.com/labstack/echo/v4"
    "github.com/labstack/echo/v4/middleware"
)

func main() {
    e := echo.New()
    
    // Middleware
    e.Use(middleware.Logger())
    e.Use(middleware.Recover())
    
    // Routes
    e.GET("/", hello)
    e.GET("/users/:id", getUser)
    e.POST("/users", createUser)
    
    e.Logger.Fatal(e.Start(":8080"))
}

func hello(c echo.Context) error {
    return c.String(http.StatusOK, "Hello, World!")
}

func getUser(c echo.Context) error {
    id := c.Param("id")
    return c.JSON(http.StatusOK, map[string]string{
        "id": id,
    })
}

type User struct {
    Name  string `json:"name" validate:"required"`
    Email string `json:"email" validate:"required,email"`
}

func createUser(c echo.Context) error {
    user := new(User)
    
    if err := c.Bind(user); err != nil {
        return err
    }
    
    if err := c.Validate(user); err != nil {
        return err
    }
    
    return c.JSON(http.StatusCreated, user)
}
```

## ⚡ Fiber Framework

Express.js-inspired framework with excellent performance.

```bash
go get github.com/gofiber/fiber/v2
```

### Basic Fiber Server

```go
package main

import (
    "github.com/gofiber/fiber/v2"
    "github.com/gofiber/fiber/v2/middleware/logger"
)

func main() {
    app := fiber.New()
    
    // Middleware
    app.Use(logger.New())
    
    // Routes
    app.Get("/", func(c *fiber.Ctx) error {
        return c.SendString("Hello, World!")
    })
    
    app.Get("/users/:id", func(c *fiber.Ctx) error {
        id := c.Params("id")
        return c.JSON(fiber.Map{
            "id": id,
        })
    })
    
    app.Post("/users", func(c *fiber.Ctx) error {
        user := new(User)
        
        if err := c.BodyParser(user); err != nil {
            return err
        }
        
        return c.Status(fiber.StatusCreated).JSON(user)
    })
    
    app.Listen(":8080")
}

type User struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}
```

## 🎨 Chi Router

Lightweight, idiomatic router built on standard net/http.

```bash
go get github.com/go-chi/chi/v5
```

### Basic Chi Server

```go
package main

import (
    "encoding/json"
    "net/http"
    
    "github.com/go-chi/chi/v5"
    "github.com/go-chi/chi/v5/middleware"
)

func main() {
    r := chi.NewRouter()
    
    // Middleware
    r.Use(middleware.Logger)
    r.Use(middleware.Recoverer)
    
    // Routes
    r.Get("/", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Hello, World!"))
    })
    
    r.Route("/users", func(r chi.Router) {
        r.Get("/", listUsers)
        r.Post("/", createUser)
        
        r.Route("/{userID}", func(r chi.Router) {
            r.Get("/", getUser)
            r.Put("/", updateUser)
            r.Delete("/", deleteUser)
        })
    })
    
    http.ListenAndServe(":8080", r)
}

func listUsers(w http.ResponseWriter, r *http.Request) {
    json.NewEncoder(w).Encode([]string{"user1", "user2"})
}

func getUser(w http.ResponseWriter, r *http.Request) {
    userID := chi.URLParam(r, "userID")
    json.NewEncoder(w).Encode(map[string]string{
        "id": userID,
    })
}
```

## 📊 Feature Comparison

### Gin Features

```go
// Binding with validation
type CreateUserRequest struct {
    Name  string `json:"name" binding:"required,min=3"`
    Email string `json:"email" binding:"required,email"`
    Age   int    `json:"age" binding:"gte=0,lte=150"`
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
}

// Route grouping
v1 := r.Group("/api/v1")
v1.Use(authMiddleware())
{
    v1.GET("/users", getUsers)
    v1.POST("/users", createUser)
}

// File upload
file, _ := c.FormFile("file")
c.SaveUploadedFile(file, dst)
```

### Echo Features

```go
// Custom validator
type CustomValidator struct {
    validator *validator.Validate
}

func (cv *CustomValidator) Validate(i interface{}) error {
    return cv.validator.Struct(i)
}

e.Validator = &CustomValidator{validator: validator.New()}

// Groups
api := e.Group("/api")
api.Use(middleware.JWT([]byte("secret")))

// Context values
c.Set("user", user)
user := c.Get("user")

// Response
c.JSON(200, data)
c.String(200, "text")
c.HTML(200, "<html>")
```

### Fiber Features

```go
// Fast routing
app.Get("/users/:id", getUser)
app.Post("/users", createUser)

// Static files
app.Static("/", "./public")

// Templates
app.Settings.Views = html.New("./views", ".html")
c.Render("index", fiber.Map{"Title": "Hello"})

// WebSocket
app.Get("/ws", websocket.New(func(c *websocket.Conn) {
    // WebSocket connection
}))
```

## 💼 Complete Example: E-commerce API (Gin)

```go
package main

import (
    "net/http"
    "time"
    
    "github.com/gin-gonic/gin"
)

type Product struct {
    ID          int       `json:"id"`
    Name        string    `json:"name" binding:"required"`
    Description string    `json:"description"`
    Price       float64   `json:"price" binding:"required,gt=0"`
    Stock       int       `json:"stock" binding:"gte=0"`
    CreatedAt   time.Time `json:"created_at"`
}

type Cart struct {
    UserID  int          `json:"user_id"`
    Items   []CartItem   `json:"items"`
    Total   float64      `json:"total"`
}

type CartItem struct {
    ProductID int     `json:"product_id"`
    Quantity  int     `json:"quantity"`
    Price     float64 `json:"price"`
}

var products = make(map[int]Product)
var carts = make(map[int]*Cart)
var nextProductID = 1

func main() {
    r := gin.Default()
    
    // CORS
    r.Use(corsMiddleware())
    
    // Public routes
    r.GET("/products", listProducts)
    r.GET("/products/:id", getProduct)
    
    // Protected routes
    authorized := r.Group("/")
    authorized.Use(authMiddleware())
    {
        // Cart operations
        authorized.GET("/cart", getCart)
        authorized.POST("/cart/items", addToCart)
        authorized.DELETE("/cart/items/:productId", removeFromCart)
        authorized.POST("/checkout", checkout)
        
        // Admin routes
        admin := authorized.Group("/admin")
        admin.Use(adminMiddleware())
        {
            admin.POST("/products", createProduct)
            admin.PUT("/products/:id", updateProduct)
            admin.DELETE("/products/:id", deleteProduct)
        }
    }
    
    r.Run(":8080")
}

func listProducts(c *gin.Context) {
    productList := make([]Product, 0, len(products))
    for _, p := range products {
        productList = append(productList, p)
    }
    
    c.JSON(http.StatusOK, productList)
}

func getProduct(c *gin.Context) {
    id := c.Param("id")
    // Get product...
}

func createProduct(c *gin.Context) {
    var product Product
    
    if err := c.ShouldBindJSON(&product); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    product.ID = nextProductID
    product.CreatedAt = time.Now()
    products[nextProductID] = product
    nextProductID++
    
    c.JSON(http.StatusCreated, product)
}

func getCart(c *gin.Context) {
    userID := c.GetInt("user_id")
    
    cart, exists := carts[userID]
    if !exists {
        cart = &Cart{
            UserID: userID,
            Items:  []CartItem{},
            Total:  0,
        }
        carts[userID] = cart
    }
    
    c.JSON(http.StatusOK, cart)
}

func addToCart(c *gin.Context) {
    userID := c.GetInt("user_id")
    
    var req struct {
        ProductID int `json:"product_id" binding:"required"`
        Quantity  int `json:"quantity" binding:"required,gt=0"`
    }
    
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    product, exists := products[req.ProductID]
    if !exists {
        c.JSON(http.StatusNotFound, gin.H{"error": "Product not found"})
        return
    }
    
    if product.Stock < req.Quantity {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Insufficient stock"})
        return
    }
    
    cart, exists := carts[userID]
    if !exists {
        cart = &Cart{
            UserID: userID,
            Items:  []CartItem{},
        }
        carts[userID] = cart
    }
    
    item := CartItem{
        ProductID: req.ProductID,
        Quantity:  req.Quantity,
        Price:     product.Price,
    }
    
    cart.Items = append(cart.Items, item)
    cart.Total += product.Price * float64(req.Quantity)
    
    c.JSON(http.StatusOK, cart)
}

func checkout(c *gin.Context) {
    userID := c.GetInt("user_id")
    
    cart, exists := carts[userID]
    if !exists || len(cart.Items) == 0 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Cart is empty"})
        return
    }
    
    // Process payment, create order, etc.
    
    // Clear cart
    delete(carts, userID)
    
    c.JSON(http.StatusOK, gin.H{
        "message": "Order placed successfully",
        "total":   cart.Total,
    })
}

func corsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
        c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }
        
        c.Next()
    }
}

func authMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        
        if token == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "No authorization header"})
            c.Abort()
            return
        }
        
        // Validate token and get user ID
        userID := 1  // From token
        c.Set("user_id", userID)
        
        c.Next()
    }
}

func adminMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        userID := c.GetInt("user_id")
        
        // Check if user is admin
        if !isAdmin(userID) {
            c.JSON(http.StatusForbidden, gin.H{"error": "Admin access required"})
            c.Abort()
            return
        }
        
        c.Next()
    }
}

func isAdmin(userID int) bool {
    // Check admin status
    return userID == 1
}
```

## 🔑 Key Takeaways

- **Gin**: Best balance of performance, features, and community
- **Echo**: Similar to Gin, slightly different API
- **Fiber**: Fastest, Express.js-like API
- **Chi**: Lightweight, idiomatic, built on net/http
- **net/http**: For learning or simple applications

Choose based on:
- Performance requirements
- Team familiarity
- Feature requirements
- Community/ecosystem

## 📖 Next Steps

Continue to [Chapter 19: Databases](19-databases.md) to learn about database integration in Go.

