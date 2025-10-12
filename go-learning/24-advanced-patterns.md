# Chapter 24: Advanced Design Patterns

## 🏗️ Builder Pattern

```go
type Server struct {
    host    string
    port    int
    timeout time.Duration
}

type ServerBuilder struct {
    server *Server
}

func NewServerBuilder() *ServerBuilder {
    return &ServerBuilder{
        server: &Server{
            host:    "localhost",
            port:    8080,
            timeout: 30 * time.Second,
        },
    }
}

func (b *ServerBuilder) Host(host string) *ServerBuilder {
    b.server.host = host
    return b
}

func (b *ServerBuilder) Port(port int) *ServerBuilder {
    b.server.port = port
    return b
}

func (b *ServerBuilder) Build() *Server {
    return b.server
}

// Usage
server := NewServerBuilder().
    Host("0.0.0.0").
    Port(3000).
    Build()
```

## 🏭 Factory Pattern

```go
type Database interface {
    Connect() error
    Query(string) ([]byte, error)
}

type PostgresDB struct{}
type MySQLDB struct{}

func (p *PostgresDB) Connect() error { return nil }
func (p *PostgresDB) Query(q string) ([]byte, error) { return nil, nil }

func NewDatabase(dbType string) Database {
    switch dbType {
    case "postgres":
        return &PostgresDB{}
    case "mysql":
        return &MySQLDB{}
    default:
        return nil
    }
}
```

## 🎯 Strategy Pattern

```go
type PaymentStrategy interface {
    Pay(amount float64) error
}

type CreditCard struct{}
type PayPal struct{}

func (c *CreditCard) Pay(amount float64) error {
    fmt.Printf("Paid %.2f with credit card\n", amount)
    return nil
}

type Checkout struct {
    strategy PaymentStrategy
}

func (c *Checkout) Process(amount float64) error {
    return c.strategy.Pay(amount)
}
```

## 🔑 Key Takeaways

- Use patterns to solve common problems
- Don't force patterns where they don't fit
- Prefer composition over inheritance
- Keep code simple and readable

## 📖 Next Steps

Continue to [Chapter 25: Best Practices](25-best-practices.md).

