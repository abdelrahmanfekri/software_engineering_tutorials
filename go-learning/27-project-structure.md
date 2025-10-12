# Chapter 27: Project Structure

## 📁 Standard Go Project Layout

```
myproject/
├── cmd/                    # Main applications
│   ├── server/
│   │   └── main.go
│   └── cli/
│       └── main.go
├── internal/               # Private application code
│   ├── api/
│   │   └── handler.go
│   ├── service/
│   │   └── user.go
│   ├── repository/
│   │   └── user_repo.go
│   └── model/
│       └── user.go
├── pkg/                    # Public library code
│   └── utils/
│       └── string.go
├── api/                    # API definitions (OpenAPI, gRPC)
│   └── openapi.yaml
├── web/                    # Web assets
│   ├── static/
│   └── templates/
├── configs/                # Configuration files
│   └── config.yaml
├── scripts/                # Build and dev scripts
│   ├── build.sh
│   └── test.sh
├── deployments/            # Docker, k8s configs
│   ├── Dockerfile
│   └── k8s/
├── test/                   # Additional test data
├── docs/                   # Documentation
├── .gitignore
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

## 🎯 Clean Architecture

```
internal/
├── domain/                 # Business entities
│   ├── user.go
│   └── order.go
├── usecase/               # Business logic
│   ├── user_usecase.go
│   └── order_usecase.go
├── repository/            # Data access
│   ├── user_repository.go
│   └── order_repository.go
└── delivery/              # Presentation layer
    ├── http/
    │   └── handler.go
    └── grpc/
        └── server.go
```

## 📦 Example Structure

```go
// internal/domain/user.go
package domain

type User struct {
    ID    int
    Name  string
    Email string
}

// internal/repository/user_repository.go
package repository

type UserRepository interface {
    Create(user *domain.User) error
    GetByID(id int) (*domain.User, error)
}

// internal/service/user_service.go
package service

type UserService struct {
    repo repository.UserRepository
}

func NewUserService(repo repository.UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) CreateUser(name, email string) (*domain.User, error) {
    user := &domain.User{Name: name, Email: email}
    if err := s.repo.Create(user); err != nil {
        return nil, err
    }
    return user, nil
}

// cmd/server/main.go
package main

func main() {
    db := setupDatabase()
    repo := repository.NewUserRepository(db)
    service := service.NewUserService(repo)
    handler := handler.NewUserHandler(service)
    
    router := gin.Default()
    handler.RegisterRoutes(router)
    
    router.Run(":8080")
}
```

## 🔑 Key Takeaways

- Follow standard layout
- Keep internal code in `internal/`
- Put reusable code in `pkg/`
- Separate concerns (handler, service, repository)
- Use dependency injection

## 📖 Next Steps

Continue to [Chapter 28: Deployment](28-deployment.md).

