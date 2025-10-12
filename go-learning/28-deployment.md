# Chapter 28: Deployment

## 🐳 Docker

### Dockerfile

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server cmd/server/main.go

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/
COPY --from=builder /app/server .

EXPOSE 8080
CMD ["./server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://postgres:secret@db:5432/mydb
    depends_on:
      - db
      
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## ☸️ Kubernetes

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## 🔧 CI/CD (GitHub Actions)

```yaml
name: Go CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Test
      run: go test -v ./...
    
    - name: Build
      run: go build -v ./...

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Deploy commands
```

## 📦 Building

```bash
# Standard build
go build -o app cmd/server/main.go

# Cross-compilation
GOOS=linux GOARCH=amd64 go build -o app-linux cmd/server/main.go
GOOS=windows GOARCH=amd64 go build -o app.exe cmd/server/main.go
GOOS=darwin GOARCH=arm64 go build -o app-mac cmd/server/main.go

# Optimized build
go build -ldflags="-s -w" -o app cmd/server/main.go

# With version info
go build -ldflags="-X main.version=1.0.0" -o app cmd/server/main.go
```

## 🔑 Key Takeaways

- Use multi-stage Docker builds
- Keep images small
- Use environment variables for configuration
- Implement health checks
- Use CI/CD for automated deployments
- Cross-compile for different platforms

## 🎉 Conclusion

Congratulations! You've completed the comprehensive Go programming tutorial. You now have the knowledge to:
- Build robust applications with Go
- Work with concurrency and goroutines
- Create REST APIs and web services
- Integrate with databases
- Apply best practices and patterns
- Deploy production-ready applications

Keep practicing and building projects!

### Resources
- [Official Go Documentation](https://golang.org/doc/)
- [Go by Example](https://gobyexample.com/)
- [Effective Go](https://golang.org/doc/effective_go.html)
- [Go Blog](https://blog.golang.org/)

