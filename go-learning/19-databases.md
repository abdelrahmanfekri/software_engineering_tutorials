# Chapter 19: Database Integration

## 📊 database/sql Package

Go's standard library provides a generic interface for SQL databases.

### PostgreSQL Setup

```bash
go get github.com/lib/pq
```

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    
    _ "github.com/lib/pq"  // Driver import
)

func main() {
    connStr := "host=localhost port=5432 user=postgres password=secret dbname=mydb sslmode=disable"
    
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
    
    // Test connection
    if err := db.Ping(); err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Connected to database!")
}
```

### MySQL Setup

```bash
go get github.com/go-sql-driver/mysql
```

```go
import _ "github.com/go-sql-driver/mysql"

connStr := "user:password@tcp(localhost:3306)/dbname"
db, err := sql.Open("mysql", connStr)
```

## 🔍 CRUD Operations

### Create (Insert)

```go
func createUser(db *sql.DB, name, email string) (int, error) {
    query := `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id`
    
    var id int
    err := db.QueryRow(query, name, email).Scan(&id)
    if err != nil {
        return 0, err
    }
    
    return id, nil
}

// Bulk insert
func createUsers(db *sql.DB, users []User) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    stmt, err := tx.Prepare(`INSERT INTO users (name, email) VALUES ($1, $2)`)
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    for _, user := range users {
        _, err := stmt.Exec(user.Name, user.Email)
        if err != nil {
            return err
        }
    }
    
    return tx.Commit()
}
```

### Read (Select)

```go
type User struct {
    ID    int
    Name  string
    Email string
}

// Query single row
func getUser(db *sql.DB, id int) (*User, error) {
    query := `SELECT id, name, email FROM users WHERE id = $1`
    
    var user User
    err := db.QueryRow(query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user not found")
        }
        return nil, err
    }
    
    return &user, nil
}

// Query multiple rows
func getAllUsers(db *sql.DB) ([]User, error) {
    query := `SELECT id, name, email FROM users`
    
    rows, err := db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var user User
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    if err := rows.Err(); err != nil {
        return nil, err
    }
    
    return users, nil
}

// Query with WHERE clause
func findUsersByName(db *sql.DB, name string) ([]User, error) {
    query := `SELECT id, name, email FROM users WHERE name LIKE $1`
    
    rows, err := db.Query(query, "%"+name+"%")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var user User
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, nil
}
```

### Update

```go
func updateUser(db *sql.DB, id int, name, email string) error {
    query := `UPDATE users SET name = $1, email = $2 WHERE id = $3`
    
    result, err := db.Exec(query, name, email, id)
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("user not found")
    }
    
    return nil
}
```

### Delete

```go
func deleteUser(db *sql.DB, id int) error {
    query := `DELETE FROM users WHERE id = $1`
    
    result, err := db.Exec(query, id)
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("user not found")
    }
    
    return nil
}
```

## 🔄 Transactions

```go
func transferMoney(db *sql.DB, fromID, toID int, amount float64) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()  // Rollback if not committed
    
    // Deduct from sender
    _, err = tx.Exec(`UPDATE accounts SET balance = balance - $1 WHERE id = $2`, amount, fromID)
    if err != nil {
        return err
    }
    
    // Add to receiver
    _, err = tx.Exec(`UPDATE accounts SET balance = balance + $1 WHERE id = $2`, amount, toID)
    if err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}
```

## 🎯 GORM (Popular ORM)

```bash
go get -u gorm.io/gorm
go get -u gorm.io/driver/postgres
```

### Basic GORM Usage

```go
package main

import (
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
)

type User struct {
    ID    uint   `gorm:"primaryKey"`
    Name  string `gorm:"not null"`
    Email string `gorm:"uniqueIndex;not null"`
    Age   int
}

func main() {
    dsn := "host=localhost user=postgres password=secret dbname=mydb sslmode=disable"
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        panic(err)
    }
    
    // Auto-migrate tables
    db.AutoMigrate(&User{})
    
    // Create
    user := User{Name: "Alice", Email: "alice@example.com", Age: 25}
    db.Create(&user)
    
    // Read
    var foundUser User
    db.First(&foundUser, user.ID)
    db.First(&foundUser, "email = ?", "alice@example.com")
    
    // Update
    db.Model(&foundUser).Update("Age", 26)
    db.Model(&foundUser).Updates(User{Name: "Alice Smith", Age: 26})
    
    // Delete
    db.Delete(&foundUser, foundUser.ID)
}
```

### GORM CRUD Operations

```go
// Create
func createUser(db *gorm.DB, user *User) error {
    return db.Create(user).Error
}

// Read one
func getUser(db *gorm.DB, id uint) (*User, error) {
    var user User
    err := db.First(&user, id).Error
    if err != nil {
        return nil, err
    }
    return &user, nil
}

// Read all
func getAllUsers(db *gorm.DB) ([]User, error) {
    var users []User
    err := db.Find(&users).Error
    return users, err
}

// Update
func updateUser(db *gorm.DB, user *User) error {
    return db.Save(user).Error
}

// Delete
func deleteUser(db *gorm.DB, id uint) error {
    return db.Delete(&User{}, id).Error
}

// Query with conditions
func findUsersByAge(db *gorm.DB, minAge int) ([]User, error) {
    var users []User
    err := db.Where("age >= ?", minAge).Find(&users).Error
    return users, err
}

// Complex query
func searchUsers(db *gorm.DB, name string, minAge int) ([]User, error) {
    var users []User
    err := db.Where("name LIKE ? AND age >= ?", "%"+name+"%", minAge).
        Order("age DESC").
        Limit(10).
        Find(&users).Error
    return users, err
}
```

### Associations (Relations)

```go
// One-to-Many
type User struct {
    ID    uint
    Name  string
    Posts []Post  // User has many posts
}

type Post struct {
    ID      uint
    Title   string
    Content string
    UserID  uint
    User    User  // Post belongs to User
}

// Create with association
user := User{
    Name: "Alice",
    Posts: []Post{
        {Title: "Post 1", Content: "Content 1"},
        {Title: "Post 2", Content: "Content 2"},
    },
}
db.Create(&user)

// Preload associations
var user User
db.Preload("Posts").First(&user, 1)

// Many-to-Many
type User struct {
    ID    uint
    Name  string
    Roles []Role `gorm:"many2many:user_roles;"`
}

type Role struct {
    ID   uint
    Name string
}

// Add association
var user User
db.First(&user, 1)
db.Model(&user).Association("Roles").Append(&Role{Name: "Admin"})
```

## 🎯 Connection Pool

```go
func setupDB(connStr string) (*sql.DB, error) {
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        return nil, err
    }
    
    // Connection pool settings
    db.SetMaxOpenConns(25)  // Max open connections
    db.SetMaxIdleConns(5)   // Max idle connections
    db.SetConnMaxLifetime(5 * time.Minute)
    db.SetConnMaxIdleTime(2 * time.Minute)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return db, nil
}
```

## 💼 Complete Example: User Repository

```go
package main

import (
    "database/sql"
    "fmt"
    "time"
    
    _ "github.com/lib/pq"
)

type User struct {
    ID        int
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

type UserRepository struct {
    db *sql.DB
}

func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) Create(user *User) error {
    query := `
        INSERT INTO users (name, email, created_at, updated_at)
        VALUES ($1, $2, $3, $4)
        RETURNING id
    `
    
    now := time.Now()
    err := r.db.QueryRow(query, user.Name, user.Email, now, now).Scan(&user.ID)
    if err != nil {
        return err
    }
    
    user.CreatedAt = now
    user.UpdatedAt = now
    
    return nil
}

func (r *UserRepository) GetByID(id int) (*User, error) {
    query := `
        SELECT id, name, email, created_at, updated_at
        FROM users
        WHERE id = $1
    `
    
    var user User
    err := r.db.QueryRow(query, id).Scan(
        &user.ID,
        &user.Name,
        &user.Email,
        &user.CreatedAt,
        &user.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("user not found")
    }
    if err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (r *UserRepository) GetByEmail(email string) (*User, error) {
    query := `
        SELECT id, name, email, created_at, updated_at
        FROM users
        WHERE email = $1
    `
    
    var user User
    err := r.db.QueryRow(query, email).Scan(
        &user.ID,
        &user.Name,
        &user.Email,
        &user.CreatedAt,
        &user.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("user not found")
    }
    if err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (r *UserRepository) GetAll() ([]User, error) {
    query := `
        SELECT id, name, email, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
    `
    
    rows, err := r.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var user User
        err := rows.Scan(
            &user.ID,
            &user.Name,
            &user.Email,
            &user.CreatedAt,
            &user.UpdatedAt,
        )
        if err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, rows.Err()
}

func (r *UserRepository) Update(user *User) error {
    query := `
        UPDATE users
        SET name = $1, email = $2, updated_at = $3
        WHERE id = $4
    `
    
    user.UpdatedAt = time.Now()
    
    result, err := r.db.Exec(query, user.Name, user.Email, user.UpdatedAt, user.ID)
    if err != nil {
        return err
    }
    
    rows, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rows == 0 {
        return fmt.Errorf("user not found")
    }
    
    return nil
}

func (r *UserRepository) Delete(id int) error {
    query := `DELETE FROM users WHERE id = $1`
    
    result, err := r.db.Exec(query, id)
    if err != nil {
        return err
    }
    
    rows, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rows == 0 {
        return fmt.Errorf("user not found")
    }
    
    return nil
}

// Usage
func main() {
    db, err := sql.Open("postgres", "connStr")
    if err != nil {
        panic(err)
    }
    defer db.Close()
    
    repo := NewUserRepository(db)
    
    // Create user
    user := &User{
        Name:  "Alice",
        Email: "alice@example.com",
    }
    
    if err := repo.Create(user); err != nil {
        panic(err)
    }
    
    fmt.Printf("Created user with ID: %d\n", user.ID)
    
    // Get user
    foundUser, err := repo.GetByID(user.ID)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Found user: %+v\n", foundUser)
    
    // Update user
    foundUser.Name = "Alice Smith"
    if err := repo.Update(foundUser); err != nil {
        panic(err)
    }
    
    // Delete user
    if err := repo.Delete(foundUser.ID); err != nil {
        panic(err)
    }
}
```

## 🔑 Key Takeaways

- Use `database/sql` for raw SQL queries
- Use GORM for ORM features
- Always use prepared statements or parameterized queries
- Close connections and rows properly
- Use transactions for data consistency
- Configure connection pool for production
- Handle `sql.ErrNoRows` explicitly

## 📖 Next Steps

Continue to [Chapter 20: ORM and Advanced DB](20-orm-advanced-db.md) for advanced database topics like migrations and query builders.

