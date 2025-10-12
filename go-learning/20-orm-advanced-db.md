# Chapter 20: ORM and Advanced Database Topics

## 🔄 Database Migrations

### golang-migrate

```bash
go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
```

Create migrations:
```bash
migrate create -ext sql -dir db/migrations -seq create_users_table
```

Migration files:
```sql
-- 000001_create_users_table.up.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 000001_create_users_table.down.sql
DROP TABLE users;
```

Run migrations:
```bash
migrate -path db/migrations -database "postgres://localhost/mydb?sslmode=disable" up
migrate -path db/migrations -database "postgres://localhost/mydb?sslmode=disable" down
```

### Migrations in Code

```go
package main

import (
    "database/sql"
    "github.com/golang-migrate/migrate/v4"
    "github.com/golang-migrate/migrate/v4/database/postgres"
    _ "github.com/golang-migrate/migrate/v4/source/file"
)

func runMigrations(db *sql.DB) error {
    driver, err := postgres.WithInstance(db, &postgres.Config{})
    if err != nil {
        return err
    }
    
    m, err := migrate.NewWithDatabaseInstance(
        "file://db/migrations",
        "postgres",
        driver,
    )
    if err != nil {
        return err
    }
    
    if err := m.Up(); err != nil && err != migrate.ErrNoChange {
        return err
    }
    
    return nil
}
```

## 🎯 GORM Migrations

```go
package main

import (
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
)

type User struct {
    gorm.Model
    Name  string `gorm:"not null"`
    Email string `gorm:"uniqueIndex;not null"`
    Age   int
    Posts []Post
}

type Post struct {
    gorm.Model
    Title   string
    Content string
    UserID  uint
}

func main() {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        panic(err)
    }
    
    // Auto-migrate
    db.AutoMigrate(&User{}, &Post{})
    
    // Create indexes
    db.Exec("CREATE INDEX idx_users_email ON users(email)")
    
    // Drop tables
    db.Migrator().DropTable(&User{}, &Post{})
    
    // Check if table exists
    if db.Migrator().HasTable(&User{}) {
        fmt.Println("Table exists")
    }
    
    // Add column
    db.Migrator().AddColumn(&User{}, "Age")
    
    // Drop column
    db.Migrator().DropColumn(&User{}, "Age")
}
```

## 📊 Advanced GORM Queries

### Hooks

```go
type User struct {
    ID        uint
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

// Before Create
func (u *User) BeforeCreate(tx *gorm.DB) error {
    if u.Name == "" {
        return errors.New("name is required")
    }
    return nil
}

// After Create
func (u *User) AfterCreate(tx *gorm.DB) error {
    log.Printf("User created: %d", u.ID)
    return nil
}

// Before Update
func (u *User) BeforeUpdate(tx *gorm.DB) error {
    u.UpdatedAt = time.Now()
    return nil
}

// After Find
func (u *User) AfterFind(tx *gorm.DB) error {
    log.Printf("User found: %s", u.Name)
    return nil
}
```

### Scopes

```go
func ActiveUsers(db *gorm.DB) *gorm.DB {
    return db.Where("active = ?", true)
}

func OrderByCreated(db *gorm.DB) *gorm.DB {
    return db.Order("created_at DESC")
}

func Paginate(page, pageSize int) func(db *gorm.DB) *gorm.DB {
    return func(db *gorm.DB) *gorm.DB {
        offset := (page - 1) * pageSize
        return db.Offset(offset).Limit(pageSize)
    }
}

// Usage
var users []User
db.Scopes(ActiveUsers, OrderByCreated, Paginate(1, 10)).Find(&users)
```

### Raw SQL

```go
// Raw query
var users []User
db.Raw("SELECT * FROM users WHERE age > ?", 18).Scan(&users)

// Execute raw SQL
db.Exec("UPDATE users SET age = ? WHERE name = ?", 25, "Alice")

// Named arguments
db.Raw("SELECT * FROM users WHERE name = @name", sql.Named("name", "Alice")).Scan(&users)
```

### Joins

```go
type Result struct {
    UserName  string
    PostTitle string
}

var results []Result

db.Table("users").
    Select("users.name as user_name, posts.title as post_title").
    Joins("left join posts on posts.user_id = users.id").
    Scan(&results)

// Preload with conditions
db.Preload("Posts", "published = ?", true).Find(&users)

// Preload nested
db.Preload("Posts.Comments").Find(&users)
```

## 🔍 Query Builder (squirrel)

```bash
go get github.com/Masterminds/squirrel
```

```go
import sq "github.com/Masterminds/squirrel"

// Select
query, args, _ := sq.Select("*").
    From("users").
    Where(sq.Eq{"age": 25}).
    Where(sq.Gt{"created_at": "2024-01-01"}).
    OrderBy("name ASC").
    Limit(10).
    PlaceholderFormat(sq.Dollar).
    ToSql()

rows, err := db.Query(query, args...)

// Insert
query, args, _ := sq.Insert("users").
    Columns("name", "email").
    Values("Alice", "alice@example.com").
    Values("Bob", "bob@example.com").
    PlaceholderFormat(sq.Dollar).
    ToSql()

// Update
query, args, _ := sq.Update("users").
    Set("name", "Alice Smith").
    Set("updated_at", time.Now()).
    Where(sq.Eq{"id": 1}).
    PlaceholderFormat(sq.Dollar).
    ToSql()

// Delete
query, args, _ := sq.Delete("users").
    Where(sq.Eq{"id": 1}).
    PlaceholderFormat(sq.Dollar).
    ToSql()

// Complex WHERE
sq.And{
    sq.Eq{"status": "active"},
    sq.Or{
        sq.Eq{"role": "admin"},
        sq.Eq{"role": "moderator"},
    },
}
```

## 💾 Caching

### Redis Integration

```bash
go get github.com/redis/go-redis/v9
```

```go
package main

import (
    "context"
    "encoding/json"
    "time"
    
    "github.com/redis/go-redis/v9"
)

type UserCache struct {
    rdb *redis.Client
    db  *gorm.DB
}

func NewUserCache(rdb *redis.Client, db *gorm.DB) *UserCache {
    return &UserCache{rdb: rdb, db: db}
}

func (uc *UserCache) GetUser(ctx context.Context, id uint) (*User, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("user:%d", id)
    
    val, err := uc.rdb.Get(ctx, cacheKey).Result()
    if err == nil {
        var user User
        if err := json.Unmarshal([]byte(val), &user); err == nil {
            return &user, nil
        }
    }
    
    // Not in cache, get from database
    var user User
    if err := uc.db.First(&user, id).Error; err != nil {
        return nil, err
    }
    
    // Store in cache
    userData, _ := json.Marshal(user)
    uc.rdb.Set(ctx, cacheKey, userData, 10*time.Minute)
    
    return &user, nil
}

func (uc *UserCache) InvalidateUser(ctx context.Context, id uint) error {
    cacheKey := fmt.Sprintf("user:%d", id)
    return uc.rdb.Del(ctx, cacheKey).Err()
}
```

## 📈 Database Performance

### Indexes

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at);

-- Unique index
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE active = true;

-- Full-text search (PostgreSQL)
CREATE INDEX idx_posts_content_fts ON posts USING gin(to_tsvector('english', content));
```

### Query Optimization

```go
// Bad: N+1 problem
var users []User
db.Find(&users)
for _, user := range users {
    var posts []Post
    db.Where("user_id = ?", user.ID).Find(&posts)  // N queries!
}

// Good: Use preload
var users []User
db.Preload("Posts").Find(&users)  // 2 queries only

// Use Select to fetch only needed columns
db.Select("id", "name", "email").Find(&users)

// Use Find in batches for large datasets
db.FindInBatches(&users, 100, func(tx *gorm.DB, batch int) error {
    // Process batch
    return nil
})
```

### Connection Pooling

```go
func setupDatabase(dsn string) (*gorm.DB, error) {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        return nil, err
    }
    
    sqlDB, err := db.DB()
    if err != nil {
        return nil, err
    }
    
    // Connection pool configuration
    sqlDB.SetMaxIdleConns(10)
    sqlDB.SetMaxOpenConns(100)
    sqlDB.SetConnMaxLifetime(time.Hour)
    
    return db, nil
}
```

## 💼 Complete Example: Blog System

```go
package main

import (
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "time"
)

type User struct {
    ID        uint      `gorm:"primaryKey"`
    Name      string    `gorm:"not null"`
    Email     string    `gorm:"uniqueIndex;not null"`
    Posts     []Post
    Comments  []Comment
    CreatedAt time.Time
    UpdatedAt time.Time
}

type Post struct {
    ID        uint      `gorm:"primaryKey"`
    Title     string    `gorm:"not null"`
    Content   string    `gorm:"type:text"`
    Published bool      `gorm:"default:false"`
    UserID    uint
    User      User
    Comments  []Comment
    Tags      []Tag     `gorm:"many2many:post_tags;"`
    CreatedAt time.Time
    UpdatedAt time.Time
}

type Comment struct {
    ID        uint      `gorm:"primaryKey"`
    Content   string    `gorm:"not null"`
    UserID    uint
    User      User
    PostID    uint
    Post      Post
    CreatedAt time.Time
}

type Tag struct {
    ID    uint   `gorm:"primaryKey"`
    Name  string `gorm:"uniqueIndex;not null"`
    Posts []Post `gorm:"many2many:post_tags;"`
}

type BlogRepository struct {
    db *gorm.DB
}

func NewBlogRepository(db *gorm.DB) *BlogRepository {
    return &BlogRepository{db: db}
}

func (r *BlogRepository) CreatePost(post *Post) error {
    return r.db.Create(post).Error
}

func (r *BlogRepository) GetPost(id uint) (*Post, error) {
    var post Post
    err := r.db.Preload("User").
        Preload("Comments.User").
        Preload("Tags").
        First(&post, id).Error
    return &post, err
}

func (r *BlogRepository) GetPublishedPosts(page, pageSize int) ([]Post, error) {
    var posts []Post
    offset := (page - 1) * pageSize
    
    err := r.db.Where("published = ?", true).
        Preload("User").
        Order("created_at DESC").
        Offset(offset).
        Limit(pageSize).
        Find(&posts).Error
    
    return posts, err
}

func (r *BlogRepository) SearchPosts(query string) ([]Post, error) {
    var posts []Post
    
    err := r.db.Where("title ILIKE ? OR content ILIKE ?", "%"+query+"%", "%"+query+"%").
        Preload("User").
        Find(&posts).Error
    
    return posts, err
}

func (r *BlogRepository) AddComment(comment *Comment) error {
    return r.db.Create(comment).Error
}

func (r *BlogRepository) GetPostsByTag(tagName string) ([]Post, error) {
    var tag Tag
    err := r.db.Preload("Posts.User").
        Where("name = ?", tagName).
        First(&tag).Error
    
    if err != nil {
        return nil, err
    }
    
    return tag.Posts, nil
}

func (r *BlogRepository) GetUserStats(userID uint) (map[string]int, error) {
    stats := make(map[string]int)
    
    r.db.Model(&Post{}).Where("user_id = ?", userID).Count(&stats["posts"])
    r.db.Model(&Comment{}).Where("user_id = ?", userID).Count(&stats["comments"])
    
    return stats, nil
}

func main() {
    dsn := "host=localhost user=postgres password=secret dbname=blog sslmode=disable"
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        panic(err)
    }
    
    // Auto-migrate
    db.AutoMigrate(&User{}, &Post{}, &Comment{}, &Tag{})
    
    repo := NewBlogRepository(db)
    
    // Create user
    user := User{
        Name:  "Alice",
        Email: "alice@example.com",
    }
    db.Create(&user)
    
    // Create post with tags
    post := Post{
        Title:     "My First Post",
        Content:   "This is the content",
        Published: true,
        UserID:    user.ID,
        Tags: []Tag{
            {Name: "golang"},
            {Name: "tutorial"},
        },
    }
    repo.CreatePost(&post)
    
    // Get posts
    posts, _ := repo.GetPublishedPosts(1, 10)
    for _, p := range posts {
        fmt.Printf("Post: %s by %s\n", p.Title, p.User.Name)
    }
    
    // Search
    results, _ := repo.SearchPosts("golang")
    fmt.Printf("Found %d posts\n", len(results))
    
    // Stats
    stats, _ := repo.GetUserStats(user.ID)
    fmt.Printf("User stats: %+v\n", stats)
}
```

## 🔑 Key Takeaways

- Use migrations for database schema management
- GORM provides hooks for lifecycle events
- Use scopes for reusable query logic
- Optimize queries with proper indexes
- Use caching for frequently accessed data
- Avoid N+1 queries with preloading
- Configure connection pools appropriately

## 📖 Next Steps

Continue to [Chapter 21: Context Package](21-context.md) to learn about Go's context for cancellation and deadlines.

