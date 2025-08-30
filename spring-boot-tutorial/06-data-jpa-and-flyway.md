### Data JPA and Flyway

### Entity and repository
```java
package com.example.bookstore.domain;

import jakarta.persistence.*;

@Entity
@Table(name = "books")
public class Book {
  @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;
  private String title;
  private String author;
  private int priceCents;

  protected Book() {}
  public Book(String title, String author, int priceCents) {
    this.title = title; this.author = author; this.priceCents = priceCents;
  }

  public Long getId() { return id; }
  public String getTitle() { return title; }
  public String getAuthor() { return author; }
  public int getPriceCents() { return priceCents; }
}
```

```java
package com.example.bookstore.domain;

import org.springframework.data.jpa.repository.JpaRepository;

public interface BookRepository extends JpaRepository<Book, Long> {
  java.util.List<Book> findByAuthorContainingIgnoreCase(String author);
}
```

### Service mapping to DTOs
```java
package com.example.bookstore.api;

import com.example.bookstore.domain.Book;
import com.example.bookstore.domain.BookRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class BookService {
  private final BookRepository repo;
  public BookService(BookRepository repo) { this.repo = repo; }

  public BookResponse create(CreateBookRequest req) {
    var saved = repo.save(new Book(req.title(), req.author(), req.priceCents()));
    return new BookResponse(saved.getId(), saved.getTitle(), saved.getAuthor(), saved.getPriceCents());
  }
  public List<BookResponse> list() {
    return repo.findAll().stream()
        .map(b -> new BookResponse(b.getId(), b.getTitle(), b.getAuthor(), b.getPriceCents()))
        .toList();
  }
}
```

### Flyway migrations
- Create `src/main/resources/db/migration/V1__init.sql`:
```sql
CREATE TABLE books (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT NOT NULL,
  price_cents INT NOT NULL
);
```

### Dev H2 profile (optional)
```yaml
spring:
  config:
    activate:
      on-profile: h2
  datasource:
    url: jdbc:h2:mem:bookstore;MODE=PostgreSQL;DB_CLOSE_DELAY=-1
    driver-class-name: org.h2.Driver
  h2:
    console:
      enabled: true
```

Next: `07-security-jwt.md`.


