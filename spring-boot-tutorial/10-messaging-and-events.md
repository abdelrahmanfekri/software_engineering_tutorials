### Messaging and Events — Domain Events, Kafka

### Spring domain events
```java
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;

record BookCreatedEvent(Long id) {}

@Service
class DomainEventService {
  private final ApplicationEventPublisher publisher;
  DomainEventService(ApplicationEventPublisher publisher) { this.publisher = publisher; }
  void onBookCreated(Long id) { publisher.publishEvent(new BookCreatedEvent(id)); }
}

@org.springframework.context.event.EventListener
void handle(BookCreatedEvent event) {
  System.out.println("Book created: " + event.id());
}
```

### Kafka basics
```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    consumer:
      group-id: bookstore
      auto-offset-reset: earliest
```

```java
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
class BookKafka {
  private final KafkaTemplate<String, String> kafka;
  BookKafka(KafkaTemplate<String, String> kafka) { this.kafka = kafka; }
  void publishCreated(Long id) { kafka.send("books.created", id.toString()); }

  @KafkaListener(topics = "books.created")
  public void onCreated(String id) { System.out.println("Kafka event book created: " + id); }
}
```

Consider the outbox pattern for reliable event publishing from DB transactions.

Next: `11-reactive-webflux.md`.


