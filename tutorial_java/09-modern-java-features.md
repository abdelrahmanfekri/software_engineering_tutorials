# Modern Java Features (Java 8–21)

## Pattern Matching and Switch Expressions
```java
import java.util.List;

public class ModernFeatures {
    // Pattern matching with instanceof (Java 16+)
    public void patternMatching(Object obj) {
        if (obj instanceof String s && s.length() > 5) {
            System.out.println("Long string: " + s.toUpperCase());
        } else if (obj instanceof Integer i && i > 0) {
            System.out.println("Positive integer: " + i);
        } else if (obj instanceof List<?> list && !list.isEmpty()) {
            System.out.println("Non-empty list with " + list.size() + " elements");
        }
    }

    // Record patterns (Java 21)
    public void recordPatterns() {
        var point = new Point(3, 4);
        switch (point) {
            case Point(int x, int y) when x == 0 && y == 0 -> System.out.println("Origin point");
            case Point(int x, int y) when x == y -> System.out.println("Point on diagonal: " + x);
            case Point(int x, int y) -> System.out.println("Point at (" + x + ", " + y + ")");
        }

        var coloredPoint = new ColoredPoint(new Point(1, 2), "red");
        switch (coloredPoint) {
            case ColoredPoint(Point(int x, int y), String color) ->
                System.out.println(color + " point at (" + x + ", " + y + ")");
        }
    }

    // String templates (Preview in Java 21)
    public void stringTemplates() {
        String name = "John";
        int age = 30;
        String traditional = String.format("Hello, %s! You are %d years old.", name, age);
        // String template (when available)
        // String template = STR."Hello, \{name}! You are \{age} years old.";
    }

    // Text blocks (Java 15+)
    public void textBlocks() {
        String json = """
            {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "Anytown",
                    "zipCode": "12345"
                }
            }
            """;

        String sql = """
            SELECT u.name, u.email, p.title
            FROM users u
            JOIN posts p ON u.id = p.user_id
            WHERE u.active = true
            ORDER BY p.created_at DESC
            LIMIT 10
            """;
    }

    record Point(int x, int y) {}
    record ColoredPoint(Point point, String color) {}
}
```

## Modules (Java 9+)
```java
// module-info.java
module com.example.backend {
    requires java.base;
    requires java.net.http;
    requires java.sql;
    requires spring.boot;
    requires spring.boot.autoconfigure;

    exports com.example.backend.api;
    exports com.example.backend.service;
    exports com.example.backend.internal to com.example.backend.test;

    provides com.example.backend.spi.PaymentProcessor 
        with com.example.backend.impl.CreditCardProcessor;

    uses com.example.backend.spi.NotificationService;
}
```


