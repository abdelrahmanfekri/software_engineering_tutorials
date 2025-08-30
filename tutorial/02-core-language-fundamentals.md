# Core Language Fundamentals

## Variables and Data Types
```java
import java.util.List;
import java.util.Map;

public class DataTypes {
    public static void main(String[] args) {
        // Primitive types
        byte b = 127;
        short s = 32767;
        int i = 2147483647;
        long l = 9223372036854775807L;
        float f = 3.14159f;
        double d = 3.141592653589793;
        char c = 'A';
        boolean bool = true;

        // Wrapper classes (auto-boxing/unboxing)
        Integer wrappedInt = 42;
        int primitiveInt = wrappedInt; // auto-unboxing

        // var keyword (type inference) - Java 10+
        var name = "John Doe";
        var numbers = List.of(1, 2, 3, 4, 5);
        var map = Map.of("key1", "value1", "key2", "value2");
    }
}
```

## Control Structures
```java
import java.util.List;

public class ControlStructures {
    public static void main(String[] args) {
        // Enhanced switch expressions (Java 14+)
        String day = "MONDAY";
        int workHours = switch (day) {
            case "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY" -> 8;
            case "SATURDAY" -> 4;
            case "SUNDAY" -> 0;
            default -> throw new IllegalArgumentException("Invalid day: " + day);
        };

        // Pattern matching with switch (Java 21, enable preview if required)
        Object obj = "Hello";
        String result = switch (obj) {
            case String s when s.length() > 5 -> "Long string: " + s;
            case String s -> "Short string: " + s;
            case Integer i when i > 0 -> "Positive number: " + i;
            case Integer i -> "Non-positive number: " + i;
            case null -> "Null value";
            default -> "Unknown type";
        };

        // Enhanced for loop
        var items = List.of("apple", "banana", "cherry");
        for (String item : items) {
            System.out.println(item);
        }
    }
}
```


