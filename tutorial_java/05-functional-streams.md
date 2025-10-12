# Functional Programming & Streams

## Lambda Expressions and Method References
```java
import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

public class FunctionalProgramming {
    public static void main(String[] args) {
        // Lambda expressions
        Predicate<String> isLongString = s -> s.length() > 5;
        Function<String, Integer> stringLength = s -> s.length();
        Consumer<String> printer = s -> System.out.println(s);
        Supplier<String> randomString = () -> "Random: " + Math.random();

        // Method references
        Predicate<String> isEmpty = String::isEmpty;
        Function<String, String> toUpperCase = String::toUpperCase;
        Consumer<String> systemOut = System.out::println;

        // Constructor references
        Supplier<List<String>> listSupplier = ArrayList::new;
        Function<String, StringBuilder> sbCreator = StringBuilder::new;

        // Complex functional interfaces
        BinaryOperator<Integer> adder = Integer::sum;
        UnaryOperator<String> trimmer = String::trim;

        // Custom functional interfaces
        TriFunction<Integer, Integer, Integer, Integer> calculator =
            (a, b, c) -> a + b + c;
    }

    @FunctionalInterface
    interface TriFunction<T, U, V, R> {
        R apply(T t, U u, V v);
    }
}
```

## Stream API Deep Dive
```java
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

public class StreamsAdvanced {
    public static void main(String[] args) {
        var products = List.of(
            new Product("Laptop", "Electronics", 999.99, 50),
            new Product("Smartphone", "Electronics", 699.99, 100),
            new Product("Book", "Education", 29.99, 200),
            new Product("Desk", "Furniture", 199.99, 30)
        );

        // Complex stream operations
        var expensiveElectronics = products.stream()
            .filter(p -> "Electronics".equals(p.category()))
            .filter(p -> p.price() > 500)
            .sorted(Comparator.comparing(Product::price).reversed())
            .limit(5)
            .collect(Collectors.toList());

        // Parallel streams for performance
        var totalValue = products.parallelStream()
            .mapToDouble(p -> p.price() * p.quantity())
            .sum();

        // Stream collectors
        Map<String, Long> productCountByCategory = products.stream()
            .collect(Collectors.groupingBy(
                Product::category,
                Collectors.counting()
            ));

        // Optional handling
        Optional<Product> mostExpensive = products.stream()
            .max(Comparator.comparing(Product::price));

        String result = mostExpensive
            .map(Product::name)
            .orElse("No products found");

        // Flat mapping
        var orders = List.of(
            new Order(List.of("item1", "item2")),
            new Order(List.of("item3", "item4", "item5"))
        );

        var allItems = orders.stream()
            .flatMap(order -> order.items().stream())
            .collect(Collectors.toList());

        // Custom collectors
        String productNames = products.stream()
            .map(Product::name)
            .collect(Collector.of(
                StringBuilder::new,
                (sb, s) -> sb.append(s).append(", "),
                StringBuilder::append,
                StringBuilder::toString
            ));
    }

    record Product(String name, String category, double price, int quantity) {}
    record Order(List<String> items) {}
}
```


