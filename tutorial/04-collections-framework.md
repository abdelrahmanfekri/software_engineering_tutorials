# Collections Framework

## Core Collections
```java
import java.util.*;
import java.util.concurrent.*;

public class CollectionsDemo {
    public static void main(String[] args) {
        // List implementations
        List<String> arrayList = new ArrayList<>();
        List<String> linkedList = new LinkedList<>();
        List<String> immutableList = List.of("a", "b", "c"); // Java 9+

        // Set implementations
        Set<String> hashSet = new HashSet<>();
        Set<String> linkedHashSet = new LinkedHashSet<>();
        Set<String> treeSet = new TreeSet<>();
        Set<String> immutableSet = Set.of("x", "y", "z");

        // Map implementations
        Map<String, Integer> hashMap = new HashMap<>();
        Map<String, Integer> linkedHashMap = new LinkedHashMap<>();
        Map<String, Integer> treeMap = new TreeMap<>();
        Map<String, Integer> immutableMap = Map.of(
            "one", 1,
            "two", 2,
            "three", 3
        );

        // Concurrent collections for multithreading
        ConcurrentMap<String, String> concurrentMap = new ConcurrentHashMap<>();
        BlockingQueue<String> blockingQueue = new ArrayBlockingQueue<>(10);

        // Modern collection operations
        var numbers = List.of(1, 2, 3, 4, 5);

        // Collection factories (Java 9+)
        var colors = List.of("red", "green", "blue");
        var scores = Map.of("Alice", 95, "Bob", 87, "Charlie", 92);

        // Collection.removeIf (Java 8+)
        var names = new ArrayList<>(List.of("Alice", "Bob", "Charlie", "David"));
        names.removeIf(name -> name.length() < 4);

        // Collection.replaceAll (Java 8+)
        names.replaceAll(String::toUpperCase);
    }
}
```

## Advanced Collection Operations
```java
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class AdvancedCollections {
    public static void main(String[] args) {
        var employees = List.of(
            new Employee("Alice", "Engineering", 75000),
            new Employee("Bob", "Marketing", 65000),
            new Employee("Charlie", "Engineering", 80000),
            new Employee("Diana", "HR", 60000)
        );

        // Grouping by department
        Map<String, List<Employee>> byDepartment = employees.stream()
            .collect(Collectors.groupingBy(Employee::department));

        // Average salary by department
        Map<String, Double> avgSalaryByDept = employees.stream()
            .collect(Collectors.groupingBy(
                Employee::department,
                Collectors.averagingDouble(Employee::salary)
            ));

        // Partitioning (true/false grouping)
        Map<Boolean, List<Employee>> highEarners = employees.stream()
            .collect(Collectors.partitioningBy(emp -> emp.salary() > 70000));

        // Custom collector
        String names = employees.stream()
            .map(Employee::name)
            .collect(Collectors.joining(", ", "[", "]"));
    }

    record Employee(String name, String department, double salary) {}
}
```


