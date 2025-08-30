# Object-Oriented Programming

## Classes and Objects
```java
// Modern class with records (Java 14+)
public record Person(String name, int age, String email) {
    // Compact constructor for validation
    public Person {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Name cannot be null or empty");
        }
        if (age < 0) {
            throw new IllegalArgumentException("Age cannot be negative");
        }
    }

    // Additional methods
    public boolean isAdult() { return age >= 18; }
    public String getDisplayName() { return name.toUpperCase(); }
}

// Traditional class with modern features
public class BankAccount {
    private final String accountNumber; // final for immutability
    private volatile double balance; // volatile for thread safety

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    // Synchronized method for thread safety
    public synchronized void deposit(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
        balance += amount;
    }

    public synchronized boolean withdraw(double amount) {
        if (amount <= 0 || amount > balance) {
            return false;
        }
        balance -= amount;
        return true;
    }

    public synchronized double getBalance() { return balance; }
}
```

## Inheritance and Polymorphism
```java
// Sealed classes (Java 17+) - controlled inheritance
public sealed class Shape permits Circle, Rectangle, Triangle {
    protected final String color;
    protected Shape(String color) { this.color = color; }
    public abstract double area();
}

public final class Circle extends Shape {
    private final double radius;
    public Circle(String color, double radius) { super(color); this.radius = radius; }
    @Override public double area() { return Math.PI * radius * radius; }
}

public final class Rectangle extends Shape {
    private final double width, height;
    public Rectangle(String color, double width, double height) { super(color); this.width = width; this.height = height; }
    @Override public double area() { return width * height; }
}

public non-sealed class Triangle extends Shape {
    private final double base, height;
    public Triangle(String color, double base, double height) { super(color); this.base = base; this.height = height; }
    @Override public double area() { return 0.5 * base * height; }
}
```

## Interfaces and Abstract Classes
```java
// Modern interface with default and static methods
public interface PaymentProcessor {
    void processPayment(double amount);

    default void logTransaction(double amount) {
        System.out.println("Processing payment of $" + amount);
    }

    static boolean isValidAmount(double amount) { return amount > 0 && amount <= 10000; }

    // Private methods in interfaces are supported since Java 9+
    private void validatePayment(double amount) {
        if (!isValidAmount(amount)) {
            throw new IllegalArgumentException("Invalid payment amount");
        }
    }
}

// Functional interface for lambdas
@FunctionalInterface
public interface TransactionValidator {
    boolean validate(Transaction transaction);

    default TransactionValidator and(TransactionValidator other) {
        return transaction -> this.validate(transaction) && other.validate(transaction);
    }
}
```


