# Reflection, MethodHandles & Dynamic Proxies

## Reflection Basics
```java
import java.lang.reflect.*;

public class ReflectionBasics {
    public static void inspectClass(Class<?> clazz) {
        System.out.println("Class: " + clazz.getName());
        for (Field f : clazz.getDeclaredFields()) {
            System.out.println("Field: " + f.getName() + " : " + f.getType());
        }
        for (Method m : clazz.getDeclaredMethods()) {
            System.out.println("Method: " + m.getName());
        }
        for (Constructor<?> c : clazz.getDeclaredConstructors()) {
            System.out.println("Ctor: " + c);
        }
    }
}
```

## MethodHandles (Faster, Safer than reflection)
```java
import java.lang.invoke.*;

public class MethodHandlesDemo {
    public static void main(String[] args) throws Throwable {
        var lookup = MethodHandles.lookup();
        MethodType type = MethodType.methodType(String.class);
        MethodHandle toStringMH = lookup.findVirtual(Object.class, "toString", type);
        String s = (String) toStringMH.invokeExact((Object) 42);
        System.out.println(s);
    }
}
```

## Dynamic Proxies (Cross-cutting concerns)
```java
import java.lang.reflect.*;

interface Repository {
    String findById(long id);
}

class RepoImpl implements Repository {
    public String findById(long id) { return "User-" + id; }
}

public class ProxyLoggingDemo {
    @SuppressWarnings("unchecked")
    public static <T> T withLogging(Class<T> intf, T target) {
        return (T) Proxy.newProxyInstance(
            intf.getClassLoader(), new Class<?>[]{intf},
            (proxy, method, args) -> {
                long start = System.nanoTime();
                try { return method.invoke(target, args); }
                finally { System.out.println(method.getName() + " took " + (System.nanoTime()-start) + " ns"); }
            }
        );
    }

    public static void main(String[] args) {
        Repository repo = withLogging(Repository.class, new RepoImpl());
        System.out.println(repo.findById(7));
    }
}
```

## Annotations via Reflection
```java
import java.lang.annotation.*;
import java.lang.reflect.*;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@interface Component { String value(); }

@Component("service:user")
class UserService {}

class AnnotationScan {
    public static void main(String[] args) {
        Component c = UserService.class.getAnnotation(Component.class);
        if (c != null) System.out.println("Found component: " + c.value());
    }
}
```


