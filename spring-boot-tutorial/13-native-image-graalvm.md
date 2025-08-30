### Native Images with GraalVM

Native images start fast and use less memory at the cost of longer builds and limited dynamic features.

### Gradle configuration
```kotlin
plugins {
  id("org.graalvm.buildtools.native") version "0.10.2"
}

graalvmNative {
  binaries {
    named("main") {
      buildArgs.add("--no-fallback")
    }
  }
}
```

### Build native
```bash
./gradlew nativeCompile
build/native/nativeCompile/bookstore
```

### Or use Cloud Native Buildpacks (recommended)
```bash
./gradlew bootBuildImage \
  -PBP_NATIVE_IMAGE=true \
  -PBP_JVM_VERSION=21 \
  -x test
docker run -p 8080:8080 ghcr.io/your/bookstore:0.0.1
```

### Runtime hints for reflection/resources
```java
import org.springframework.aot.hint.RuntimeHints;
import org.springframework.aot.hint.RuntimeHintsRegistrar;
import org.springframework.context.annotation.ImportRuntimeHints;

@ImportRuntimeHints(MyHints.class)
class App {}

class MyHints implements RuntimeHintsRegistrar {
  @Override public void registerHints(RuntimeHints hints, ClassLoader cl) {
    hints.resources().registerPattern("db/migration/**");
  }
}
```

Next: `14-docker-and-deploy.md`.


