# Packaging: jlink, jpackage, GraalVM Native Image

## jlink (Custom Runtime Image)
Create a small runtime containing only required modules.
```bash
javac -d out --module-source-path src $(find src -name "*.java")
jlink \
  --module-path $JAVA_HOME/jmods:out \
  --add-modules com.example.app \
  --output build/runtime \
  --strip-debug --compress=2 --no-header-files --no-man-pages
```

Run with:
```bash
build/runtime/bin/java -m com.example.app/com.example.app.Main
```

## jpackage (Installers/DMG/MSI)
```bash
jpackage \
  --name MyApp \
  --input build/libs \
  --main-jar myapp-all.jar \
  --type dmg \
  --icon app.icns
```

## GraalVM Native Image
Build a native binary with fast startup and low RSS.
```bash
# Install GraalVM and native-image tool, then:
native-image -jar myapp-all.jar myapp
./myapp
```

Notes:
- Reflection, dynamic proxies, and resources may require configuration (`reflect-config.json`, etc.).
- For Spring Boot, prefer AOT processing (Spring Native/Spring Boot 3 AOT) to optimize native image.


