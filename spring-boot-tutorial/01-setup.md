### Setup — Java 21, Build Tools, Boot CLI

### Install Java 21
- macOS (Homebrew):
```bash
brew install --cask temurin
java -version
```

- Or use SDKMAN (cross-platform):
```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 21.0.4-tem
java -version
```

### Install Gradle or Maven
- Gradle (recommended):
```bash
sdk install gradle 8.9
gradle -v
```

- Maven:
```bash
brew install maven
mvn -v
```

### Install Spring Boot CLI (optional)
```bash
sdk install springboot 3.5.0
spring --version
```

If `3.5.0` is not available yet, install the latest `3.5.x`.

### IDE setup
- IntelliJ IDEA Community/Ultimate (recommended) or VS Code with Java extensions
- Enable annotation processing (for Lombok if you use it)

### Docker
- Install Docker Desktop for your OS. Verify:
```bash
docker version
```

Continue to `02-create-project.md` to scaffold your project.


