# Java 21 Overview & Setup

## What's New in Java 21 (LTS)
- **Virtual Threads (Project Loom)**: Lightweight threads for massive concurrency
- **Pattern Matching enhancements**: More powerful `switch` expressions
- **Record Patterns**: Destructuring records in pattern matching
- **String Templates (Preview)**: Safe string interpolation
- **Sequenced Collections**: New collection interfaces
- **Key Encapsulation Mechanism API**: Enhanced cryptography

## Installation & Setup
```bash
# Recommended: SDKMAN
# Using Homebrew (recommended)
brew install openjdk@21

# Add to your shell profile (~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Alternative: Download from Oracle
# 1. Visit https://www.oracle.com/java/technologies/downloads/#java21
# 2. Download the macOS .dmg installer
# 3. Run the installer package

# Verify installation
java --version
```

## Notes on Preview Features
Some features (e.g., switch pattern matching guards, string templates) may be preview in Java 21.

Compile and run with preview enabled when needed:
```bash
javac --enable-preview --release 21 MyPreviewDemo.java
java --enable-preview MyPreviewDemo
```


