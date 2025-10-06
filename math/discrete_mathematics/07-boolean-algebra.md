# Boolean Algebra

## Overview
Boolean algebra is the mathematical foundation of digital logic and computer design. It deals with binary variables and logical operations, providing the theoretical basis for digital circuits, programming logic, and computer architecture. This chapter covers Boolean functions, operations, and their applications.

## Learning Objectives
- Understand Boolean functions and expressions
- Master Boolean operations and laws
- Learn to use Karnaugh maps for simplification
- Understand logic gates and circuits
- Apply minimization techniques
- Connect Boolean algebra to computer design

## 1. Boolean Functions

### Definition
A **Boolean function** is a function f: {0, 1}^n → {0, 1} that maps n binary inputs to a single binary output.

### Boolean Variables
- **Binary variables**: Can take values 0 (false) or 1 (true)
- **Notation**: x, y, z, etc.

### Examples
1. f(x) = x (identity function)
2. f(x, y) = x ∧ y (AND function)
3. f(x, y) = x ∨ y (OR function)
4. f(x) = ¬x (NOT function)

## 2. Boolean Operations

### Basic Operations

#### AND (Conjunction)
- Symbol: ∧ or ·
- Truth table:
  | x | y | x ∧ y |
  |---|---|---|
  | 0 | 0 |  0   |
  | 0 | 1 |  0   |
  | 1 | 0 |  0   |
  | 1 | 1 |  1   |

#### OR (Disjunction)
- Symbol: ∨ or +
- Truth table:
  | x | y | x ∨ y |
  |---|---|---|
  | 0 | 0 |  0   |
  | 0 | 1 |  1   |
  | 1 | 0 |  1   |
  | 1 | 1 |  1   |

#### NOT (Negation)
- Symbol: ¬ or '
- Truth table:
  | x | ¬x |
  |---|---|
  | 0 | 1  |
  | 1 | 0  |

### Derived Operations

#### XOR (Exclusive OR)
- Symbol: ⊕
- Truth table:
  | x | y | x ⊕ y |
  |---|---|---|
  | 0 | 0 |  0   |
  | 0 | 1 |  1   |
  | 1 | 0 |  1   |
  | 1 | 1 |  0   |

#### NAND (NOT AND)
- Symbol: ↑
- Truth table:
  | x | y | x ↑ y |
  |---|---|---|
  | 0 | 0 |  1   |
  | 0 | 1 |  1   |
  | 1 | 0 |  1   |
  | 1 | 1 |  0   |

#### NOR (NOT OR)
- Symbol: ↓
- Truth table:
  | x | y | x ↓ y |
  |---|---|---|
  | 0 | 0 |  1   |
  | 0 | 1 |  0   |
  | 1 | 0 |  0   |
  | 1 | 1 |  0   |

## 3. Boolean Laws

### Commutative Laws
- x ∧ y = y ∧ x
- x ∨ y = y ∨ x

### Associative Laws
- (x ∧ y) ∧ z = x ∧ (y ∧ z)
- (x ∨ y) ∨ z = x ∨ (y ∨ z)

### Distributive Laws
- x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)
- x ∨ (y ∧ z) = (x ∨ y) ∧ (x ∨ z)

### Identity Laws
- x ∧ 1 = x
- x ∨ 0 = x

### Domination Laws
- x ∧ 0 = 0
- x ∨ 1 = 1

### Idempotent Laws
- x ∧ x = x
- x ∨ x = x

### Double Negation Law
- ¬(¬x) = x

### De Morgan's Laws
- ¬(x ∧ y) = ¬x ∨ ¬y
- ¬(x ∨ y) = ¬x ∧ ¬y

### Absorption Laws
- x ∧ (x ∨ y) = x
- x ∨ (x ∧ y) = x

### Complement Laws
- x ∧ ¬x = 0
- x ∨ ¬x = 1

## 4. Boolean Expressions

### Definition
A **Boolean expression** is a combination of Boolean variables and operations.

### Examples
1. x ∧ y
2. (x ∨ y) ∧ ¬z
3. x ∧ (y ∨ z) ∧ (¬x ∨ ¬y)

### Canonical Forms

#### Sum of Products (SOP)
A Boolean expression written as a sum (OR) of product (AND) terms.

**Example**: f(x, y, z) = xyz + x¬yz + ¬xyz

#### Product of Sums (POS)
A Boolean expression written as a product (AND) of sum (OR) terms.

**Example**: f(x, y, z) = (x + y + z)(x + ¬y + z)(¬x + y + z)

### Minterms and Maxterms

#### Minterm
A product term that contains all variables exactly once.

**Example**: For variables x, y, z: xyz, x¬y¬z, ¬x¬y¬z

#### Maxterm
A sum term that contains all variables exactly once.

**Example**: For variables x, y, z: x + y + z, x + ¬y + ¬z, ¬x + ¬y + ¬z

## 5. Karnaugh Maps

### Definition
A **Karnaugh map** (K-map) is a graphical method for simplifying Boolean expressions.

### 2-Variable K-map
```
     y
     0   1
x 0 | 0 | 1 |
   1 | 1 | 0 |
```

### 3-Variable K-map
```
     yz
     00  01  11  10
x 0 | 0 | 1 | 1 | 0 |
   1 | 1 | 0 | 0 | 1 |
```

### 4-Variable K-map
```
     yz
     00  01  11  10
wx 00 | 0 | 1 | 1 | 0 |
   01 | 1 | 0 | 0 | 1 |
   11 | 1 | 0 | 0 | 1 |
   10 | 0 | 1 | 1 | 0 |
```

### K-map Rules
1. **Adjacent cells**: Differ in exactly one variable
2. **Grouping**: Group 1s in powers of 2 (1, 2, 4, 8, ...)
3. **Don't cares**: Can be treated as 0 or 1 to simplify
4. **Wrapping**: Top and bottom edges are adjacent, left and right edges are adjacent

### Example
Simplify f(x, y, z) = Σ(0, 2, 4, 6) using K-map:

```
     yz
     00  01  11  10
x 0 | 1 | 0 | 0 | 1 |
   1 | 1 | 0 | 0 | 1 |
```

Grouping: f(x, y, z) = ¬y

## 6. Logic Gates

### Basic Gates

#### AND Gate
- Symbol: 
- Output: 1 only when all inputs are 1

#### OR Gate
- Symbol: 
- Output: 1 when any input is 1

#### NOT Gate (Inverter)
- Symbol: 
- Output: Opposite of input

#### NAND Gate
- Symbol: 
- Output: NOT of AND

#### NOR Gate
- Symbol: 
- Output: NOT of OR

#### XOR Gate
- Symbol: 
- Output: 1 when inputs differ

### Universal Gates
- **NAND**: Can implement any Boolean function
- **NOR**: Can implement any Boolean function

### Gate Implementation
Any Boolean function can be implemented using only NAND gates or only NOR gates.

## 7. Minimization Techniques

### Algebraic Minimization
Use Boolean laws to simplify expressions.

**Example**: Simplify (x ∧ y) ∨ (x ∧ ¬y)
- (x ∧ y) ∨ (x ∧ ¬y) = x ∧ (y ∨ ¬y) = x ∧ 1 = x

### K-map Minimization
Use Karnaugh maps to find minimal expressions.

### Quine-McCluskey Algorithm
Systematic method for minimizing Boolean functions with many variables.

## 8. Practice Problems

### Boolean Operations
1. Construct truth tables for:
   - (x ∧ y) ∨ (¬x ∧ ¬y)
   - (x ∨ y) ∧ (¬x ∨ ¬y)
   - x ⊕ y

2. Prove the following identities:
   - x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)
   - ¬(x ∧ y) = ¬x ∨ ¬y

### Boolean Expressions
3. Convert to SOP form: (x + y)(¬x + z)

4. Convert to POS form: xy + ¬xz

5. Find the minterm expansion for f(x, y) = x ⊕ y

### Karnaugh Maps
6. Simplify using K-map: f(x, y, z) = Σ(0, 1, 2, 3, 6, 7)

7. Simplify using K-map: f(w, x, y, z) = Σ(0, 1, 2, 3, 8, 9, 10, 11)

8. Use K-map to minimize: f(x, y, z) = xyz + x¬yz + ¬xyz + ¬x¬yz

### Logic Gates
9. Implement f(x, y, z) = (x ∧ y) ∨ (¬x ∧ z) using:
   - Basic gates
   - Only NAND gates
   - Only NOR gates

10. Design a 2-bit adder using logic gates.

### Advanced Problems
11. Minimize the function f(x, y, z, w) = Σ(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

12. Implement a 4-to-1 multiplexer using logic gates.

## 9. Applications

### Digital Circuits
- **Combinational circuits**: Output depends only on current inputs
- **Sequential circuits**: Output depends on current inputs and previous state
- **Arithmetic circuits**: Adders, multipliers, comparators
- **Memory circuits**: Flip-flops, registers, RAM

### Computer Architecture
- **CPU design**: ALU, control unit, registers
- **Memory systems**: Cache, main memory, storage
- **I/O systems**: Interfaces, controllers
- **Buses**: Data, address, control buses

### Programming
- **Boolean logic**: Conditional statements, loops
- **Bitwise operations**: AND, OR, XOR, NOT
- **Control flow**: Branching, decision making
- **Error detection**: Parity bits, checksums

### Cryptography
- **Stream ciphers**: XOR-based encryption
- **Hash functions**: Boolean operations in hashing
- **Random number generation**: Linear feedback shift registers
- **Error correction**: Hamming codes, Reed-Solomon codes

## Key Takeaways

1. **Boolean algebra is fundamental**: It's the basis of digital logic
2. **Operations have properties**: Learn the laws and use them for simplification
3. **K-maps are powerful**: They provide visual simplification methods
4. **Logic gates implement functions**: Any Boolean function can be built from gates
5. **Minimization is important**: It reduces circuit complexity and cost
6. **Applications are everywhere**: From computers to cryptography
7. **Universal gates exist**: NAND and NOR can implement any function

## Next Steps
- Master Boolean operations and laws
- Practice with Karnaugh maps
- Learn to implement functions using logic gates
- Study minimization techniques
- Apply Boolean algebra to digital circuit design
- Connect Boolean algebra to programming and computer architecture
