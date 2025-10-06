# College Algebra Tutorial 07: Complex Numbers

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the definition and notation of complex numbers
- Perform arithmetic operations with complex numbers
- Convert between rectangular and polar forms
- Apply De Moivre's theorem
- Find roots of complex numbers
- Understand applications in engineering and physics

## Introduction to Complex Numbers

### Definition
A complex number is a number of the form a + bi, where:
- a and b are real numbers
- i is the imaginary unit with i² = -1
- a is the real part
- b is the imaginary part

**Notation**: z = a + bi

**Examples**:
- 3 + 4i (real part: 3, imaginary part: 4)
- -2 - 5i (real part: -2, imaginary part: -5)
- 7i (real part: 0, imaginary part: 7)
- 5 (real part: 5, imaginary part: 0)

## Basic Operations

### Addition and Subtraction
Add/subtract real parts and imaginary parts separately.

**Example**: (3 + 4i) + (2 - 7i) = (3 + 2) + (4 - 7)i = 5 - 3i

**Example**: (3 + 4i) - (2 - 7i) = (3 - 2) + (4 - (-7))i = 1 + 11i

### Multiplication
Use the distributive property and remember that i² = -1.

**Example**: (3 + 4i)(2 - 7i)
= 3(2) + 3(-7i) + 4i(2) + 4i(-7i)
= 6 - 21i + 8i - 28i²
= 6 - 13i - 28(-1)
= 6 - 13i + 28
= 34 - 13i

### Division
Multiply numerator and denominator by the complex conjugate of the denominator.

**Complex conjugate**: If z = a + bi, then z̄ = a - bi

**Example**: (3 + 4i)/(2 - 7i)
= (3 + 4i)(2 + 7i)/(2 - 7i)(2 + 7i)
= (6 + 21i + 8i + 28i²)/(4 + 14i - 14i - 49i²)
= (6 + 29i - 28)/(4 + 49)
= (-22 + 29i)/53
= -22/53 + (29/53)i

## Complex Plane

### Rectangular Form
Plot complex numbers as points (a, b) in the plane.

**Example**: Plot z = 3 + 4i
- Point: (3, 4)
- Distance from origin: |z| = √(3² + 4²) = 5

### Polar Form
z = r(cos θ + i sin θ) = r cis θ

Where:
- r = |z| = √(a² + b²) (modulus)
- θ = arg(z) = arctan(b/a) (argument)

**Example**: Convert 3 + 4i to polar form
- r = √(3² + 4²) = 5
- θ = arctan(4/3) ≈ 53.13°
- Polar form: 5(cos 53.13° + i sin 53.13°)

### Converting Between Forms
**Rectangular to Polar**:
- r = √(a² + b²)
- θ = arctan(b/a) (adjust quadrant as needed)

**Polar to Rectangular**:
- a = r cos θ
- b = r sin θ

## De Moivre's Theorem

### Statement
If z = r(cos θ + i sin θ), then:
zⁿ = rⁿ(cos nθ + i sin nθ)

### Applications
1. **Powers**: Easily compute powers of complex numbers
2. **Roots**: Find nth roots of complex numbers

**Example**: Find (1 + i)⁶
1. Convert to polar: 1 + i = √2(cos 45° + i sin 45°)
2. Apply De Moivre's theorem: (√2)⁶(cos 270° + i sin 270°)
3. Simplify: 8(cos 270° + i sin 270°) = 8(0 + i(-1)) = -8i

## Roots of Complex Numbers

### Finding nth Roots
For z = r(cos θ + i sin θ), the nth roots are:
z^(1/n) = r^(1/n)[cos((θ + 2kπ)/n) + i sin((θ + 2kπ)/n)]

Where k = 0, 1, 2, ..., n-1

**Example**: Find the cube roots of 8
1. 8 = 8(cos 0° + i sin 0°)
2. Cube roots: 8^(1/3)[cos(0° + 2kπ)/3 + i sin(0° + 2kπ)/3]
3. For k = 0: 2(cos 0° + i sin 0°) = 2
4. For k = 1: 2(cos 120° + i sin 120°) = -1 + √3i
5. For k = 2: 2(cos 240° + i sin 240°) = -1 - √3i

## Properties of Complex Numbers

### Modulus Properties
- |z₁z₂| = |z₁||z₂|
- |z₁/z₂| = |z₁|/|z₂|
- |z̄| = |z|
- |z|² = zz̄

### Argument Properties
- arg(z₁z₂) = arg(z₁) + arg(z₂)
- arg(z₁/z₂) = arg(z₁) - arg(z₂)
- arg(z̄) = -arg(z)

## Applications

### Electrical Engineering
Complex numbers represent AC circuits:
- Real part: Resistance
- Imaginary part: Reactance
- Impedance: Z = R + iX

### Signal Processing
Complex numbers represent signals:
- Real part: In-phase component
- Imaginary part: Quadrature component
- Frequency domain analysis

### Physics
- Quantum mechanics uses complex wave functions
- Electromagnetic field theory
- Fluid dynamics

## Practice Problems

### Problem 1
Simplify (2 + 3i)(4 - i)

**Solution**:
(2 + 3i)(4 - i) = 8 - 2i + 12i - 3i² = 8 + 10i - 3(-1) = 11 + 10i

### Problem 2
Find (1 - i)⁴

**Solution**:
1. Convert to polar: 1 - i = √2(cos 315° + i sin 315°)
2. Apply De Moivre's theorem: (√2)⁴(cos 1260° + i sin 1260°)
3. Simplify: 4(cos 180° + i sin 180°) = 4(-1 + i(0)) = -4

### Problem 3
Find all fourth roots of -16

**Solution**:
1. -16 = 16(cos 180° + i sin 180°)
2. Fourth roots: 16^(1/4)[cos(180° + 2kπ)/4 + i sin(180° + 2kπ)/4]
3. For k = 0: 2(cos 45° + i sin 45°) = √2 + √2i
4. For k = 1: 2(cos 135° + i sin 135°) = -√2 + √2i
5. For k = 2: 2(cos 225° + i sin 225°) = -√2 - √2i
6. For k = 3: 2(cos 315° + i sin 315°) = √2 - √2i

### Problem 4
Convert 5(cos 60° + i sin 60°) to rectangular form

**Solution**:
- Real part: 5 cos 60° = 5(1/2) = 2.5
- Imaginary part: 5 sin 60° = 5(√3/2) = 5√3/2
- Rectangular form: 2.5 + (5√3/2)i

## Key Takeaways
- Complex numbers extend real numbers with imaginary unit i
- Operations follow algebraic rules with i² = -1
- Polar form simplifies multiplication and powers
- De Moivre's theorem enables easy computation of powers and roots
- Complex numbers have wide applications in science and engineering

## Next Steps
In the next tutorial, we'll explore matrices and determinants, learning about matrix operations and their applications in solving systems of equations.
