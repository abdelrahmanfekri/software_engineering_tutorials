# College Algebra Tutorial 03: Exponential and Logarithmic Functions

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand exponential functions and their properties
- Work with logarithmic functions and their properties
- Use the change of base formula
- Solve exponential and logarithmic equations
- Apply these functions to real-world problems

## Exponential Functions

### Definition
An exponential function has the form f(x) = aˣ, where:
- a > 0 and a ≠ 1 (base)
- x is any real number (exponent)

### Properties of Exponential Functions
1. **Domain**: All real numbers (-∞, ∞)
2. **Range**: (0, ∞) if a > 1, (0, ∞) if 0 < a < 1
3. **y-intercept**: (0, 1) for all exponential functions
4. **Horizontal asymptote**: y = 0
5. **Monotonicity**: 
   - Increasing if a > 1
   - Decreasing if 0 < a < 1

### Common Exponential Functions
- **Natural exponential**: f(x) = eˣ (where e ≈ 2.718)
- **Base 2**: f(x) = 2ˣ
- **Base 10**: f(x) = 10ˣ

### Laws of Exponents
1. aˣ · aʸ = aˣ⁺ʸ
2. aˣ/aʸ = aˣ⁻ʸ
3. (aˣ)ʸ = aˣʸ
4. (ab)ˣ = aˣbˣ
5. (a/b)ˣ = aˣ/bˣ
6. a⁰ = 1
7. a⁻ˣ = 1/aˣ

## Logarithmic Functions

### Definition
The logarithmic function is the inverse of the exponential function:
y = logₐ(x) if and only if aʸ = x

Where:
- a > 0, a ≠ 1 (base)
- x > 0 (argument)

### Common Logarithmic Functions
- **Natural logarithm**: f(x) = ln(x) = logₑ(x)
- **Common logarithm**: f(x) = log(x) = log₁₀(x)

### Properties of Logarithmic Functions
1. **Domain**: (0, ∞)
2. **Range**: (-∞, ∞)
3. **x-intercept**: (1, 0)
4. **Vertical asymptote**: x = 0
5. **Monotonicity**: 
   - Increasing if a > 1
   - Decreasing if 0 < a < 1

### Laws of Logarithms
1. logₐ(xy) = logₐ(x) + logₐ(y)
2. logₐ(x/y) = logₐ(x) - logₐ(y)
3. logₐ(xʸ) = y logₐ(x)
4. logₐ(1) = 0
5. logₐ(a) = 1
6. logₐ(aˣ) = x
7. a^(logₐ(x)) = x

## Change of Base Formula

### Formula
logₐ(x) = log_b(x)/log_b(a)

This allows us to evaluate logarithms with any base using common bases.

**Example**: Evaluate log₃(7) using natural logarithms
log₃(7) = ln(7)/ln(3) ≈ 1.9459/1.0986 ≈ 1.771

## Solving Exponential Equations

### Method 1: Same Base
If both sides have the same base, set exponents equal.

**Example**: Solve 2ˣ = 8
- 2ˣ = 2³
- x = 3

### Method 2: Different Bases
Take the logarithm of both sides.

**Example**: Solve 3ˣ = 7
- ln(3ˣ) = ln(7)
- x ln(3) = ln(7)
- x = ln(7)/ln(3) ≈ 1.771

### Method 3: Quadratic Form
Sometimes exponential equations can be written as quadratics.

**Example**: Solve 2^(2x) - 3(2ˣ) + 2 = 0
- Let u = 2ˣ, then u² - 3u + 2 = 0
- (u - 1)(u - 2) = 0
- u = 1 or u = 2
- 2ˣ = 1 → x = 0
- 2ˣ = 2 → x = 1

## Solving Logarithmic Equations

### Method 1: Single Logarithm
Convert to exponential form.

**Example**: Solve log₂(x) = 3
- x = 2³ = 8

### Method 2: Multiple Logarithms
Use logarithm properties to combine terms.

**Example**: Solve log(x) + log(x - 3) = 1
- log(x(x - 3)) = 1
- x(x - 3) = 10¹ = 10
- x² - 3x - 10 = 0
- (x - 5)(x + 2) = 0
- x = 5 or x = -2
- Check: x = -2 is not in domain, so x = 5

### Method 3: Logarithmic Properties
Use properties to simplify before solving.

**Example**: Solve log₃(x²) = 2
- 2 log₃(x) = 2
- log₃(x) = 1
- x = 3¹ = 3

## Applications

### Compound Interest
A = P(1 + r/n)^(nt)
Where:
- A = final amount
- P = principal
- r = annual interest rate
- n = number of times compounded per year
- t = time in years

**Example**: $1000 invested at 5% compounded quarterly for 3 years
A = 1000(1 + 0.05/4)^(4·3) = 1000(1.0125)¹² ≈ $1161.18

### Population Growth
P(t) = P₀e^(rt)
Where:
- P(t) = population at time t
- P₀ = initial population
- r = growth rate
- t = time

**Example**: A population of 1000 grows at 2% per year. Find population after 5 years.
P(5) = 1000e^(0.02·5) = 1000e^0.1 ≈ 1105

### Radioactive Decay
A(t) = A₀e^(-λt)
Where:
- A(t) = amount remaining at time t
- A₀ = initial amount
- λ = decay constant
- t = time

## Practice Problems

### Problem 1
Solve 4ˣ = 64.

**Solution**:
- 4ˣ = 4³
- x = 3

### Problem 2
Solve log₅(x + 2) = 2.

**Solution**:
- x + 2 = 5² = 25
- x = 23

### Problem 3
Solve 2ˣ⁺¹ = 3ˣ⁻¹.

**Solution**:
- ln(2ˣ⁺¹) = ln(3ˣ⁻¹)
- (x + 1)ln(2) = (x - 1)ln(3)
- x ln(2) + ln(2) = x ln(3) - ln(3)
- x(ln(2) - ln(3)) = -ln(3) - ln(2)
- x = -(ln(3) + ln(2))/(ln(2) - ln(3)) ≈ 4.419

### Problem 4
A bacteria culture doubles every 3 hours. If there are 100 bacteria initially, how many will there be after 12 hours?

**Solution**:
- Growth model: P(t) = 100(2)^(t/3)
- P(12) = 100(2)^(12/3) = 100(2)⁴ = 100(16) = 1600 bacteria

## Key Takeaways
- Exponential functions model growth and decay
- Logarithmic functions are inverses of exponential functions
- Laws of exponents and logarithms simplify calculations
- Change of base formula allows evaluation with any base
- These functions have many real-world applications
- Always check solutions for domain restrictions

## Next Steps
In the next tutorial, we'll explore systems of equations and inequalities, learning how to solve multiple equations simultaneously and work with linear programming.
