# College Algebra Tutorial 04: Systems of Equations and Inequalities

## Learning Objectives
By the end of this tutorial, you will be able to:
- Solve systems of linear equations using various methods
- Solve systems of nonlinear equations
- Use matrix methods for solving systems
- Solve systems of linear inequalities
- Apply linear programming techniques
- Interpret solutions in context

## Systems of Linear Equations

### Definition
A system of linear equations is a collection of two or more linear equations with the same variables.

**Example**:
2x + 3y = 7
x - y = 1

### Types of Solutions
1. **Consistent and Independent**: Exactly one solution (lines intersect)
2. **Consistent and Dependent**: Infinitely many solutions (same line)
3. **Inconsistent**: No solution (parallel lines)

## Solving Methods

### Method 1: Substitution
1. Solve one equation for one variable
2. Substitute into the other equation
3. Solve for the remaining variable
4. Back-substitute to find the other variable

**Example**: Solve the system
2x + 3y = 7
x - y = 1

**Solution**:
1. From second equation: x = y + 1
2. Substitute into first: 2(y + 1) + 3y = 7
3. Simplify: 2y + 2 + 3y = 7 → 5y = 5 → y = 1
4. Back-substitute: x = 1 + 1 = 2
5. Solution: (2, 1)

### Method 2: Elimination
1. Multiply equations to get opposite coefficients
2. Add equations to eliminate one variable
3. Solve for remaining variable
4. Back-substitute

**Example**: Solve the system
3x + 2y = 8
2x - y = 1

**Solution**:
1. Multiply second equation by 2: 4x - 2y = 2
2. Add to first equation: (3x + 2y) + (4x - 2y) = 8 + 2
3. Simplify: 7x = 10 → x = 10/7
4. Substitute: 2(10/7) - y = 1 → 20/7 - y = 1 → y = 13/7
5. Solution: (10/7, 13/7)

### Method 3: Graphing
1. Graph both equations
2. Find intersection point
3. Verify solution

## Systems of Three Variables

### General Form
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃

### Solving Process
1. Use elimination to reduce to two equations with two variables
2. Solve the 2×2 system
3. Back-substitute to find the third variable

**Example**: Solve the system
x + y + z = 6
2x - y + z = 3
x + 2y - z = 2

**Solution**:
1. Add first and second: 3x + 2z = 9
2. Add first and third: 2x + 3y = 8
3. From first equation: y = 6 - x - z
4. Substitute into step 2: 2x + 3(6 - x - z) = 8
5. Simplify: 2x + 18 - 3x - 3z = 8 → -x - 3z = -10 → x + 3z = 10
6. Now solve: 3x + 2z = 9 and x + 3z = 10
7. Solution: x = 1, y = 2, z = 3

## Matrix Methods

### Augmented Matrix
Convert system to matrix form:
[2  3 | 7]
[1 -1 | 1]

### Gaussian Elimination
Use row operations to get reduced row echelon form:
1. Swap rows
2. Multiply row by constant
3. Add multiple of one row to another

**Example**: Solve using matrices
2x + 3y = 7
x - y = 1

**Solution**:
1. Augmented matrix: [2  3 | 7]
                     [1 -1 | 1]

2. R₁ ↔ R₂: [1 -1 | 1]
            [2  3 | 7]

3. R₂ - 2R₁: [1 -1 | 1]
             [0  5 | 5]

4. (1/5)R₂: [1 -1 | 1]
            [0  1 | 1]

5. R₁ + R₂: [1  0 | 2]
            [0  1 | 1]

6. Solution: x = 2, y = 1

## Systems of Nonlinear Equations

### Substitution Method
1. Solve one equation for one variable
2. Substitute into the other equation
3. Solve the resulting equation
4. Find corresponding values

**Example**: Solve
x² + y² = 25
x + y = 7

**Solution**:
1. From second equation: y = 7 - x
2. Substitute: x² + (7 - x)² = 25
3. Expand: x² + 49 - 14x + x² = 25
4. Simplify: 2x² - 14x + 24 = 0 → x² - 7x + 12 = 0
5. Factor: (x - 3)(x - 4) = 0 → x = 3 or x = 4
6. Solutions: (3, 4) and (4, 3)

## Systems of Linear Inequalities

### Graphing Linear Inequalities
1. Graph the boundary line (solid for ≤, ≥; dashed for <, >)
2. Test a point to determine which side to shade
3. Shade the appropriate region

### Systems of Inequalities
The solution is the intersection of all individual solutions.

**Example**: Solve the system
x + y ≤ 4
x - y ≥ 1
x ≥ 0, y ≥ 0

**Solution**:
1. Graph x + y = 4 (solid line, shade below)
2. Graph x - y = 1 (solid line, shade above)
3. Graph x = 0 and y = 0 (axes, shade first quadrant)
4. Solution region is the intersection

## Linear Programming

### Definition
Linear programming finds the maximum or minimum value of a linear objective function subject to linear constraints.

### Standard Form
Maximize (or minimize): z = ax + by
Subject to: constraints (linear inequalities)

### Solving Process
1. Graph the feasible region (solution to constraints)
2. Find corner points of the feasible region
3. Evaluate objective function at each corner point
4. Identify optimal solution

**Example**: Maximize z = 3x + 2y
Subject to:
x + y ≤ 4
2x + y ≤ 6
x ≥ 0, y ≥ 0

**Solution**:
1. Corner points: (0, 0), (0, 4), (2, 2), (3, 0)
2. Evaluate z at each point:
   - z(0, 0) = 0
   - z(0, 4) = 8
   - z(2, 2) = 10
   - z(3, 0) = 9
3. Maximum value: 10 at (2, 2)

## Applications

### Mixture Problems
**Example**: A chemist needs 100ml of a 15% acid solution. Available are 10% and 20% solutions. How much of each should be mixed?

**Solution**:
Let x = amount of 10% solution, y = amount of 20% solution
x + y = 100
0.10x + 0.20y = 0.15(100) = 15

Solving: x = 50ml, y = 50ml

### Investment Problems
**Example**: $10,000 invested in two accounts earning 5% and 8% annually. Total interest is $680. How much in each account?

**Solution**:
Let x = amount at 5%, y = amount at 8%
x + y = 10000
0.05x + 0.08y = 680

Solving: x = $4000, y = $6000

## Practice Problems

### Problem 1
Solve the system:
3x - 2y = 7
x + 4y = 1

**Solution**:
Using elimination: Multiply first by 2, second by 1
6x - 4y = 14
x + 4y = 1
Adding: 7x = 15 → x = 15/7
Substituting: 15/7 + 4y = 1 → 4y = -8/7 → y = -2/7
Solution: (15/7, -2/7)

### Problem 2
Solve the system:
x² + y² = 13
x - y = 1

**Solution**:
From second equation: x = y + 1
Substituting: (y + 1)² + y² = 13
Expanding: y² + 2y + 1 + y² = 13 → 2y² + 2y - 12 = 0 → y² + y - 6 = 0
Factoring: (y + 3)(y - 2) = 0 → y = -3 or y = 2
Solutions: (3, 2) and (-2, -3)

### Problem 3
Maximize z = 2x + 3y subject to:
x + y ≤ 6
2x + y ≤ 8
x ≥ 0, y ≥ 0

**Solution**:
Corner points: (0, 0), (0, 6), (2, 4), (4, 0)
Evaluating z: z(0, 0) = 0, z(0, 6) = 18, z(2, 4) = 16, z(4, 0) = 8
Maximum: 18 at (0, 6)

## Key Takeaways
- Systems can have one, infinitely many, or no solutions
- Multiple methods exist for solving systems
- Matrix methods are efficient for larger systems
- Nonlinear systems require different approaches
- Linear programming optimizes under constraints
- Real-world problems often involve systems of equations

## Next Steps
In the next tutorial, we'll explore sequences and series, learning about arithmetic and geometric progressions and their applications in finance and science.
