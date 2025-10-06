# College Algebra Tutorial 06: Conic Sections

## Learning Objectives
By the end of this tutorial, you will be able to:
- Identify and graph circles
- Identify and graph ellipses
- Identify and graph hyperbolas
- Identify and graph parabolas
- Understand focus-directrix properties
- Apply translation and rotation to conic sections

## Introduction to Conic Sections

### What are Conic Sections?
Conic sections are curves formed by the intersection of a plane with a double-napped cone. The four main types are:
1. **Circle**: Plane perpendicular to cone's axis
2. **Ellipse**: Plane at an angle to axis (but not parallel to side)
3. **Parabola**: Plane parallel to one side of cone
4. **Hyperbola**: Plane cuts both nappes of cone

## Circles

### Standard Form
(x - h)² + (y - k)² = r²
- Center: (h, k)
- Radius: r

### General Form
x² + y² + Dx + Ey + F = 0

### Converting Between Forms
**General to Standard**: Complete the square for both x and y terms.

**Example**: Convert x² + y² - 4x + 6y - 3 = 0 to standard form
1. Group x and y terms: (x² - 4x) + (y² + 6y) = 3
2. Complete the square: (x² - 4x + 4) + (y² + 6y + 9) = 3 + 4 + 9
3. Factor: (x - 2)² + (y + 3)² = 16
4. Center: (2, -3), Radius: 4

### Graphing Circles
1. Plot the center
2. Use the radius to mark points in all directions
3. Draw a smooth circle through these points

## Ellipses

### Standard Forms
**Horizontal major axis**: (x - h)²/a² + (y - k)²/b² = 1
**Vertical major axis**: (x - h)²/b² + (y - k)²/a² = 1

Where a > b and:
- Center: (h, k)
- Major axis length: 2a
- Minor axis length: 2b
- Foci: c = √(a² - b²)

### Properties
- **Foci**: Two fixed points inside the ellipse
- **Sum of distances**: From any point on ellipse to foci = 2a
- **Eccentricity**: e = c/a (0 < e < 1)

**Example**: Graph (x - 2)²/9 + (y + 1)²/4 = 1
- Center: (2, -1)
- a = 3, b = 2 (horizontal major axis)
- c = √(9 - 4) = √5
- Foci: (2 ± √5, -1)
- Vertices: (2 ± 3, -1) = (-1, -1), (5, -1)
- Co-vertices: (2, -1 ± 2) = (2, -3), (2, 1)

## Hyperbolas

### Standard Forms
**Horizontal transverse axis**: (x - h)²/a² - (y - k)²/b² = 1
**Vertical transverse axis**: (y - k)²/a² - (x - h)²/b² = 1

Where:
- Center: (h, k)
- Transverse axis length: 2a
- Conjugate axis length: 2b
- Foci: c = √(a² + b²)

### Properties
- **Foci**: Two fixed points outside the hyperbola
- **Difference of distances**: From any point on hyperbola to foci = 2a
- **Asymptotes**: Lines the hyperbola approaches
- **Eccentricity**: e = c/a (e > 1)

**Example**: Graph (x - 1)²/4 - (y + 2)²/9 = 1
- Center: (1, -2)
- a = 2, b = 3
- c = √(4 + 9) = √13
- Foci: (1 ± √13, -2)
- Vertices: (1 ± 2, -2) = (-1, -2), (3, -2)
- Asymptotes: y + 2 = ±(3/2)(x - 1)

## Parabolas

### Standard Forms
**Vertical axis**: (x - h)² = 4p(y - k)
**Horizontal axis**: (y - k)² = 4p(x - h)

Where:
- Vertex: (h, k)
- Focus: p units from vertex
- Directrix: Line p units from vertex (opposite side of focus)

### Properties
- **Focus**: Fixed point
- **Directrix**: Fixed line
- **Distance property**: From any point on parabola to focus = distance to directrix

**Example**: Graph (x - 2)² = 8(y + 1)
- Vertex: (2, -1)
- 4p = 8, so p = 2
- Focus: (2, -1 + 2) = (2, 1)
- Directrix: y = -1 - 2 = -3

## Focus-Directrix Properties

### General Definition
A conic section is the set of points P such that:
distance(P, focus) = e × distance(P, directrix)

Where e is the eccentricity:
- Circle: e = 0
- Ellipse: 0 < e < 1
- Parabola: e = 1
- Hyperbola: e > 1

## Translation and Rotation

### Translation
Moving a conic section by (h, k):
- Replace x with (x - h)
- Replace y with (y - k)

**Example**: Translate x² + y² = 9 by (2, -3)
- New equation: (x - 2)² + (y + 3)² = 9

### Rotation
For rotated conic sections, use rotation formulas:
x' = x cos θ + y sin θ
y' = -x sin θ + y cos θ

## Applications

### Real-World Examples
1. **Circles**: Wheels, orbits (circular)
2. **Ellipses**: Planetary orbits, satellite dishes
3. **Hyperbolas**: Sonic booms, radio navigation
4. **Parabolas**: Satellite dishes, projectile motion

### Engineering Applications
- **Optics**: Reflecting telescopes use parabolic mirrors
- **Architecture**: Arches often use parabolic shapes
- **Antennas**: Parabolic reflectors focus signals

## Practice Problems

### Problem 1
Find the center and radius of x² + y² - 6x + 8y - 11 = 0

**Solution**:
- Complete the square: (x² - 6x + 9) + (y² + 8y + 16) = 11 + 9 + 16
- (x - 3)² + (y + 4)² = 36
- Center: (3, -4), Radius: 6

### Problem 2
Graph the ellipse (x + 1)²/16 + (y - 2)²/9 = 1

**Solution**:
- Center: (-1, 2)
- a = 4, b = 3 (horizontal major axis)
- c = √(16 - 9) = √7
- Foci: (-1 ± √7, 2)
- Vertices: (-1 ± 4, 2) = (-5, 2), (3, 2)
- Co-vertices: (-1, 2 ± 3) = (-1, -1), (-1, 5)

### Problem 3
Find the focus and directrix of y² = 12x

**Solution**:
- Standard form: (y - 0)² = 4(3)(x - 0)
- Vertex: (0, 0)
- p = 3
- Focus: (0 + 3, 0) = (3, 0)
- Directrix: x = 0 - 3 = -3

### Problem 4
Graph the hyperbola (y - 1)²/4 - (x + 2)²/9 = 1

**Solution**:
- Center: (-2, 1)
- a = 2, b = 3 (vertical transverse axis)
- c = √(4 + 9) = √13
- Foci: (-2, 1 ± √13)
- Vertices: (-2, 1 ± 2) = (-2, -1), (-2, 3)
- Asymptotes: y - 1 = ±(2/3)(x + 2)

## Key Takeaways
- Conic sections are curves formed by intersecting a plane with a cone
- Each type has distinct geometric properties
- Focus-directrix definition unifies all conic sections
- Translation and rotation can transform conic sections
- Real-world applications span many fields

## Next Steps
In the next tutorial, we'll explore complex numbers, learning about their properties, operations, and applications in engineering and physics.
