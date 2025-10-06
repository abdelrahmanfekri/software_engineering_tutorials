# Practice Problems and Exercises

## Overview
This collection provides comprehensive practice problems for all topics in discrete mathematics. Problems are organized by topic and difficulty level, with solutions and explanations to help reinforce understanding and develop problem-solving skills.

## Problem Categories

### 1. Logic and Proof Techniques

#### Basic Logic
1. **Truth Tables**: Construct truth tables for:
   - (p → q) ∧ (q → r)
   - ¬(p ∨ q) → (¬p ∧ ¬q)
   - (p ∧ q) ∨ (¬p ∧ ¬q)

2. **Logical Equivalences**: Prove the following:
   - p → q ≡ ¬p ∨ q
   - p ↔ q ≡ (p → q) ∧ (q → p)
   - ¬(p ∧ q) ≡ ¬p ∨ ¬q

3. **Predicate Logic**: Translate to English:
   - ∀x (Student(x) → Takes(x, Math))
   - ∃x (Student(x) ∧ Takes(x, CS))
   - ∀x ∃y (x < y)

#### Proof Techniques
4. **Direct Proof**: Prove "If a and b are odd integers, then a + b is even"

5. **Contrapositive**: Prove "If n² is even, then n is even"

6. **Contradiction**: Prove "There are infinitely many prime numbers"

7. **Mathematical Induction**: Prove "2ⁿ > n for all n ≥ 1"

8. **Strong Induction**: Prove "Every positive integer n ≥ 2 can be written as a product of primes"

### 2. Set Theory

#### Basic Operations
9. **Set Operations**: Let A = {1, 2, 3, 4, 5}, B = {3, 4, 5, 6, 7}, C = {2, 4, 6, 8}
   - Find A ∪ B
   - Find A ∩ B
   - Find A - B
   - Find A △ B

10. **Set Laws**: Prove the distributive law: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)

11. **Venn Diagrams**: Draw Venn diagrams for:
    - A ∩ (B ∪ C)
    - (A ∪ B) ∩ (A ∪ C)
    - A' ∩ B'

#### Cardinality
12. **Finite Sets**: If |A| = 5 and |B| = 3, find:
    - |A × B|
    - |P(A)|
    - |A ∪ B| (assuming A ∩ B = ∅)

13. **Power Sets**: Find P({a, b, c})

14. **Infinite Sets**: Prove that the set of rational numbers is countably infinite

### 3. Functions and Relations

#### Functions
15. **Function Properties**: Determine if the following functions are injective, surjective, or bijective:
    - f: ℝ → ℝ, f(x) = x²
    - g: ℝ → ℝ, g(x) = x³
    - h: ℤ → ℤ, h(n) = n + 1

16. **Function Composition**: Find f ∘ g and g ∘ f for:
    - f(x) = 2x + 1, g(x) = x²

17. **Inverse Functions**: Find the inverse of f(x) = 3x - 2

#### Relations
18. **Relation Properties**: Determine which properties (reflexive, symmetric, antisymmetric, transitive) the following relations have:
    - R = {(1, 1), (2, 2), (3, 3)} on {1, 2, 3}
    - "divides" on ℤ⁺
    - "is perpendicular to" on lines in a plane

19. **Equivalence Relations**: Show that "congruent modulo 3" is an equivalence relation on ℤ

20. **Equivalence Classes**: Find the equivalence classes for the relation "has the same remainder when divided by 3" on ℤ

### 4. Combinatorics

#### Basic Counting
21. **Counting Principles**: How many ways can:
    - 5 people sit in a row?
    - 3 people be chosen from 10?
    - 4 books be arranged on a shelf if 2 are identical?

22. **Permutations**: How many ways can the letters in "MATHEMATICS" be arranged?

23. **Combinations**: A committee of 5 is to be chosen from 12 people. How many ways if:
    - No restrictions?
    - 2 specific people must be included?
    - 2 specific people cannot both be included?

#### Advanced Counting
24. **Pigeonhole Principle**: Prove that in any group of 6 people, either 3 are mutual friends or 3 are mutual strangers.

25. **Inclusion-Exclusion**: How many integers from 1 to 1000 are divisible by 2, 3, or 5?

26. **Generating Functions**: Find the generating function for the sequence {1, 2, 3, 4, ...}

27. **Recurrence Relations**: Solve: a_n = 3a_{n-1} - 2a_{n-2}, a₀ = 1, a₁ = 2

### 5. Graph Theory

#### Basic Concepts
28. **Graph Properties**: Draw all non-isomorphic graphs with 4 vertices.

29. **Connectivity**: Find all connected components of the graph with edges: {(1,2), (2,3), (4,5), (6,7), (7,8)}

30. **Trees**: Prove that a tree with n vertices has n-1 edges.

#### Graph Algorithms
31. **BFS and DFS**: Implement BFS and DFS for the graph:
    - Vertices: {1, 2, 3, 4, 5}
    - Edges: {(1,2), (2,3), (3,4), (4,5), (5,1)}

32. **Shortest Paths**: Find the shortest path from vertex 1 to vertex 5 in a weighted graph.

33. **Minimum Spanning Tree**: Find the MST for the graph with edges and weights:
    - (1,2): 3, (1,3): 1, (2,3): 2, (2,4): 4, (3,4): 5

#### Graph Coloring
34. **Chromatic Number**: Find the chromatic number of:
    - Complete graph K₅
    - Cycle C₆
    - Bipartite graph K_{3,3}

35. **Graph Coloring**: Color the following graph using the greedy algorithm:
    - Vertices: {1, 2, 3, 4, 5}
    - Edges: {(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)}

### 6. Number Theory

#### Divisibility and GCD
36. **Euclidean Algorithm**: Find gcd(123, 456) using the Euclidean algorithm.

37. **Extended Euclidean**: Find integers x and y such that 123x + 456y = gcd(123, 456).

38. **Modular Arithmetic**: Compute 7^100 mod 13.

#### Prime Numbers
39. **Primality Testing**: Use the Sieve of Eratosthenes to find all primes up to 50.

40. **Prime Factorization**: Find the prime factorization of 1001.

41. **Chinese Remainder Theorem**: Solve the system:
    - x ≡ 1 (mod 3)
    - x ≡ 2 (mod 5)
    - x ≡ 3 (mod 7)

#### Cryptography
42. **RSA**: Generate RSA keys with p = 5, q = 7, e = 5.

43. **RSA Encryption**: Encrypt the message m = 3 using the public key from problem 42.

44. **Diffie-Hellman**: Implement Diffie-Hellman key exchange.

### 7. Boolean Algebra

#### Boolean Operations
45. **Truth Tables**: Construct truth tables for:
    - (x ∧ y) ∨ (¬x ∧ ¬y)
    - (x ∨ y) ∧ (¬x ∨ ¬y)
    - x ⊕ y

46. **Boolean Laws**: Prove the following identities:
    - x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)
    - ¬(x ∧ y) = ¬x ∨ ¬y

#### Karnaugh Maps
47. **K-map Simplification**: Simplify using K-map: f(x, y, z) = Σ(0, 1, 2, 3, 6, 7)

48. **Logic Gates**: Implement f(x, y, z) = (x ∧ y) ∨ (¬x ∧ z) using:
    - Basic gates
    - Only NAND gates
    - Only NOR gates

### 8. Algorithms and Complexity

#### Algorithm Analysis
49. **Time Complexity**: Analyze the time complexity of:
    - Finding the maximum element in an array
    - Matrix multiplication
    - Tower of Hanoi

50. **Sorting**: Implement and compare the performance of:
    - Bubble sort vs. insertion sort
    - Merge sort vs. quick sort

#### Graph Algorithms
51. **Shortest Paths**: Implement Dijkstra's algorithm for finding shortest paths.

52. **Minimum Spanning Tree**: Compare Kruskal's and Prim's algorithms.

#### Dynamic Programming
53. **Fibonacci**: Implement Fibonacci numbers using dynamic programming.

54. **Longest Common Subsequence**: Solve the LCS problem using dynamic programming.

55. **Knapsack Problem**: Implement the 0/1 knapsack problem using dynamic programming.

### 9. Advanced Topics

#### Advanced Combinatorics
56. **Stirling Numbers**: Calculate s(5, 3) and S(5, 3).

57. **Bell Numbers**: Calculate B₇ using the recurrence relation.

58. **Partition Theory**: Find all partitions of 6.

#### Advanced Graph Theory
59. **Network Flows**: Find the maximum flow in a given network.

60. **Planar Graphs**: Determine if K₅ and K_{3,3} are planar.

61. **Graph Coloring**: Find the chromatic number of the Petersen graph.

#### Cryptography
62. **Hash Functions**: Implement SHA-256 hash function.

63. **Digital Signatures**: Implement RSA digital signature scheme.

64. **Elliptic Curves**: Find points on the elliptic curve y² = x³ + 2x + 3 over GF(7).

#### Computational Geometry
65. **Convex Hulls**: Find the convex hull of a set of points.

66. **Voronoi Diagrams**: Construct the Voronoi diagram for a set of points.

67. **Closest Pair**: Find the closest pair among a set of points.

## Problem Solutions

### Solution 1: Truth Table for (p → q) ∧ (q → r)
| p | q | r | p→q | q→r | (p→q)∧(q→r) |
|---|---|---|-----|-----|-------------|
| T | T | T |  T  |  T  |      T      |
| T | T | F |  T  |  F  |      F      |
| T | F | T |  F  |  T  |      F      |
| T | F | F |  F  |  T  |      F      |
| F | T | T |  T  |  T  |      T      |
| F | T | F |  T  |  F  |      F      |
| F | F | T |  T  |  T  |      T      |
| F | F | F |  T  |  T  |      T      |

### Solution 4: Direct Proof
**Statement**: If a and b are odd integers, then a + b is even.

**Proof**: Let a and b be odd integers. Then a = 2k + 1 and b = 2m + 1 for some integers k and m.
- a + b = (2k + 1) + (2m + 1) = 2k + 2m + 2 = 2(k + m + 1)
- Since k + m + 1 is an integer, a + b is even.

### Solution 9: Set Operations
Given A = {1, 2, 3, 4, 5}, B = {3, 4, 5, 6, 7}, C = {2, 4, 6, 8}:
- A ∪ B = {1, 2, 3, 4, 5, 6, 7}
- A ∩ B = {3, 4, 5}
- A - B = {1, 2}
- A △ B = {1, 2, 6, 7}

### Solution 21: Counting Principles
- 5 people in a row: 5! = 120 ways
- 3 people from 10: C(10, 3) = 120 ways
- 4 books with 2 identical: 4!/2! = 12 ways

### Solution 36: Euclidean Algorithm
Find gcd(123, 456):
- 456 = 123 × 3 + 87
- 123 = 87 × 1 + 36
- 87 = 36 × 2 + 15
- 36 = 15 × 2 + 6
- 15 = 6 × 2 + 3
- 6 = 3 × 2 + 0
- gcd(123, 456) = 3

## Practice Tips

1. **Start with basics**: Master fundamental concepts before moving to advanced topics
2. **Work systematically**: Solve problems in order of difficulty
3. **Check your work**: Verify solutions and understand mistakes
4. **Practice regularly**: Consistent practice builds understanding
5. **Apply concepts**: Connect problems to real-world applications
6. **Seek help**: Don't hesitate to ask questions or seek clarification
7. **Review regularly**: Reinforce learning through periodic review

## Additional Resources

### Online Practice
- LeetCode: Algorithm problems
- HackerRank: Discrete mathematics problems
- Codeforces: Competitive programming problems
- Project Euler: Mathematical programming problems

### Books for Practice
- "Discrete Mathematics and Its Applications" by Kenneth Rosen
- "Concrete Mathematics" by Graham, Knuth, and Patashnik
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein

### Study Groups
- Form study groups with classmates
- Join online communities
- Participate in math competitions
- Attend workshops and seminars

Remember: Practice is essential for mastering discrete mathematics. Work through problems regularly, understand the solutions, and apply concepts to solve new problems.
