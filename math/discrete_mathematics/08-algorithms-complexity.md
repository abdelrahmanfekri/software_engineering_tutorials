# Algorithms and Complexity

## Overview
Algorithm analysis and complexity theory are fundamental to computer science. They provide tools for understanding the efficiency of algorithms, comparing different approaches to solving problems, and understanding the limits of computation. This chapter covers algorithm analysis, complexity classes, and important algorithmic techniques.

## Learning Objectives
- Understand algorithm analysis and big-O notation
- Learn about sorting and searching algorithms
- Master graph algorithms
- Understand dynamic programming
- Learn greedy algorithms
- Study NP-completeness and computational complexity
- Apply complexity analysis to real problems

## 1. Algorithm Analysis

### Time Complexity
The **time complexity** of an algorithm describes how the running time grows as the input size increases.

### Space Complexity
The **space complexity** of an algorithm describes how much memory the algorithm uses as the input size increases.

### Big-O Notation
**Big-O notation** describes the upper bound of an algorithm's complexity.

**Definition**: f(n) = O(g(n)) if there exist constants c and n₀ such that f(n) ≤ c·g(n) for all n ≥ n₀.

### Common Complexity Classes
- **O(1)**: Constant time
- **O(log n)**: Logarithmic time
- **O(n)**: Linear time
- **O(n log n)**: Linearithmic time
- **O(n²)**: Quadratic time
- **O(n³)**: Cubic time
- **O(2ⁿ)**: Exponential time
- **O(n!)**: Factorial time

### Examples
```python
# O(1) - Constant time
def get_first_element(arr):
    return arr[0]

# O(n) - Linear time
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

# O(n²) - Quadratic time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

## 2. Sorting Algorithms

### Comparison-Based Sorting

#### Bubble Sort
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: Yes
- **In-place**: Yes

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

#### Selection Sort
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: No
- **In-place**: Yes

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

#### Insertion Sort
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: Yes
- **In-place**: Yes

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

#### Merge Sort
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(n)
- **Stable**: Yes
- **In-place**: No

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

#### Quick Sort
- **Time Complexity**: O(n log n) average, O(n²) worst case
- **Space Complexity**: O(log n)
- **Stable**: No
- **In-place**: Yes

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```

### Non-Comparison-Based Sorting

#### Counting Sort
- **Time Complexity**: O(n + k) where k is the range of input
- **Space Complexity**: O(k)
- **Stable**: Yes
- **In-place**: No

#### Radix Sort
- **Time Complexity**: O(d(n + k)) where d is the number of digits
- **Space Complexity**: O(n + k)
- **Stable**: Yes
- **In-place**: No

## 3. Searching Algorithms

### Linear Search
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

### Binary Search
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1)
- **Requirement**: Array must be sorted

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## 4. Graph Algorithms

### Breadth-First Search (BFS)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Depth-First Search (DFS)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

### Shortest Path Algorithms

#### Dijkstra's Algorithm
- **Time Complexity**: O((V + E) log V)
- **Space Complexity**: O(V)
- **Requirement**: Non-negative edge weights

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

## 5. Dynamic Programming

### Definition
**Dynamic programming** is a method for solving complex problems by breaking them down into simpler subproblems and storing the solutions to avoid redundant calculations.

### Key Principles
1. **Optimal substructure**: The optimal solution contains optimal solutions to subproblems
2. **Overlapping subproblems**: The same subproblems are solved multiple times

### Examples

#### Fibonacci Numbers
```python
def fibonacci(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]
```

#### Longest Common Subsequence
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

## 6. Greedy Algorithms

### Definition
A **greedy algorithm** makes the locally optimal choice at each step, hoping to find a global optimum.

### Examples

#### Activity Selection Problem
```python
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    
    return selected
```

#### Huffman Coding
```python
import heapq

def huffman_coding(frequencies):
    heap = [[freq, [char, ""]] for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
```

## 7. NP-Completeness

### Complexity Classes

#### P
Problems solvable in polynomial time by a deterministic Turing machine.

#### NP
Problems solvable in polynomial time by a non-deterministic Turing machine, or problems where a solution can be verified in polynomial time.

#### NP-Complete
Problems that are:
1. In NP
2. NP-hard (every problem in NP can be reduced to them in polynomial time)

#### NP-Hard
Problems that are at least as hard as the hardest problems in NP.

### Examples of NP-Complete Problems
- **SAT**: Boolean satisfiability
- **3-SAT**: 3-CNF satisfiability
- **Clique**: Finding a clique of size k
- **Vertex Cover**: Finding a vertex cover of size k
- **Hamiltonian Cycle**: Finding a cycle that visits each vertex exactly once
- **Traveling Salesman Problem**: Finding the shortest tour visiting all cities

### Reduction
A problem A is **reducible** to problem B if there's a polynomial-time algorithm that converts instances of A to instances of B.

## 8. Practice Problems

### Algorithm Analysis
1. Analyze the time complexity of the following algorithms:
   - Finding the maximum element in an array
   - Matrix multiplication
   - Tower of Hanoi

2. Compare the time complexity of different sorting algorithms for:
   - Small arrays (n < 10)
   - Large arrays (n > 1000)
   - Nearly sorted arrays

### Sorting and Searching
3. Implement and compare the performance of:
   - Bubble sort vs. insertion sort
   - Merge sort vs. quick sort
   - Linear search vs. binary search

4. Design an algorithm to find the k-th largest element in an array.

### Graph Algorithms
5. Implement BFS and DFS for:
   - Adjacency list representation
   - Adjacency matrix representation

6. Find the shortest path between two vertices in a weighted graph.

### Dynamic Programming
7. Solve the following problems:
   - Longest increasing subsequence
   - Edit distance
   - Coin change problem

8. Implement the knapsack problem using dynamic programming.

### Greedy Algorithms
9. Solve the following problems:
   - Fractional knapsack
   - Minimum spanning tree (Kruskal's algorithm)
   - Job scheduling

10. Design a greedy algorithm for the set cover problem.

### NP-Completeness
11. Show that the clique problem is NP-complete.

12. Reduce the vertex cover problem to the independent set problem.

## 9. Applications

### Computer Science
- **Algorithm design**: Choosing the right algorithm for a problem
- **Performance optimization**: Improving algorithm efficiency
- **System design**: Scalability and resource management
- **Database systems**: Query optimization and indexing

### Software Engineering
- **Code optimization**: Improving program performance
- **Testing**: Algorithm correctness and performance
- **Maintenance**: Understanding algorithm behavior
- **Documentation**: Explaining algorithm complexity

### Research
- **Complexity theory**: Understanding computational limits
- **Algorithm design**: Developing new efficient algorithms
- **Optimization**: Finding better solutions to problems
- **Machine learning**: Algorithm selection and optimization

## Key Takeaways

1. **Complexity matters**: Understanding algorithm efficiency is crucial
2. **Big-O notation is essential**: It provides a way to compare algorithms
3. **Different problems need different approaches**: Choose the right algorithm
4. **Dynamic programming is powerful**: It solves many optimization problems
5. **Greedy algorithms are simple**: But they don't always give optimal solutions
6. **NP-completeness is important**: It helps understand problem difficulty
7. **Practice is essential**: Implement algorithms to understand them better

## Next Steps
- Master algorithm analysis and complexity
- Implement and compare different algorithms
- Study advanced algorithmic techniques
- Learn about approximation algorithms
- Apply algorithms to solve real-world problems
- Connect algorithm theory to practical programming
