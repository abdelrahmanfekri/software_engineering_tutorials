# Advanced Graph Theory

## Overview
Advanced graph theory extends the basic concepts to more sophisticated algorithms, structures, and applications. This chapter covers network flows, advanced graph algorithms, planar graph theory, graph coloring algorithms, random graphs, and spectral graph theory - topics that are essential for modern computer science and network analysis.

## Learning Objectives
- Understand network flows and max-flow min-cut theorem
- Master advanced graph algorithms
- Learn about planar graph theory and Kuratowski's theorem
- Study graph coloring algorithms
- Understand random graphs and their properties
- Learn spectral graph theory
- Apply advanced graph techniques to real problems

## 1. Network Flows

### Flow Networks
**Definition**: A flow network is a directed graph G = (V, E) with:
- A source vertex s
- A sink vertex t
- Each edge (u, v) has a capacity c(u, v) ≥ 0
- Each edge (u, v) has a flow f(u, v) ≤ c(u, v)

### Flow Constraints
1. **Capacity constraint**: f(u, v) ≤ c(u, v)
2. **Flow conservation**: Σ f(u, v) = Σ f(v, w) for all v ≠ s, t
3. **Skew symmetry**: f(u, v) = -f(v, u)

### Maximum Flow Problem
Find the maximum flow from source s to sink t.

### Ford-Fulkerson Algorithm
```
FORD-FULKERSON(G, s, t):
    for each edge (u, v) in G.E:
        f(u, v) = 0
    while there exists a path p from s to t in residual graph G_f:
        c_f(p) = min{c_f(u, v) : (u, v) is in p}
        for each edge (u, v) in p:
            f(u, v) = f(u, v) + c_f(p)
            f(v, u) = f(v, u) - c_f(p)
```

### Max-Flow Min-Cut Theorem
The maximum flow from s to t equals the minimum capacity of any s-t cut.

### Applications
- **Network routing**: Maximum data flow
- **Bipartite matching**: Maximum matching
- **Image segmentation**: Graph cuts
- **Supply chain**: Maximum flow optimization

## 2. Advanced Graph Algorithms

### All-Pairs Shortest Paths

#### Floyd-Warshall Algorithm
Finds shortest paths between all pairs of vertices.

```
FLOYD-WARSHALL(W):
    n = W.rows
    D^(0) = W
    for k = 1 to n:
        for i = 1 to n:
            for j = 1 to n:
                D^(k)[i, j] = min(D^(k-1)[i, j], D^(k-1)[i, k] + D^(k-1)[k, j])
    return D^(n)
```

**Time Complexity**: O(V³)
**Space Complexity**: O(V²)

#### Johnson's Algorithm
Finds shortest paths between all pairs using Dijkstra's algorithm.

**Time Complexity**: O(V² log V + VE)

### Strongly Connected Components

#### Tarjan's Algorithm
Finds strongly connected components using DFS.

```
TARJAN(G):
    for each vertex v in G:
        if v.index is undefined:
            STRONG-CONNECT(v)

STRONG-CONNECT(v):
    v.index = index
    v.lowlink = index
    index = index + 1
    S.push(v)
    v.onStack = true
    
    for each edge (v, w) in G:
        if w.index is undefined:
            STRONG-CONNECT(w)
            v.lowlink = min(v.lowlink, w.lowlink)
        else if w.onStack:
            v.lowlink = min(v.lowlink, w.index)
    
    if v.lowlink == v.index:
        start a new strongly connected component
        repeat:
            w = S.pop()
            w.onStack = false
            add w to current component
        until w == v
```

### Minimum Spanning Trees

#### Kruskal's Algorithm
```
KRUSKAL(G):
    A = empty set
    for each vertex v in G.V:
        MAKE-SET(v)
    sort edges by weight
    for each edge (u, v) in sorted order:
        if FIND-SET(u) != FIND-SET(v):
            A = A ∪ {(u, v)}
            UNION(u, v)
    return A
```

#### Prim's Algorithm
```
PRIM(G, r):
    for each vertex v in G.V:
        v.key = ∞
        v.π = NIL
    r.key = 0
    Q = G.V
    while Q is not empty:
        u = EXTRACT-MIN(Q)
        for each vertex v in G.Adj[u]:
            if v in Q and w(u, v) < v.key:
                v.π = u
                v.key = w(u, v)
```

## 3. Planar Graph Theory

### Planar Graphs
**Definition**: A graph is planar if it can be drawn in the plane without edge crossings.

### Euler's Formula
For a connected planar graph with V vertices, E edges, and F faces:
V - E + F = 2

### Kuratowski's Theorem
A graph is planar if and only if it doesn't contain K₅ or K_{3,3} as a minor.

### Planarity Testing

#### Hopcroft-Tarjan Algorithm
Linear-time algorithm for testing planarity.

**Time Complexity**: O(V)

### Applications
- **Circuit design**: VLSI layout
- **Map coloring**: Geographic maps
- **Network topology**: Network design
- **Computer graphics**: 3D modeling

## 4. Graph Coloring Algorithms

### Graph Coloring
**Definition**: A k-coloring of a graph assigns k colors to vertices so that adjacent vertices have different colors.

### Chromatic Number
The minimum number of colors needed to color a graph.

### Coloring Algorithms

#### Greedy Coloring
```
GREEDY-COLORING(G):
    for each vertex v in G:
        color[v] = 0
    for each vertex v in G:
        used = set of colors used by neighbors
        color[v] = smallest color not in used
```

#### Welsh-Powell Algorithm
```
WELSH-POWELL(G):
    sort vertices by degree in decreasing order
    for each vertex v in sorted order:
        color[v] = smallest available color
```

### Advanced Coloring

#### Kempe Chains
Technique for proving the four-color theorem.

#### Edge Coloring
Coloring edges so that adjacent edges have different colors.

### Applications
- **Scheduling**: Resource allocation
- **Register allocation**: Compiler optimization
- **Map coloring**: Geographic maps
- **Network optimization**: Channel assignment

## 5. Random Graphs

### Erdős-Rényi Model
**Definition**: G(n, p) is a random graph with n vertices where each edge appears with probability p.

### Properties
- **Threshold functions**: Properties that appear suddenly
- **Phase transitions**: Sharp changes in graph properties
- **Giant component**: Large connected component

### Random Graph Theorems

#### Connectivity Threshold
For p = (log n + c)/n, the probability that G(n, p) is connected approaches e^(-e^(-c)) as n → ∞.

#### Giant Component
For p = c/n with c > 1, G(n, p) has a giant component of size Θ(n).

### Applications
- **Network analysis**: Social networks, internet
- **Algorithm analysis**: Average-case complexity
- **Cryptography**: Random graph-based protocols
- **Biology**: Protein interaction networks

## 6. Spectral Graph Theory

### Graph Laplacian
**Definition**: The Laplacian matrix L = D - A where D is the degree matrix and A is the adjacency matrix.

### Eigenvalues and Eigenvectors
The eigenvalues of the Laplacian provide information about graph structure.

### Properties
- **First eigenvalue**: λ₁ = 0 (always)
- **Second eigenvalue**: λ₂ > 0 if and only if the graph is connected
- **Algebraic connectivity**: λ₂ measures graph connectivity

### Applications
- **Graph partitioning**: Spectral clustering
- **Network analysis**: Community detection
- **Machine learning**: Graph-based learning
- **Computer vision**: Image segmentation

## 7. Practice Problems

### Network Flows
1. Find the maximum flow in the following network:
   - Vertices: {s, a, b, c, t}
   - Edges with capacities: (s, a): 10, (s, b): 5, (a, b): 3, (a, c): 8, (b, c): 2, (c, t): 10

2. Implement the Ford-Fulkerson algorithm.

3. Find the minimum cut for the network in problem 1.

### Advanced Algorithms
4. Implement Floyd-Warshall algorithm for all-pairs shortest paths.

5. Find strongly connected components in the graph:
   - Vertices: {1, 2, 3, 4, 5, 6}
   - Edges: {(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)}

6. Compare Kruskal's and Prim's algorithms for finding MST.

### Planar Graphs
7. Determine if the following graph is planar:
   - K₅ (complete graph on 5 vertices)
   - K_{3,3} (complete bipartite graph)

8. Use Euler's formula to find the number of faces in a planar graph with 8 vertices and 12 edges.

9. Test planarity of the graph with edges: {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 4), (2, 5), (3, 6)}

### Graph Coloring
10. Find the chromatic number of:
    - Complete graph K₅
    - Cycle C₆
    - Petersen graph

11. Implement the greedy coloring algorithm.

12. Use Kempe chains to prove that the Petersen graph is not 3-colorable.

### Random Graphs
13. Generate a random graph G(100, 0.1) and analyze its properties.

14. Find the probability that G(10, 0.5) is connected.

15. Analyze the giant component in G(1000, 0.01).

### Spectral Graph Theory
16. Find the eigenvalues of the Laplacian matrix for:
    - Complete graph K₃
    - Cycle C₄
    - Path P₃

17. Use spectral clustering to partition a graph.

18. Analyze the algebraic connectivity of different graph structures.

## 8. Applications

### Computer Science
- **Network algorithms**: Routing, flow optimization
- **Graph algorithms**: Shortest paths, spanning trees
- **Planar algorithms**: VLSI design, circuit layout
- **Coloring algorithms**: Resource allocation, scheduling

### Network Analysis
- **Social networks**: Community detection, influence analysis
- **Internet topology**: Routing, congestion control
- **Biological networks**: Protein interactions, gene regulation
- **Transportation**: Traffic flow, route optimization

### Machine Learning
- **Graph neural networks**: Node classification, link prediction
- **Spectral clustering**: Data clustering, dimensionality reduction
- **Random graphs**: Model validation, algorithm testing
- **Graph embeddings**: Representation learning

## Key Takeaways

1. **Network flows are powerful**: They solve many optimization problems
2. **Advanced algorithms are efficient**: They handle large-scale problems
3. **Planar graphs are special**: They have unique properties and applications
4. **Graph coloring is challenging**: It's related to many optimization problems
5. **Random graphs model reality**: They capture real-world network properties
6. **Spectral theory is deep**: It connects graphs to linear algebra
7. **Applications are everywhere**: From networks to machine learning

## Next Steps
- Master network flow algorithms
- Implement advanced graph algorithms
- Study planar graph theory
- Learn graph coloring techniques
- Explore random graph models
- Apply spectral graph theory
- Connect advanced graph theory to real-world problems
