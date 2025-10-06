# Graph Theory

## Overview
Graph theory is the study of graphs, which are mathematical structures used to model pairwise relations between objects. Graphs are fundamental in computer science, network analysis, optimization, and many other fields. This chapter covers the essential concepts and algorithms in graph theory.

## Learning Objectives
- Understand graphs and their representations
- Learn about paths, cycles, and connectivity
- Master trees and spanning trees
- Understand graph coloring
- Learn about planar graphs
- Study Eulerian and Hamiltonian paths
- Apply graph algorithms to solve problems

## 1. Basic Concepts

### Definition of a Graph
A **graph** G = (V, E) consists of:
- **Vertices (nodes)**: V = {v₁, v₂, ..., v_n}
- **Edges**: E = {e₁, e₂, ..., e_m} where each edge connects two vertices

### Types of Graphs

#### Undirected Graph
Edges have no direction. If (u, v) is an edge, then (v, u) is the same edge.

#### Directed Graph (Digraph)
Edges have direction. (u, v) and (v, u) are different edges.

#### Weighted Graph
Edges have weights (costs, distances, etc.).

#### Simple Graph
No loops (edges from a vertex to itself) and no multiple edges between the same pair of vertices.

### Graph Terminology

#### Degree
- **Degree of a vertex**: Number of edges incident to it
- **In-degree**: Number of edges entering a vertex (directed graphs)
- **Out-degree**: Number of edges leaving a vertex (directed graphs)

#### Adjacency
- **Adjacent vertices**: Connected by an edge
- **Incident edge**: An edge that connects to a vertex

#### Path
A sequence of vertices v₁, v₂, ..., v_k where (v_i, v_{i+1}) is an edge for all i.

#### Cycle
A path that starts and ends at the same vertex.

#### Connected Graph
There's a path between any two vertices.

## 2. Graph Representations

### Adjacency Matrix
A matrix A where A[i][j] = 1 if there's an edge from vertex i to vertex j, 0 otherwise.

**Example**: For graph with vertices {1, 2, 3, 4} and edges {(1,2), (2,3), (3,4), (4,1)}:
```
  1 2 3 4
1 0 1 0 0
2 0 0 1 0
3 0 0 0 1
4 1 0 0 0
```

### Adjacency List
For each vertex, maintain a list of adjacent vertices.

**Example**: Same graph as above:
```
1: [2]
2: [3]
3: [4]
4: [1]
```

### Edge List
A list of all edges.

**Example**: [(1,2), (2,3), (3,4), (4,1)]

## 3. Connectivity

### Connected Components
A **connected component** is a maximal connected subgraph.

### Strongly Connected Components (Directed Graphs)
A **strongly connected component** is a maximal subgraph where there's a path from any vertex to any other vertex.

### Cut Vertices and Cut Edges
- **Cut vertex**: Removing it increases the number of connected components
- **Cut edge**: Removing it increases the number of connected components

## 4. Trees

### Definition
A **tree** is a connected graph with no cycles.

### Properties of Trees
1. There's exactly one path between any two vertices
2. A tree with n vertices has n-1 edges
3. Adding any edge creates exactly one cycle
4. Removing any edge disconnects the graph

### Rooted Trees
A tree with one vertex designated as the root.

#### Terminology
- **Parent**: Vertex connected to current vertex on path to root
- **Child**: Vertex connected to current vertex away from root
- **Leaf**: Vertex with no children
- **Height**: Length of longest path from root to leaf
- **Depth**: Length of path from root to current vertex

### Binary Trees
A tree where each vertex has at most two children.

#### Types
- **Complete binary tree**: All levels filled except possibly the last
- **Full binary tree**: Every vertex has 0 or 2 children
- **Perfect binary tree**: All levels completely filled

## 5. Spanning Trees

### Definition
A **spanning tree** of a connected graph G is a subgraph that:
- Contains all vertices of G
- Is a tree (connected with no cycles)

### Minimum Spanning Tree (MST)
A spanning tree with minimum total edge weight.

#### Kruskal's Algorithm
1. Sort edges by weight
2. Add edges in order, skipping those that create cycles
3. Stop when n-1 edges are added

#### Prim's Algorithm
1. Start with any vertex
2. Add minimum weight edge that connects to current tree
3. Repeat until all vertices are included

## 6. Graph Coloring

### Definition
A **coloring** of a graph assigns colors to vertices so that adjacent vertices have different colors.

### Chromatic Number
The minimum number of colors needed to color a graph.

### Examples
- **Complete graph K_n**: χ(K_n) = n
- **Bipartite graph**: χ(G) = 2
- **Cycle C_n**: χ(C_n) = 2 if n is even, 3 if n is odd
- **Tree**: χ(T) = 2

### Greedy Coloring Algorithm
1. Order vertices arbitrarily
2. For each vertex, assign the smallest available color
3. A color is available if no adjacent vertex has that color

## 7. Planar Graphs

### Definition
A **planar graph** can be drawn in the plane without edge crossings.

### Euler's Formula
For a connected planar graph with V vertices, E edges, and F faces:
V - E + F = 2

### Kuratowski's Theorem
A graph is planar if and only if it doesn't contain K₅ or K_{3,3} as a minor.

### Planarity Testing
- **Kuratowski's theorem**: Check for K₅ and K_{3,3} minors
- **Planarity algorithms**: Use depth-first search and face detection

## 8. Eulerian and Hamiltonian Paths

### Eulerian Path
A path that uses every edge exactly once.

### Eulerian Circuit
An Eulerian path that starts and ends at the same vertex.

### Euler's Theorem
A connected graph has an Eulerian circuit if and only if every vertex has even degree.

### Hamiltonian Path
A path that visits every vertex exactly once.

### Hamiltonian Cycle
A Hamiltonian path that starts and ends at the same vertex.

### Note
There's no simple characterization for Hamiltonian paths/cycles like there is for Eulerian paths.

## 9. Graph Algorithms

### Breadth-First Search (BFS)
```
BFS(G, s):
    for each vertex v in G:
        color[v] = WHITE
        d[v] = ∞
        π[v] = NIL
    color[s] = GRAY
    d[s] = 0
    Q = {s}
    while Q is not empty:
        u = DEQUEUE(Q)
        for each v in Adj[u]:
            if color[v] == WHITE:
                color[v] = GRAY
                d[v] = d[u] + 1
                π[v] = u
                ENQUEUE(Q, v)
        color[u] = BLACK
```

### Depth-First Search (DFS)
```
DFS(G):
    for each vertex v in G:
        color[v] = WHITE
        π[v] = NIL
    time = 0
    for each vertex v in G:
        if color[v] == WHITE:
            DFS-VISIT(G, v)

DFS-VISIT(G, u):
    time = time + 1
    d[u] = time
    color[u] = GRAY
    for each v in Adj[u]:
        if color[v] == WHITE:
            π[v] = u
            DFS-VISIT(G, v)
    color[u] = BLACK
    time = time + 1
    f[u] = time
```

### Shortest Path Algorithms

#### Dijkstra's Algorithm
Finds shortest paths from a source vertex to all other vertices in a weighted graph with non-negative weights.

#### Floyd-Warshall Algorithm
Finds shortest paths between all pairs of vertices.

## 10. Practice Problems

### Basic Concepts
1. Draw all non-isomorphic graphs with 4 vertices.

2. For a graph with n vertices, what's the maximum number of edges in:
   - A simple undirected graph?
   - A simple directed graph?

### Connectivity
3. Find all connected components of the graph with edges: {(1,2), (2,3), (4,5), (6,7), (7,8)}.

4. Determine if the following graph is connected:
   - Vertices: {1, 2, 3, 4, 5}
   - Edges: {(1,2), (2,3), (3,4), (4,5), (5,1)}

### Trees
5. Prove that a tree with n vertices has n-1 edges.

6. Find the number of leaves in a binary tree with 15 internal vertices.

### Spanning Trees
7. Find a minimum spanning tree for the graph with edges and weights:
   - (1,2): 3, (1,3): 1, (2,3): 2, (2,4): 4, (3,4): 5

8. Apply Kruskal's algorithm to find the MST.

### Graph Coloring
9. Find the chromatic number of:
   - Complete graph K₅
   - Cycle C₆
   - Bipartite graph K_{3,3}

10. Color the following graph using the greedy algorithm:
    - Vertices: {1, 2, 3, 4, 5}
    - Edges: {(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)}

### Eulerian and Hamiltonian
11. Determine if the following graph has an Eulerian circuit:
    - Vertices: {1, 2, 3, 4, 5}
    - Edges: {(1,2), (2,3), (3,4), (4,5), (5,1), (1,3), (2,4)}

12. Find a Hamiltonian cycle in the complete graph K₅.

## 11. Applications

### Computer Science
- **Network routing**: Shortest path algorithms
- **Social networks**: Community detection
- **Web graphs**: PageRank algorithm
- **Compiler design**: Control flow graphs

### Operations Research
- **Transportation**: Network optimization
- **Scheduling**: Task dependencies
- **Resource allocation**: Assignment problems
- **Project management**: Critical path method

### Biology
- **Protein networks**: Protein-protein interactions
- **Phylogenetic trees**: Evolutionary relationships
- **Neural networks**: Brain connectivity
- **Ecosystems**: Food webs

## Key Takeaways

1. **Graphs model relationships**: They're powerful for representing complex systems
2. **Connectivity is fundamental**: Understanding paths and cycles is crucial
3. **Trees are special**: They have unique properties and applications
4. **Algorithms matter**: BFS, DFS, and shortest path algorithms are essential
5. **Coloring is challenging**: It's related to many optimization problems
6. **Planarity is important**: It affects algorithm complexity and visualization
7. **Applications are everywhere**: From computer networks to social systems

## Next Steps
- Master basic graph concepts and terminology
- Practice with graph algorithms (BFS, DFS)
- Learn about trees and spanning trees
- Study graph coloring and planarity
- Apply graph theory to solve real-world problems
- Connect graph theory to algorithms and data structures
