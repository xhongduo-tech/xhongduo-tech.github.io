---
pageClass: plain-doc
---

# Algorithm Design & Analysis

Guided by *Introduction to Algorithms* (CLRS), this track covers the classic scope of algorithm design and analysis: from asymptotic notation, divide-and-conquer and randomization, to dynamic programming, greedy algorithms, graph algorithms, NP-completeness, and onward to approximation algorithms, string matching, computational geometry, number-theoretic algorithms, and parallel and external-memory algorithms.

## Topic Plan

<ProgressGrid cat="cs/algorithms" />


### Part 1: Algorithm Foundations and Asymptotic Notation

- [ ] The definition and properties of algorithms, and their standing as a technology
- [ ] Insertion Sort: loop invariants and correctness proofs
- [ ] The framework of algorithm analysis: worst case, average case, and asymptotic efficiency
- [ ] Asymptotic Notation: the rigorous definitions and properties of Θ, O, Ω, o, and ω
- [ ] Growth-rate comparison of common functions and summation techniques (series, integral approximation, harmonic numbers)

### Part 2: Divide and Conquer and Recurrences

- [ ] Merge Sort: the divide-and-conquer paradigm and recursion-tree analysis
- [ ] The Substitution Method for solving recurrences
- [ ] The Recursion-Tree Method: expansion and summation
- [ ] The Master Theorem: its three cases and discussion of the gap between them
- [ ] A classic divide-and-conquer case: the maximum subarray problem
- [ ] A classic divide-and-conquer case: matrix multiplication (Strassen's algorithm)

### Part 3: Randomized Algorithms

- [ ] Basic concepts of randomized algorithms: the hiring problem and indicator random variables
- [ ] Random permutation algorithms: sorting by random priorities and in-place shuffling (Fisher–Yates)
- [ ] Advanced probabilistic analysis: balls and bins
- [ ] The Birthday Paradox and its algorithmic implications
- [ ] Expected-time analysis of randomized quicksort

### Part 4: Heapsort and Quicksort

- [ ] Binary Heaps: maintaining the heap structure and HEAPIFY
- [ ] Heapsort: the O(n) analysis of building a heap
- [ ] Priority queues: max-heaps, min-heaps, and d-ary heaps
- [ ] Quicksort: the partitioning procedure and its correctness
- [ ] Performance analysis of quicksort: worst case and balanced partitions
- [ ] Proof of the expected running time of randomized quicksort
- [ ] Engineering optimizations for quicksort: median-of-three, three-way partitioning, insertion sort for small arrays

### Part 5: Sorting in Linear Time

- [ ] The lower bound for comparison sorting: the decision-tree model and the Ω(n log n) proof
- [ ] Counting Sort: stability and conditions for applicability
- [ ] Radix Sort: least-significant-digit-first and the correctness induction
- [ ] Bucket Sort: expected analysis under the uniform-distribution assumption

### Part 6: Medians and Order Statistics

- [ ] Minimum and maximum: the exact lower bound on the number of comparisons
- [ ] Selection in expected linear time: RANDOMIZED-SELECT
- [ ] Selection in worst-case linear time: SELECT and the Median of Medians
- [ ] Applications of order statistics: weighted median and the nearest post-office problem

### Part 7: Dynamic Programming

- [ ] Principles of dynamic programming: optimal substructure and overlapping subproblems
- [ ] Rod Cutting: top-down with memoization and bottom-up approaches
- [ ] Matrix-Chain Multiplication: the optimal order of parenthesization
- [ ] Longest Common Subsequence: the recurrence and backtracking reconstruction
- [ ] A variant of LCS: Edit Distance
- [ ] Optimal BST: minimizing expected search cost
- [ ] The 0-1 Knapsack problem: pseudo-polynomial time and its distinction from fully polynomial
- [ ] Space optimization for 0-1 Knapsack and solution backtracking
- [ ] Fractional Knapsack: its greedy nature and the fundamental difference from 0-1 Knapsack
- [ ] Longest Increasing Subsequence (LIS): the O(n²) DP and the O(n log n) optimization

### Part 8: Greedy Algorithms

- [ ] Theoretical foundations of greedy algorithms: the greedy-choice property and optimal substructure
- [ ] Activity-Selection: proof of greedy choice by sorting by finish time
- [ ] Huffman Coding: construction and correctness of optimal prefix codes
- [ ] Matroids: an abstract framework for greedy theory
- [ ] Task scheduling through the matroid lens: scheduling unit-time tasks with deadlines

### Part 9: Amortized Analysis

- [ ] Introduction to amortized analysis: the aggregate method and stack operations
- [ ] The Accounting Method: credit assignment in binary counters
- [ ] The Potential Method: analysis of dynamic-table expansion and contraction
- [ ] Dynamic Tables: amortized costs of insertion and deletion
- [ ] Amortized analysis vs. average-case analysis

### Part 10: Advanced Graph Algorithms

- [ ] Review of graph representation and traversal: BFS, DFS, and their properties
- [ ] Topological Sort: the proof using reverse order of DFS finish times
- [ ] Strongly Connected Components: Kosaraju's algorithm and the component graph
- [ ] Single-source shortest paths: Bellman-Ford and shortest paths on DAGs
- [ ] All-pairs shortest paths: Floyd-Warshall and Johnson's algorithm
- [ ] Maximum Flow: flow networks, residual networks, and augmenting paths
- [ ] The Max-Flow Min-Cut Theorem and its corollaries
- [ ] The Ford-Fulkerson method and its variants: the Edmonds-Karp algorithm
- [ ] Maximum Bipartite Matching: reduction to maximum flow
- [ ] An introduction to the Push-Relabel algorithm

### Part 11: NP-Completeness

- [ ] Polynomial time: the class P and the influence of encodings
- [ ] Polynomial-time verification: the class NP and certificates
- [ ] Reductions: polynomial-time reductions and proofs of the lemma
- [ ] NP-Completeness: definition, properties, and the Cook-Levin theorem
- [ ] A template for NP-completeness proofs: the Vertex Cover example
- [ ] Classic NP-complete problems: Hamiltonian Cycle
- [ ] Classic NP-complete problems: the Traveling Salesman Problem (TSP)
- [ ] Classic NP-complete problems: Subset-Sum and 3-CNF satisfiability

### Part 12: Approximation Algorithms

- [ ] Performance ratios of approximation algorithms: absolute approximation ratios and relative error bounds
- [ ] A 2-approximation algorithm for the vertex cover problem
- [ ] Approximation algorithms for TSP: a 2-approximation when the triangle inequality holds
- [ ] Set Cover: the ln n approximation ratio of the greedy algorithm
- [ ] A fully polynomial-time approximation scheme (FPTAS) for Subset-Sum

### Part 13: Advanced String Matching

- [ ] Naive string matching and analysis of its shortcomings
- [ ] The Rabin-Karp algorithm: rolling hashes
- [ ] String matching with finite automata (DFA): constructing the transition function
- [ ] The KMP algorithm: the prefix function and linear-time matching
- [ ] The Boyer-Moore algorithm: the bad-character and good-suffix rules

### Part 14: Introduction to Computational Geometry

- [ ] Geometric properties of line segments: the cross product and determining direction
- [ ] Segment intersection testing: the straddling test
- [ ] Detecting intersections among arbitrary pairs of segments: the sweep-line algorithm
- [ ] Convex Hull: Graham's scan
- [ ] Convex Hull: Jarvis's march (gift wrapping)
- [ ] The closest-pair problem: divide-and-conquer and the O(n log n) analysis

### Part 15: Number-Theoretic Algorithms

- [ ] Elementary number-theoretic concepts: divisibility, the greatest common divisor, and Euclid's algorithm
- [ ] Modular arithmetic and the extended Euclidean algorithm: solving linear congruences
- [ ] The Chinese Remainder Theorem and modular exponentiation
- [ ] The RSA public-key cryptosystem: key generation, encryption, and correctness
- [ ] Primality testing: Fermat's little theorem and the Miller-Rabin test
- [ ] Integer factorization: Pollard's rho heuristic

### Part 16: Introduction to Parallel Algorithms

- [ ] The multithreaded computing model: dynamic multithreading, work, and span
- [ ] Parallel computation of Fibonacci numbers: performance measures and scheduling
- [ ] Parallel matrix multiplication and parallel merging
- [ ] The greedy scheduling theorem and race conditions in parallel algorithms

### Part 17: External-Memory Algorithms

- [ ] The external-memory model: I/O complexity and the cost of disk-block access
- [ ] External sorting: External Merge Sort
- [ ] B-trees: definition, properties, and a disk-friendly search structure
- [ ] Insertion into a B-tree: the split operation
- [ ] Deletion from a B-tree: the merge and borrow operations

> After writing is done: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
