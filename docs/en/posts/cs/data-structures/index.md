---
pageClass: plain-doc
---

# Data Structures

Finishing data structures = writing blog posts for every chapter of Yan Weimin's *Data Structures (C Language Edition)*, plus rounding out the common algorithm topics. Each post corresponds to one section of the textbook; check off an item once its post is written.

## Topic Plan

<ProgressGrid cat="cs/data-structures" />


### Part 1 · Introduction

- [ ] What is a data structure
- [ ] Basic concepts and terminology
- [ ] Representation and implementation of abstract data types
- [ ] Algorithms and algorithm analysis
- [ ] How to compute time complexity and space complexity

### Part 2 · Linear Lists

- [ ] Type definition of a linear list
- [ ] Sequential representation and implementation of a linear list
- [ ] Linked representation and implementation of a linear list (singly linked list)
- [ ] Circular linked lists and doubly linked lists
- [ ] Representation of a polynomial in one variable and its addition

### Part 3 · Stacks and Queues

- [ ] Definition of a stack and implementation of a sequential stack
- [ ] Linked representation of a stack
- [ ] Stacks and recursion (Tower of Hanoi)
- [ ] Evaluating expressions
- [ ] Sequential representation of a queue and the circular queue
- [ ] Linked queues
- [ ] Deques

### Part 4 · Strings

- [ ] Type definition of a string
- [ ] Representation and implementation of strings (fixed-length sequential storage, heap-allocated storage, block-chain storage)
- [ ] String pattern-matching algorithms (naive matching)
- [ ] Example applications of string operations (text editing)

### Part 5 · Arrays and Generalized Lists

- [ ] Definition and sequential representation of arrays
- [ ] Compressed storage of matrices (special matrices)
- [ ] Triple-table sequential representation of sparse matrices
- [ ] Orthogonal-list representation of sparse matrices
- [ ] Definition of generalized lists
- [ ] Storage structure of generalized lists
- [ ] Representation of polynomials in m variables and recursive algorithms on generalized lists

### Part 6 · Trees and Binary Trees

- [ ] Definition of trees and basic terminology
- [ ] Definition and properties of binary trees
- [ ] Storage structure of binary trees
- [ ] Traversing binary trees (preorder, inorder, postorder, level-order)
- [ ] Threaded binary trees
- [ ] Trees and forests (storage structure and conversion to/from binary trees)
- [ ] Traversing trees and forests
- [ ] Huffman trees and their applications
- [ ] Huffman coding
- [ ] Backtracking and tree traversal (the eight queens problem)
- [ ] Counting trees

### Part 7 · Graphs

- [ ] Definition of graphs and terminology
- [ ] Storage structure of graphs (array representation, adjacency lists)
- [ ] Orthogonal lists and adjacency multilists
- [ ] Graph traversal (depth-first search)
- [ ] Graph traversal (breadth-first search)
- [ ] Connectivity problems in graphs (connected components and spanning trees of undirected graphs)
- [ ] Directed acyclic graphs and their applications (topological sorting)
- [ ] Critical paths
- [ ] Shortest paths (Dijkstra's algorithm)
- [ ] Shortest paths (Floyd's algorithm)
- [ ] Minimum spanning trees (Prim's algorithm)
- [ ] Minimum spanning trees (Kruskal's algorithm)

### Part 8 · Dynamic Storage Management

- [ ] Overview of dynamic storage management
- [ ] Available-space lists and allocation methods
- [ ] Boundary-tag method
- [ ] Buddy system
- [ ] Garbage collection of unused cells
- [ ] Storage compaction

### Part 9 · Searching

- [ ] Static search tables (sequential search)
- [ ] Binary search on ordered lists
- [ ] Indexed sequential tables (block search)
- [ ] Binary search trees
- [ ] Balanced binary trees
- [ ] B-trees and B+ trees
- [ ] Key trees (tries)
- [ ] Basic concepts of hash tables and constructing hash functions
- [ ] Methods for resolving hash collisions
- [ ] Searching a hash table and its analysis

### Part 10 · Internal Sorting

- [ ] Sorting overview and stability
- [ ] Straight insertion sort and binary insertion sort
- [ ] Shell sort
- [ ] Bubble sort
- [ ] Quicksort
- [ ] Simple selection sort
- [ ] Tree selection sort and heap sort
- [ ] Merge sort
- [ ] Radix sort
- [ ] Comparison and discussion of internal sorting methods

### Part 11 · External Sorting

- [ ] Accessing information in external storage
- [ ] Methods of external sorting
- [ ] Implementation of multiway balanced merging
- [ ] Replacement-selection sort
- [ ] Optimal merge trees

### Part 12 · Files

- [ ] Basic concepts of files
- [ ] Sequential files
- [ ] Indexed files
- [ ] ISAM files and VSAM files
- [ ] Direct-access files (hash files)
- [ ] Multi-key files (multilist files, inverted files)

### Topic · Hashing

- [ ] Consistent hashing and applications in distributed scenarios
- [ ] Bloom filters
- [ ] Engineering implementation of hash tables (open addressing vs. separate chaining, resizing and load factor)

### Topic · Disjoint-Set Union

- [ ] Basic implementation of the union-find and path compression
- [ ] Union by rank and complexity analysis
- [ ] Weighted union-find and species union-find
- [ ] Classic applications of union-find (connectivity checking, use in Kruskal's algorithm)

### Topic · Heaps and Priority Queues

- [ ] Sift-up and sift-down in a binary heap
- [ ] Heap sort and complexity analysis of heap construction
- [ ] Typical applications of priority queues (Top K, merging K sorted linked lists)
- [ ] Two-heap trick for the dynamic median

### Topic · Balanced Trees

- [ ] AVL tree rotations and balance maintenance
- [ ] Properties of red-black trees and insertion adjustments
- [ ] Deletion adjustments in red-black trees
- [ ] Engineering applications of red-black trees (TreeMap, epoll, process scheduling)

### Topic · Skip Lists

- [ ] How skip lists work and randomized levels
- [ ] Implementing skip lists and complexity analysis
- [ ] Skip lists vs. balanced trees (why Redis chose skip lists)

### Topic · B/B+ Trees

- [ ] Definition of B-trees and insertion with splitting
- [ ] Deletion from B-trees and merging
- [ ] B+ trees and database indexing
- [ ] How B+ trees are implemented in MySQL InnoDB

### Topic · Segment Trees and Fenwick Trees

- [ ] How Fenwick trees work and range summation
- [ ] Range updates with Fenwick trees and counting inversions
- [ ] Building a segment tree, point updates, and range queries
- [ ] Lazy propagation and range updates
- [ ] Typical applications of segment trees (range max/min, sweep line)

### Topic · Tries and String Matching

- [ ] Building and querying a Trie
- [ ] Applications of Tries (prefix statistics, maximum XOR pair)
- [ ] The KMP algorithm and the next array
- [ ] Correctness proof of KMP and complexity analysis
- [ ] String hashing and Rabin-Karp

### Topic · Monotonic Stacks and Monotonic Queues

- [ ] Monotonic stacks and next greater element
- [ ] Applications of monotonic stacks (largest rectangle in a histogram, trapping rain water)
- [ ] Monotonic queues and the sliding-window maximum
- [ ] An introduction to optimizing dynamic programming with monotonic queues

### Topic · Advanced Graph Algorithms

- [ ] Bellman-Ford and SPFA
- [ ] Difference constraint systems
- [ ] Second-shortest path and K-shortest paths
- [ ] Tarjan's algorithm for strongly connected components
- [ ] Articulation points and bridges
- [ ] Bipartite matching (the Hungarian algorithm)
- [ ] Network flow basics (max flow and Ford-Fulkerson)
- [ ] The Dinic algorithm and minimum cuts

> After writing a post: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [title](./xxx)`.
