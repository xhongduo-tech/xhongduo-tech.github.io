---
pageClass: plain-doc
---

# Discrete Mathematics

Finishing a discipline means writing out all the blog posts that correspond to its classic textbook. This category aligns with Kenneth H. Rosen's *Discrete Mathematics and Its Applications*, covering the complete body of knowledge from the foundations of logic to models of computation — each post corresponds to one section of the textbook.

## Topic Plan

<ProgressGrid cat="intermediate/discrete-math" />


### Part 1 · Logic and Proofs

- [ ] Propositional logic: propositions, logical connectives, and truth tables
- [ ] Propositional equivalences: logical equivalence and De Morgan's laws
- [ ] Predicates and quantifiers: universal quantifiers, existential quantifiers, and domains
- [ ] Nested quantifiers: the order and negation of quantifiers
- [ ] Rules of inference: valid arguments and common fallacies
- [ ] Rules of inference for propositional logic: modus ponens, modus tollens, and resolution
- [ ] Introduction to proofs: theorems, axioms, and conjectures
- [ ] Direct proofs, proof by contraposition, and proof by contradiction
- [ ] Methods of proof overview: proof by cases, existence proofs, and uniqueness proofs

### Part 2 · Sets

- [ ] Basic concepts of sets: membership, subsets, and the empty set
- [ ] Set operations: union, intersection, complement, difference, and symmetric difference
- [ ] Set identities and Venn diagrams
- [ ] Power sets and Cartesian products
- [ ] Partitions of sets and set operations with computer representations

### Part 3 · Functions

- [ ] The definition of a function: domain, codomain, and image
- [ ] Injective, surjective, and bijective functions (one-to-one, onto, and one-to-one correspondence)
- [ ] Inverse functions and function composition
- [ ] Floor and ceiling functions
- [ ] Cardinality and countability: countable sets and Cantor's diagonalization argument

### Part 4 · Sequences and Summations

- [ ] Sequences: arithmetic sequences, geometric sequences, and recursively defined sequences
- [ ] Summation notation and common summation formulas
- [ ] Double summations and telescoping sums

### Part 5 · Algorithms and Complexity

- [ ] The concept of algorithms and pseudocode descriptions
- [ ] Searching algorithms: linear search and binary search
- [ ] Sorting algorithms: bubble sort and insertion sort
- [ ] Greedy algorithms and correctness arguments
- [ ] Algorithm growth: big-O notation
- [ ] Big-Ω and big-Θ notation
- [ ] Algorithm complexity: analysis of time complexity and space complexity

### Part 6 · Number Theory and Cryptography

- [ ] Divisibility: divisibility, factors, and the division algorithm
- [ ] Modular arithmetic: congruences and their arithmetic properties
- [ ] Representations of integers: binary, hexadecimal, and base conversion
- [ ] Primes: definition, distribution, and primality testing
- [ ] Greatest common divisors and the Euclidean algorithm
- [ ] The Fundamental Theorem of Arithmetic and unique factorization
- [ ] The extended Euclidean algorithm and modular inverses
- [ ] Linear congruences and the Chinese remainder theorem
- [ ] Fermat's little theorem and Euler's theorem
- [ ] Introduction to cryptography: classical ciphers and the Caesar cipher
- [ ] Public-key cryptography and the RSA encryption algorithm

### Part 7 · Induction and Recursion

- [ ] Mathematical induction: the principle and basic proofs
- [ ] Strong induction and the well-ordering principle
- [ ] Recursive definitions: recursively defined functions and sequences
- [ ] Recursively defined sets and structural induction
- [ ] Recursive algorithms: design and correctness proofs
- [ ] Program correctness: preconditions, postconditions, and loop invariants

### Part 8 · Counting

- [ ] Basic counting principles: the product rule and the sum rule
- [ ] The subtraction rule (complement counting) and the division rule
- [ ] The pigeonhole principle and its generalizations
- [ ] Permutations: permutations without and with repetition
- [ ] Combinations: binomial coefficients
- [ ] The binomial theorem and its corollaries
- [ ] Pascal's identity and Vandermonde's identity
- [ ] Permutations and combinations with repetition: counting multisets
- [ ] Distributing objects in permutations and combinations
- [ ] The inclusion–exclusion principle: the two-set and three-set cases
- [ ] The general form of the inclusion–exclusion principle and derangements

### Part 9 · Advanced Counting Techniques

- [ ] Recurrence relations: modeling and applications
- [ ] Linear homogeneous recurrence relations: the constant-coefficient case and characteristic equations
- [ ] The case of repeated roots of characteristic equations
- [ ] Solving linear non-homogeneous recurrence relations
- [ ] Divide-and-conquer recurrence relations and the master theorem
- [ ] Generating functions: definition and basic operations
- [ ] Solving recurrence relations with generating functions
- [ ] Proving identities with generating functions
- [ ] The generalized binomial theorem and counting applications

### Part 10 · Relations

- [ ] Binary relations: definition and n-ary relations
- [ ] Properties of relations: reflexive, symmetric, antisymmetric, and transitive
- [ ] Operations on relations: composition and inverse relations
- [ ] Representing relations: matrices and directed graphs
- [ ] Closures of relations: reflexive, symmetric, and transitive closures
- [ ] Warshall's algorithm
- [ ] Equivalence relations and partitions
- [ ] Partial orders and Hasse diagrams
- [ ] Maximal elements, minimal elements, upper bounds, and lower bounds
- [ ] Lattices and topological sorting

### Part 11 · Graphs

- [ ] Basic concepts of graphs: graphs, directed graphs, and multigraphs
- [ ] Graph terminology: degrees, degree sequences, and the handshaking theorem
- [ ] Special graphs: complete graphs, cycles, wheels, and bipartite graphs
- [ ] Graph operations and subgraphs
- [ ] Representing graphs: adjacency matrices and incidence matrices
- [ ] Determining graph isomorphism
- [ ] Connectivity: paths, circuits, and connected components
- [ ] Euler paths and Euler circuits
- [ ] Hamilton paths and Hamilton circuits
- [ ] The shortest-path problem: Dijkstra's algorithm
- [ ] The traveling salesman problem
- [ ] Planar graphs: Euler's formula and Kuratowski's theorem
- [ ] Graph coloring: the four-color theorem and chromatic numbers

### Part 12 · Trees

- [ ] Basic concepts of trees: rooted trees, ordered rooted trees, and properties of trees
- [ ] m-ary trees and counting properties of trees
- [ ] Applications of trees: binary search trees and decision trees
- [ ] Prefix codes and Huffman coding
- [ ] Tree traversal: preorder, inorder, and postorder traversal
- [ ] Infix, prefix, and postfix notation
- [ ] Spanning trees: depth-first search and breadth-first search
- [ ] Backtracking and its applications
- [ ] Minimum spanning trees: Prim's algorithm and Kruskal's algorithm

### Part 13 · Boolean Algebra

- [ ] Boolean functions: Boolean operations and Boolean expressions
- [ ] Identities of Boolean algebra and the duality principle
- [ ] Representing Boolean functions: sum-of-products expansion (disjunctive normal form)
- [ ] Functional completeness: NAND and NOR gates
- [ ] Logic gate circuits: design of combinational circuits
- [ ] Simplification with Karnaugh maps
- [ ] The Quine–McCluskey method

### Part 14 · Models of Computation

- [ ] Languages and grammars: phrase-structure grammars and derivations
- [ ] Types of grammars: the Chomsky hierarchy
- [ ] Backus–Naur form (BNF)
- [ ] Finite-state machines: finite-state machines with output (Mealy and Moore machines)
- [ ] Finite-state machines without output and language recognition
- [ ] Finite-state automata and regular languages
- [ ] Non-deterministic finite-state automata and Kleene's theorem
- [ ] Turing machines: definition and computation
- [ ] Computability and the halting problem

> After a post is written: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [标题](./xxx)`.
