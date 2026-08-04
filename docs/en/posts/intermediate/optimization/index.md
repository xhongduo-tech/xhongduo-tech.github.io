---
pageClass: plain-doc
---

# Optimization Theory

Benchmarked against courses on Optimization Methods and the introductory chapters of Boyd's *Convex Optimization*, this track starts from the foundations of convex analysis and systematically covers classical algorithms for linear programming, unconstrained and constrained optimization, extending to introductory material on integer programming, multi-objective, stochastic, and global optimization. Upon completing this track, you will be equipped to read the optimization literature and to model and solve real-world problems.

## Topic Plan

<ProgressGrid cat="intermediate/optimization" />


### Part 1 · Convex Sets and Convex Functions

- [ ] Affine and convex sets: definitions and basic examples
- [ ] Convex combinations, convex hulls, and convex cones
- [ ] Hyperplanes, half-spaces, and polyhedra
- [ ] Operations preserving convexity: affine transformations, perspective functions, and linear-fractional functions
- [ ] The separating hyperplane theorem and the supporting hyperplane theorem
- [ ] Definition of convex functions and equivalent characterizations
- [ ] First-order and second-order conditions for convexity
- [ ] The epigraph and lower-level sets
- [ ] Jensen's inequality and its consequences
- [ ] Examples of common convex and concave functions
- [ ] Operations preserving convexity: nonnegative weighted sums, pointwise maximum, and composition rules
- [ ] The conjugate function
- [ ] A first look at quasiconvex functions

### Part 2 · Convex Optimization Problems

- [ ] Standard form of an optimization problem: objective, constraints, and optimal value
- [ ] Definition of a convex optimization problem and why a local optimum is a global optimum
- [ ] Equivalent transformations: eliminating equality constraints, introducing slack variables, epigraph form
- [ ] Standard form of linear programming (LP) and common modeling patterns
- [ ] Quadratic programming (QP) and quadratically constrained quadratic programming (QCQP)
- [ ] A first look at second-order cone programming (SOCP)
- [ ] A first look at semidefinite programming (SDP)
- [ ] A first look at geometric programming (GP)
- [ ] Modeling examples: least squares, regularization, and robust optimization

### Part 3 · Duality Theory and Optimality Conditions

- [ ] The Lagrangian and the Lagrange dual function
- [ ] Concavity of the dual function and the lower bound on the optimal value
- [ ] The Lagrange dual problem
- [ ] Weak duality and strong duality
- [ ] Slater's constraint qualification
- [ ] Complementary slackness conditions
- [ ] KKT conditions: necessity and sufficiency
- [ ] Sensitivity analysis from the dual perspective
- [ ] Farkas' lemma and theorems of the alternative

### Part 4 · Linear Programming

- [ ] Geometric intuition for linear programming: feasible region, vertices, and optimal solutions
- [ ] The one-to-one correspondence between bases, basic feasible solutions, and vertices
- [ ] The basic idea of the simplex method: basis-exchange iteration
- [ ] Constructing the simplex tableau and the pivot operation
- [ ] Degeneracy, cycling, and Bland's rule
- [ ] The two-phase method and the big-M method: finding an initial basic feasible solution
- [ ] The dual of a linear program and the dual simplex method
- [ ] Shadow prices and sensitivity analysis
- [ ] An overview of interior-point methods: the central path and primal-dual algorithms
- [ ] The ellipsoid method and polynomial solvability of linear programming

### Part 5 · Unconstrained Optimization

- [ ] First-order and second-order optimality conditions for unconstrained problems
- [ ] Descent directions and the general framework of iterative algorithms
- [ ] Line search: exact line search and the Armijo backtracking rule
- [ ] Wolfe conditions and strong Wolfe conditions
- [ ] Gradient descent and its convergence analysis
- [ ] Steepest descent and the idea of preconditioning
- [ ] Newton's method: derivation, local quadratic convergence, and damped Newton's method
- [ ] Quasi-Newton methods: the secant condition and the DFP formula
- [ ] The BFGS formula and limited-memory L-BFGS
- [ ] Conjugate gradient methods: linear conjugate gradient
- [ ] Nonlinear conjugate gradient: the FR and PRP formulas
- [ ] Trust-region methods and the Levenberg–Marquardt algorithm

### Part 6 · Constrained Optimization

- [ ] The method of Lagrange multipliers for equality-constrained problems
- [ ] Inequality constraints and the geometric interpretation of the KKT conditions
- [ ] The exterior penalty method
- [ ] The interior-point barrier method (log barrier method)
- [ ] The augmented Lagrangian method and the method of multipliers
- [ ] A first look at the alternating direction method of multipliers (ADMM)
- [ ] Projected gradient methods and proximal operators
- [ ] A first look at sequential quadratic programming (SQP)

### Part 7 · Quadratic Programming

- [ ] Standard form of quadratic programming and detecting convexity
- [ ] Analytic solutions to equality-constrained quadratic programs
- [ ] The active-set method
- [ ] Interior-point methods for quadratic programming
- [ ] Typical applications of quadratic programming: SVMs and portfolio optimization

### Part 8 · Integer Programming and Combinatorial Optimization

- [ ] Integer programming modeling: 0-1 variables and logical constraints
- [ ] Linear relaxation and the integrality gap
- [ ] Branch and bound
- [ ] Cutting-plane methods and Gomory cuts
- [ ] The knapsack problem and the assignment problem
- [ ] Classical problems on graphs: shortest paths, maximum flow, and minimum cut
- [ ] Matching problems and the Hungarian algorithm

### Part 9 · Multi-Objective Optimization

- [ ] Pareto optimality and the efficient frontier
- [ ] The weighted-sum method and its limitations
- [ ] The ε-constraint method
- [ ] A first look at goal programming

### Part 10 · Stochastic Optimization

- [ ] Stochastic approximation and the Robbins–Monro algorithm
- [ ] Stochastic gradient descent (SGD): from full-batch gradients to random sampling
- [ ] Step-size rules and convergence analysis for SGD
- [ ] Mini-batches, momentum, and the idea of variance reduction (SVRG)
- [ ] Expectation constraints and a first look at stochastic programming

### Part 11 · Global Optimization and Heuristic Algorithms

- [ ] The difficulty of global optimization and its NP-hard background
- [ ] Simulated annealing: the Metropolis criterion and annealing schedules
- [ ] Genetic algorithms: encoding, selection, crossover, and mutation
- [ ] Tabu search and local search strategies
- [ ] A first look at particle swarm optimization (PSO)
- [ ] Evaluating heuristic algorithms: convergence guarantees and experimental comparison

> After writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
