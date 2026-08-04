---
pageClass: plain-doc
---

# Numerical Analysis

Modeled on Li Qingyang's *Numerical Analysis*, covering the basic theory of numerical algorithms across nine core topics, written section by section.

## Topic Planning

<ProgressGrid cat="intermediate/numerical-analysis" />


### Errors and the Stability of Numerical Algorithms

- [ ] The research object and characteristics of numerical analysis
- [ ] Sources and classification of errors: model error, observation error, truncation error, rounding error
- [ ] Absolute error, relative error, and significant digits
- [ ] Error estimation in function evaluation
- [ ] Ill-conditioned problems and the condition number
- [ ] Numerical stability of algorithms
- [ ] Principles for avoiding the hazards of error: avoid subtracting nearly equal numbers, avoid small numbers being absorbed by large ones, reduce the number of operations
- [ ] Qin Jiushao's algorithm (Horner's algorithm) and polynomial evaluation

### Interpolation

- [ ] Statement of the interpolation problem and existence/uniqueness of polynomial interpolation
- [ ] Lagrange interpolation polynomial
- [ ] Interpolation remainder and error estimation
- [ ] Divided differences and Newton interpolation
- [ ] Finite differences and interpolation formulas with equally spaced nodes: Newton forward and backward interpolation formulas
- [ ] Hermite interpolation
- [ ] Piecewise low-degree interpolation: piecewise linear interpolation and piecewise cubic Hermite interpolation
- [ ] The Runge phenomenon and the limitations of high-degree interpolation
- [ ] Cubic spline interpolation: the three-moment equations
- [ ] Cubic spline interpolation: three-slope equations and boundary conditions

### Function Approximation and Curve Fitting

- [ ] Basic concepts of function approximation: norms and inner product spaces
- [ ] Orthogonal polynomials: Legendre polynomials
- [ ] Chebyshev polynomials and their properties
- [ ] Other commonly used orthogonal polynomials: Laguerre and Hermite polynomials
- [ ] Best uniform approximation polynomial
- [ ] Best square approximation
- [ ] Chebyshev interpolation and near-best approximation
- [ ] Curve fitting by the least squares method
- [ ] Least squares fitting with orthogonal polynomials
- [ ] Inconsistent systems of equations and the linear least squares problem

### Numerical Integration and Numerical Differentiation

- [ ] Basic ideas of numerical quadrature and algebraic precision
- [ ] Newton–Cotes formulas
- [ ] Remainder terms of the trapezoidal rule and Simpson's rule
- [ ] Composite quadrature formulas: composite trapezoidal and composite Simpson
- [ ] Error estimation for composite quadrature and the successive bisection method
- [ ] Romberg integration: Richardson extrapolation
- [ ] Gaussian quadrature and its construction
- [ ] Gauss–Legendre quadrature
- [ ] Gauss–Chebyshev quadrature and weighted Gaussian formulas
- [ ] Numerical differentiation: interpolation-type differentiation formulas and three-point formulas

### Direct Methods for Linear Systems

- [ ] Gaussian elimination
- [ ] The relation between the computational cost of Gaussian elimination and matrix triangular factorization
- [ ] Partial pivoting and complete pivoting elimination
- [ ] Gauss–Jordan elimination and matrix inversion
- [ ] LU factorization of a matrix: Doolittle and Crout factorizations
- [ ] The Thomas algorithm for tridiagonal systems
- [ ] The square root method for symmetric positive definite matrices (Cholesky decomposition)
- [ ] The improved square root method (LDLᵀ factorization)
- [ ] Vector norms and matrix norms
- [ ] Error analysis of linear systems: condition numbers and ill-conditioned systems
- [ ] Iterative refinement

### Iterative Methods for Linear Systems

- [ ] Basic ideas of iterative methods and splitting schemes
- [ ] The Jacobi method
- [ ] The Gauss–Seidel method
- [ ] The successive over-relaxation method (SOR) and choice of the relaxation factor
- [ ] The fundamental convergence theorem for iterative methods: the spectral radius condition
- [ ] Convergence for strictly diagonally dominant and irreducibly diagonally dominant matrices
- [ ] Convergence of SOR on symmetric positive definite matrices
- [ ] Error estimation and stopping criteria for iterative methods

### Matrix Eigenvalue Computation

- [ ] Properties and estimation of eigenvalues: the Gershgorin circle theorem
- [ ] The power method for the dominant eigenvalue and dominant eigenvector
- [ ] Accelerating the power method: origin shifting and Rayleigh quotient acceleration
- [ ] The inverse power method
- [ ] Householder transformation and reduction of a matrix to tridiagonal form
- [ ] QR decomposition of a matrix
- [ ] The basic QR algorithm and its convergence
- [ ] The QR algorithm with origin shift and the Hessenberg matrix
- [ ] The Jacobi method for computing all eigenvalues of a real symmetric matrix

### Root Finding for Nonlinear Equations

- [ ] Basic steps of root finding and root searching
- [ ] The bisection method and its convergence
- [ ] Fixed-point iteration
- [ ] Convergence theorems for fixed-point iteration and local convergence
- [ ] The concept and determination of the order of convergence
- [ ] Accelerating iteration: Aitken's Δ² method and Steffensen's method
- [ ] Newton's method and its local convergence
- [ ] Modifying Newton's method for multiple roots
- [ ] The secant method
- [ ] Müller's method
- [ ] An introduction to Newton's method for nonlinear systems

### Numerical Methods for Ordinary Differential Equations

- [ ] Basic concepts of numerical methods for initial value problems: single-step and multistep methods
- [ ] Euler's method and its local truncation error
- [ ] The backward Euler method and the trapezoidal method
- [ ] The improved Euler method and predictor–corrector schemes
- [ ] The basic idea of Runge–Kutta methods
- [ ] Second- and third-order Runge–Kutta methods
- [ ] The classical fourth-order Runge–Kutta method
- [ ] Variable-step Runge–Kutta methods and error control
- [ ] Convergence and stability of single-step methods
- [ ] Linear multistep methods: Adams explicit and implicit schemes
- [ ] Adams predictor–corrector systems
- [ ] Numerical methods for first-order systems and higher-order equations
- [ ] Stiff problems and regions of absolute stability

> After writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
