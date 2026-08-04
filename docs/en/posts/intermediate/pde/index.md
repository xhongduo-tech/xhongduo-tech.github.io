---
pageClass: plain-doc
---

# Partial Differential Equations

Partial differential equations are the mathematical language for describing the motion of continuous media and fields in nature. This category mirrors the chapter structure of *Equations of Mathematical Physics* (Gu Chaohao, Jiang Lishang, et al.), systematically organizing the full introductory content on the three classical equations (wave, heat, Laplace) along with Fourier methods, special functions, distributions, and the theory of weak solutions.

## Topic Plan

<ProgressGrid cat="intermediate/pde" />

### Part 1 Basic Concepts of Partial Differential Equations

- [ ] Definitions and examples of partial differential equations (three types of classical equations in mathematical physics)
- [ ] Order of an equation, linear vs. nonlinear, homogeneous vs. nonhomogeneous
- [ ] The concepts of solution, general solution, and particular solution
- [ ] Well-posed conditions: initial conditions and boundary conditions
- [ ] Three types of boundary conditions (Dirichlet, Neumann, Robin)
- [ ] Formulation of well-posed problems and well-posedness (existence, uniqueness, stability)
- [ ] Superposition principle

### Part 2 Classification and Canonical Forms of Second-Order Linear Equations

- [ ] General form and characteristic equation of second-order linear equations in two independent variables
- [ ] Discriminating hyperbolic, parabolic, and elliptic types
- [ ] Canonical form of hyperbolic equations (vibrating string equation)
- [ ] Canonical form of parabolic equations (heat equation)
- [ ] Canonical form of elliptic equations (Laplace equation)
- [ ] Reduction of constant-coefficient equations and examples of mixed-type equations (Tricomi equation)
- [ ] Classification of second-order linear equations in several independent variables

### Part 3 First-Order Partial Differential Equations

- [ ] First-order linear homogeneous equations and first integrals of systems of ODEs
- [ ] The method of characteristics for solving the Cauchy problem
- [ ] Geometric theory of first-order quasilinear equations
- [ ] First-order nonlinear equations and Charpit's method
- [ ] Complete integrals, singular integrals, and envelopes
- [ ] Introduction to the Hamilton–Jacobi equation

### Part 4 The Wave Equation

- [ ] Derivation of the vibrating string equation and its physical background
- [ ] d'Alembert's formula for the one-dimensional wave equation
- [ ] Traveling waves, intervals of dependence, domains of determination, and domains of influence
- [ ] The semi-infinite string problem and the method of extension (odd and even extensions)
- [ ] Initial-boundary value problems on a bounded string: separation of variables
- [ ] The homogeneity (Duhamel) principle for nonhomogeneous equations
- [ ] Handling nonhomogeneous boundary conditions
- [ ] The method of spherical means for higher-dimensional wave equations (Poisson formula)
- [ ] The method of descent and the two-dimensional wave equation
- [ ] Huygens' principle and wave dispersion
- [ ] Energy inequalities and uniqueness of solutions
- [ ] Energy methods and stability of solutions

### Part 5 The Heat Equation

- [ ] Derivation of the heat equation (Fourier's law of heat conduction)
- [ ] Initial-boundary value problems on a bounded bar: separation of variables
- [ ] Convergence of Fourier series solutions
- [ ] Heat conduction problems on a circular domain
- [ ] Poisson's formula for the Cauchy problem of the heat equation
- [ ] Solving the Cauchy problem by the Fourier transform method
- [ ] Extremum principle (maximum principle)
- [ ] Proving uniqueness and stability of initial-boundary value problems via the extremum principle
- [ ] Uniqueness and stability of solutions to the Cauchy problem
- [ ] Asymptotic behavior of solutions (decay as t → ∞)

### Part 6 The Laplace Equation (Potential Equation)

- [ ] Physical background of the Laplace and Poisson equations (gravitational fields, electrostatic fields)
- [ ] Definition of harmonic functions and basic examples
- [ ] Green's formula and its corollaries
- [ ] Fundamental solutions (r⁻¹ in three dimensions and ln r in two dimensions)
- [ ] Integral representation of harmonic functions
- [ ] Mean value theorem for harmonic functions
- [ ] Extremum principle and uniqueness of solutions
- [ ] Analyticity of harmonic functions and the removable singularity theorem
- [ ] Harnack's inequality and Liouville's theorem
- [ ] The method of electrostatic images (method of images)
- [ ] Green's function for the sphere and Poisson's formula
- [ ] Green's functions for special regions such as the half-space and the two-dimensional disk
- [ ] Compatibility condition for the solvability of the interior Neumann problem
- [ ] Solving boundary value problems on special regions by trial (guess) methods

### Part 7 Fourier Transform Methods

- [ ] Review of Fourier series: systems of orthogonal functions and expansion theorems
- [ ] Derivation of the Fourier integral
- [ ] Definitions of the Fourier transform and its inverse
- [ ] Basic properties of the Fourier transform (translation, differentiation, convolution)
- [ ] The convolution theorem
- [ ] Solving the Cauchy problem of the heat equation by the Fourier transform
- [ ] Fourier transforms for problems on semi-infinite domains (sine and cosine transforms)
- [ ] The Laplace transform and its application to well-posed problems
- [ ] Fourier methods for higher-dimensional heat and wave equations

### Part 8 Special Functions

- [ ] Separation of variables in cylindrical coordinates and derivation of the Bessel equation
- [ ] Series solutions of the Bessel equation and Bessel functions
- [ ] Recurrence formulas for Bessel functions
- [ ] Zeros and asymptotic properties of Bessel functions
- [ ] Fourier–Bessel series expansions
- [ ] Applications of Bessel functions (heat conduction in a cylinder and vibration of a circular membrane)
- [ ] Separation of variables in spherical coordinates and derivation of the Legendre equation
- [ ] Legendre polynomials and their properties
- [ ] Orthogonality of Legendre polynomials and expansion theorems
- [ ] Associated Legendre polynomials and spherical harmonics
- [ ] Applications of Legendre polynomials (potential problems on spherical regions)

### Part 9 Distributions and Fundamental Solutions: An Introduction

- [ ] Physical introduction to concentrated quantities and the δ function
- [ ] Definitions of test function spaces and distributions
- [ ] Limits, derivatives, and multiplication operators for distributions
- [ ] Convolution and the Fourier transform of distributions
- [ ] Introduction to Sobolev spaces
- [ ] The concept of fundamental solutions and fundamental solutions of the three classical equations

### Part 10 Variational Methods and Weak Solutions: An Introduction

- [ ] Variational problems and the Euler equation
- [ ] Equivalence of boundary value problems and variational problems
- [ ] Definition of weak (generalized) solutions
- [ ] The Ritz method
- [ ] The Galerkin method
- [ ] Introduction to the finite element method

> After finishing your writing: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
