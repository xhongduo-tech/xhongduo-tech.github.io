---
pageClass: plain-doc
---

# Stochastic Processes

Stochastic processes study random phenomena that evolve over time, extending probability theory to dynamical systems. This category follows the chapter structure of Zhang Bo's *Applied Stochastic Processes* and Ross's *Stochastic Processes*, moving from Poisson processes and Markov chains all the way to martingales, Brownian motion, and stochastic integration.

## Topic Planning

<ProgressGrid cat="intermediate/stochastic-processes" />


### Part 1 · Preliminaries

- [ ] Basic concepts of stochastic processes: sample paths and finite-dimensional distribution families
- [ ] Classification of stochastic processes: discrete/continuous time and discrete/continuous state
- [ ] Review of conditional probability and conditional expectation
- [ ] Rigorous definition of conditional expectation: expectation with respect to a σ-algebra
- [ ] Properties of conditional expectation: tower property, taking out what is known, independence
- [ ] Law of total expectation and conditional variance formula
- [ ] Moment generating functions and characteristic functions
- [ ] Moment generating functions of common distributions and convolution of independent sums
- [ ] Modes of convergence: almost sure, in probability, in distribution, and in mean square

### Part 2 · Poisson Processes

- [ ] Counting processes and independent stationary increments
- [ ] Three equivalent definitions of a Poisson process
- [ ] Exponential distribution of interarrival times and the memoryless property
- [ ] Conditional distribution of arrival times: order statistics of a uniform distribution
- [ ] Superposition and decomposition of Poisson processes (thinning theorem)
- [ ] Nonhomogeneous Poisson processes: intensity function and cumulative intensity
- [ ] Compound Poisson processes: definition, mean, and variance
- [ ] Conditional Poisson processes and mixed Poisson models
- [ ] Simulation and parameter estimation of Poisson processes

### Part 3 · Renewal Processes

- [ ] Definition of a renewal process: a counting process with i.i.d. interarrival times
- [ ] Distribution of N(t) and the renewal function m(t)
- [ ] Renewal equations and asymptotic properties of the renewal function
- [ ] Elementary Renewal Theorem
- [ ] Key Renewal Theorem and Blackwell's theorem
- [ ] Renewal Reward processes and long-run average cost
- [ ] Age and residual life: equilibrium renewal processes
- [ ] Alternating renewal processes and their applications in reliability
- [ ] Delayed renewal processes and regenerative processes

### Part 4 · Discrete-Time Markov Chains

- [ ] Markov property and the definition of a Markov chain
- [ ] One-step transition probability matrix and the Chapman-Kolmogorov equations
- [ ] Computing n-step transition probabilities and matrix powers
- [ ] Classification of states: communication, closed sets, and irreducibility
- [ ] Periodicity: determining periodic states and equivalence classes
- [ ] Recurrence and transience: first-passage probabilities and expected number of visits
- [ ] Criterion for recurrence: recurrence is equivalent to the divergence of ∑pⁿᵢᵢ
- [ ] Positive recurrence and null recurrence: mean recurrence times
- [ ] Methods for computing first-passage probabilities and hitting times
- [ ] Existence and uniqueness of invariant (stationary) distributions
- [ ] Limit theorems: ergodic theorem for irreducible aperiodic positive recurrent chains
- [ ] Reversible Markov chains and the detailed balance condition
- [ ] Branching processes: extinction probabilities and moments
- [ ] Applications of Markov chains: random walks and PageRank

### Part 5 · Continuous-Time Markov Chains

- [ ] Definition of continuous-time Markov chains and transition probability functions
- [ ] Exponential holding times and the embedded chain
- [ ] Transition rate matrices (Q matrices / infinitesimal generators)
- [ ] Kolmogorov backward and forward equations
- [ ] Stationary distributions and long-run behavior: the continuous-time case
- [ ] Pure birth processes and the Yule process
- [ ] Birth-death processes: definition and stationary distributions
- [ ] Introduction to queueing theory: the M/M/1 queue
- [ ] M/M/s and M/M/∞ queueing systems
- [ ] Little's law and performance measures of queueing systems

### Part 6 · Martingales

- [ ] Definition of a martingale: discrete-time martingales and examples
- [ ] Supermartingales, submartingales, and equivalent characterizations of martingales
- [ ] Definition and properties of stopping times
- [ ] Optional Stopping Theorem
- [ ] Applications of the optional stopping theorem: the gambler's ruin problem
- [ ] Wald's equation and hitting times of random walks
- [ ] Martingale Convergence Theorem
- [ ] Doob's inequalities and maximal inequalities
- [ ] Martingale difference sequences and Azuma's inequality
- [ ] Applications of martingales in algorithm analysis

### Part 7 · Brownian Motion

- [ ] Historical background of Brownian motion: from pollen movement to the Wiener process
- [ ] Definition of Brownian motion: independent stationary normal increments
- [ ] Continuity of Brownian motion sample paths
- [ ] Non-differentiability and quadratic variation of Brownian motion sample paths
- [ ] Markov and martingale properties of Brownian motion
- [ ] Distribution of hitting times and the Reflection Principle
- [ ] Distribution of the maximum and the arcsine law
- [ ] Brownian bridge and the Gaussian process perspective
- [ ] Variants of Brownian motion: Brownian motion with drift and geometric Brownian motion

### Part 8 · Introduction to Stochastic Integration

- [ ] Why the Riemann-Stieltjes integral is not enough
- [ ] Construction of the Itô integral: starting from simple processes
- [ ] Properties of the Itô integral: isometry and martingale property
- [ ] Itô's Lemma: the single-variable form
- [ ] Multidimensional Itô's formula and the product rule
- [ ] Existence and uniqueness of solutions to stochastic differential equations (SDEs)
- [ ] Solving common SDEs: geometric Brownian motion and the OU process

### Part 9 · Stationary Processes

- [ ] Definitions of strict stationarity and wide-sense (weak) stationarity
- [ ] Properties and estimation of the autocorrelation function (ACF)
- [ ] Ergodicity: time averages versus ensemble averages
- [ ] Power spectral density and the Wiener-Khinchin theorem
- [ ] Linear transformations and filtering of stationary processes
- [ ] White noise, moving average, and autoregressive processes
- [ ] Introduction to the spectral decomposition of stationary processes

### Part 10 · Applications

- [ ] Stochastic models in finance: from Brownian motion to Black-Scholes
- [ ] Risk-neutral pricing and an introduction to martingale measures
- [ ] Stochastic interest rate models: Vasicek and CIR models
- [ ] Stochastic processes in insurance: the Cramér-Lundberg ruin model
- [ ] Stochastic models in inventory management and queueing networks
- [ ] Markov decision processes (MDPs): states, actions, and rewards
- [ ] A stochastic-process view of reinforcement learning: Bellman equations and the Markov property
- [ ] MCMC: the basic principle of sampling with Markov chains

> After writing: create a `xxx.md` file in this directory, then change the corresponding entry above to `- [x] [Title](./xxx)`.
