---
pageClass: plain-doc
---

# Quantum Computing

Completing this series on quantum computing means writing out all the blog posts that map to Nielsen & Chuang's *Quantum Computation and Quantum Information* and the IBM Qiskit curriculum. Starting from the foundations of quantum mechanics and linear algebra, it covers quantum circuits, quantum algorithms, quantum error correction, quantum hardware, and the variational algorithms of the NISQ era, through to hands-on quantum programming and a look at what lies ahead.

## Topic Map

<ProgressGrid cat="advanced/quantum-computing" />


### Part 0 — Overview of Quantum Computing

- [ ] The origins of quantum computing: Feynman's proposal and quantum simulation
- [ ] A brief history of quantum computing: from Deutsch to NISQ
- [ ] Why quantum computing can be faster: quantum parallelism and entanglement
- [ ] Models of quantum computation: circuit model, adiabatic computing, and measurement-based quantum computing
- [ ] Clearing up misconceptions: exponential speedup is not free
- [ ] The fundamental tasks of quantum information: computing, communication, cryptography, and simulation

### Part 1 — Linear Algebra and the Foundations of Quantum Mechanics

- [ ] Vector spaces over the complex numbers and inner products
- [ ] Dirac notation (bra-ket): kets, bras, and inner/outer products
- [ ] Linear operators and matrix representations
- [ ] Hermitian operators and unitary operators
- [ ] Eigenvalues, eigenvectors, and spectral decomposition
- [ ] Tensor products: state spaces of many-body systems
- [ ] The fundamental postulates of quantum mechanics: state, evolution, measurement
- [ ] Projective measurements and generalized measurements (POVM)
- [ ] Density operators: mixed states and the partial trace
- [ ] Schmidt decomposition and purification

### Part 2 — Qubits and the Bloch Sphere

- [ ] Qubits: |0⟩, |1⟩, and superposition states
- [ ] The Bloch sphere representation
- [ ] Measurement of single-qubit states and basis choice
- [ ] Multi-qubit systems and entangled states
- [ ] The no-cloning theorem
- [ ] Quantum teleportation
- [ ] Superdense coding

### Part 3 — Quantum Gates and Quantum Circuits

- [ ] The quantum circuit model and circuit-diagram conventions
- [ ] Single-qubit gates: X, Y, Z, and the Pauli gates
- [ ] The Hadamard gate and phase gates (S and T gates)
- [ ] Rotation gates: Rx, Ry, Rz, and the decomposition of arbitrary single-qubit gates
- [ ] Controlled gates: CNOT, CZ, and controlled-U
- [ ] The Toffoli gate and the Fredkin gate
- [ ] Circuit representations of measurement and the principle of deferred measurement
- [ ] Universal gate sets: CNOT plus single-qubit gates
- [ ] The Solovay-Kitaev theorem and gate approximation
- [ ] Circuit depth, width, and complexity

### Part 4 — Quantum Entanglement and Bell's Inequalities

- [ ] Definition of entanglement: separable states and entangled states
- [ ] Bell states and Bell measurement
- [ ] The EPR paradox and hidden-variable theories
- [ ] The CHSH inequality and its quantum violation
- [ ] Measures of entanglement: concurrence and entanglement entropy
- [ ] GHZ states and W states: two types of multipartite entanglement

### Part 5 — Foundations of Quantum Algorithms

- [ ] Quantum query complexity and the black-box model
- [ ] Deutsch's algorithm and the Deutsch-Jozsa algorithm
- [ ] The Bernstein-Vazirani algorithm
- [ ] Simon's algorithm
- [ ] The quantum Fourier transform (QFT) and its implementation
- [ ] The phase estimation algorithm
- [ ] Quantum amplitude amplification and amplitude estimation

### Part 6 — Shor's Algorithm

- [ ] The relation between integer factorization and RSA cryptography
- [ ] Reducing integer factorization to period finding
- [ ] Circuit implementations of modular exponentiation
- [ ] A complete analysis of Shor's algorithm's flow
- [ ] The complexity of Shor's algorithm versus classical algorithms
- [ ] The quantum algorithm for discrete logarithms and its threat to elliptic-curve cryptography

### Part 7 — Grover's Algorithm and Search

- [ ] The unstructured search problem and classical lower bounds
- [ ] The Grover iteration: the oracle and the diffusion operator
- [ ] The geometric interpretation of Grover's algorithm: rotation and amplitude amplification
- [ ] Searching for multiple solutions and partial solutions
- [ ] Complexity analysis of Grover's algorithm: optimality of O(√N)
- [ ] Applications of Grover's algorithm: collision finding, counting, and minimum finding

### Part 8 — Quantum Error Correction

- [ ] Classical error correction and the difficulty of quantum error correction
- [ ] The three-qubit bit-flip code
- [ ] The three-qubit phase-flip code
- [ ] The Shor nine-qubit code
- [ ] Conditions for quantum error correction and error discretization
- [ ] The stabilizer formalism
- [ ] CSS codes and the Steane code
- [ ] An introduction to surface codes and the fault-tolerant threshold
- [ ] Fault-tolerant quantum computing and the threshold theorem

### Part 9 — Quantum Hardware

- [ ] Physical requirements for qubit implementation: the DiVincenzo criteria
- [ ] Superconducting qubits: Transmon and flux qubits
- [ ] Trapped-ion quantum computing
- [ ] Photonic quantum computing and the linear-optical scheme
- [ ] Neutral atoms and Rydberg atom arrays
- [ ] Other approaches: semiconductor spins and topological quantum computing
- [ ] Decoherence, noise, and quantum gate fidelity

### Part 10 — NISQ and Variational Quantum Algorithms

- [ ] The definition and limitations of the NISQ era
- [ ] The framework of variational quantum algorithms: parameterized circuits and classical optimization
- [ ] The variational quantum eigensolver (VQE)
- [ ] The quantum approximate optimization algorithm (QAOA)
- [ ] Cost-function landscapes and barren plateaus
- [ ] Quantum annealing and adiabatic quantum computing

### Part 11 — An Introduction to Quantum Machine Learning

- [ ] Problem setting and data encoding in quantum machine learning
- [ ] Amplitude encoding, angle encoding, and basis encoding
- [ ] Quantum kernel methods
- [ ] Quantum neural networks (QNN) and parameterized quantum circuits
- [ ] The HHL algorithm and quantum linear algebra
- [ ] The speedup debate in quantum machine learning and dequantization

### Part 12 — Quantum Programming in Practice (Qiskit)

- [ ] Setting up Qiskit and your first quantum circuit
- [ ] Implementing single-qubit gates on the Bloch sphere in Qiskit
- [ ] Constructing Bell states with Qiskit and verifying entanglement
- [ ] Implementing the Deutsch-Jozsa algorithm in Qiskit
- [ ] Implementing QFT and phase estimation in Qiskit
- [ ] Implementing Grover search in Qiskit
- [ ] Implementing VQE and QAOA instances in Qiskit
- [ ] Simulators versus real backends: IBM Quantum cloud experiments
- [ ] Compilation, transpilation, and noise simulation of quantum circuits

### Part 13 — The Present State and Outlook of Quantum Computing

- [ ] Quantum supremacy: random circuit sampling and boson sampling
- [ ] Responding with post-quantum cryptography
- [ ] The quantum internet and quantum repeaters
- [ ] Prospects for quantum computing in chemistry, materials, and optimization
- [ ] Open problems in quantum computing and a learning roadmap

> After finishing each post: create a `xxx.md` in this directory, then change the corresponding item above to `- [x] [title](./xxx)`.
