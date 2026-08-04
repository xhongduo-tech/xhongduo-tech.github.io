---
pageClass: plain-doc
---

# Computer Architecture

A systematic, quantitative approach to computer architecture at an advanced level, aligned with the chapter structure of Hennessy & Patterson's *Computer Architecture: A Quantitative Approach*, covering everything from instruction-level parallelism to warehouse-scale computers. For the undergraduate-level foundations in "Principles of Computer Organization", see [/en/posts/cs/computer-organization/](/en/posts/cs/computer-organization/); this category focuses on the advanced quantitative perspective.

## Topic Plan

<ProgressGrid cat="cs/computer-architecture" />


### Part 1 · Architecture Fundamentals and Quantitative Analysis

- [ ] Definition of computer architecture: the division of labor between instruction set, organization, and implementation
- [ ] Classification of computers and Flynn's taxonomy
- [ ] The principle of quantitative analysis: making the common case fast
- [ ] Performance metrics: CPU time, CPI, and the pitfalls of MIPS
- [ ] Benchmark suites: SPEC, TPC, and choosing benchmarks
- [ ] Amdahl's law: quantitative derivation and application of speedup
- [ ] The principle of locality and estimating speedup in common scenarios
- [ ] The power wall: dynamic and static power
- [ ] The end of Dennard scaling and the dark silicon problem
- [ ] From single-core to multi-core: the design shift under power constraints
- [ ] Quantitative measures of cost, reliability, and availability
- [ ] Pitfalls and fallacies in computer architecture

### Part 2 · Instruction-Level Parallelism

- [ ] Instruction-level parallelism (ILP): the concept and its limiting factors
- [ ] Distinguishing data, name, and control dependencies
- [ ] Dynamic scheduling fundamentals: the scoreboard algorithm
- [ ] Tomasulo's algorithm: reservation stations and the common data bus (CDB)
- [ ] Register renaming: eliminating WAR and WAW dependencies
- [ ] Hardware speculation: the reorder buffer (ROB)
- [ ] Precise exceptions and speculation recovery mechanisms
- [ ] Branch prediction: 1-bit/2-bit predictors and correlating predictors
- [ ] Tournament predictors and TAGE
- [ ] Branch target buffer (BTB) and the return address stack
- [ ] Multiple-issue processors: superscalar and superpipelining
- [ ] Static multiple issue: VLIW and explicitly parallel instruction computing (EPIC)
- [ ] Combining dynamic scheduling and multiple issue: modern superscalar pipelines
- [ ] The limits of speculation and the ILP ceiling study
- [ ] Case study: the microarchitectures of Intel Core and ARM Cortex

### Part 3 · Advanced Instruction Set Architectures

- [ ] Classification of ISAs: stack, accumulator, and register-based
- [ ] RISC design philosophy: the historical evolution of reduced instruction sets
- [ ] RISC-V ISA overview: modular and extensible design
- [ ] RISC-V integer ISA: the encoding and semantics of RV32I/RV64I
- [ ] RISC-V privileged architecture and exception handling
- [ ] The RISC-V compressed instruction extension (C extension)
- [ ] Vector instruction architecture: the vector-length-agnostic programming model
- [ ] The RISC-V vector extension (RVV): configuration instructions and mask operations
- [ ] Advantages of vector architectures: a fundamental comparison with SIMD
- [ ] Quantitative evaluation of ISA design: code density and performance trade-offs

### Part 4 · Memory Hierarchy in Depth

- [ ] A quantitative review of the memory hierarchy: hit time and average memory access time (AMAT)
- [ ] Advanced cache optimizations (1): small and simple first-level caches
- [ ] Advanced cache optimizations (2): way prediction and pseudo-associative caches
- [ ] Advanced cache optimizations (3): non-blocking caches and pipelining misses
- [ ] Advanced cache optimizations (4): hardware prefetching and compiler prefetching
- [ ] Advanced cache optimizations (5): compiler optimizations (loop interchange, blocking)
- [ ] Advanced cache optimizations (6): critical word first and early restart
- [ ] Advanced cache optimizations (7): merging write buffers and pipelined access
- [ ] Summary of cache optimizations: their impact on hit time, miss rate, and miss penalty
- [ ] Virtual memory in depth: TLB optimizations and multi-level page tables
- [ ] Memory protection for virtual machines and accelerating address translation
- [ ] Main memory technology: the internal organization of DRAM (banks, row buffers)
- [ ] The evolution of SDRAM and the DDR family and bandwidth enhancement mechanisms
- [ ] Memory controllers and scheduling policies
- [ ] Flash and emerging non-volatile memories (PCM, 3D XPoint)
- [ ] Storage reliability: RAID levels and error-correcting codes (ECC)
- [ ] Case study: the memory hierarchy of modern servers

### Part 5 · Thread-Level Parallelism

- [ ] Thread-level parallelism (TLP) and an overview of multiprocessor architectures
- [ ] Symmetric multiprocessors (SMP) and distributed shared-memory multiprocessors (DSM)
- [ ] The cache coherence problem: snooping protocols
- [ ] State transitions in the MESI/MOESI coherence protocols
- [ ] Directory-based coherence protocols
- [ ] Synchronization primitives: atomic operations, locks, and barriers
- [ ] Memory consistency models: sequential consistency
- [ ] Weak consistency models and release consistency
- [ ] Performance modeling of multicore processors and scalability limits
- [ ] The principles and trade-offs of simultaneous multithreading (SMT/Hyper-Threading)

### Part 6 · Data-Level Parallelism

- [ ] Data-level parallelism (DLP) overview: revisiting the SIMD versus vector comparison
- [ ] SIMD instruction extensions: the evolution from MMX to AVX-512
- [ ] SIMD programming models and automatic vectorization
- [ ] GPU architectures: from the graphics pipeline to general-purpose computing (GPGPU)
- [ ] The GPU SIMT execution model: warps and warp scheduling
- [ ] The GPU memory hierarchy: shared memory, register files, and global memory
- [ ] GPU branch divergence and masked execution
- [ ] Tensor cores and matrix operation acceleration
- [ ] Loop-level parallelism and dependency analysis

### Part 7 · Domain-Specific Architectures

- [ ] The rise of domain-specific architectures (DSAs): the end of Moore's law
- [ ] DSA design principles: specialized memory, simplified control, and domain matching
- [ ] Google TPU: the principles of systolic arrays
- [ ] The evolution of the TPU: from inference chips to training clusters
- [ ] Design trade-offs of NPUs and edge AI accelerators
- [ ] Dataflows in neural network acceleration: weight/output/row stationary
- [ ] Exploiting sparsity and quantization in DSAs
- [ ] Programming models for DSAs and software stack challenges

### Part 8 · Warehouse-Scale Computers and Interconnection

- [ ] Warehouse-scale computers (WSCs): the data center as a computer
- [ ] WSC architecture: server, storage, and network organization
- [ ] The latency-throughput trade-off: tail latency in online services
- [ ] Energy efficiency in WSCs: PUE and total cost of ownership (TCO)
- [ ] Architectural support for cluster computing frameworks such as MapReduce and Spark
- [ ] The economics of cloud computing and WSCs
- [ ] Interconnection network basics: topologies (mesh, torus, fat tree)
- [ ] Switching techniques: wormhole routing and virtual channels
- [ ] Cluster interconnection case study: InfiniBand versus Ethernet
- [ ] Key design considerations of networks-on-chip (NoCs)

### Part 9 · Emerging Topics

- [ ] Processing-in-memory (PIM): the revival of near-data computing
- [ ] PIM architectures based on emerging memories (ReRAM computing)
- [ ] Chiplets and advanced packaging: going beyond single-chip scaling
- [ ] Chiplet interconnect standards: UCIe and heterogeneous integration
- [ ] The RISC-V ecosystem: how an open ISA reshapes the industry landscape
- [ ] The CXL interconnect protocol: cache-coherent device interconnect and memory pooling
- [ ] A first look at quantum computing architectures
- [ ] Architectural security: Spectre, Meltdown, and microarchitectural side channels

> After the writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [title](./xxx)`.
