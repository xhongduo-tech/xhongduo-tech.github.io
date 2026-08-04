---
pageClass: plain-doc
---

# Computer Organization

This category maps to the chapter structure of Tang Shuofei's *Principles of Computer Organization* and *Computer Systems: A Programmer's Perspective* (CS:APP), covering the full content of a computer organization course: from data representation to the memory hierarchy, instruction set, CPU, pipelining, buses, and the input/output system. Study through it and it's written.

## Topic Plan

<ProgressGrid cat="cs/computer-organization" />


### Part 1 Computer System Overview

- [ ] The origin and evolution of computers: from vacuum tubes to VLSI
- [ ] Classification of computers and development trends
- [ ] The basic ideas and characteristics of the von Neumann computer
- [ ] Basic computer components: ALU, control unit, memory, I/O devices
- [ ] How a computer works: fetch, decode, and execute instructions
- [ ] The hierarchical structure of computer systems: from microprogram machine level to high-level language machine level
- [ ] The difference between computer architecture and computer organization
- [ ] Key hardware metrics: machine word length, memory capacity, and operation speed
- [ ] Performance evaluation: CPU execution time, CPI, MIPS, and MFLOPS
- [ ] Amdahl's law and improving system performance
- [ ] Compilation, assembly, linking, and loading of programs
- [ ] Binary representation of information and the machine-level view of programs

### Part 2 Data Representation and Operations

- [ ] Positional number systems and their conversions
- [ ] Machine representation of unsigned and signed numbers
- [ ] Sign-magnitude, one's complement, two's complement, and biased representation
- [ ] Fixed-point representation: fixed-point fractions and fixed-point integers
- [ ] Shift operations and sign extension in two's complement
- [ ] Fixed-point addition and subtraction and their implementation
- [ ] The concept of overflow and detection methods: single sign-bit and double sign-bit methods
- [ ] Fixed-point multiplication: one-bit-at-a-time sign-magnitude and two's complement (Booth's algorithm)
- [ ] Fixed-point division: restoring division and non-restoring (alternating add/subtract) methods
- [ ] Array multipliers and array dividers
- [ ] Floating-point representation: exponent, mantissa, and normalization
- [ ] The IEEE 754 floating-point standard: single precision, double precision, and special values
- [ ] Floating-point addition and subtraction: alignment, mantissa addition, normalization, and rounding
- [ ] Floating-point multiplication and division
- [ ] Rounding modes and precision issues in floating-point arithmetic
- [ ] Structure of the floating-point unit and the floating-point pipeline
- [ ] Integers and floating-point numbers in C: type conversion, integer overflow, and floating-point pitfalls
- [ ] Basic structure of the arithmetic logic unit (ALU)
- [ ] Adder design: ripple-carry, carry-lookahead, and carry-select adders
- [ ] Character and string representation: ASCII, Chinese character encoding, and BCD
- [ ] Parity check codes
- [ ] Hamming codes
- [ ] Cyclic redundancy check codes (CRC)

### Part 3 Memory Hierarchy

- [ ] Classification of memory: by medium, access mode, and role
- [ ] The memory hierarchy: registers, cache, main memory, and auxiliary storage
- [ ] The principle of locality: temporal locality and spatial locality
- [ ] Semiconductor random access memory: SRAM structure and operation
- [ ] DRAM structure, read/write principles, and refresh methods
- [ ] Read-only memory ROM: mask ROM, PROM, EPROM, EEPROM, and flash memory
- [ ] Connecting main memory to the CPU
- [ ] Memory capacity expansion: bit expansion, word expansion, and simultaneous word-bit expansion
- [ ] Improving memory access speed: dual-port memory and multi-bank interleaved memory
- [ ] The operating principle and basic structure of cache
- [ ] Cache address mapping: direct mapping, fully associative mapping, and set-associative mapping
- [ ] Cache replacement algorithms: random, FIFO, LRU, and least frequently used
- [ ] Cache write policies: write-through, write-back, and the write buffer
- [ ] Cache performance analysis: hit rate, average access time, and multi-level caches
- [ ] Separation of instruction and data caches
- [ ] Basic concepts of virtual memory
- [ ] Paged virtual memory and page tables
- [ ] Segmented virtual memory and segmented-paged virtual memory
- [ ] The translation lookaside buffer TLB and multi-level page tables
- [ ] Page replacement algorithms: OPT, FIFO, LRU, and Clock
- [ ] Storage protection: bounds checking and access permission protection
- [ ] Virtual memory as a cache and a memory management tool: mmap, dynamic memory allocation, and garbage collection

### Part 4 Instruction Set Architecture

- [ ] The general format of machine instructions: opcode and address fields
- [ ] The relationship between instruction word length and machine word length
- [ ] Operand types and operation types
- [ ] Instruction addressing: sequential addressing and jump addressing
- [ ] Data addressing modes: immediate, direct, indirect, register, and register indirect
- [ ] Data addressing modes: relative, base, indexed, and stack addressing
- [ ] Opcode expansion techniques and instruction format design
- [ ] Characteristics of RISC and CISC and their comparison
- [ ] The x86 instruction set architecture: data transfer, arithmetic/logical, and control-transfer instructions
- [ ] Data alignment and big-endian vs. little-endian storage

### Part 5 Central Processing Unit

- [ ] CPU functions and organization: ALU, control unit, and register file
- [ ] Major CPU registers: PC, IR, MAR, MDR, PSW
- [ ] The instruction cycle: fetch, indirect, execute, and interrupt cycles
- [ ] Data-flow analysis of the instruction cycle
- [ ] Datapath structures and functions: single-bus, dual-bus, and triple-bus designs
- [ ] The timing system and multi-level timing: machine cycles, beats, and working pulses
- [ ] Control modes: synchronous, asynchronous, and hybrid control
- [ ] Design principles of a hardwired controller
- [ ] The basic ideas of microprogrammed control: microcommands, micro-operations, and microinstructions
- [ ] Organization and operation of the microprogrammed controller
- [ ] Microinstruction encoding: direct encoding, field-direct encoding, and field-indirect encoding
- [ ] Microaddress generation and microinstruction formats
- [ ] Hardwired vs. microprogrammed control
- [ ] Machine-level program representation: the correspondence between assembly instructions and machine code
- [ ] Machine-level implementation of procedure calls: stack frames, argument passing, and return addresses
- [ ] Exceptions and control flow: traps, faults, aborts, and context switches

### Part 6 Instruction Pipelining and Advanced Topics

- [ ] Basic concepts of instruction pipelining and pipeline stage partitioning
- [ ] Pipeline performance metrics: throughput, speedup, and efficiency
- [ ] Structural hazards and their resolution
- [ ] Data hazards and their resolution: stalling and data forwarding (bypassing)
- [ ] Control hazards and their resolution: branch delay slots and delayed branching
- [ ] Branch prediction: static prediction, dynamic prediction, and the branch target buffer (BTB)
- [ ] Interrupt handling and precise exceptions in pipelined processors
- [ ] Superscalar pipelines and dynamic scheduling
- [ ] Superpipelining and very long instruction word (VLIW)
- [ ] Out-of-order execution: the scoreboard and Tomasulo's algorithms
- [ ] Register renaming and the reorder buffer (ROB)
- [ ] Multicore processors and multiprocessor systems
- [ ] Cache coherence and the MESI protocol
- [ ] Simultaneous multithreading (SMT) and hyper-threading

### Part 7 Buses

- [ ] Basic concepts and characteristics of buses
- [ ] Classification of buses: on-chip, system, and communication buses
- [ ] Bus structures and performance metrics: width, bandwidth, and clock frequency
- [ ] Bus arbitration: daisy-chaining, counter-timed polling, and independent request methods
- [ ] Distributed arbitration
- [ ] Bus communication control: synchronous, asynchronous, half-synchronous, and split transactions
- [ ] Bus standards: ISA, EISA, PCI, PCIe, and USB

### Part 8 Input/Output Systems

- [ ] Overview of the I/O system: I/O devices and I/O software
- [ ] Peripheral devices: keyboards, mice, displays, and printers
- [ ] Functions, organization, and types of I/O interfaces
- [ ] I/O ports and their addressing: memory-mapped and isolated I/O
- [ ] Programmed I/O and its interface
- [ ] Basic concepts of program interrupts
- [ ] Interrupt requests, prioritization, and response
- [ ] Interrupt service routines and the interrupt-handling process
- [ ] Multiple interrupts and interrupt masking
- [ ] Basic concepts of DMA and the DMA interface
- [ ] DMA transfer modes and the transfer process
- [ ] DMA vs. interrupts
- [ ] Channel I/O: channel types and the channel work process
- [ ] Magnetic disk storage: structure, performance metrics, and disk scheduling algorithms
- [ ] Redundant arrays of independent disks (RAID): levels and principles
- [ ] Solid-state drive (SSD) structure and read/write characteristics
- [ ] System-level I/O: Unix I/O, file reading/writing, and sharing

> When a post is finished: create a `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
