---
pageClass: plain-doc
---

# Digital Logic

This category mirrors every chapter of Yan Shi's *Fundamentals of Digital Electronic Technology*, with two additional advanced tracks covering hardware description languages and CPU construction. Goal: to write blog posts for the entire textbook — in other words, to fully master the discipline of digital logic.

## Topic Planning

<ProgressGrid cat="cs/digital-logic" />


### Part 1 · Number Systems and Codes

- [ ] Overview of digital signals and digital circuits
- [ ] Common number systems: binary, octal, decimal, hexadecimal
- [ ] Conversions between different number systems
- [ ] Binary arithmetic: sign-magnitude, one's complement, and two's complement
- [ ] Common codes: BCD code, Gray code, ASCII, and parity-check codes

### Part 2 · Foundations of Boolean Algebra

- [ ] The three basic operations of Boolean algebra: AND, OR, NOT
- [ ] Compound logic operations: NAND, NOR, XOR, XNOR
- [ ] Basic and commonly used identities of Boolean algebra
- [ ] Fundamental theorems of Boolean algebra: substitution theorem, De Morgan's theorem, duality theorem
- [ ] Logic functions and their representations: truth tables, expressions, logic diagrams, waveform diagrams
- [ ] The two standard forms of logic functions: sum of minterms and product of maxterms
- [ ] Algebraic simplification of logic functions
- [ ] Karnaugh map representation and Karnaugh map simplification
- [ ] Logic functions with don't-care terms and their simplification

### Part 3 · Gate Circuits

- [ ] Switching characteristics of semiconductor diodes and transistors
- [ ] Discrete gate circuits: diode AND/OR gates and transistor NOT gate
- [ ] Circuit structure and operating principle of the TTL inverter
- [ ] Static input and output characteristics of the TTL inverter
- [ ] Other TTL gate types: NAND, NOR, open-collector (OC) gates, and tri-state gates
- [ ] Circuit structure and operating principle of the CMOS inverter
- [ ] Other CMOS gate types: transmission gates, open-drain (OD) gates, and tri-state gates
- [ ] TTL/CMOS interfacing and usage considerations

### Part 4 · Combinational Logic Circuits

- [ ] Characteristics and functional description of combinational logic circuits
- [ ] Analysis methods for combinational logic circuits
- [ ] Design methods for combinational logic circuits
- [ ] Common combinational modules: encoders, basic encoders, and priority encoders
- [ ] Common combinational modules: decoders, BCD-to-decimal decoders, and display decoders
- [ ] Common combinational modules: multiplexers and demultiplexers
- [ ] Common combinational modules: adders, half adders, and full adders
- [ ] Common combinational modules: magnitude comparators
- [ ] Designing combinational logic circuits with medium-scale integrated (MSI) circuits
- [ ] Race-hazard phenomena in combinational logic circuits and their elimination

### Part 5 · Flip-Flops

- [ ] Flip-flops overview: bistability and memory function
- [ ] SR latch (basic RS flip-flop)
- [ ] Level-triggered flip-flops: synchronous SR flip-flop and D latch
- [ ] Pulse-triggered and edge-triggered flip-flops
- [ ] Edge-triggered D flip-flop and JK flip-flop
- [ ] T flip-flop and T' flip-flop
- [ ] Logic-function classification of flip-flops and conversion between types
- [ ] Dynamic characteristics of flip-flops: setup time, hold time, and propagation delay

### Part 6 · Sequential Logic Circuits

- [ ] Sequential logic overview: structural models (Mealy and Moore)
- [ ] Analysis of synchronous sequential logic circuits
- [ ] Analysis of asynchronous sequential logic circuits
- [ ] Registers and shift registers
- [ ] Counters (I): synchronous binary up/down counters
- [ ] Counters (II): asynchronous counters and decimal counters
- [ ] Building arbitrary-modulus counters from MSI counters: the reset method and the preset method
- [ ] Shift-register counters: ring counters and Johnson counters
- [ ] Sequence pulse generators and sequence signal generators
- [ ] Design of synchronous sequential logic circuits: state minimization, state assignment, and self-starting checks

### Part 7 · Pulse Waveform Generation and Shaping

- [ ] Pulse waveform parameters and an overview of shaping circuits
- [ ] Schmitt trigger: construction from gates and operating principle
- [ ] Schmitt trigger applications: waveform conversion, shaping, and amplitude discrimination
- [ ] Monostable multivibrators: differentiator type, integrator type, and integrated monostable multivibrators
- [ ] Astable multivibrators: symmetric, asymmetric, and ring oscillators
- [ ] Circuit structure and functions of the 555 timer
- [ ] Building Schmitt triggers, monostable multivibrators, and astable multivibrators from the 555 timer

### Part 8 · Semiconductor Memory

- [ ] Semiconductor memory overview and classification
- [ ] Read-only memory (ROM): fixed ROM and PROM
- [ ] Erasable programmable read-only memory: EPROM, E²PROM, and flash memory
- [ ] Random-access memory (RAM): SRAM and DRAM memory cells
- [ ] Memory capacity expansion: bit expansion and word expansion
- [ ] Implementing combinational logic functions with memory

### Part 9 · Programmable Logic Devices

- [ ] Programmable logic devices (PLDs): overview and basic structure
- [ ] Programmable array logic (PAL) and generic array logic (GAL)
- [ ] Structure and principles of complex programmable logic devices (CPLDs)
- [ ] Structure and principles of field-programmable gate arrays (FPGAs)
- [ ] PLD development flow and programming techniques

### Part 10 · D/A and A/D Conversion

- [ ] Overview of D/A and A/D conversion
- [ ] D/A converters: weighted-resistor network and inverted R-2R ladder DACs
- [ ] Key specifications of D/A converters: resolution and conversion accuracy
- [ ] Fundamentals of A/D conversion: sampling, holding, quantizing, and encoding
- [ ] Sample-and-hold circuits
- [ ] A/D converters: flash (parallel-comparator) and successive-approximation ADCs
- [ ] A/D converters: dual-slope and other indirect ADCs

### Part 11 · Introduction to the Hardware Description Language Verilog

- [ ] HDL overview: from schematic design to hardware description languages
- [ ] Basic Verilog structure: modules, ports, and signal declarations
- [ ] Verilog data types and operators
- [ ] Modeling combinational logic: assign statements and always blocks
- [ ] Modeling sequential logic: always @(posedge clk), blocking and non-blocking assignments
- [ ] Describing state machines in Verilog: one-block, two-block, and three-block styles
- [ ] Writing testbenches and simulation verification

### Part 12 · From Logic Gates to the CPU

- [ ] Building a 1-bit full adder and a multi-bit arithmetic logic unit (ALU) from logic gates
- [ ] From flip-flops to register files: the CPU's storage path
- [ ] Program counter, instruction register, and instruction fetch circuits
- [ ] Instruction encoding and decoding: from decoders to control signals
- [ ] Hardwired controllers and microprogrammed controllers
- [ ] Datapath organization: bus structures and single-cycle datapaths
- [ ] A minimal CPU: full system integration and running a program

> Once written: create a new `xxx.md` in this directory, then change the corresponding entry above to `- [x] [标题](./xxx)`.
