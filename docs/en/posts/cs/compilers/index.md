---
pageClass: plain-doc
---

# Compilers

Following the classic textbook *Compilers: Principles, Techniques, and Tools* (the "Dragon Book"), this track runs from lexical analysis all the way to code optimization, with additional modern topics such as garbage collection, JIT, and LLVM.

## Topic Planning

<ProgressGrid cat="cs/compilers" />


### Part 1 · Compiler Overview

- [ ] The structure of a compiler: analysis and synthesis
- [ ] Compiler phases: lexical, syntax, and semantic analysis, intermediate code generation, optimization, code generation
- [ ] Symbol table management and error handling
- [ ] Compilers, interpreters, and hybrid execution models
- [ ] Evolution of compilers: from single-pass to multi-pass compilation
- [ ] Compiler construction tools and compiler-compilers

### Part 2 · Lexical Analysis

- [ ] Role and interface of the lexical analyzer
- [ ] Tokens, patterns, and lexemes
- [ ] Input buffering and sentinel techniques
- [ ] Regular expressions and regular definitions
- [ ] Finite automata: NFA and DFA
- [ ] From regular expressions to NFA: Thompson's construction
- [ ] From NFA to DFA: subset construction
- [ ] DFA state minimization
- [ ] Lexer generator tools Lex/Flex: usage and implementation principles

### Part 3 · Syntax Analysis

- [ ] Context-free grammars (CFG) and derivations
- [ ] Parse trees and grammar ambiguity
- [ ] Eliminating ambiguity, left recursion, and left factoring
- [ ] Overview of top-down parsing: recursive descent
- [ ] Computing FIRST and FOLLOW sets
- [ ] LL(1) grammars and predictive parsing tables
- [ ] Non-recursive table-driven predictive parsers
- [ ] Overview of bottom-up parsing: shift-reduce parsing
- [ ] How LR parsers work and the structure of LR parsing tables
- [ ] LR(0) item sets and SLR parsing table construction
- [ ] Canonical LR(1) parsing table construction
- [ ] LALR(1) parsing table construction
- [ ] Using ambiguous grammars in LR parsing
- [ ] Parser generator tools Yacc/Bison: usage and implementation principles

### Part 4 · Syntax-Directed Translation

- [ ] Syntax-directed definitions (SDD): synthesized and inherited attributes
- [ ] Attribute grammar evaluation order and dependency graphs
- [ ] S-attributed and L-attributed definitions
- [ ] Syntax-directed translation schemes (SDT)
- [ ] Implementing SDTs in LL parsing
- [ ] Implementing SDTs in LR parsing
- [ ] Building a simple syntax-directed translator

### Part 5 · Intermediate Code Generation

- [ ] Forms of intermediate representation: syntax trees, DAGs, and three-address code
- [ ] Three-address code instruction types: quadruples and triples
- [ ] Translation of expressions and generation of temporary variables
- [ ] Translation of array addressing
- [ ] Type systems and type expressions
- [ ] Type checking: type equivalence and type conversions
- [ ] Translation of Boolean expressions: numeric representation and short-circuit evaluation
- [ ] Translation of control-flow statements and backpatching
- [ ] Translation of procedure calls and switch statements

### Part 6 · Runtime Environments

- [ ] Storage organization: code, static, stack, and heap regions
- [ ] Activation records and calling sequences
- [ ] Stack allocation and activation trees
- [ ] Access to nonlocal names: access links and nesting depth
- [ ] Heap management and memory fragmentation
- [ ] Parameter passing mechanisms: by value, by reference, by name, and copy-restore

### Part 7 · Code Generation

- [ ] Code generator design issues: input, target program, and instruction selection
- [ ] Target machine model and instruction costs
- [ ] Construction of basic blocks and flow graphs
- [ ] Basic-block optimizations: local common-subexpression elimination and dead-code elimination
- [ ] A simple code generator: next-use information
- [ ] Register allocation: graph coloring
- [ ] Instruction selection by tree pattern matching
- [ ] Peephole optimization

### Part 8 · Machine-Independent Optimizations

- [ ] Sources of optimization and the organization of optimizing compilers
- [ ] Foundations of data-flow analysis: reaching definitions
- [ ] Live-variable analysis and available-expression analysis
- [ ] Constant propagation and constant folding
- [ ] Common-subexpression elimination
- [ ] Copy propagation and dead-code elimination
- [ ] Loop optimizations: code hoisting and induction-variable elimination
- [ ] Strength reduction and partial redundancy elimination
- [ ] Dominators and loop identification
- [ ] Introduction to interprocedural analysis

### Part 9 · Special Topic: Garbage Collection

- [ ] Basic concepts of garbage collection: reachability and safe points
- [ ] Mark-and-sweep collection
- [ ] Reference counting and its cycle-collection problem
- [ ] Copying collection and the semispace strategy
- [ ] Generational garbage collection
- [ ] Incremental and concurrent garbage collection

### Part 10 · Special Topic: JIT Compilation and Dynamic Compilation

- [ ] Principles of JIT compilation: the trade-off between interpretation and just-in-time compilation
- [ ] Hot-spot detection, method-level JIT, and tracing JIT
- [ ] Tiered compilation: JVM's C1/C2 as an example
- [ ] Speculative optimization and deoptimization
- [ ] JIT for dynamic languages: V8 and PyPy as examples

### Part 11 · Special Topic: The LLVM Architecture

- [ ] Overall LLVM architecture: frontend, optimizer, and backend
- [ ] LLVM IR: intermediate representation in SSA form
- [ ] The LLVM pass mechanism and optimization pipeline
- [ ] Observing the stages of compilation with Clang
- [ ] Implementing a language frontend on LLVM: a walkthrough of the Kaleidoscope tutorial
- [ ] LLVM and the modern compiler ecosystem: Clang, Rust, and Swift

### Part 12 · Special Topic: Writing a Toy Compiler from Scratch

- [ ] Designing the toy language's syntax and feature set
- [ ] Hand-writing a lexer: from character stream to token stream
- [ ] Hand-writing a recursive-descent parser
- [ ] Building the abstract syntax tree (AST) and symbol table
- [ ] Semantic analysis and simple type checking
- [ ] Generating target code: compiling to C, assembly, or bytecode
- [ ] Writing tests and error diagnostics for the toy compiler

> When done writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [标题](./xxx)`.
