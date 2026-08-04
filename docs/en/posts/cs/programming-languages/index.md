---
pageClass: plain-doc
---

# Programming Languages

This category is organized around *Concepts of Programming Languages* (Sebesta) and the PLT curriculum, systematically covering programming language design principles, syntax and semantics, type systems, programming paradigms, and runtime mechanisms.

## Topic Plan

<ProgressGrid cat="cs/programming-languages" />

### Part 1 Language Design and Evaluation

- [ ] The significance of studying programming language principles
- [ ] The four evaluation criteria for programming languages: readability, writability, reliability, and cost
- [ ] Factors influencing language design: computer architecture and programming methodology
- [ ] Language implementation approaches: compilation, interpretation, and hybrid implementation
- [ ] The evolution of languages: from Plankalkül to modern multi-paradigm languages

### Part 2 Describing Syntax

- [ ] Basic concepts of syntax and semantics
- [ ] Context-free grammars and derivations
- [ ] BNF and Extended BNF (EBNF)
- [ ] Ambiguity, operator precedence, and associativity
- [ ] Attribute grammars: a method for describing static semantics

### Part 3 Foundations of Semantics

- [ ] The motivation for and taxonomy of formal semantics
- [ ] Operational Semantics
- [ ] An introduction to Denotational Semantics
- [ ] Axiomatic semantics and program correctness verification
- [ ] The λ calculus as a metalanguage for semantic description

### Part 4 Data Types

- [ ] The concept of data types and type checking
- [ ] Primitive data types: numeric, character, boolean, and string
- [ ] Enumeration types and subrange types
- [ ] Array types: subscripts, bindings, and initialization
- [ ] Record types and tuple types
- [ ] List types and associative arrays
- [ ] Union types, discriminated unions, and option types (Option/Maybe)
- [ ] Pointer and reference types

### Part 5 A Deeper Look at Type Systems

- [ ] Strong typing vs. weak typing: the boundary of type safety
- [ ] Static type checking and dynamic type checking
- [ ] Type equivalence: name equivalence and structural equivalence
- [ ] Type inference and the Hindley-Milner algorithm
- [ ] Parametric polymorphism (generics), ad hoc polymorphism, and subtype polymorphism
- [ ] Covariance and contravariance

### Part 6 Names, Bindings, and Scopes

- [ ] Distinguishing the concepts of names, variables, and values
- [ ] The concept of binding and binding time
- [ ] Static scoping (lexical scoping) and dynamic scoping
- [ ] Block structure, nested subprograms, and scope holes
- [ ] Storage binding and the lifetime of variables

### Part 7 Expressions and Assignment Statements

- [ ] Arithmetic expressions: operand evaluation order and precedence
- [ ] Mixed-mode expressions and type conversion
- [ ] Relational expressions, boolean expressions, and short-circuit evaluation
- [ ] Forms of assignment statements and compound assignment
- [ ] Side effects in expressions and function side effects

### Part 8 Statement-Level Control Structures

- [ ] Selection criteria for control structures: single entry, single exit
- [ ] Compound statements and conditional statements
- [ ] Multiple selection structures: if-elif and switch/match
- [ ] Iteration statements: counting loops, logical loops, and iterators
- [ ] The unconditional branch (goto) problem and structured programming
- [ ] Guarded commands and nondeterministic control structures

### Part 9 Subprograms

- [ ] Basic concepts and definitions of subprograms
- [ ] Formal and actual parameters: positional parameters, keyword parameters, and default parameters
- [ ] Parameter passing mechanisms: pass-by-value, pass-by-reference, and pass-by-result
- [ ] Coroutines
- [ ] Nested subprograms, static chains, and access to nonlocal variables
- [ ] Closures and first-class functions

### Part 10 Abstract Data Types and Encapsulation

- [ ] The concept of abstraction and data abstraction
- [ ] Design requirements for abstract data types
- [ ] Data abstraction implementations in Ada, C++, and Java
- [ ] Named encapsulation: namespaces and packages
- [ ] Parameterized abstract data types (generic containers)

### Part 11 Object-Oriented Programming

- [ ] Basic object-oriented concepts: objects, classes, and message passing
- [ ] Design issues of inheritance: single inheritance and multiple inheritance
- [ ] Dynamic binding and message polymorphism
- [ ] The problem of multiple inheritance: diamond inheritance and its solutions
- [ ] Smalltalk's pure object-oriented model
- [ ] Interfaces, Protocols, and Mixins

### Part 12 Functional Programming

- [ ] Basic functional programming concepts: mathematical functions and referential transparency
- [ ] λ calculus basics: abstraction, application, and Church encoding
- [ ] Higher-order functions, currying, and function composition
- [ ] Scheme basics: functions, lists, and recursion
- [ ] Lazy evaluation
- [ ] An introduction to Haskell: type classes and pattern matching
- [ ] An introduction to Monads: IO and computational contexts
- [ ] Implementation issues of functional languages: tail recursion and evaluation strategies

### Part 13 Logic Programming

- [ ] Basics of predicate calculus and clause form
- [ ] The resolution principle
- [ ] Basic elements of Prolog: terms, facts, rules, and goals
- [ ] Unification, backtracking, and the shortcomings of Prolog

### Part 14 Concurrent Programming

- [ ] Basic concepts of concurrency: tasks, synchronization, and races
- [ ] Shared-memory concurrency models: semaphores and Monitors
- [ ] Message passing models
- [ ] The Actor model
- [ ] CSP (Communicating Sequential Processes) and Go's channels
- [ ] Software Transactional Memory (STM) and asynchronous programming models (async/await)

### Part 15 Memory Management

- [ ] Managing stack allocation and heap allocation
- [ ] Reference Counting and the problem of cyclic references
- [ ] Tracing garbage collection: mark-and-sweep, copying, and generational collection
- [ ] Ownership systems and borrow checking (Rust's ownership model)
- [ ] Typical problems of manual memory management: dangling pointers and memory leaks

### Part 16 Metaprogramming

- [ ] Macro systems: textual macros and Hygienic Macros
- [ ] Reflection and Introspection
- [ ] Template metaprogramming and compile-time computation

### Part 17 Anatomy of Mainstream Language Mechanisms

- [ ] Python: dynamic typing, the object model, and the GIL
- [ ] Rust: ownership, lifetimes, and zero-cost abstractions
- [ ] Go: interfaces, goroutines, and a minimalist type system
- [ ] Java vs. C++: generic erasure, RAII, and value semantics compared
- [ ] A horizontal comparison of the type systems of five languages
- [ ] A comparison of error handling in five languages: exceptions, Result, and multiple return values

### Part 18 Language Virtual Machines and Runtimes

- [ ] The concept of language virtual machines: from p-code to modern VMs
- [ ] The JVM: class loading, bytecode, and Just-In-Time compilation (JIT)
- [ ] The JVM memory model and its garbage collector architecture
- [ ] BEAM: the Erlang virtual machine and its fault-tolerant concurrency model
- [ ] WebAssembly: a general-purpose virtual instruction set for the Web

> After writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
