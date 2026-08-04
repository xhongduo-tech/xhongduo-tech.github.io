---
pageClass: plain-doc
---

# Database Systems

This section covers the full content of a university database course, following *Database System Concepts* (Silberschatz) as the main thread, plus a special topic on distributed databases. The goal: after finishing each chapter, write the corresponding blog post for every section.

## Topic Planning

<ProgressGrid cat="cs/database" />


### Part I Relational Databases and SQL

#### Chapter 1 Introduction to Database Systems

- [ ] Goals of database systems: from file systems to a DBMS
- [ ] Data views: data abstraction, instances and schemas, data models
- [ ] Database languages: DDL and DML
- [ ] Relational databases: tables, DML, and database design
- [ ] Database engine: storage manager and query processor
- [ ] Database and application architecture: two-tier and three-tier architectures
- [ ] Database users and administrators (DBA)
- [ ] History and evolution of database systems

#### Chapter 2 The Relational Model

- [ ] Structure of relational databases: relations, tuples, and attributes
- [ ] Database schemas and relation instances
- [ ] Keys: superkeys, candidate keys, primary keys, and foreign keys
- [ ] Schema diagrams and relational schema design
- [ ] Overview of relational query languages
- [ ] Relational algebra: basic operations
- [ ] Relational algebra: additional and extended operations

#### Chapter 3 SQL Fundamentals

- [ ] Overview of the SQL query language and data definition (DDL)
- [ ] Single-relation queries: basic structure of SELECT
- [ ] Multi-relation queries: joins and the Cartesian product
- [ ] Set operations: union, intersection, and difference
- [ ] Aggregate functions and grouping (GROUP BY / HAVING)
- [ ] Nested subqueries and set-membership comparisons
- [ ] Null values and three-valued logic
- [ ] Database modification: INSERT, UPDATE, DELETE

#### Chapter 4 Intermediate SQL

- [ ] Join expressions: inner, outer, and natural joins
- [ ] Views: definition, querying, and updating
- [ ] SQL semantics of transactions
- [ ] Integrity constraints: primary keys, foreign keys, CHECK, and assertions
- [ ] SQL data types and schemas
- [ ] Authorization: privileges, roles, and revoking
- [ ] Recursion of views and authorization

#### Chapter 5 Advanced SQL

- [ ] Functions and procedures: PL/SQL-style stored procedures
- [ ] Triggers: definition, events, and semantics
- [ ] Recursive queries: WITH RECURSIVE
- [ ] Advanced aggregation: window functions, ranking, and bucketing
- [ ] OLAP operations: CUBE, ROLLUP, and data cubes

#### Chapter 6 Formal Relational Query Languages

- [ ] Tuple relational calculus
- [ ] Domain relational calculus
- [ ] Expressive equivalence of relational algebra and relational calculus

### Part II Database Design

#### Chapter 7 The Entity-Relationship Model (ER Model)

- [ ] Overview of the design process and requirements analysis
- [ ] Entity sets, attributes, and keys
- [ ] Relationship sets and mapping cardinalities
- [ ] Participation constraints and weak entity sets
- [ ] Eliminating redundancy: attribute design and composite attributes
- [ ] ER diagram notation and examples
- [ ] Extended ER features: specialization, generalization, aggregation
- [ ] Conversion from ER diagrams to relational schemas
- [ ] Other database design issues and UML

#### Chapter 8 Normalization Theory

- [ ] Features of good relational design and counterexamples
- [ ] Functional dependencies: definition, closure, and trivial dependencies
- [ ] The relationship between keys and functional dependencies
- [ ] Overview of normal forms: 1NF, 2NF, 3NF, BCNF
- [ ] Theory of functional dependencies: Armstrong's axioms and canonical covers
- [ ] Lossless-join decomposition and dependency preservation
- [ ] BCNF decomposition algorithm
- [ ] Third normal form decomposition algorithm
- [ ] Multivalued dependencies and fourth normal form
- [ ] Other normal forms and design trade-offs: denormalization

### Part III Data Storage and Query Engine

#### Chapter 9 Data Storage and File Organization

- [ ] Overview of physical storage media: disks, SSDs, memory hierarchy
- [ ] Disk structure and the cost of disk block access
- [ ] File organization: heap files and sequential files
- [ ] Data dictionary and system catalog
- [ ] Buffer management: buffer pool and replacement policies
- [ ] Column-oriented vs. row-oriented storage organization

#### Chapter 10 Indexing and Hashing

- [ ] Basic indexing concepts: ordered and secondary indexes
- [ ] Structure and search of B+ trees
- [ ] Insertion and deletion in B+ trees
- [ ] B+ tree variants: B trees, B* trees, and in-memory optimization
- [ ] LSM trees: principles, compaction, and read/write amplification
- [ ] Static hashing: hash functions and bucket overflow
- [ ] Dynamic hashing: extendible hashing and linear hashing
- [ ] Bitmap indexes and other secondary indexes

#### Chapter 11 Query Processing

- [ ] Query processing steps and execution plans
- [ ] Measuring query cost
- [ ] Algorithms for selection: linear scan and index scan
- [ ] Sorting: external merge sort
- [ ] Join operations: nested-loop and block nested-loop joins
- [ ] Join operations: index nested-loop, merge, and hash joins
- [ ] Implementation of aggregation and duplicate elimination
- [ ] Expression evaluation: materialization and pipelining
- [ ] In-memory queries and column-oriented execution models

#### Chapter 12 Query Optimization

- [ ] Overview of query optimization: logical and physical optimization
- [ ] Transformation rules for relational expressions (equivalent transformations)
- [ ] Query rewriting: predicate pushdown and join ordering
- [ ] Statistics and cost estimation: cardinality estimation
- [ ] Cost estimation: selectivity, histograms, and sampling
- [ ] Cost-based optimizers and dynamic-programming enumeration
- [ ] Materialized views and query result caching

### Part IV Transactions and Failure Recovery

#### Chapter 13 Transactions

- [ ] Transaction concepts and the ACID properties
- [ ] Transaction state model
- [ ] The need for and problems of concurrent execution
- [ ] Serializability: conflict serializability and view serializability
- [ ] Recoverable and cascadeless schedules
- [ ] Transaction isolation levels: from read uncommitted to serializable

#### Chapter 14 Concurrency Control

- [ ] Lock-based protocols: shared and exclusive locks
- [ ] Two-phase locking (2PL) and its variants
- [ ] Deadlock handling: detection, prevention, and wait-for graphs
- [ ] Lock granularity and intention locks
- [ ] Multiversion concurrency control (MVCC): snapshot isolation
- [ ] The write-skew problem in snapshot isolation
- [ ] Timestamp-ordering protocols
- [ ] Optimistic concurrency control (validation based)
- [ ] Predicate reads, the phantom phenomenon, and index locks

#### Chapter 15 Recovery System

- [ ] Failure classification and the storage hierarchy
- [ ] Log-based recovery: WAL (write-ahead logging)
- [ ] Deferred and immediate modification
- [ ] Checkpointing
- [ ] Recovery algorithms: UNDO and REDO
- [ ] Buffer management policies: STEAL and NO-FORCE
- [ ] ARIES recovery algorithm: log records, analysis and redo phases
- [ ] ARIES recovery algorithm: undo phase and compensation log records (CLR)
- [ ] Fuzzy checkpoints and high availability: backups and remote disaster recovery

### Part V Special Topic: Distributed Databases

#### Chapter 16 Replication and Consensus

- [ ] Motivations for replication: availability, latency, and read scaling
- [ ] Leader-based and multi-leader replication
- [ ] Replication logs and replication lag
- [ ] The consensus problem and the FLP impossibility
- [ ] Paxos: roles, proposals, and quorums
- [ ] Raft: leader election and log replication
- [ ] Raft membership changes and safety guarantees

#### Chapter 17 Sharding and Distributed Transactions

- [ ] Data partitioning: range, hash, and consistent hashing
- [ ] Rebalancing and routing: metadata management
- [ ] Distributed transactions and the atomic commit problem
- [ ] Two-phase commit (2PC): process and failure handling
- [ ] Three-phase commit (3PC) and its limitations
- [ ] TCC: the Try-Confirm-Cancel compensation model
- [ ] The Saga pattern: orchestration and choreography
- [ ] Distributed deadlock detection and global snapshots

#### Chapter 18 CAP and Consistency Models

- [ ] The CAP theorem: meaning and common misconceptions
- [ ] Linearizability and sequential consistency
- [ ] Causal consistency and session guarantees
- [ ] Eventual consistency and convergence
- [ ] The spectrum of consistency models and trade-offs

#### Chapter 19 NoSQL and NewSQL

- [ ] The rise of NoSQL and data model classification
- [ ] Key-value stores and document databases
- [ ] Wide-column stores: BigTable and Cassandra
- [ ] Graph databases and graph queries
- [ ] NewSQL: horizontally scalable relational databases
- [ ] Google Spanner: TrueTime and external consistency
- [ ] TiDB: architecture, TiKV, and distributed SQL execution

#### Chapter 20 Column Stores and OLAP

- [ ] Workload differences between OLTP and OLAP
- [ ] Columnar storage: compression, late materialization, and vectorized execution
- [ ] Data warehouses, data lakes, and lakehouses
- [ ] Star schemas and snowflake schemas
- [ ] Column-store practice: ClickHouse and DuckDB
- [ ] HTAP: hybrid transactional and analytical processing

> After writing: create a `xxx.md` file in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
