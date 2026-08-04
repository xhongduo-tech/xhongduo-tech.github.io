---
pageClass: plain-doc
---

# Distributed Systems

This article maps to the chapter structure of MIT 6.824 and *Designing Data-Intensive Applications* (DDIA), systematically covering the core theory and classic engineering practice of distributed systems — from communication, clocks, and consistency to consensus, replication, partitioning, transactions, and fault tolerance — while dissecting milestone systems such as GFS, MapReduce, Spanner, and Dynamo one by one.

## Topic Planning

<ProgressGrid cat="cs/distributed-systems" />


### Part 1 Distributed Systems Overview

- [ ] What is a distributed system: motivation and definition
- [ ] Core challenges of distributed systems: partial failure and uncertainty
- [ ] Fallacies of Distributed Computing
- [ ] System models: synchronous, asynchronous, and partially asynchronous
- [ ] Failure models: crash failures, Byzantine failures, and network partitions
- [ ] Scalability, availability, and performance metrics
- [ ] The CAP theorem and its correct interpretation
- [ ] BASE and eventual consistency

### Part 2 RPC and Communication

- [ ] Fundamentals of remote procedure calls (RPC)
- [ ] Interface definition languages (IDL) and serialization: Protocol Buffers and Thrift
- [ ] gRPC design and streaming calls
- [ ] Message-passing models: point-to-point, publish-subscribe, and message queues
- [ ] Timeouts, retries, and idempotency
- [ ] The dilemma of network partitioning and timeout detection
- [ ] The trade-off between REST and RPC

### Part 3 Logical Clocks and Event Ordering

- [ ] Limitations of physical clocks and clock synchronization (NTP)
- [ ] The happens-before relation and partial order
- [ ] Lamport timestamps
- [ ] Vector clocks and causality detection
- [ ] Version vectors in replica systems
- [ ] Hybrid logical clocks (HLC) and TrueTime

### Part 4 Consistency Issues

- [ ] Overview of consistency models: from strong to weak
- [ ] Definition and verification of linearizability
- [ ] Sequential consistency
- [ ] Causal consistency
- [ ] Eventual consistency and session guarantees
- [ ] Practical trade-offs between consistency and availability

### Part 5 Consensus Algorithms

- [ ] Formal definition of the consensus problem and the FLP impossibility theorem
- [ ] Intuitive understanding of Paxos: a deep dive into Basic Paxos
- [ ] Multi-Paxos and leader election optimization
- [ ] The Raft algorithm in detail: leader election
- [ ] The Raft algorithm in detail: log replication
- [ ] The Raft algorithm in detail: safety and membership changes
- [ ] Comparing Paxos and Raft, and key points for engineering implementation
- [ ] Byzantine fault tolerance and a brief introduction to PBFT

### Part 6 Replication

- [ ] Goals and difficulties of replication
- [ ] Leader-based replication: synchronous and asynchronous
- [ ] Replication lag and read-your-writes consistency
- [ ] Multi-leader replication and write conflict handling
- [ ] Conflict resolution: last-write-wins (LWW) and conflict-free replicated data types (CRDTs)
- [ ] Leaderless replication: quorums, read repair, and anti-entropy
- [ ] The split-brain problem and fencing

### Part 7 Partitioning and Rebalancing

- [ ] Motivation for partitioning and partition key selection
- [ ] Range partitioning by key and hash partitioning
- [ ] Consistent hashing and virtual nodes
- [ ] Partitioning secondary indexes: local indexes and global indexes
- [ ] Rebalancing strategies: fixed partitioning, dynamic partitioning, and proportional to nodes
- [ ] Request routing and service discovery

### Part 8 Distributed Transactions

- [ ] Motivation and limitations of distributed transactions
- [ ] Two-phase commit (2PC) in detail and its blocking problem
- [ ] Three-phase commit (3PC) and its assumptions
- [ ] Snapshot isolation and MVCC
- [ ] The write skew problem in snapshot isolation
- [ ] Serializable and SSI (serializable snapshot isolation)
- [ ] The Saga pattern and compensating transactions
- [ ] Percolator-style optimistic distributed transactions

### Part 9 Fault Tolerance and Failure Detection

- [ ] Failure detectors: heartbeats and timeouts
- [ ] The Phi accrual failure detector
- [ ] Membership management: the gossip protocol
- [ ] Checkpointing and log recovery
- [ ] State machine replication
- [ ] Chaos engineering: fault injection and validating system resilience

### Part 10 Classic System Case Studies

- [ ] GFS: the architecture and consistency model of the Google File System
- [ ] MapReduce: the programming model and fault-tolerance mechanisms
- [ ] Bigtable: the data model and SSTables
- [ ] Dynamo: the design of an eventually consistent key-value store
- [ ] Spanner: TrueTime and a globally distributed database
- [ ] Kafka: distributed log and stream storage architecture
- [ ] Chubby: a Paxos-based lock service

### Part 11 Distributed Locks and Coordination Services

- [ ] Correctness issues of distributed locks: leases and fencing tokens
- [ ] The ZooKeeper data model and the Zab protocol
- [ ] Typical ZooKeeper applications: leader election, configuration, and queues
- [ ] etcd and Raft-based coordination services
- [ ] The Redlock controversy and database-lock-based alternatives

### Part 12 Stream Processing and Batch Processing

- [ ] Reviewing the batch processing model: from MapReduce to Spark/Flink
- [ ] Stream processing semantics: event time and processing time
- [ ] Windows and watermarks
- [ ] Implementing exactly-once semantics
- [ ] Unified stream-batch processing and the Lambda/Kappa architectures

### Part 13 Formal Verification of Distributed Systems

- [ ] Why formal verification is needed
- [ ] TLA+ basics: state machines and temporal logic
- [ ] Describing consensus protocols with TLA+
- [ ] Model checking and TLC
- [ ] PlusCal and formal methods in engineering practice

> After the writing is complete: create a new `xxx.md` in this directory, then change the corresponding entry above to `- [x] [Title](./xxx)`.
