---
pageClass: plain-doc
---

# Operating Systems

Learning a subject = writing all the posts corresponding to the classic textbook for that subject. The Operating Systems section tracks the chapter structure of *Operating System Concepts* (the Dinosaur Book) and OSTEP, covering everything in a university operating systems course, plus dedicated topics on the Linux kernel and virtualization/containers.

## Topic Plan

<ProgressGrid cat="cs/os" />


### Part 1 Introduction to Operating Systems

- [ ] Goals and roles of the operating system
- [ ] Computer system components and hardware architecture
- [ ] Interrupt mechanism and interrupt handling flow
- [ ] Storage structure and the memory hierarchy
- [ ] I/O structure: DMA and device controllers
- [ ] History of operating systems: batch, multiprogramming, time-sharing, and real-time systems
- [ ] Computing environments and architectures: single-processor, multiprocessor, and clustered systems
- [ ] User mode and kernel mode: dual-mode operation and privileged instructions

### Part 2 Operating System Structures

- [ ] Overview of operating system services
- [ ] Types of system calls and how they are implemented
- [ ] Overall operating system structure: simple and layered structures
- [ ] Microkernel structure and its pros and cons
- [ ] Modular kernels, loadable kernel modules, and hybrid kernels
- [ ] Virtual machines and hypervisors: Type 1 and Type 2
- [ ] Operating system generation, booting, and the startup process

### Part 3 Processes

- [ ] Process concept: the difference between a process and a program
- [ ] Process state model and state transition diagram
- [ ] Composition and role of the process control block (PCB)
- [ ] Process context switching and scheduling queues
- [ ] Process creation and termination: fork, copy-on-write, zombie processes, and orphan processes
- [ ] Interprocess communication overview: shared memory and message passing
- [ ] Classic IPC mechanisms: pipes, FIFOs, message queues, and shared memory segments
- [ ] Sockets, remote procedure calls (RPC), and the client-server model

### Part 4 Threads and Concurrency

- [ ] Thread concept and the advantages of multithreading
- [ ] Multithreading models: many-to-one, one-to-one, many-to-many
- [ ] Thread libraries: Pthreads, Java threads, and Windows threads
- [ ] Implicit threading: thread pools, Fork-Join, and OpenMP
- [ ] Thread-related issues: signal handling, thread cancellation, and thread-local storage
- [ ] Shared data and race conditions in concurrent execution
- [ ] Parallelism and concurrency: multicore environments and Amdahl's Law

### Part 5 CPU Scheduling

- [ ] Basic scheduling concepts: CPU bursts, I/O bursts, and scheduling criteria
- [ ] First-Come, First-Served (FCFS) scheduling algorithm
- [ ] Shortest-Job-First (SJF/SRTF) scheduling algorithm
- [ ] Priority scheduling and the starvation problem
- [ ] Round-Robin (RR) scheduling algorithm
- [ ] Multilevel queue and multilevel feedback queue scheduling
- [ ] Multiprocessor scheduling: load balancing and processor affinity
- [ ] Real-time scheduling: rate monotonic (RM) and earliest deadline first (EDF)
- [ ] The Linux scheduler: the Completely Fair Scheduler (CFS)

### Part 6 Process Synchronization

- [ ] The critical-section problem and the three mutual exclusion requirements
- [ ] Peterson's algorithm and software solutions
- [ ] Hardware synchronization primitives: Test-and-Set, Compare-and-Swap, and atomic variables
- [ ] Mutex locks and spinlocks
- [ ] Semaphore mechanisms: binary semaphores and counting semaphores
- [ ] Monitor mechanism and condition variables
- [ ] Classic synchronization problem: producer-consumer (bounded buffer)
- [ ] Classic synchronization problem: readers-writers
- [ ] Classic synchronization problems: dining philosophers, sleeping barber, and more
- [ ] Lock-free programming, the ABA problem of CAS, and priority inversion

### Part 7 Deadlocks

- [ ] Deadlock concept and the four necessary conditions for deadlock
- [ ] Resource allocation graphs and their analysis
- [ ] Overview of deadlock handling: prevention, avoidance, detection, and recovery
- [ ] Deadlock prevention: breaking the four necessary conditions
- [ ] Deadlock avoidance: safe states and safe sequences
- [ ] Banker's algorithm: single-resource and multi-resource versions
- [ ] Deadlock detection algorithm and when to run detection
- [ ] Deadlock recovery: process termination and resource preemption

### Part 8 Memory Management

- [ ] Basic memory management concepts: logical addresses and physical addresses
- [ ] Program linking and loading: absolute, relocatable, and dynamic run-time loading
- [ ] Memory protection mechanisms: base and limit registers
- [ ] Contiguous allocation: single contiguous allocation and fixed partition allocation
- [ ] Dynamic partition allocation: first fit, best fit, worst fit, and the fragmentation problem
- [ ] Swapping: process swap-in and swap-out
- [ ] Paging: page tables, page frames, and the address translation process
- [ ] TLB: the hardware mechanism that speeds up address translation
- [ ] Page table structures: hierarchical, hashed, and inverted page tables
- [ ] Segmentation and segmented paging
- [ ] The Buddy System allocation algorithm

### Part 9 Virtual Memory

- [ ] Virtual memory concept and the principle of locality
- [ ] Demand paging: page faults and the page-in process
- [ ] Page-fault rate and effective access time
- [ ] Copy-on-Write mechanism
- [ ] Page replacement: the optimal algorithm (OPT)
- [ ] Page replacement: First-In, First-Out (FIFO) and Belady's anomaly
- [ ] Page replacement: Least Recently Used (LRU) and its implementation
- [ ] Page replacement: clock, second-chance, and enhanced clock algorithms
- [ ] Frame allocation policy and thrashing: the working-set model
- [ ] Memory-mapped files (mmap) and their applications
- [ ] Huge Pages and transparent huge pages

### Part 10 Mass Storage and Disk Scheduling

- [ ] Disk structure and the principles of magnetic recording
- [ ] Disk performance parameters: seek time, rotational latency, and transfer rate
- [ ] Disk scheduling: FCFS and SSTF
- [ ] Disk scheduling: SCAN, C-SCAN, LOOK, and C-LOOK (the elevator algorithm family)
- [ ] SSDs: flash memory principles and wear leveling
- [ ] RAID: RAID 0/1/5/6 and data redundancy
- [ ] NVM (non-volatile memory) and the future of the storage hierarchy

### Part 11 File System Interface

- [ ] File concept: file attributes, types, and structure
- [ ] File operations: create, read, write, truncate, and delete
- [ ] File access methods: sequential, direct (random), and indexed sequential access
- [ ] Directory structure: single-level, two-level, tree-structured, and acyclic-graph directories
- [ ] File system mounting: mount points and the mount table
- [ ] File sharing: network file systems and consistency semantics
- [ ] File protection: access control lists (ACLs) and the access matrix

### Part 12 File System Implementation

- [ ] File system hierarchy: VFS, logical layer, and physical layer
- [ ] File control blocks and directory implementation
- [ ] File allocation methods: contiguous allocation
- [ ] File allocation methods: linked allocation and FAT
- [ ] File allocation methods: indexed allocation and multilevel indexes
- [ ] Free-space management: bitmaps, linked lists, and grouped linking
- [ ] File system performance optimization: buffering, caching, and read-ahead
- [ ] Journaling file systems and crash recovery
- [ ] NFS (network file system) architecture
- [ ] Deep dive into ext4: inodes, block groups, and journaling

### Part 13 I/O Systems

- [ ] I/O hardware: device controllers, device drivers, and character/block devices
- [ ] I/O control methods: program polling, interrupt-driven I/O, DMA, and channels
- [ ] The layered structure of I/O software
- [ ] Buffering: single buffering, double buffering, and circular buffering
- [ ] Spooling (SPOOLing)
- [ ] The kernel I/O subsystem and I/O performance optimization

### Part 14 Protection and Security

- [ ] Goals of protection, protection domains, and the principle of least privilege
- [ ] The access matrix and its implementation: access control lists and capability lists
- [ ] Comparing access control models and the trusted computing base
- [ ] Security threats: trojans, viruses, and worms
- [ ] Cryptography fundamentals: symmetric encryption, asymmetric encryption, and digital signatures
- [ ] User authentication mechanisms: passwords, two-factor authentication, and biometrics
- [ ] Program threats: buffer overflow attacks and defenses
- [ ] System and network threats: port scanning, denial of service, firewalls, and intrusion detection
- [ ] SELinux and mandatory access control (MAC)

### Part 15 Linux Special Topics: System Calls

- [ ] System call mechanics: the complete path from user mode to kernel mode
- [ ] The evolution of int 0x80, sysenter, and the syscall instruction
- [ ] The system call table, system call numbers, and argument passing
- [ ] vDSO and user-space acceleration of gettimeofday
- [ ] strace: tracing a process's system calls
- [ ] Quick reference for common system calls: open/read/write/fork/exec/wait

### Part 16 Linux Special Topics: Process Management and IPC

- [ ] Detailed look at the Linux process descriptor, task_struct
- [ ] Differences between fork, vfork, and clone
- [ ] The exec family and program loading: parsing the ELF format
- [ ] Process exit, the wait family, and daemon processes
- [ ] The Linux signal mechanism: signal generation, delivery, and handling
- [ ] Signal handlers and the reentrancy problem
- [ ] Implementing and using pipes and FIFOs
- [ ] System V and POSIX IPC: shared memory, semaphores, and message queues
- [ ] Linux thread implementation: NPTL and pthreads from the kernel's perspective

### Part 17 Linux Special Topics: epoll and I/O Multiplexing

- [ ] I/O models: blocking, non-blocking, synchronous, and asynchronous
- [ ] The principles and limitations of select and poll
- [ ] The epoll trio: epoll_create/epoll_ctl/epoll_wait
- [ ] The kernel implementation of epoll: red-black trees and the ready list
- [ ] The difference between level-triggered (LT) and edge-triggered (ET) modes and their pitfalls
- [ ] The Reactor pattern and event-driven programming
- [ ] io_uring: the new paradigm for asynchronous I/O in Linux

### Part 18 Linux Special Topics: Virtualization and Containers

- [ ] Virtualization basics: full virtualization, paravirtualization, and hardware-assisted virtualization
- [ ] The KVM architecture: cooperation between the kernel module and QEMU
- [ ] Memory and I/O virtualization: EPT, shadow page tables, virtio, and device passthrough
- [ ] Linux Namespaces: the six namespaces for process isolation
- [ ] A detailed look at Mount and Network namespaces
- [ ] Cgroups: CPU, memory, and I/O resource control (v1 and v2)
- [ ] How Docker works: images, union file systems, and the container runtime
- [ ] Handwriting a minimal container: Namespace + Cgroup + rootfs
- [ ] Container security: Capabilities, Seccomp, and AppArmor

> When the writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
