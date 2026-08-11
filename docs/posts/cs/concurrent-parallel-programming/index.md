---
pageClass: plain-doc
---

# 并发与并行编程

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Herlihy & Shavit, "The Art of Multiprocessor Programming" (2nd, 2020)
- Bryant & O'Hallaron, "Computer Systems: A Programmer's Perspective" (3rd, 2015)
- Tanenbaum & Van Steen, "Distributed Systems" (3rd, 2017)

## 主题规划

<ProgressGrid cat="cs/concurrent-parallel-programming" />

### 第1篇

- [x] [线程与并发执行 (CS:APP §12.1)](./threads-and-concurrent-execution)
- [x] [共享变量与互斥 (CS:APP §12.5)](./shared-variables-and-mutual-exclusion)
- [x] [信号量与生产者-消费者 (CS:APP §12.5.5)](./semaphores-and-producer-consumer)
- [x] [并发对象与同步原语相对能力 (Herlihy §3-6)](./concurrent-data-structures-lists-queues-stacks)
- [x] [锁的实现与 CAS (Herlihy §7)](./lock-implementation-and-cas)
- [x] [条件变量与读者-写者 (Herlihy §8)](./condition-variables-and-reader-writer)
- [x] [并发数据结构：链表/队列/栈 (Herlihy §9-10)](./concurrent-data-structures-lists-queues-stacks)
- [x] [并发哈希表与跳表 (Herlihy §11-12)](./concurrent-hash-tables-and-skip-lists)

### 第2篇

- [x] [死锁检测与预防 (CS:APP §12.7)](./deadlock-detection-and-prevention)
- [x] [并行算法与工作窃取 (Herlihy §15)](./parallel-algorithms-and-work-stealing)
- [x] [同步屏障 Barrier (Herlihy §16)](./synchronization-barriers)
- [x] [事务内存 TM (Herlihy §17)](./transactional-memory)
- [x] [内存一致性与顺序 (Herlihy §3)](./memory-consistency-and-ordering)
