---
pageClass: plain-doc
---

# 操作系统

学完一个学科 = 写完该学科经典教材对应的全部博文。操作系统篇对标《操作系统概念》（恐龙书）与 OSTEP 的章节体系，覆盖大学操作系统课程的全部内容，并补充 Linux 内核与虚拟化容器专题。

## 主题规划

<ProgressGrid cat="cs/os" />


### 第一篇 操作系统引论

- [x] [操作系统的目标与作用](./goals-and-roles)
- [x] [计算机系统组成与硬件结构](./computer-system-structure)
- [x] [中断机制与中断处理流程](./interrupts)
- [x] [存储结构与多级存储层次](./storage-structure)
- [x] [输入输出结构：DMA 与设备控制器](./io-structure-dma)
- [x] [操作系统的发展历史：批处理、多道程序、分时与实时系统](./os-history)
- [x] [计算环境与体系结构：单处理器、多处理器与集群](./computing-environments)
- [x] [用户态与内核态：双模式操作与特权指令](./dual-mode-operation)

### 第二篇 操作系统结构

- [x] [操作系统服务概览](./os-services)
- [x] [系统调用的类型与实现原理](./system-call-types)
- [x] [操作系统的整体结构：简单结构与分层结构](./os-structure-layered)
- [x] [微内核结构及其优缺点](./microkernel)
- [x] [模块化内核、可加载内核模块与混合内核](./modular-kernel-lkm)
- [x] [虚拟机与 Hypervisor：Type-1 与 Type-2](./hypervisor)
- [x] [操作系统的生成、引导与启动流程](./os-boot)

### 第三篇 进程

- [x] [进程概念：进程与程序的区别](./process-concept)
- [x] [进程状态模型与状态转换图](./process-states)
- [x] [进程控制块 PCB 的组成与作用](./pcb)
- [x] [进程上下文切换与调度队列](./context-switch)
- [x] [进程创建与终止：fork、写时复制、僵尸进程与孤儿进程](./process-create-terminate)
- [x] [进程间通信概述：共享内存与消息传递](./ipc-overview)
- [x] [IPC 经典机制：管道、FIFO、消息队列与共享内存段](./ipc-mechanisms)
- [x] [套接字、远程过程调用 RPC 与客户机-服务器模型](./sockets-rpc)

### 第四篇 线程与并发

- [x] [线程概念与多线程的优势](./thread-concept)
- [x] [多线程模型：多对一、一对一、多对多](./threading-models)
- [x] [线程库：Pthreads、Java 线程与 Windows 线程](./thread-libraries)
- [x] [隐式线程：线程池、Fork-Join 与 OpenMP](./implicit-threading)
- [x] [线程相关问题：信号处理、线程取消与线程局部存储](./thread-issues)
- [x] [并发执行中的共享数据与竞态条件](./shared-data-race)
- [x] [并行与并发：多核环境与 Amdahl 定律](./parallelism-amdahl)

### 第五篇 CPU 调度

- [x] [调度基本概念：CPU 突发、I/O 突发与调度准则](./scheduling-basics)
- [x] [先来先服务（FCFS）调度算法](./fcfs)
- [x] [短作业优先（SJF/SRTF）调度算法](./sjf-srtf)
- [x] [优先级调度与饥饿问题](./priority-scheduling)
- [x] [时间片轮转（RR）调度算法](./round-robin)
- [x] [多级队列与多级反馈队列调度](./mlq-mlfq)
- [x] [多处理器调度：负载均衡与处理器亲和性](./multiprocessor-scheduling)
- [x] [实时调度：速率单调（RM）与最早截止时间优先（EDF）](./realtime-scheduling)
- [x] [Linux 调度器：CFS 完全公平调度器原理](./linux-cfs)

### 第六篇 进程同步

- [x] [临界区问题与互斥三原则](./critical-section)
- [x] [Peterson 算法与软件解决方案](./peterson)
- [x] [硬件同步原语：Test-and-Set、Compare-and-Swap 与原子变量](./hardware-sync)
- [x] [互斥锁 Mutex 与自旋锁 Spinlock](./mutex-spinlock)
- [x] [信号量机制：二元信号量与计数信号量](./semaphores)
- [x] [管程机制与条件变量](./monitors)
- [x] [经典同步问题：生产者-消费者（有界缓冲）问题](./producer-consumer)
- [x] [经典同步问题：读者-写者问题](./readers-writers)
- [x] [经典同步问题：哲学家进餐、理发师睡觉等问题](./dining-philosophers)
- [x] [无锁编程、CAS 的 ABA 问题与优先级反转](./lockfree-cas)

### 第七篇 死锁

- [x] [死锁的概念与产生死锁的四个必要条件](./deadlock-concept)
- [x] [资源分配图及其分析方法](./resource-allocation-graph)
- [x] [死锁处理方法概览：预防、避免、检测与恢复](./deadlock-handling)
- [x] [死锁预防：破坏四个必要条件](./deadlock-prevention)
- [x] [死锁避免：安全状态与安全序列](./deadlock-avoidance)
- [x] [银行家算法：单资源与多资源版本](./bankers-algorithm)
- [x] [死锁检测算法与检测时机](./deadlock-detection)
- [x] [死锁恢复：进程终止与资源剥夺](./deadlock-recovery)

### 第八篇 内存管理

- [x] [内存管理基本概念：逻辑地址与物理地址](./logical-physical-address)
- [x] [程序链接与装入：绝对装入、可重定位装入与动态运行时装入](./linking-loading)
- [x] [内存保护机制：基址寄存器与界限寄存器](./base-limit-registers)
- [x] [连续分配：单一连续分配与固定分区分配](./contiguous-allocation)
- [x] [动态分区分配：首次适应、最佳适应、最坏适应与碎片问题](./dynamic-partitioning)
- [x] [交换技术：进程换入与换出](./swapping)
- [x] [分页存储管理：页表、页框与地址转换过程](./paging)
- [x] [快表 TLB：加速地址转换的硬件机制](./tlb)
- [x] [页表结构：分层页表、哈希页表与倒置页表](./page-table-structures)
- [x] [分段存储管理与段页式存储管理](./segmentation)
- [x] [伙伴系统 Buddy System 分配算法](./buddy-system)

### 第九篇 虚拟内存

- [x] [虚拟内存的概念与局部性原理](./virtual-memory)
- [x] [请求分页：缺页中断与页面调入流程](./demand-paging)
- [x] [缺页率与有效访问时间](./page-fault-eat)
- [x] [写时复制（Copy-on-Write）机制](./copy-on-write)
- [x] [页面置换：最优置换算法 OPT](./opt-replacement)
- [x] [页面置换：先进先出 FIFO 与 Belady 异常](./fifo-belady)
- [x] [页面置换：最近最少使用 LRU 及其实现](./lru)
- [x] [页面置换：时钟、第二次机会与改进型时钟算法](./clock-replacement)
- [x] [页框分配策略与抖动（Thrashing）：工作集模型](./thrashing-working-set)
- [x] [内存映射文件 mmap 及其应用](./mmap)
- [x] [大页 Huge Page 与透明大页](./huge-pages)

### 第十篇 大容量存储与磁盘调度

- [x] [磁盘结构与磁记录原理](./disk-structure)
- [x] [磁盘性能参数：寻道时间、旋转延迟与传输率](./disk-performance)
- [x] [磁盘调度：FCFS 与 SSTF](./disk-fcfs-sstf)
- [x] [磁盘调度：SCAN、C-SCAN、LOOK 与 C-LOOK（电梯算法家族）](./disk-scan-cscan)
- [x] [SSD 固态硬盘：闪存原理与磨损均衡](./ssd)
- [x] [RAID 技术：RAID 0/1/5/6 与数据冗余](./raid)
- [x] [NVM 非易失性存储与存储器层次的未来](./nvm-storage)

### 第十一篇 文件系统接口

- [x] [文件概念：文件属性、类型与结构](./file-concept)
- [x] [文件操作：创建、读写、截断与删除](./file-operations)
- [x] [文件访问方法：顺序、直接（随机）与索引顺序访问](./file-access)
- [x] [目录结构：单层、双层、树形与无环图目录](./directory-structure)
- [x] [文件系统挂载：挂载点与挂载表](./mounting)
- [x] [文件共享：网络文件系统与一致性语义](./file-sharing)
- [x] [文件保护：访问控制列表 ACL 与访问矩阵](./file-protection)

### 第十二篇 文件系统实现

- [x] [文件系统层次结构：VFS、逻辑层与物理层](./fs-layers-vfs)
- [x] [文件控制块与目录实现](./fcb-directory-implementation)
- [x] [文件分配方法：连续分配](./contiguous-file-allocation)
- [x] [文件分配方法：链接分配与 FAT](./linked-allocation-fat)
- [x] [文件分配方法：索引分配与多级索引](./indexed-allocation)
- [x] [空闲空间管理：位图、链接表与成组链接](./free-space-management)
- [x] [文件系统性能优化：缓冲、缓存与预读](./fs-performance)
- [x] [日志文件系统与崩溃恢复](./journaling)
- [x] [NFS 网络文件系统架构](./nfs)
- [x] [ext4 文件系统深度剖析：Inode、块组与日志](./ext4)

### 第十三篇 I/O 系统

- [x] [I/O 硬件：设备控制器、设备驱动与字符/块设备](./io-hardware)
- [x] [I/O 控制方式：程序轮询、中断驱动、DMA 与通道](./io-control)
- [x] [I/O 软件的层次结构](./io-layers)
- [x] [缓冲技术：单缓冲、双缓冲与循环缓冲](./buffering)
- [x] [假脱机技术 SPOOLing](./spooling)
- [x] [内核 I/O 子系统与 I/O 性能优化](./kernel-io-subsystem)

### 第十四篇 保护与安全

- [x] [保护的目标、保护域与最小特权原则](./protection-domains)
- [x] [访问矩阵及其实现：访问控制表与能力表](./access-matrix)
- [x] [访问控制模型的比较与可信计算基](./access-control-models)
- [x] [安全威胁：木马、病毒与蠕虫](./security-threats)
- [x] [密码学基础：对称加密、非对称加密与数字签名](./crypto-basics)
- [x] [用户认证机制：口令、双因素与生物识别](./authentication)
- [x] [程序威胁：缓冲区溢出攻击与防御](./buffer-overflow)
- [x] [系统与网络威胁：端口扫描、拒绝服务、防火墙与入侵检测](./network-threats)
- [x] [SELinux 与强制访问控制 MAC](./selinux-mac)

### 第十五篇 Linux 专题：系统调用

- [x] [系统调用原理：从用户态到内核态的完整路径](./syscall-principle)
- [x] [软中断 int 0x80、sysenter 与 syscall 指令的演进](./syscall-instructions)
- [x] [系统调用表、系统调用号与参数传递](./syscall-table)
- [x] [vDSO 与 gettimeofday 的用户态加速](./vdso)
- [x] [strace：跟踪进程的系统调用](./strace)
- [x] [常见系统调用速查：open/read/write/fork/exec/wait](./syscall-cheatsheet)

### 第十六篇 Linux 专题：进程管理与 IPC

- [x] [Linux 进程描述符 task_struct 详解](./task-struct)
- [x] [fork、vfork 与 clone 的区别](./fork-vfork-clone)
- [x] [exec 家族与程序加载：ELF 格式解析](./exec-elf)
- [x] [进程退出、wait 家族与守护进程 Daemon](./process-exit-daemon)
- [x] [Linux 信号机制：信号的产生、递送与处理](./linux-signals)
- [x] [信号处理函数与可重入问题](./signal-reentrancy)
- [x] [管道与 FIFO 的实现与使用](./pipe-fifo)
- [x] [System V 与 POSIX IPC：共享内存、信号量与消息队列](./sysv-posix-ipc)
- [x] [Linux 线程实现：NPTL 与 pthread 的内核视角](./nptl-pthread)

### 第十七篇 Linux 专题：epoll 与 I/O 多路复用

- [x] [I/O 模型：阻塞、非阻塞、同步与异步](./io-models)
- [x] [select 与 poll 的原理与局限](./select-poll)
- [x] [epoll 三剑客：epoll_create/epoll_ctl/epoll_wait](./epoll-api)
- [x] [epoll 的内核实现：红黑树与就绪链表](./epoll-implementation)
- [x] [水平触发 LT 与边缘触发 ET 的区别与陷阱](./lt-et)
- [x] [Reactor 模式与事件驱动编程](./reactor)
- [x] [io_uring：Linux 异步 I/O 的新范式](./io-uring)

### 第十八篇 Linux 专题：虚拟化与容器

- [x] [虚拟化基础：全虚拟化、半虚拟化与硬件辅助虚拟化](./virtualization-basics)
- [x] [KVM 架构：内核模块与 QEMU 的协作](./kvm-qemu)
- [x] [内存与 I/O 虚拟化：EPT、影子页表、virtio 与设备直通](./memory-io-virtualization)
- [x] [Linux Namespace：进程隔离的六大命名空间](./linux-namespaces)
- [x] [Mount 与 Network Namespace 详解](./mount-network-namespace)
- [x] [Cgroup：CPU、内存与 I/O 资源控制（v1 与 v2）](./cgroup)
- [x] [Docker 原理：镜像、联合文件系统与容器运行时](./docker-principles)
- [x] [手写一个简易容器：Namespace + Cgroup + rootfs](./build-mini-container)
- [x] [容器安全：Capability、Seccomp 与 AppArmor](./container-security)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
