---
pageClass: plain-doc
---

# 计算机组成原理

本分类对标唐朔飞《计算机组成原理》与《深入理解计算机系统》（CS:APP）的章节体系，覆盖计算机组成原理课程的全部内容：从数据表示到存储器层次、指令系统、CPU、流水线、总线与输入输出系统，学完即写完。

## 主题规划

<ProgressGrid cat="cs/computer-organization" />


### 第一篇 计算机系统概述

- [x] [计算机的产生与发展：从电子管到超大规模集成电路](./computer-history)
- [x] [计算机的分类与发展趋势](./computer-classification)
- [x] [冯·诺依曼计算机的基本思想与特点](./von-neumann)
- [x] [计算机的基本组成：运算器、控制器、存储器、输入输出设备](./computer-composition)
- [x] [计算机的工作过程：取指令、分析指令、执行指令](./instruction-cycle)
- [x] [计算机系统的层次结构：从微程序机器级到高级语言机器级](./computer-hierarchy)
- [x] [计算机体系结构与计算机组成的区别](./architecture-vs-organization)
- [x] [计算机硬件的主要技术指标：机器字长、存储容量、运算速度](./hardware-indicators)
- [x] [性能评价：CPU 执行时间、CPI、MIPS 与 MFLOPS](./performance-evaluation)
- [x] [阿姆达尔定律与系统性能改进](./amdahls-law)
- [x] [程序的编译、汇编、链接与装载过程](./compile-assemble-link-load)
- [x] [信息的二进制表示与程序的机器级视角](./binary-representation-machine-view)

### 第二篇 数据的表示与运算

- [x] [进位计数制及其相互转换](./positional-numeral-systems)
- [x] [无符号数与有符号数的机器表示](./signed-unsigned-representation)
- [x] [原码、反码、补码与移码](./sign-magnitude-ones-complement-twos-complement)
- [x] [定点数的表示：定点小数与定点整数](./fixed-point-representation)
- [x] [补码的移位运算与符号扩展](./shift-operations-sign-extension)
- [x] [定点加减运算及其实现](./fixed-point-add-sub)
- [x] [溢出的概念与判断方法：单符号位法与双符号位法](./overflow-detection)
- [x] [定点乘法运算：原码一位乘与补码一位乘（Booth 算法）](./fixed-point-multiplication-booth)
- [x] [定点除法运算：恢复余数法与加减交替法](./fixed-point-division)
- [x] [阵列乘法器与阵列除法器](./array-multiplier-divider)
- [x] [浮点数的表示：阶码、尾数与规格化](./floating-point-representation)
- [x] [IEEE 754 浮点数标准：单精度、双精度与特殊值](./ieee754)
- [x] [浮点数的加减运算：对阶、尾数求和、规格化与舍入](./floating-point-add-sub)
- [x] [浮点数的乘除运算](./floating-point-mul-div)
- [x] [浮点运算的舍入方式与精度问题](./floating-point-rounding)
- [x] [浮点运算器的结构与浮点运算流水线](./floating-point-unit)
- [x] [C 语言中的整数与浮点数：类型转换、整数溢出与浮点数陷阱](./c-integer-float-tricks)
- [x] [算术逻辑单元 ALU 的基本结构](./alu-structure)
- [x] [加法器设计：串行进位、并行进位与先行进位](./adder-design)
- [x] [字符与字符串的表示：ASCII、汉字编码与 BCD 码](./character-encoding)
- [x] [奇偶校验码](./parity-check-code)
- [x] [海明校验码](./hamming-code)
- [x] [循环冗余校验码 CRC](./crc-code)

### 第三篇 存储器层次结构

- [x] [存储器的分类：按介质、存取方式与作用分类](./memory-classification)
- [x] [存储器层次结构：寄存器、Cache、主存、辅存](./memory-hierarchy)
- [x] [程序访问的局部性原理：时间局部性与空间局部性](./locality-principle)
- [x] [半导体随机存取存储器：SRAM 的结构与工作原理](./sram)
- [x] [DRAM 的结构、读写原理与刷新方式](./dram)
- [x] [只读存储器 ROM：掩膜 ROM、PROM、EPROM、EEPROM 与闪存](./rom-flash)
- [x] [主存储器与 CPU 的连接](./main-memory-cpu-connection)
- [x] [存储器容量扩展：位扩展、字扩展与字位同时扩展](./memory-capacity-expansion)
- [x] [提高访存速度：双端口存储器与多体交叉存储器](./dual-port-multibank-memory)
- [x] [Cache 的工作原理与基本结构](./cache-principle)
- [x] [Cache 与主存的地址映射：直接映射、全相联映射与组相联映射](./cache-address-mapping)
- [x] [Cache 的替换算法：随机、先进先出、LRU 与最不经常使用](./cache-replacement)
- [x] [Cache 的写策略：写直达、写回与写缓冲](./cache-write-policy)
- [x] [Cache 的性能分析：命中率、平均访问时间与多级 Cache](./cache-performance)
- [x] [指令 Cache 与数据 Cache 的分离](./instruction-data-cache)
- [x] [虚拟存储器的基本概念](./virtual-memory-basics)
- [x] [页式虚拟存储器与页表](./paging-page-table)
- [x] [段式虚拟存储器与段页式虚拟存储器](./segmentation-segmented-paging)
- [x] [快表 TLB 与多级页表](./tlb-multilevel-page-table)
- [x] [页面替换算法：OPT、FIFO、LRU 与 Clock](./page-replacement-algorithms)
- [x] [存储保护：越界保护与访问权限保护](./memory-protection)
- [x] [虚拟内存作为缓存与内存管理工具：mmap、动态内存分配与垃圾回收](./virtual-memory-as-cache-malloc)

### 第四篇 指令系统

- [x] [机器指令的一般格式：操作码与地址码](./instruction-format)
- [x] [指令字长与机器字长的关系](./instruction-word-length)
- [x] [操作数类型与操作类型](./operand-types-operation-types)
- [x] [指令寻址：顺序寻址与跳跃寻址](./instruction-addressing)
- [x] [数据寻址方式：立即、直接、间接、寄存器、寄存器间接](./data-addressing-modes-1)
- [x] [数据寻址方式：相对寻址、基址寻址、变址寻址与堆栈寻址](./data-addressing-modes-2)
- [x] [操作码扩展技术与指令格式设计](./opcode-extension)
- [x] [RISC 与 CISC 的特点及比较](./risc-cisc)
- [x] [x86 指令体系结构：数据传送、算术逻辑与控制转移指令](./x86-instruction-set)
- [x] [数据的对齐存放与大端、小端存储模式](./alignment-endianness)

### 第五篇 中央处理器

- [x] [CPU 的功能与组成：运算器、控制器与寄存器组](./cpu-function-composition)
- [x] [CPU 中的主要寄存器：PC、IR、MAR、MDR、PSW](./cpu-registers)
- [x] [指令周期：取指、间址、执行与中断周期](./instruction-cycle-interrupt)
- [x] [指令周期的数据流分析](./instruction-cycle-dataflow)
- [x] [数据通路的结构与功能：单总线、双总线与三总线结构](./datapath-structures)
- [x] [时序系统与多级时序：机器周期、节拍与工作脉冲](./timing-system)
- [x] [控制方式：同步控制、异步控制与联合控制](./control-methods)
- [x] [硬布线控制器的设计原理](./hardwired-control)
- [x] [微程序控制的基本思想：微命令、微操作与微指令](./microprogram-control-basics)
- [x] [微程序控制器的组成与工作原理](./microprogram-controller)
- [x] [微指令的编码方式：直接编码、字段直接编码与字段间接编码](./microinstruction-encoding)
- [x] [微地址的形成方式与微指令格式](./microaddress-format)
- [x] [硬布线控制与微程序控制的比较](./hardwired-vs-microprogram)
- [x] [机器级程序的表示：汇编指令与机器代码的对应](./machine-level-program-representation)
- [x] [过程调用的机器级实现：栈帧、参数传递与返回地址](./procedure-call-stack-frame)
- [x] [异常与控制流：陷阱、故障、终止与上下文切换](./exceptions-control-flow)

### 第六篇 指令流水线与高级专题

- [x] [指令流水线的基本概念与流水段划分](./pipeline-basics)
- [x] [流水线的性能指标：吞吐率、加速比与效率](./pipeline-performance)
- [x] [结构冒险及其解决](./structural-hazard)
- [x] [数据冒险及其解决：停顿与数据前递（旁路）](./data-hazard-forwarding)
- [x] [控制冒险及其解决：分支延迟槽与延迟分支](./control-hazard-branch)
- [x] [分支预测：静态预测、动态预测与分支目标缓冲器 BTB](./branch-prediction)
- [x] [流水线的中断处理与精确异常](./pipeline-interrupt-precise-exception)
- [x] [超标量流水线与动态调度](./superscalar-dynamic-scheduling)
- [x] [超流水线与超长指令字 VLIW](./superpipeline-vliw)
- [x] [乱序执行：记分牌算法与 Tomasulo 算法](./tomasulo-scoreboard)
- [x] [寄存器重命名与重排序缓冲 ROB](./register-renaming-rob)
- [x] [多核处理器与多处理器系统](./multicore-multiprocessor)
- [x] [缓存一致性问题与 MESI 协议](./cache-coherence-mesi)
- [x] [同时多线程 SMT 与超线程技术](./smt-hyperthreading)

### 第七篇 总线

- [x] [总线的基本概念与特性](./bus-basics)
- [x] [总线的分类：片内总线、系统总线与通信总线](./bus-classification)
- [x] [总线结构与总线性能指标：宽度、带宽与时钟频率](./bus-structure-performance)
- [x] [总线仲裁：链式查询、计数器定时查询与独立请求方式](./bus-arbitration)
- [x] [分布式仲裁](./distributed-arbitration)
- [x] [总线通信控制：同步通信、异步通信、半同步通信与分离式通信](./bus-communication-control)
- [x] [总线标准：ISA、EISA、PCI、PCIe 与 USB](./bus-standards)

### 第八篇 输入输出系统

- [x] [输入输出系统概述：I/O 设备与 I/O 软件](./io-system-overview)
- [x] [外部设备：键盘、鼠标、显示器与打印机](./external-devices)
- [x] [I/O 接口的功能、组成与类型](./io-interface)
- [x] [I/O 端口及其编址：统一编址与独立编址](./io-ports)
- [x] [程序查询方式及其接口](./programmed-io)
- [x] [程序中断方式的基本概念](./interrupt-basics)
- [x] [中断请求、中断判优与中断响应](./interrupt-request-arbitration)
- [x] [中断服务程序与中断处理过程](./interrupt-service)
- [x] [多重中断与中断屏蔽](./multiple-interrupts-masking)
- [x] [DMA 方式的基本概念与 DMA 接口](./dma-basics)
- [x] [DMA 的传送方式与传送过程](./dma-transfer)
- [x] [DMA 方式与中断方式的比较](./dma-vs-interrupt)
- [x] [通道方式：通道类型与通道工作过程](./channel-io)
- [x] [磁盘存储器：结构、性能指标与磁盘调度算法](./disk-storage)
- [x] [磁盘阵列 RAID 的分级与原理](./raid)
- [x] [固态硬盘 SSD 的结构与读写特性](./ssd)
- [x] [系统级 I/O：Unix I/O、文件的读写与共享](./system-level-io)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
