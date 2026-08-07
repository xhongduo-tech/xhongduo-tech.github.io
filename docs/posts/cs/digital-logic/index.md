---
pageClass: plain-doc
---

# 数字逻辑

本分类对标阎石《数字电子技术基础》的全部章节，并补充硬件描述语言与 CPU 构成两条进阶路线。目标：写完这本教材对应的全部博文，即学完数字逻辑这门学科。

## 主题规划

<ProgressGrid cat="cs/digital-logic" />


### 第一篇 数制与码制

- [x] [数字信号与数字电路概述](./digital-signals-overview)
- [x] [常用数制：二进制、八进制、十进制、十六进制](./number-systems)
- [x] [不同数制之间的相互转换](./radix-conversion)
- [x] [二进制算术运算：原码、反码与补码](./signed-number-representation)
- [x] [常用编码：BCD 码、格雷码（Gray Code）、ASCII 码与奇偶校验码](./binary-codes)

### 第二篇 逻辑代数基础

- [x] [逻辑代数的三种基本运算：与、或、非](./boolean-algebra-basic-operations)
- [x] [复合逻辑运算：与非、或非、异或、同或](./compound-logic-operations)
- [x] [逻辑代数的基本公式与常用公式](./boolean-algebra-formulas)
- [x] [逻辑代数的基本定理：代入定理、反演定理、对偶定理](./boolean-algebra-theorems)
- [x] [逻辑函数及其表示方法：真值表、逻辑式、逻辑图、波形图](./logic-functions-representations)
- [x] [逻辑函数的两种标准形式：最小项之和与最大项之积](./canonical-forms-minterm-maxterm)
- [x] [逻辑函数的公式化简法](./boolean-algebra-simplification)
- [x] [卡诺图（Karnaugh Map）表示法与卡诺图化简法](./karnaugh-map)
- [x] [具有无关项的逻辑函数及其化简](./dont-care-conditions)

### 第三篇 门电路

- [x] [半导体二极管与三极管的开关特性](./semiconductor-switching-characteristics)
- [x] [分立元件门电路：二极管与门、或门与三极管非门](./discrete-gate-circuits)
- [x] [TTL 反相器的电路结构与工作原理](./ttl-inverter-circuit)
- [x] [TTL 反相器的静态输入特性与输出特性](./ttl-inverter-characteristics)
- [x] [TTL 门电路的其他类型：与非门、或非门、OC 门（集电极开路门）与三态门](./ttl-gate-types)
- [x] [CMOS 反相器的电路结构与工作原理](./cmos-inverter-circuit)
- [x] [CMOS 门电路的其他类型：传输门、漏极开路门（OD 门）与三态门](./cmos-gate-types)
- [x] [TTL 与 CMOS 电路的接口与使用注意事项](./ttl-cmos-interfacing)

### 第四篇 组合逻辑电路

- [x] [组合逻辑电路的特点与功能描述](./combinational-logic-overview)
- [x] [组合逻辑电路的分析方法](./combinational-analysis)
- [x] [组合逻辑电路的设计方法](./combinational-design)
- [x] [常用组合逻辑模块：编码器与普通编码器、优先编码器](./encoders)
- [x] [常用组合逻辑模块：译码器、二-十进制译码器与显示译码器](./decoders)
- [x] [常用组合逻辑模块：数据选择器与数据分配器](./multiplexers-demultiplexers)
- [x] [常用组合逻辑模块：加法器、半加器与全加器](./adders)
- [x] [常用组合逻辑模块：数值比较器](./magnitude-comparators)
- [x] [用中规模集成电路（MSI）设计组合逻辑电路](./msi-based-design)
- [x] [组合逻辑电路中的竞争-冒险现象及其消除](./hazards-race)

### 第五篇 触发器

- [x] [触发器概述：双稳态与记忆功能](./flip-flop-overview)
- [x] [SR 锁存器（基本 RS 触发器）](./sr-latch)
- [x] [电平触发的触发器：同步 SR 触发器与 D 锁存器](./level-triggered-flip-flops)
- [x] [脉冲触发与边沿触发的触发器](./pulse-edge-triggered-flip-flops)
- [x] [边沿 D 触发器与 JK 触发器](./edge-triggered-d-jk)
- [x] [T 触发器与 T' 触发器](./t-flip-flops)
- [x] [触发器的逻辑功能分类及相互转换](./flip-flop-conversion)
- [x] [触发器的动态特性：建立时间、保持时间与传输延迟](./flip-flop-timing-characteristics)

### 第六篇 时序逻辑电路

- [x] [时序逻辑电路概述：结构模型（Mealy 型与 Moore 型）](./sequential-logic-overview)
- [x] [同步时序逻辑电路的分析方法](./synchronous-sequential-analysis)
- [x] [异步时序逻辑电路的分析方法](./asynchronous-sequential-analysis)
- [x] [寄存器与移位寄存器](./registers-shift-registers)
- [x] [计数器（一）：同步二进制加法/减法计数器](./synchronous-counters)
- [x] [计数器（二）：异步计数器与十进制计数器](./asynchronous-counters-decimal)
- [x] [用中规模集成计数器构成任意进制计数器：置零法与置数法](./msi-counters-arbitrary-modulus)
- [x] [移位寄存器型计数器：环形计数器与扭环形计数器](./shift-register-counters)
- [x] [顺序脉冲发生器与序列信号发生器](./pulse-sequence-generators)
- [x] [同步时序逻辑电路的设计方法：状态化简、状态分配与自启动检查](./synchronous-sequential-design)

### 第七篇 脉冲波形的产生与整形

- [x] [脉冲波形参数与整形电路概述](./pulse-waveform-parameters)
- [x] [施密特触发器（Schmitt Trigger）：门电路构成与工作原理](./schmitt-trigger-circuit)
- [x] [施密特触发器的应用：波形变换、整形与幅度鉴别](./schmitt-trigger-applications)
- [x] [单稳态触发器：微分型、积分型与集成单稳态触发器](./monostable-multivibrator)
- [x] [多谐振荡器：对称式、非对称式与环形振荡器](./astable-multivibrator)
- [x] [555 定时器的电路结构与功能](./555-timer)
- [x] [用 555 定时器构成施密特触发器、单稳态触发器与多谐振荡器](./555-timer-applications)

### 第八篇 半导体存储器

- [x] [半导体存储器概述与分类](./semiconductor-memory-overview)
- [x] [只读存储器（ROM）：固定 ROM 与 PROM](./rom-prom)
- [x] [可擦除可编程只读存储器：EPROM、E²PROM 与 Flash Memory](./eprom-eeprom-flash)
- [x] [随机存取存储器（RAM）：SRAM 与 DRAM 的存储单元](./sram-dram)
- [x] [存储器容量的扩展：位扩展与字扩展](./memory-expansion)
- [x] [用存储器实现组合逻辑函数](./memory-based-logic)

### 第九篇 可编程逻辑器件

- [x] [可编程逻辑器件（PLD）概述与基本结构](./pld-overview)
- [x] [可编程阵列逻辑（PAL）与通用阵列逻辑（GAL）](./pal-gal)
- [x] [复杂可编程逻辑器件（CPLD）的结构与原理](./cpld)
- [x] [现场可编程门阵列（FPGA）的结构与原理](./fpga)
- [x] [PLD 的开发流程与编程技术](./pld-development-flow)

### 第十篇 数-模和模-数转换

- [x] [D/A 与 A/D 转换概述](./adc-dac-overview)
- [x] [D/A 转换器：权电阻网络与倒 T 形电阻网络 DAC](./resistor-network-dac)
- [x] [D/A 转换器的主要技术指标：分辨率与转换精度](./dac-specifications)
- [x] [A/D 转换的基本原理：取样、保持、量化与编码](./adc-principles)
- [x] [取样-保持电路](./sample-and-hold)
- [x] [A/D 转换器：并联比较型与逐次逼近型 ADC](./flash-sar-adc)
- [x] [A/D 转换器：双积分型与其他间接型 ADC](./dual-slope-adc)

### 第十一篇 硬件描述语言 Verilog 入门

- [x] [HDL 概述：从原理图设计到硬件描述语言](./hdl-overview)
- [x] [Verilog 的基本结构：模块、端口与信号声明](./verilog-module-structure)
- [x] [Verilog 的数据类型与运算符](./verilog-data-types-operators)
- [x] [组合逻辑的建模：assign 语句与 always 块](./verilog-combinational-modeling)
- [x] [时序逻辑的建模：always @(posedge clk)、阻塞与非阻塞赋值](./verilog-sequential-modeling)
- [x] [状态机的 Verilog 描述：一段式、两段式与三段式写法](./verilog-fsm)
- [x] [Testbench 编写与仿真验证](./verilog-testbench)

### 第十二篇 从逻辑门到 CPU

- [x] [用逻辑门搭建一位全加器与多位算术逻辑单元（ALU）](./alu-design)
- [x] [从触发器到寄存器堆：CPU 的存储通路](./registers-register-file)
- [x] [程序计数器、指令寄存器与取指电路](./pc-instruction-fetch)
- [x] [指令的编码与译码：从译码器到控制信号](./instruction-decode)
- [x] [硬布线控制器与微程序控制器](./hardwired-microprogrammed-control)
- [x] [数据通路的组织：总线结构与单周期数据通路](./datapath-organization)
- [x] [一台最简单的 CPU：整体联调与运行一段程序](./simplest-cpu)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
