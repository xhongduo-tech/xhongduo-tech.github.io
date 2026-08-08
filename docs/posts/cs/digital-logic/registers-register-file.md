---
title: 从触发器到寄存器堆：CPU 的存储通路
date: 2026-08-07
---

# 从触发器到寄存器堆

<div class="epigraph">
<p>CPU 的快节奏，靠一排寄存器托底。</p>
<footer>—— 佚名（寄存器堆格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 12 章 §12.2 ｜ 2026-08-07</p>
</div>

## 为什么从寄存器堆开始

CPU 执行指令时，操作数要「随取随用」——内存太慢，必须有一组**最快的存储**放在 CPU 内部，这就是**寄存器堆（register file）**。它由一排触发器组成，支持「同时读两个、写一个」的多端口访问。<span class="marginnote">寄存器堆是 CPU 的「工作台」：<strong>ALU 的两个操作数从它读，结果写回它</strong>。RISC-V、ARM、x86 的指令集里「寄存器」就是指它。数量不多（16~32 个）但速度最快——因为它是触发器阵列，紧贴 ALU。</span>这一节从单个触发器讲到多端口寄存器堆，以及它的 Verilog 实现。

## 1 从触发器到寄存器

单个边沿 D 触发器存 1 位；n 个触发器共享时钟就构成 **n 位寄存器**（存一个数据字）：

```verilog
module reg_en #(parameter N = 8) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,            // 写使能
    input  wire [N-1:0] d,
    output reg  [N-1:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 0;
        else if (en)                  // 使能有效才更新
            q <= d;
    end
endmodule
```

**带使能的意义**：不是每个时钟都强制写，只有使能有效（<code>en == 1'b1</code>`）才更新——这是「条件写入」的基础，也是寄存器堆写控制的雏形。<span class="marginnote">使能寄存器是「<strong>写控制</strong>」的最小形式：<code>en</code>` 决定「这拍写不写」。CPU 里每条指令是否写寄存器、写哪个寄存器，都是由控制信号控制使能——整个寄存器堆的写端口就建立在这个机制上。

## 2 寄存器堆：多端口读写

**寄存器堆（register file）**：一组寄存器 + 地址译码 + 多端口访问。典型 32 个寄存器、每个 32 位，支持：

**两个读端口**：同时读两个寄存器（ALU 两个操作数）。
**一个写端口**：一个时钟写一个寄存器。

**为什么双读**：一条指令通常要两个操作数（如 <code>ADD r3, r1, r2</code> 读 r1、r2）。两个读口让 ALU 一拍拿到两个操作数。

**Verilog 实现**：

```verilog
module regfile #(
    parameter ADDR_W = 5,        // 32 个寄存器
    parameter DATA_W = 32
) (
    input  wire              clk,
    input  wire              we,            // 写使能
    input  wire [ADDR_W-1:0] raddr1,        // 读端口 1
    input  wire [ADDR_W-1:0] raddr2,        // 读端口 2
    input  wire [ADDR_W-1:0] waddr,         // 写地址
    input  wire [DATA_W-1:0] wdata,         // 写数据
    output wire [DATA_W-1:0] rdata1,        // 读数据 1
    output wire [DATA_W-1:0] rdata2         // 读数据 2
);
    reg [DATA_W-1:0] regs [0:2**ADDR_W-1];

    // 读是组合的：地址一变，数据立刻出
    assign rdata1 = regs[raddr1];
    assign rdata2 = regs[raddr2];

    // 写是时序的：时钟边沿才更新
    always @(posedge clk) begin
        if (we)
            regs[waddr] <= wdata;
    end
endmodule
```

<span class="marginnote">寄存器堆的「<strong>双读单写</strong>」是 RISC 处理器的标准结构：ALU 一拍读两个操作数、算完一拍写回。Verilog 里数组 <code>reg [DATA_W-1:0] regs [0:N-1]</code> 就是寄存器堆，读是组合的、写是时序的——两个端口并行工作，一拍完成「读读写」。</span>

## 3 读写同时进行：经典技巧

寄存器堆的一个经典设计细节：**同一拍内，写地址与读地址相同时，读应该读到「新值」还是「旧值」？**

两种约定：

**读旧写新**：先读后写，读端口读到旧值——大多数 RISC 处理器（如 MIPS）采用。
**写直达读**：写的同时读端口直接看到新值。

Verilog 中，由于**非阻塞赋值**的「同时更新」语义，写在同一拍生效于下一个读：

```verilog
// 同一拍：写地址 == 读地址
// 非阻塞赋值 → 读端口仍读到旧值（读旧写新，RISC 约定）
assign rdata1 = regs[raddr1];

always @(posedge clk) begin
    if (we)
        regs[waddr] <= wdata;   // 边沿之后才更新，本拍读端口不变
end
```

**这正是非阻塞赋值的正确用法**——读组合、写时序，天然实现「同拍读旧值」的 RISC 约定。<span class="marginnote">这个细节是「<strong>非阻塞赋值的工程胜利</strong>」：如果是阻塞赋值，写会立即生效、读可能读到新值——与 RISC 的「读旧值」约定冲突。用非阻塞赋值，硬件行为与指令集语义自动对齐。这就是为什么时序块必须用非阻塞。</span>

## 4 公式解析：寄存器堆的端口带宽

寄存器堆的设计核心是「端口带宽」——每周期能读写多少次。设 $N$ 个寄存器、$W$ 位宽、$R$ 个读端口、$W_p$ 个写端口：

**第一步，看面积**：寄存器堆面积 $\propto N \times W$——寄存器多、位宽宽，面积大。

**第二步，看端口成本**：每个读端口是一套地址译码 + 输出数据线；读端口越多，面积与延迟越大。**双读比单读面积大、但性能翻倍**（ALU 一拍拿两操作数）。

**第三步，看写冲突**：多个写端口同时写不同地址要分别译码；同一地址多写则冲突。

**第四步，工程权衡**：RISC-V 常用 32 个 32 位寄存器、2 读 1 写——这是面积与性能的经典平衡点。<span class="marginnote">「<strong>多端口 = 多带宽 = 多面积</strong>」是寄存器堆的铁律。现代超标量 CPU 用 6~8 个读端口、4~6 个写端口支持每周期发射多条指令——端口数直接反映 CPU 的「并行胃口」，也直接推高面积。理解端口-面积权衡，就看懂了为什么处理器核越强越大。</span>

## 5 寄存器堆在 CPU 数据通路中的位置

寄存器堆连接 ALU 与控制逻辑，构成 CPU 的「数据通路」核心：

```
   rs1 ──▶ raddr1 ──┐
   rs2 ──▶ raddr2 ──┤         ┌────────┐
                    ├────────▶│  ALU   │──▶ 结果
   rd  ──▶ waddr    │         └────────┘       │
   we  ──▶ 写使能    │                          ▼
        ┌───────────────┐                  写回数据
        │    寄存器堆     │◀─────────────────────┘
        └───────────────┘
   读（组合）→ 算（ALU）→ 写（时序），一拍完成
```

**读**：指令里的源寄存器号（rs1、rs2）→ 读端口 → ALU。
**算**：ALU 用两个操作数计算。
**写**：目标寄存器号（rd）+ ALU 结果 → 写端口写回。

这条「读-算-写」循环是每一拍指令执行的基本节拍——**RISC 处理器一拍完成一次循环**。<span class="marginnote">这一节与下一节的衔接：<strong>取指（下一节）把指令送来，指令里的寄存器号驱动寄存器堆读写</strong>。寄存器堆 + ALU + 取指 + 控制 = 数据通路。你在搭的，正是一个最简 RISC 处理器的核心。</span>

## 6 小结

- 触发器 → 寄存器（共享时钟）→ 寄存器堆（一组寄存器 + 端口）。
- 寄存器堆：**双读单写**（2 读 1 写）是 RISC 标准，ALU 一拍取两操作数。
- 读是组合、写是时序；非阻塞赋值天然实现「同拍读旧值」约定。
- 端口带宽权衡：多端口 = 多带宽 = 多面积，超标量 CPU 靠多端口支持多发射。
- 寄存器堆 + ALU + 控制 = 数据通路，「读-算-写」循环是每拍指令的节拍。

在下一节，我们将搭 CPU 的「指挥器官」——程序计数器、指令寄存器与取指电路，看 CPU 如何拿到下一条指令。
