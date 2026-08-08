---
title: 用逻辑门搭建一位全加器与多位算术逻辑单元（ALU）
date: 2026-08-07
---

# 用逻辑门搭建一位全加器与多位算术逻辑单元（ALU）

<div class="epigraph">
<p>从两个与门开始，搭出会计算的机器。</p>
<footer>—— 佚名（ALU 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 12 章 §12.1 ｜ 2026-08-07</p>
</div>

## 为什么从 ALU 开始

至此，我们学完了从门到存储器的全部积木。最后三篇的目标是**把它们组装成一台 CPU**——而 CPU 的心脏是 **ALU（算术逻辑单元）**：它负责全部加减与逻辑运算。从一位全加器到多位 ALU，是从「逻辑门」走向「计算机」的第一大步。<span class="marginnote">第十二篇的路线图：<strong>ALU（运算）→ 寄存器堆（存储）→ 取指与控制（指挥）→ 数据通路（连接）→ 整机（运行程序）</strong>。一台 CPU = 算（ALU）+ 存（寄存器）+ 控（控制逻辑）+ 连（总线）。这一节先攻克「算」。</span>这一节从门搭全加器，再扩展成多功能 ALU。

## 1 一位全加器：从门开始

第四篇我们学过全加器，现在用「真门」把它搭出来。全加器逻辑：

$$S = A\oplus B\oplus C_i, \qquad C_{o} = AB + C_i(A\oplus B)$$

**用门实现**：

- 两个异或门算 $S$：$X = A\oplus B$，$S = X \oplus C_i$。
- 两个与门加一个或门算 $C_o$：$AB$ 与 $(A\oplus B)C_i$。

**Verilog 行为描述**（综合器自动生成门）：

```verilog
module full_adder (
    input  a, b, cin,
    output s, cout
);
    assign s    = a ^ b ^ cin;                   // 和
    assign cout = (a & b) | (cin & (a ^ b));     // 进位
endmodule
```

一行 `assign`、一行 `assign`、一行 `assign`——综合后就是一张标准全加器门级网表。<span class="marginnote">从「手画门」到「写行为」的转变：<strong>第四篇你用手算真值表搭全加器，现在写行为级 Verilog 综合器替你搭</strong>。但理解底层（哪几个门、怎么进位）仍然重要——它决定你能不能在综合结果里找出问题。</span>

## 2 多位加法器：行波进位

把 $n$ 个全加器级联成 $n$ 位行波进位加法器：

```verilog
module ripple_carry_adder #(parameter N = 4) (
    input  [N-1:0] a, b,
    input  cin,
    output [N-1:0] sum,
    output cout
);
    assign {cout, sum} = a + b + cin;   // 行为级：综合器自动生成进位结构
endmodule
```

或者用 generate 结构显式级联全加器（更贴近门级）：

```verilog
module ripple_carry_adder #(parameter N = 4) (
    input  [N-1:0] a, b,
    input  cin,
    output [N-1:0] sum,
    output cout
);
    wire [N:0] c;
    assign c[0] = cin;
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : fa
            full_adder fa_i (
                .a(a[i]), .b(b[i]), .cin(c[i]),
                .s(sum[i]), .cout(c[i+1])
            );
        end
    endgenerate
    assign cout = c[N];
endmodule
```

**两种写法**：`a + b` 直接（行为级，综合器选最优结构）或显式级联（结构级，可控但笨）。工程上用 `a + b`，工具会做进位优化（超前进位）。<span class="marginnote">「写 `a+b` 还是搭门」体现了两层设计：<strong>行为级让工具优化（自动超前进位），结构级自己控制（固定行波进位）</strong>。现代综合器对 `a + b` 的优化远超手工，所以除非特殊要求，写 `a + b` 就好——但理解行波进位（第四篇）能让你读懂综合报告里的关键路径。</span>

## 3 ALU：一个能算多种运算的单元

**ALU（Arithmetic Logic Unit）**：一个多功能运算单元，用**功能选择信号**决定执行哪种运算。经典 ALU 支持：加、减、与、或、异或、比较等。

**ALU 的结构**：运算器（多种运算电路）+ **多路选择器**（按功能选择输出）。

**Verilog 行为描述**：

```verilog
module alu #(parameter N = 8) (
    input  [N-1:0] a, b,
    input  [2:0]   op,             // 功能选择信号
    output reg [N-1:0] y,
    output reg         zero
);
    always @(*) begin
        case (op)
            3'b000: y = a + b;              // 加
            3'b001: y = a - b;              // 减
            3'b010: y = a & b;              // 与
            3'b011: y = a | b;              // 或
            3'b100: y = a ^ b;              // 异或
            3'b101: y = (a < b) ? 1 : 0;    // 比较
            default: y = {N{1'b0}};
        endcase
        zero = (y == {N{1'b0}});            // 结果为 0 时置位
    end
endmodule
```

<span class="marginnote">ALU 的精髓：<strong>多种运算电路「并行算」，选择信号「挑一个输出」</strong>——就像多路选择器选数据。`op` 对应「运算选择」，综合器为每种运算生成电路、再用 MUX 合并。`zero` 标志供条件跳转使用——CPU 判断「是否相等/为零」全靠它。</span>

## 4 公式解析：减法与标志位

ALU 的减法用补码实现：$a - b = a + \overline{b} + 1$。这在硬件上只要在加法器的 $b$ 输入端加反相器、并把进位输入置 1：

```verilog
module add_sub #(parameter N = 8) (
    input  [N-1:0] a, b,
    input  sub,               // 减法标志：1 表示减法
    output [N-1:0] y,
    output cout
);
    wire [N-1:0] b_sel = sub ? ~b : b;   // 减时对 b 取反
    assign {cout, y} = a + b_sel + sub;  // 进位输入补上的 +1
endmodule
```

**第一步，看补码**：`~b + 1` 就是 $b$ 的补码（反码 + 1）。

**第二步，看加法器**：`a + (~b + 1)` 即 $a + (-b) = a - b$——加法器同时做加与减，靠一个「减法标志」切换。

**第三步，看标志位**：ALU 输出的标志（status flags）：
`zero`：结果为 0（相等判断）。
`carry`：进位/借位（无符号比较）。
`negative`：最高位（有符号判断负）。
`overflow`：符号位溢出（有符号加减溢出）。

**第四步，应用**：CPU 的条件跳转（beq、bne、blt）全部基于这些标志——比较指令算 `a - b`，看 zero/negative 就知道谁大谁小。<span class="marginnote">标志位是 ALU 的「副产物」，却决定了 CPU 的分支能力：`<strong>`a - b` 的结果 + zero/negative 标志 = 完整的比较器</strong>。现代 CPU 用「比较指令」（cmp/sub + 标志）实现 if/while，原理就是这里的一行 `assign`。</span>

## 5 从 ALU 到 CPU 的第一块拼图

ALU 是 CPU 的「运算器官」，它需要与其他部件协同：

**操作数来源**：寄存器堆或立即数（下一节）。
**结果去向**：写回寄存器或送内存地址。
**标志输出**：送控制逻辑做分支判断。

一台最小 CPU 的 ALU 需求：支持加/减（运算）、与/或/异或（逻辑）、比较（分支）——这正好是上面 ALU 的功能集。<span class="marginnote">回顾全书：<strong>异或（第二篇）→ 半加器（第四篇）→ 全加器 → 加法器 → ALU</strong>——一条从「逻辑运算」到「CPU 运算器」的完整链条。你正在亲手把学过的每一个积木拼成一台计算机。</span>

## 6 小结

- 全加器：$S = A\oplus B\oplus C_i$、$C_o = AB + (A\oplus B)C_i$，用异或/与/或门实现。
- 多位加法器：行为级 `a + b` 或显式级联；综合器自动优化进位。
- **ALU** = 多种运算电路 + 功能选择 MUX，`case` 实现运算选择。
- 减法 = 补码加法：$a - b = a + \overline{b} + 1$；标志位（zero/negative/carry/overflow）支撑分支。
- ALU 是 CPU 的运算核心，与寄存器堆、控制逻辑、数据通路共同构成整机。

在下一节，我们将搭 CPU 的「存储器官」——从触发器到寄存器堆，看 CPU 的工作存储如何构建。
