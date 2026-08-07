---
title: 一台最简单的 CPU：整体联调与运行一段程序
date: 2026-08-07
---

# 一台最简单的 CPU

<div class="epigraph">
<p>把十二篇的知识，凝成一颗会计算的芯。</p>
<footer>—— 佚名（整机联调格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 12 章 §12.7 ｜ 2026-08-07</p>
</div>

## 为什么从整机联调开始

最后一步，把 ALU、寄存器堆、取指、译码、控制、数据通路**全部拼起来**，得到一台能运行程序的最小 CPU——然后写一段程序，看它真的算出结果。这是全书的高潮：从「与或非」到「能跑程序的计算机」，一条完整的路走完了。<span class="marginnote">这一节是对全书的「总复习」：<strong>与或非（第二篇）、门电路（第三篇）、组合与时序（第四~六篇）、存储（第八篇）、Verilog（第十一篇）——全都在一颗小小的 CPU 里交汇</strong>。搭出它，你就理解了冯·诺依曼机器的一切。</span>这一节给出整机结构、Verilog 顶层，并运行一段小程序验证。

## 1 整机结构：所有部件的总装

这台最小 CPU（单周期、简化 RISC）的组成：

| 部件 | 功能 | 对应章节 |
|---|---|---|
| 程序计数器（PC） | 指令地址 | §12.3 |
| 指令存储器 | 存程序 | §12.3 |
| 寄存器堆 | 工作寄存器 | §12.2 |
| ALU | 运算 | §12.1 |
| 数据存储器 | 数据存取 | §12.2/12.6 |
| 译码/控制器 | 产生控制信号 | §12.4/12.5 |
| 多路选择器 | 选择数据来源 | §12.6 |

**数据流闭环**：

```
取指 → 译码 → 读寄存器 → ALU → (访存) → 写回 → 更新 PC → 下一拍
```

每一拍执行一条指令，PC 顺序推进或跳转——程序就这样被「跑」起来。

## 2 Verilog 顶层：把模块拼起来

用 Verilog 把各模块例化拼成整机（简化版）：

```verilog
module tiny_cpu(
    input  clk, rst_n,
    output [7:0] debug_result
);
    // 内部信号
    wire [31:0] pc, instr, r_data1, r_data2, alu_result, w_data;
    wire [31:0] imm, mem_data, mem_out;
    wire        reg_write, alu_src, mem_write, branch, zero;
    wire [3:0]  alu_op;

    // 1. 取指
    fetch    u_fetch (.clk(clk), .rst_n(rst_n),
                      .branch_taken(branch & zero),
                      .branch_target(imm), .pc(pc), .instr(instr));

    // 2. 译码/控制
    decoder  u_dec (.instr(instr), .reg_write(reg_write),
                    .alu_src(alu_src), .alu_op(alu_op),
                    .mem_write(mem_write), .branch(branch));

    // 3. 寄存器堆
    regfile  u_rf (.clk(clk), .we(reg_write),
                   .r_addr1(instr[19:15]), .r_addr2(instr[24:20]),
                   .w_addr(instr[11:7]), .w_data(w_data),
                   .r_data1(r_data1), .r_data2(r_data2));

    // 4. ALU（第二操作数选择）
    assign imm = {{20{instr[31]}}, instr[31:20]};  // 立即数扩展
    assign alu_b = alu_src ? imm : r_data2;
    alu u_alu (.a(r_data1), .b(alu_b), .op(alu_op),
               .y(alu_result), .zero(zero));

    // 5. 数据存储器（简化）
    assign w_data = mem_write ? r_data2 : alu_result;
    // ... load/store 控制 ...

    // 调试：看最终结果
    assign debug_result = r_data1[7:0];
endmodule
```

<span class="marginnote">这个顶层就是「<strong>把前面每一节的模块像乐高一样拼起来</strong>」：fetch、decoder、regfile、alu 依次例化，连线靠内部的 wire。每一行都对应前面学过的一个模块——搭 CPU 不是「新知识」，而是「复用全部旧知识」。</span>

## 3 公式解析：运行一段程序

写一段小程序验证 CPU——计算 $5 + 7$ 并存储：

**汇编**（简化指令集）：

```asm
# 程序：计算 5 + 7，结果存到寄存器 5
addi r1, x0, 5      # r1 = 5
addi r2, x0, 7      # r2 = 7
add  r3, r1, r2     # r3 = 5 + 7 = 12
```

**机器码**（编码后存入指令存储器）：

```verilog
initial begin
    imem[0] = 32'b000000000101_00000_000_00001_0010011;  // addi r1,x0,5
    imem[1] = 32'b000000000111_00000_000_00010_0010011;  // addi r2,x0,7
    imem[2] = 32'b0000000_00010_00001_000_00011_0110011; // add  r3,r1,r2
end
```

**执行过程**：

- **拍 1**：取 `addi r1,x0,5` → 译码 → ALU 算 `0+5` → 写回 r1=5。
- **拍 2**：取 `addi r2,x0,7` → r2=7。
- **拍 3**：取 `add r3,r1,r2` → ALU 算 `5+7=12` → 写回 r3=12。

**结果**：$r3 = 12$——验证通过，CPU 正确地执行了程序。<span class="marginnote">这一步的震撼：<strong>从「与或非」到「算出 5+7」</strong>——你搭的机器真的「会算」了。加法是 ALU 里一堆门干的活，指挥是译码器给的控制信号，存储是寄存器堆的触发器——全部是最初学的东西。</span>

## 4 验证与调试：仿真看波形

整机联调的关键是**仿真验证**：

1. **写 Testbench**：给 clk、rst_n，加载程序。
2. **跑仿真**：看每拍的 PC、指令、寄存器值。
3. **对照预期**：检查 r1=5、r2=7、r3=12 是否按时出现。
4. **修 bug**：若结果不对，追查是取指、译码还是 ALU 的问题。

**常见 bug**：

- **控制信号错**：reg_write 没开，寄存器没写入。
- **立即数扩展错**：符号扩展写成零扩展。
- **分支判断错**：branch 信号与 zero 标志组合错了。
- **时序错**：非阻塞赋值用成阻塞，读到了旧值。

调试经验：**从最简单的指令（addi）开始，逐步增加指令类型**——先跑通一条，再跑通全部。<span class="marginnote">「从简到繁」是整机验证的铁律：<strong>先让 `addi`（最简单）跑对，再让 `add`（双寄存器）、`lw/sw`（访存）、`beq`（分支）逐类打通</strong>。每加一类指令，控制信号就多一组——bug 出现的位置也能定位到「新加的部分」。这就像软件开发的「增量集成」。</span>

## 5 回顾：从极限到大模型，这条路走了多远

回顾全书十二篇，从最低层到最高层：

- **数制与码制**：数字世界的语言。
- **逻辑代数**：0/1 的运算规则。
- **门电路**：运算的物理实现。
- **组合逻辑**：无记忆的计算。
- **触发器**：记忆的种子。
- **时序逻辑**：会记忆、会计数的系统。
- **脉冲电路**：制造与整形波形。
- **存储器**：大容量记忆。
- **PLD/FPGA**：可编程的硬件。
- **D/A-A/D**：数字与模拟的桥。
- **Verilog**：描述硬件的语言。
- **CPU**：所有知识的总装。

一台最小 CPU 的完成，意味着你**从底层理解了计算机**——而理解底层，是理解大模型（Transformer 在 GPU 上算矩阵乘法）的最坚实地基。<span class="marginnote">「从极限到大模型」的终点回望：<strong>大模型的每一次矩阵乘，最终都落到 GPU 里的 ALU 与寄存器堆</strong>——你搭过的最小 CPU，就是那座庞大计算之塔的第一块砖。地基稳，塔才高。</span>

## 6 小结

- 整机 = 取指 + 译码/控制 + 寄存器堆 + ALU + 数据存储器 + 数据通路，全部模块拼装。
- Verilog 顶层把各模块例化连接，内部 wire 传递数据与控制信号。
- 运行程序验证：`addi r1,0,5 → addi r2,0,7 → add r3,r1,r2` → r3=12。
- 调试从简到繁：先 addi，再 add、访存、分支逐类打通。
- 十二篇知识在 CPU 里总装：这台最小机器，是从「与或非」到「大模型」整条路的基石。

至此，第十二篇「从逻辑门到 CPU」全部完成，数字逻辑这一学科也全部写完了。
