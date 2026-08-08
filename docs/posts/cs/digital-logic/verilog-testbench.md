---
title: Testbench 编写与仿真验证
date: 2026-08-07
---

# Testbench 编写与仿真验证

<div class="epigraph">
<p>在硅片之前，先在电脑里验证一切。</p>
<footer>—— 佚名（验证格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 11 章 §11.7 ｜ 2026-08-07</p>
</div>

## 为什么从 Testbench 开始

写 HDL 只是设计的一半，另一半是**验证**——确认代码行为正确。**Testbench（测试平台）**就是「给设计施加激励、观察响应」的测试环境，它本身也是 Verilog 代码。仿真（simulation）让我们在流片/下载之前就发现 bug，是最便宜的调试手段。<span class="marginnote">Testbench 的本质：<strong>一个没有端口、只用来「伺候」被测模块（DUT）的模块</strong>——它生成时钟、给激励、检查输出。它是「硬件设计师的测试用例」，就像软件里的单元测试。</span>这一节写一个完整的 Testbench，并讲清仿真流程与验证技巧。

## 1 Testbench 的结构

一个 Testbench 由三部分组成：

1. **被测模块例化（DUT）**：把要测试的设计接进来。
2. **激励生成**：`initial` 块生成输入信号、时钟等。
3. **响应检查**：观察输出（打印波形/用断言检查）。

**结构模板**：

```verilog
module tb_and_gate;
    reg  a, b;        // 信号声明
    wire y;

    and_gate dut (    // DUT 例化
        .a(a),
        .b(b),
        .y(y)
    );

    initial begin     // 激励生成
        a = 0; b = 0;
        #10 a = 0; b = 1;
        #10 a = 1; b = 0;
        #10 a = 1; b = 1;
        #10 $finish;
    end
endmodule
```

<span class="marginnote">Testbench 的「可综合性」：<strong>Testbench 只用于仿真，不用于综合</strong>——它用 `initial`、`#延迟`、`$display` 这些<strong>仿真专用语句</strong>（综合器不认）。设计代码与测试代码分开是工程惯例：一个文件夹放设计（可综合），一个放 testbench（仅仿真）。</span>

## 2 激励生成：initial 与时钟

**`initial` 块**：仿真开始时执行一次，用于初始化与生成激励。

```verilog
initial begin
    rst_n = 0;          // 初始复位
    #100 rst_n = 1;     // 100ns 后释放复位
    #500 $finish;       // 600ns 结束仿真
end
```

**`#` 延迟**：`#10` 表示等 10 个时间单位——仿真专用，不可综合。

**生成时钟**：用 `always` 或 `forever` 产生周期信号：

```verilog
initial clk = 0;
always #5 clk = ~clk;      // 周期 10ns 的时钟
```

**复位信号**：

```verilog
initial begin
    rst_n = 1'b1;
    #10 rst_n = 1'b0;      // 拉低复位
    #10 rst_n = 1'b1;      // 释放复位
end
```

**时间单位**：在文件头用 `` `timescale 1ns/1ps`` 指定：

```verilog
`timescale 1ns/1ps
```

<span class="marginnote">`#` 延迟让 Testbench 能「按时间安排激励」——这在综合代码里不存在，但在仿真里是必需品。`<strong>`#5` 就是 5ns。跨文件时 timescale 不一致会造成仿真时间错乱，这是仿真常见的坑。</span>

## 3 响应检查：波形、display 与断言

**查看响应**的三种方式：

**① 波形（最常用）**：仿真器（Vivado、ModelSim、Verilator）把信号画成波形图，肉眼观察是否正确。

**② 打印（$display）**：

```verilog
initial begin
    $display("a=%b b=%b y=%b", a, b, y);
end
```

`$monitor` 在信号变化时自动打印。

**③ 断言（自检）**：Testbench 自动检查输出是否正确：

```verilog
initial begin
    #40
    if (y !== 1'b1)
        $display("ERROR: y 应为 1，实际为 %b", y);
    else
        $display("PASS");
    $finish;
end
```

断言让 Testbench 能「自动判对错」，无需人盯波形——现代验证（SystemVerilog 断言）在此基础上发展成完整的验证方法学。<span class="marginnote">从「看波形」到「写断言」是验证水平的进阶：<strong>波形只能看「发生了什么」，断言能自动报「哪里错了」</strong>。大型设计靠断言 + 自动比对（reference model），而不是人肉盯波形——这也是 UVM 验证方法学的核心思想。</span>

## 4 公式解析：完整 Testbench 实例

写一个完整的 Testbench 测试「2 输入与门」，走一遍所有步骤：

```verilog
`timescale 1ns/1ps

module tb_and_gate;
    reg  a, b;
    wire y;

    and_gate dut (.a(a), .b(b), .y(y));

    initial begin
        $monitor("t=%0t a=%b b=%b y=%b", $time, a, b, y);
        a = 0; b = 0; #10;
        a = 0; b = 1; #10;
        a = 1; b = 0; #10;
        a = 1; b = 1; #10;
        if (y !== 1'b1)
            $display("FAIL: y 应为 1");
        else
            $display("PASS");
        $finish;
    end
endmodule
```

**第一步，看 timescale**：1ns/1ps，`#10` = 10ns。

**第二步，看例化**：被测模块 `and_gate dut` 接好端口。

**第三步，看激励**：`initial` 按时间依次给四组输入——覆盖与门全部真值表。

**第四步，看检查**：`$monitor` 打印全部变化；断言检查最后一组输出。

**仿真流程**：用仿真器编译 DUT + Testbench → 运行 → 看波形/输出——通过则设计正确。<span class="marginnote">验证的「<strong>完备性</strong>」：激励要覆盖所有关键情况——边界值（全 0、全 1）、典型值、随机值、时序边界（建立保持）。只测「正常情况」的 Testbench 会漏掉边界 bug。对真值表电路，最好穷举全部输入组合。</span>

## 5 仿真与验证的工程实践

**功能仿真 vs 时序仿真**：功能仿真验证逻辑（无延迟），时序仿真验证时序（含延迟）。先功能后时序。
**模块级 vs 系统级**：先测小模块（单元测试），再测整系统（集成测试）。
**覆盖率**：代码覆盖率、功能覆盖率——衡量测试「测到多全」。
**回归测试**：改代码后重跑全部 Testbench，防「修好这个、弄坏那个」。

**仿真工具**：Vivado（Xilinx）、Quartus（Intel）、ModelSim/Questa（Mentor）、Icarus Verilog（开源）、Verilator（开源，快）。

<span class="marginnote">验证在硬件设计里的比重：<strong>现代芯片设计 60~70% 的时间花在验证上</strong>——因为流片一次成本百万级，bug 必须在仿真里找出来。Testbench 是验证的第一课，UVM/形式验证是进阶——但「给激励、看响应、查断言」的基本思想不变。</span>

## 6 小结

- **Testbench** = 测试平台：信号声明 + DUT 例化 + 激励生成 + 响应检查。
- **initial 块与 `#` 延迟**生成激励；`` `timescale 1ns/1ps`` 定时间单位。
- 响应检查三方式：波形、`$display`/`$monitor` 打印、断言自动比对。
- 仿真流程：编译 DUT + Testbench → 运行 → 看结果；先功能仿真后时序仿真。
- 工程实践：覆盖完备、模块先于系统、回归测试、覆盖率度量。

至此，第十一篇「硬件描述语言 Verilog 入门」全部完成。在下一篇，我们将迎来全书的高潮——用逻辑门搭出一台能运行程序的 CPU。
