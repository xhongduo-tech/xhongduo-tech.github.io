---
title: 状态机的 Verilog 描述：一段式、两段式与三段式写法
date: 2026-08-07
---

# 状态机的 Verilog 描述

<div class="epigraph">
<p>状态机，是控制逻辑的灵魂。</p>
<footer>—— 佚名（FSM 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 11 章 §11.6 ｜ 2026-08-07</p>
</div>

## 为什么从状态机建模开始

第六篇我们用手工方法设计了状态机，这一节用 **Verilog 描述状态机**——这是现代控制逻辑设计的标准方式。FSM 的 Verilog 描述有三种风格：**一段式、两段式、三段式**，它们在可读性、时序质量、面积上各有取舍。<span class="marginnote">状态机的本质回顾：<strong>状态寄存器（记忆）+ 次态逻辑（组合）+ 输出逻辑（组合）</strong>。三种写法区别在于「这三个部分怎么组织成 always 块」——一段式全塞一起，两段式状态/输出分开，三段式再加寄存器输出。</span>这一节用同一个例子（序列检测器）演示三种写法，并总结选型建议。

## 1 状态机的通用结构回顾

一个有限状态机（FSM）由三部分组成：

1. **状态寄存器**：保存当前状态（触发器）。
2. **次态逻辑**：根据当前状态与输入，计算次态（组合逻辑）。
3. **输出逻辑**：根据当前状态（及输入）产生输出（组合逻辑）。

Mealy 型输出依赖「状态 + 输入」，Moore 型只依赖「状态」。

**状态编码**（第六篇讲过）：二进制、独热码、格雷码。Verilog 里用 `parameter` 定义状态：

```verilog
parameter S0 = 2'b00, S1 = 2'b01, S2 = 2'b10, S3 = 2'b11;  // 自然二进制编码
```

**状态寄存器**：

```verilog
reg [1:0] state, next_state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= S0;   // 复位回到初始状态
    else        state <= next_state;
end
```

<span class="marginnote">编码选择的工程影响：<strong>二进制省寄存器、独热码省译码逻辑</strong>。FPGA 里触发器多、独热码常用（一个状态一个 bit）；CPLD/ASIC 里触发器贵，二进制常用。综合工具可自动选，但手写时要知道取舍——用 `parameter` 定义编码，改起来只需改参数。</span>

## 2 一段式：全在一块的紧凑写法

**一段式（one-segment）**：状态寄存器、次态逻辑、输出逻辑**全写在一个 always 块**里。

```verilog
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S0;
        y     <= 1'b0;
    end
    else begin
        y <= 1'b0;                    // 默认输出
        case (state)
            S0: state <= (x) ? S1 : S0;
            S1: state <= (x) ? S2 : S0;
            S2: state <= (x) ? S2 : S3;
            S3: begin
                y <= 1'b1;            // 检测到 110，输出拉高
                state <= (x) ? S1 : S0;
            end
            default: state <= S0;     // 未用状态归位（自启动）
        endcase
    end
end
```

**特点**：代码最短，但状态转移与输出混在一起，**输出是寄存器输出**（无毛刺），时序干净。

**缺点**：可读性差（状态与输出纠缠），且输出逻辑写在时序块里，**所有输出都是寄存器的**——如果设计者想要组合输出，一段式做不到。<span class="marginnote">一段式的本质：<strong>把「次态 + 输出」都放进时序块，所有输出被寄存器锁存</strong>。好处是输出干净无毛刺，坏处是输出有一拍延迟、且代码难维护。适合输出不多、追求极简的小状态机。</span>

## 3 两段式：状态与输出分离

**两段式（two-segment）**：**时序块只做状态寄存器**，**组合块做次态逻辑 + 输出逻辑**。

```verilog
// 第一段：状态寄存器（时序）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= S0;
    else        state <= next_state;
end

// 第二段：次态逻辑 + 输出逻辑（组合）
always @(*) begin
    next_state = state;               // 默认值，防锁存器
    y          = 1'b0;
    case (state)
        S0: if (x) next_state = S1;
        S1: next_state = (x) ? S2 : S0;
        S2: next_state = (x) ? S2 : S3;
        S3: begin
            y = 1'b1;                 // 组合输出：检测到 110 立即拉高
            next_state = (x) ? S1 : S0;
        end
        default: next_state = S0;
    endcase
end
```

**特点**：状态转移与输出在组合块里（阻塞赋值 `=`），输出是**组合输出**（无寄存器锁存）——响应快，但可能有毛刺（组合逻辑直接输出）。

**优点**：结构清晰，状态与输出分离，可读性好。

**缺点**：组合输出可能有毛刺，且组合 always 要小心锁存器（要补默认值）。<span class="marginnote">两段式是「<strong>教科书标准写法</strong>」：状态寄存器 + 组合逻辑块，职责分明。它的组合输出快但可能有毛刺，适合输出不敏感、追求结构清晰的场合。注意组合块里「先给默认值、再 case 覆盖」——既防锁存器，又简化逻辑。</span>

## 4 三段式：状态、次态、输出各归其位

**三段式（three-segment）**：**时序块管状态寄存器，组合块管次态逻辑，再加一个时序块管输出（寄存器输出）**。

```verilog
// 第一段：状态寄存器（时序）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= S0;
    else        state <= next_state;
end

// 第二段：次态逻辑（组合）
always @(*) begin
    next_state = state;               // 默认值，防锁存器
    case (state)
        S0: if (x) next_state = S1;
        S1: next_state = (x) ? S2 : S0;
        S2: next_state = (x) ? S2 : S3;
        S3: next_state = (x) ? S1 : S0;
        default: next_state = S0;     // 未用状态归位
    endcase
end

// 第三段：输出寄存器（时序，寄存器输出无毛刺）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) y <= 1'b0;
    else        y <= (state == S3);
end
```

**特点**：三段各司其职——**状态寄存器、次态组合、输出寄存器**。输出被寄存器锁存，**无毛刺**，且结构与状态机模型一一对应。

**优点**：时序质量最好（输出干净）、结构清晰、适合复杂状态机。

**缺点**：代码最长。<span class="marginnote">三段式是工程推荐：<strong>「组合算次态、寄存器存状态、寄存器出输出」</strong>——输出与状态同步更新、无毛刺，是 FPGA 设计的事实标准。代价是多一拍输出延迟（寄存器输出），但换取时序干净，几乎总是值得。</span>

## 5 公式解析：三种写法的对比与选型

把三种写法放在一起对比，理解各自的定位：

| 对比 | 一段式 | 两段式 | 三段式 |
|---|---|---|---|
| always 块数 | 1 | 2 | 3 |
| 状态寄存器 | 时序块 | 独立时序块 | 独立时序块 |
| 次态逻辑 | 时序块内 | 组合块 | 独立组合块 |
| 输出 | 寄存器输出 | 组合输出 | 寄存器输出 |
| 输出毛刺 | 无 | 可能有 | 无 |
| 输出延迟 | 一拍 | 无 | 一拍 |
| 可读性 | 差 | 中 | 好 |
| 适用 | 极小状态机 | 教学/简单 | 复杂、工程标准 |

**第一步，看输出质量**：要无毛刺 → 一段式或三段式（寄存器输出）；要零延迟 → 两段式（组合输出）。

**第二步，看结构清晰**：复杂状态机 → 三段式（职责分明、易维护）。

**第三步，看代码量**：极小状态机 → 一段式（省代码）。

**第四步，工程推荐**：**默认三段式**——它兼顾时序质量与可维护性，是大多数设计的选择。<span class="marginnote">工程上的成熟结论：<strong>「复杂状态机用三段式，简单状态机用一段式，两段式主要用于教学演示」</strong>。核心权衡是「输出延迟 vs 输出毛刺」与「代码量 vs 可维护性」——三段式在这两点上都表现均衡，所以成了默认选项。</span>

## 6 小结

- FSM 三部分：状态寄存器 + 次态逻辑 + 输出逻辑；Verilog 三种写法对应不同的组织方式。
- **一段式**：全塞一个时序块，输出寄存器化、无毛刺，但难维护。
- **两段式**：状态寄存器 + 组合块（次态+输出），输出组合化、快但有毛刺风险。
- **三段式**：状态寄存器 + 次态组合 + 输出寄存器，无毛刺、结构清晰，工程推荐。
- 状态编码用 `parameter` 定义；输出质量与延迟是三种写法的核心权衡。

在下一节，我们将学如何「测试」设计——Testbench 编写与仿真验证，让硬件在电脑里先跑起来。
