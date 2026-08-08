---
title: LLVM Pass 机制与优化流水线
date: 2026-08-07
---

# LLVM Pass 机制与优化流水线

<div class="epigraph">
<p>优化是一队小工人，每人只干一件事，按巧妙的顺序排好，共同把 IR 打磨干净。</p>
<footer>—— 仿自克里斯 · 拉特纳（Chris Lattner）对 LLVM pass 架构的描述</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · LLVM 专题 ｜ 2026-08-07</p>
</div>

## 为什么从 Pass 机制开始

第八篇讲了「pass 流水线」的概念——LLVM 把它变成了工程现实：**pass** 是对 IR 做一次变换的最小模块，几十个 pass 按序排列成流水线。理解 LLVM 的 pass 机制，就是理解「优化如何被组织、复用、排序」——从「写一个优化」到「搭一条流水线」的完整过程。<span class="marginnote">「pass = 一个优化模块」是 LLVM 的模块化精髓：`GVN` 做值编号、`LICM` 做循环外提、`SimplifyCFG` 做控制流化简——每个 pass 只做一件事，却可被所有语言与后端共享。LLVM 的优化能力不是「一个巨大的优化器」，而是「一堆小 pass 的流水线」——这是它可维护、可扩展的根源。</span>

## 1 Pass 的类型

LLVM pass 按作用对象分几类：

| 类型 | 作用范围 | 例子 |
| --- | --- | --- |
| **Module pass** | 整个模块（跨函数） | 过程间优化、死全局消除 |
| **Function pass** | 单个函数 | 绝大多数优化（CSE、常量传播） |
| **Loop pass** | 单个循环 | 循环外提、归纳变量 |
| **CallGraph pass** | 调用图 | 内联分析、函数属性推断 |

**pass 依赖**：一个 pass 可能需要另一个 pass 的分析结果（如「活跃变量分析」的结果供「死代码消除」用）。LLVM 用 **pass 依赖管理** 保证「被依赖的 pass 先跑」。

**分析 pass vs 变换 pass**：

**分析 pass**（`LoopInfo`、`DominatorTree`）只计算信息、不改 IR——供其他 pass 查询。
**变换 pass**（`GVN`、`InstCombine`）修改 IR——可能需要分析 pass 的结果。

**重点是**：pass 分「干活」与「侦察」——分析 pass 是情报员，变换 pass 是施工队，依赖管理保证情报先行。<span class="marginnote">「分析 pass 与变换 pass 分离」是 LLVM 的经典设计：分析结果可被多个变换复用（`DominatorTree` 一次计算，`LICM`、`GVN` 都查），变换之间则按顺序串行。这个「情报与施工分开」的模式，让优化器的组织像「先测绘、再施工」的工程流程。</span>

## 2 优化流水线的组织

`opt -O2` 背后的流水线（简化）：

```
SimplifyCFG       /* 化简控制流 */
Inliner           /* 内联 */
InstCombine       /* 指令组合 */
GVN               /* 值编号 */
LICM              /* 循环外提 */
DCE               /* 死代码消除 */
```

关键要素：

**遍历**：函数 → 基本块 → 指令。
**匹配**：用 pattern match 识别目标模式（`x + 0`、`x * 1` 等）。
**变换**：`x + 0 → x` 替换使用。
**保留分析**：声明哪些分析结果未失效（`PreservedAnalyses`）。<span class="marginnote">「pass 的骨架是遍历 + 匹配 + 替换 + 保留声明」——看懂一个 LLVM pass，就等于看懂了一半的优化 pass。`PreservedAnalyses` 是 pass 与依赖管理的契约：pass 必须诚实声明「我改了哪些结构」，分析 pass 才能判断「我的结果还作不作数」——谎报会导致分析失效、优化出错。</span>

## 4 公式解析：pass 流水线的正确性

设流水线为 pass 序列 $P_1, P_2, \ldots, P_k$，每个 pass 是语义保持变换：

$$\text{语义}(IR) = \text{语义}(P_1(IR)) = \text{语义}(P_2(P_1(IR))) = \cdots$$

- **第一步，单 pass 保持**：每个 pass 正确 = 它不改程序语义（行为等价，第五十一节的判据）。
- **第二步，组合保持**：语义保持变换的复合仍保持语义——流水线整体正确。
- **第三步，顺序的收益**：顺序不影响正确性，但影响**收益**——不同顺序可能优化出不同质量的代码。

**「单 pass 正确 ⇒ 流水线正确」是 pass 架构的正确性基石**——开发者只需保证每个 pass 局部正确，流水线整体就安全；顺序则留给调优。

## 5 现代 PassManager 与「优化管道」

LLVM 的 pass 管理几经演进：

- **Legacy PassManager**：旧版，pass 对象化、需显式注册。
- **New PassManager**（LLVM 17+ 默认）：基于**函数分析管理**，pass 是「轻量 mixin」，分析按需惰性计算、缓存自动失效——更快、更安全。

现代优化管道的特点：

- **模块 pass 与函数 pass 分层**：模块级先做过程间优化，再对每个函数跑函数级流水线。
- **循环 pass 单独编排**：`LICM`、`IndVarSimplify`、`LoopUnroll` 在循环层循环执行。
- **优化等级预置**：`-O1`/`-O2`/`-O3` 各映射一套流水线。

**重点是**：pass 机制是 LLVM 优化的「操作系统」——pass 是进程，分析是资源，流水线是调度策略。<span class="marginnote">「新 PassManager 是 LLVM 的一次大重构」：旧版每次 pass 要整模块跑、分析缓存手动管理；新版按函数粒度惰性分析、自动失效、并行友好。这解释了为什么 LLVM 近年的编译速度提升一大截——pass 基础设施的效率，直接决定整个工具链的效率。</span>

## 6 小结

- **Pass** 是 LLVM 优化的最小模块：分析 pass（侦察）与变换 pass（施工）分离，依赖管理保证顺序。
- pass 按作用分 Module/Function/Loop/CallGraph 四类；变换 pass 是优化的主力。
- **流水线**按「化简 → 内联 → 全局 → 循环 → 清理」组织，常迭代多轮；顺序影响收益。
- 写 pass 的骨架：**遍历 + 匹配 + 替换 + 保留声明**；单 pass 正确 ⇒ 流水线正确。
- 现代 **New PassManager**：函数级惰性分析、自动失效、优化等级预置流水线。

在下一节，我们用 Clang 亲手观察整条流水线：**用 Clang 观察编译流程的各个阶段**。
