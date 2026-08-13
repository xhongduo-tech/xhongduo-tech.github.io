---
pageClass: plain-doc
---

# 程序分析（静态分析/符号执行）

程序分析是连接编译优化与软件安全验证的桥梁，以数据流分析、抽象解释、符号执行、污点分析等手段在程序源码或中间表示上自动化地推断程序性质、指导优化并发现漏洞。学完这些章节，就掌握了一整套"让机器理解程序"的原理性工具。

## 对标教材

- Nielson, Nielson & Hankin, "Principles of Program Analysis" (Springer, 1999)
- Keith Cooper & Linda Torczon, "Engineering a Compiler" (Morgan Kaufmann, 2nd ed., 2012)
- Aho, Lam, Sethi & Ullman, 《编译原理》(机械工业出版社)

## 主题规划

<ProgressGrid cat="cs/program-analysis" />

### 第1篇

- [x] [程序分析概述：目标、分类与应用（Nielson 第1章）](./program-analysis-overview)
- [x] [程序表示：控制流图与中间表示（Cooper 第9章）](./control-flow-graphs-and-ir)
- [x] [格论基础：偏序、格与不动点定理（Nielson 第1章）](./lattice-theory-foundations)
- [x] [形式语义：操作语义与抽象语义（Nielson 第1章）](./formal-semantics)
- [x] [单调数据流框架与传递函数（Nielson 第2章）](./monotone-dataflow-framework)

### 第2篇

- [x] [经典数据流分析：活跃变量、到达定值与可用表达式（Nielson 第2章）](./classic-dataflow-analysis)
- [x] [数据流方程求解：迭代算法与位向量实现（Nielson 第2章）](./dataflow-iterative-algorithms)
- [x] [静态单赋值形式（SSA）（Cooper 第9章）](./static-single-assignment)
- [x] [支配者与支配树（Cooper 第9章）](./dominators-and-dominator-trees)
- [x] [常量传播与复写传播（Cooper 第10章）](./constant-and-copy-propagation)
- [x] [死代码消除（Cooper 第10章）](./dead-code-elimination)
- [x] [循环分析与自然循环（龙书 第9章）](./loop-analysis-natural-loops)
- [x] [循环优化：外提、强度削减与归纳变量消除（龙书 第9章）](./loop-optimizations)

### 第3篇

- [x] [抽象解释：具体语义到抽象语义（Nielson 第4章）](./abstract-interpretation)
- [x] [Galois 连接与抽象不动点（Nielson 第4章）](./galois-connections)
- [x] [经典抽象域：区间、符号与多面体（Nielson 第4章）](./abstract-domains)
- [x] [基于约束的分析（Nielson 第3章）](./constraint-based-analysis)
- [x] [过程间分析：调用字符串与环境（Nielson 第6章）](./interprocedural-analysis)
- [x] [指针分析与别名分析（Cooper 第9章）](./pointer-alias-analysis)
- [x] [类型与效应系统（Nielson 第5章）](./type-and-effect-systems)

### 第4篇

- [x] [约束求解与 SAT/SMT 基础（Nielson 第3章扩展）](./sat-smt-basics)
- [x] [符号执行：符号状态与路径条件（Nielson 第6章相关）](./symbolic-execution)
- [x] [动态符号执行（Concolic）与路径探索（书目外）](./concolic-execution)
- [x] [污点分析与数据流安全（Nielson 第2章应用）](./taint-analysis)
- [x] [程序切片与依赖分析（Nielson 第7章）](./program-slicing)
- [x] [控制流分析（CFA）与程序转换（Nielson 第7章）](./control-flow-analysis-cfa)
