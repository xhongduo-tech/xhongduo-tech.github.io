---
pageClass: plain-doc
---

# 约束求解与自动定理证明（SAT/SMT）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Biere, Heule, van Maaren & Walsh (eds.), "Handbook of Satisfiability" (2nd ed., 2021)
- Kroening & Strichman, "Decision Procedures" (2nd ed., 2016)
- de Moura & Bjørner, "Z3: An Efficient SMT Solver" (TACAS 2008) 及 Z3 官方文档

## 主题规划

<ProgressGrid cat="cs/constraint-solving-sat-smt" />

### 第1篇

- [x] [SAT 问题与 NP 完全性（Cook-Levin 定理的工程意义）](./sat-np-completeness)
- [x] [DPLL 算法（回溯搜索、单元传播、纯文字消除）](./dpll-algorithm)
- [x] [CDCL 革命（冲突驱动子句学习、VSIDS 启发式）](./cdcl-revolution)
- [x] [现代 SAT 求解器（MiniSat/CaDiCaL/Kissat 的架构）](./modern-sat-solvers)
- [x] [从 SAT 到 SMT（背景理论：算术/数组/位向量/未解释函数）](./from-sat-to-smt)
- [x] [DPLL(T) 框架（布尔骨架与理论求解器的协同）](./dpll-t-framework)
- [x] [Z3 实战（符号执行的引擎、程序验证的条件生成）](./z3-in-practice)
- [x] [约束规划 CP（全局约束、传播器、与 SAT 的融合）](./constraint-programming-cp)

### 第2篇

- [x] [MaxSAT 与优化（软约束、 UNSAT 核的利用）](./maxsat-optimization)
- [x] [自动定理证明（一阶逻辑的 Superposition、Lean/Isabelle 交互证明）](./automated-theorem-proving)
- [x] [应用场景（EDA 验证、配置求解、调度规划、密码分析）](./applications)
- [x] [前沿（神经引导的启发式、证明生成与验证）](./frontiers)
