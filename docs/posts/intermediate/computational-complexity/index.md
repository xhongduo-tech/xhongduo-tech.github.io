---
pageClass: plain-doc
---

# 计算复杂性理论

Sipser《计算理论导引》、Arora & Barak《计算复杂性》。按照「学完一个学科 = 写完该学科权威教材对应的全部博文」的标准，每写完一篇勾掉一条。

## 主题规划

<ProgressGrid cat="intermediate/computational-complexity" />

### 第一篇 形式语言与自动机

- [x] [有穷自动机与正则语言](./dfa-regular-languages)
- [x] [正则表达式等价性](./regular-expressions-equivalence)
- [x] [泵引理与正则语言性质](./pumping-lemma-regular-languages)
- [x] [上下文无关文法](./context-free-grammars)
- [x] [下推自动机](./pushdown-automata)
- [x] [上下文无关语言性质](./context-free-language-properties)
- [x] [图灵机模型](./turing-machine)
- [x] [图灵机变体与丘奇-图灵论题](./turing-machine-variants-church-turing)
- [x] [可判定性](./decidable-languages)
- [x] [停机问题与不可判定问题](./halting-problem-undecidability)

### 第二篇 复杂性类

- [x] [时间复杂性与大 O 记号](./time-complexity-big-o)
- [x] [P 类与多项式时间](./p-class-polynomial-time)
- [x] [验证者与 NP 类](./verifiers-np)
- [x] [NP 完全性](./np-completeness)
- [x] [SAT 与可满足性问题](./sat-satisfiability)
- [x] [经典 NP 完全问题](./classic-np-complete-problems)
- [x] [归约方法](./reductions)
- [x] [多项式层次](./polynomial-hierarchy)
- [x] [空间复杂性与 PSPACE](./space-complexity-pspace)
- [x] [萨维奇定理](./savitch-theorem)
- [x] [随机化复杂性类](./randomized-complexity-classes)

### 第三篇 下界与高级专题

- [x] [时间层次定理](./time-hierarchy-theorem)
- [x] [空间层次定理](./space-hierarchy-theorem)
- [x] [电路复杂性](./circuit-complexity)
- [x] [交互式证明系统](./interactive-proof-systems)
- [x] [零知识证明](./zero-knowledge-proofs)
- [x] [计数复杂性（#P）](./counting-complexity-sharp-p)
- [x] [近似算法与不可近似性](./approximation-algorithms-inapproximability)
- [x] [参数化复杂性](./parameterized-complexity)
- [x] [去随机化](./derandomization)
- [x] [量子计算复杂性](./quantum-computational-complexity)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [ ] 计算模型（图灵机、可计算性回顾）
- [ ] 时间复杂性与 P 类（多项式时间）
- [ ] NP 与 NP 完全性（Cook-Levin 定理、归约技术）
- [ ] 经典 NP 完全问题（SAT、团、哈密顿回路）
- [ ] 空间复杂性（PSPACE、Savitch 定理）
- [ ] 对角化与相对化（时间层次定理、障碍结果）
- [ ] 随机化计算（BPP、RP、去随机化）
- [ ] 电路复杂性与非一致性（P/poly、下界方法）
- [ ] 交互证明与 PCP 定理（IP=PSPACE、不可近似性）
- [ ] 前沿专题（密码学基础、量子复杂性 BQP）
