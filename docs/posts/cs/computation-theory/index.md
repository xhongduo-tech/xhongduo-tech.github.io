---
pageClass: plain-doc
---

# 计算理论（可计算性与计算复杂性）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。以自动机、可计算性与计算复杂性为理论计算机科学主干，系统重建从正则语言到 NP 完全性与现代复杂性类的完整图景。

## 对标教材

- Michael Sipser, "Introduction to the Theory of Computation" (Cengage, 3rd ed.)
- Hopcroft, Motwani & Ullman, 《自动机理论、语言和计算导论》(Pearson, 3rd ed.)
- Sanjeev Arora & Boaz Barak, "Computational Complexity: A Modern Approach" (Cambridge University Press)

## 主题规划

<ProgressGrid cat="cs/computation-theory" />

### 第1篇

- [x] [有限自动机与正则语言](./finite-automata-regular-languages)
- [x] [非确定性自动机 NFA 与等价转化](./nfa-and-equivalence)
- [x] [正则表达式与正则语言](./regular-expressions)
- [x] [泵引理与正则语言判定](./pumping-lemma-regular-languages)
- [x] [上下文无关文法与派生树](./context-free-grammars-parse-trees)
- [x] [下推自动机与上下文无关语言](./pushdown-automata-context-free-languages)
- [x] [上下文无关语言的泵引理与性质](./pumping-lemma-context-free-languages)
- [x] [乔姆斯基范式与形式语言谱系](./chomsky-normal-form-language-hierarchy)

### 第2篇

- [x] [图灵机模型与 Church-Turing 论题](./turing-machine-church-turing-thesis)
- [x] [图灵机的变体与计算等价性](./turing-machine-variants-equivalence)
- [x] [可判定语言与递归可枚举语言](./decidable-and-recognizable-languages)
- [x] [停机问题与不可判定性](./halting-problem-undecidability)
- [x] [归约方法与不可判定问题](./reductions-undecidable-problems)
- [x] [递归定理与自引用](./recursion-theorem-self-reference)
- [x] [哥德尔编码与计算中的可表示性](./godel-encoding-representability)

### 第3篇

- [x] [时间复杂性类与 P](./time-complexity-and-p)
- [x] [非确定性时间与 NP](./nondeterministic-time-np)
- [x] [NP 完全性与 Cook-Levin 定理](./np-completeness-cook-levin)
- [x] [常见 NP 完全问题与归约](./common-np-complete-problems)
- [x] [空间复杂性类与 PSPACE](./space-complexity-pspace)
- [x] [L、NL 与对数空间归约](./l-nl-logspace-reductions)
- [x] [复杂性层次定理与类间关系](./complexity-hierarchy-theorems)

### 第4篇

- [x] [概率图灵机与随机复杂性类](./probabilistic-turing-machines)
- [x] [交互证明系统与 IP=PSPACE](./interactive-proofs-ip-pspace)
- [x] [PCP 定理与不可近似性](./pcp-theorem-inapproximability)
- [x] [电路复杂性理论](./circuit-complexity)
- [x] [多项式层次 PH](./polynomial-hierarchy)
- [x] [单向函数与密码学复杂性基础](./one-way-functions-cryptography)
- [x] [去随机化与伪随机发生器](./derandomization-pseudorandomness)
