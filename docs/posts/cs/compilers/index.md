---
pageClass: plain-doc
---

# 编译原理

对标《编译原理》（龙书，Compilers: Principles, Techniques, and Tools），从词法分析一路写到代码优化，并补充垃圾回收、JIT、LLVM 等现代专题。

## 主题规划

<ProgressGrid cat="cs/compilers" />


### 第一篇 编译器概述

- [x] [编译器的结构：分析与综合](./compiler-structure)
- [x] [编译器各阶段：词法、语法、语义、中间代码生成、优化、代码生成](./compiler-phases)
- [x] [符号表管理与错误处理](./symbol-table-error-handling)
- [x] [编译器、解释器与混合执行模型](./compiler-interpreter-models)
- [x] [编译器的演化：从单遍编译到多遍编译](./compiler-evolution)
- [x] [编译器构造工具与编译器-编译器（Compiler-Compiler）](./compiler-construction-tools)

### 第二篇 词法分析

- [x] [词法分析器的作用与接口](./lexer-role-interface)
- [x] [记号、模式与词素（Token、Pattern、Lexeme）](./token-pattern-lexeme)
- [x] [输入缓冲与哨兵（Sentinel）技术](./input-buffering-sentinel)
- [x] [正则表达式与正则定义](./regex-regular-definitions)
- [x] [有限自动机：NFA 与 DFA](./finite-automata-nfa-dfa)
- [x] [从正则表达式到 NFA：Thompson 构造法](./regex-to-nfa-thompson)
- [x] [从 NFA 到 DFA：子集构造法](./nfa-to-dfa-subset-construction)
- [x] [DFA 状态最小化](./dfa-state-minimization)
- [x] [词法分析器生成工具 Lex/Flex 的使用与实现原理](./lex-flex-generator)

### 第三篇 语法分析

- [x] [上下文无关文法（CFG）与推导](./context-free-grammar-derivations)
- [x] [语法分析树与文法的二义性](./parse-trees-ambiguity)
- [x] [消除二义性、左递归与提取左公因子](./removing-ambiguity-left-recursion)
- [x] [自顶向下分析概述：递归下降分析](./topdown-parsing-recursive-descent)
- [x] [FIRST 集与 FOLLOW 集的计算](./first-follow-sets)
- [x] [LL(1) 文法与预测分析表](./ll1-grammars-predictive-table)
- [x] [非递归的表驱动预测分析器](./table-driven-predictive-parser)
- [x] [自底向上分析概述：移进-归约分析](./bottomup-parsing-shift-reduce)
- [x] [LR 分析器的工作原理与 LR 分析表结构](./lr-parser-algorithm)
- [x] [LR(0) 项集与 SLR 分析表构造](./lr0-items-slr)
- [x] [规范 LR(1) 分析表构造](./canonical-lr1)
- [x] [LALR(1) 分析表构造](./lalr1-construction)
- [x] [二义性文法在 LR 分析中的应用](./ambiguous-grammar-lr)
- [x] [语法分析器生成工具 Yacc/Bison 的使用与实现原理](./yacc-bison-parser-generator)

### 第四篇 语法制导翻译

- [x] [语法制导定义（SDD）：综合属性与继承属性](./sdd-synthesized-inherited-attributes)
- [x] [属性文法的求值顺序与依赖图](./attribute-dependency-graph)
- [x] [S-属性定义与 L-属性定义](./s-attributed-l-attributed)
- [x] [语法制导翻译方案（SDT）](./syntax-directed-translation-schemes)
- [x] [在 LL 分析中实现 SDT](./sdt-in-ll-parsing)
- [x] [在 LR 分析中实现 SDT](./sdt-in-lr-parsing)
- [x] [一个简单的语法制导翻译器实战](./simple-syntax-directed-translator)

### 第五篇 中间代码生成

- [x] [中间表示的形式：语法树、DAG 与三地址码](./intermediate-representations-tree-dag-3ac)
- [x] [三地址码的指令类型与四元式、三元式表示](./three-address-code-instructions)
- [x] [表达式的翻译与临时变量生成](./translation-expressions-temporaries)
- [x] [数组寻址的翻译](./translation-array-addressing)
- [x] [类型系统与类型表达式](./type-systems-type-expressions)
- [x] [类型检查：类型等价与类型转换](./type-checking-equivalence-conversion)
- [x] [布尔表达式的翻译：数值表示法与短路求值](./translation-boolean-expressions)
- [x] [控制流语句的翻译与回填（Backpatching）技术](./control-flow-backpatching)
- [x] [过程调用与 switch 语句的翻译](./translation-procedure-calls-switch)

### 第六篇 运行时环境

- [x] [存储组织：代码区、静态区、栈区与堆区](./storage-organization)
- [x] [活动记录（Activation Record）与调用序列](./activation-records-call-sequences)
- [x] [栈式分配与活动树](./stack-allocation-activation-trees)
- [x] [非局部名字的访问：访问链与嵌套深度](./nonlocal-access-access-links)
- [x] [堆管理与内存碎片问题](./heap-management-fragmentation)
- [x] [参数传递机制：传值、传引用、传名与复制恢复](./parameter-passing-mechanisms)

### 第七篇 代码生成

- [x] [代码生成器的设计问题：输入、目标程序与指令选择](./code-generator-design-issues)
- [x] [目标机器模型与指令代价](./target-machine-instruction-costs)
- [x] [基本块与流图的构造](./basic-blocks-flow-graphs)
- [x] [基本块的优化：局部公共子表达式消除与死代码消除](./local-optimizations-basic-blocks)
- [x] [一个简单的代码生成器：下次引用信息](./simple-code-generator-next-use)
- [x] [寄存器分配：图着色算法](./register-allocation-graph-coloring)
- [x] [基于树的模式匹配指令选择](./tree-based-instruction-selection)
- [x] [窥孔优化（Peephole Optimization）](./peephole-optimization)

### 第八篇 机器无关优化

- [x] [优化的来源与优化编译器的组织](./sources-of-optimization)
- [x] [数据流分析基础：到达定值分析](./dataflow-analysis-reaching-definitions)
- [x] [活跃变量分析与可用表达式分析](./live-variables-available-expressions)
- [x] [常量传播与常量折叠](./constant-propagation-folding)
- [x] [公共子表达式消除](./common-subexpression-elimination)
- [x] [复写传播与死代码消除](./copy-propagation-dead-code)
- [x] [循环优化：代码外提与归纳变量消除](./loop-optimizations)
- [x] [强度削弱与部分冗余消除](./strength-reduction-partial-redundancy)
- [x] [支配结点与循环的识别](./dominance-loop-identification)
- [x] [过程间分析简介](./interprocedural-analysis)

### 第九篇 专题：垃圾回收

- [x] [垃圾回收的基本概念：可达性与安全点](./gc-concepts-reachability-safepoints)
- [x] [标记-清扫（Mark-Sweep）算法](./mark-sweep-gc)
- [x] [引用计数及其循环引用问题](./reference-counting-gc)
- [x] [复制式回收与半区（Semispace）策略](./copying-collector-semispace)
- [x] [分代垃圾回收（Generational GC）](./generational-gc)
- [x] [增量式与并发垃圾回收](./incremental-concurrent-gc)

### 第十篇 专题：JIT 编译与动态编译

- [x] [JIT 编译的原理：解释执行与即时编译的权衡](./jit-compilation-principles)
- [x] [热点探测与方法级 JIT、跟踪 JIT（Tracing JIT）](./hotspot-detection-tracing-jit)
- [x] [分层编译：以 JVM 的 C1/C2 为例](./tiered-compilation-c1-c2)
- [x] [推测优化与去优化（Deoptimization）](./speculative-optimization-deoptimization)
- [x] [动态语言 JIT：以 V8 与 PyPy 为例](./dynamic-language-jit-v8-pypy)

### 第十一篇 专题：LLVM 架构

- [x] [LLVM 的整体架构：前端、优化器与后端](./llvm-overall-architecture)
- [x] [LLVM IR：SSA 形式的中间表示](./llvm-ir-ssa)
- [x] [LLVM Pass 机制与优化流水线](./llvm-pass-optimization-pipeline)
- [x] [用 Clang 观察编译流程的各个阶段](./clang-compilation-stages)
- [x] [基于 LLVM 实现一门语言的前端：Kaleidoscope 教程解读](./kaleidoscope-frontend-llvm)
- [x] [LLVM 与现代编译器生态：Clang、Rust 与 Swift](./llvm-ecosystem-clang-rust-swift)

### 第十二篇 专题：手写一个玩具编译器

- [x] [设计玩具语言的语法与特性集](./toy-language-design)
- [x] [手写词法分析器：从字符流到记号流](./handwritten-lexer)
- [x] [手写递归下降语法分析器](./handwritten-recursive-descent-parser)
- [x] [构建抽象语法树（AST）与符号表](./ast-symbol-table)
- [x] [语义分析与简单类型检查](./semantic-analysis-type-checking)
- [x] [生成目标代码：编译到 C、汇编或字节码](./codegen-to-c-asm-bytecode)
- [x] [为玩具编译器编写测试与错误诊断](./toy-compiler-testing-diagnostics)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
