---
pageClass: plain-doc
---

# 程序设计语言

本分类以《程序设计语言原理》（Sebesta）与 PLT 课程体系为纲，系统覆盖程序设计语言的设计原理、语法语义、类型系统、编程范式与运行时机制。

## 主题规划

<ProgressGrid cat="cs/programming-languages" />


### 第一篇 语言设计与评价

- [x] [学习程序设计语言原理的意义](./why-study-pl)
- [x] [编程语言的四大评价标准：可读性、可写性、可靠性与成本](./language-evaluation-criteria)
- [x] [影响语言设计的因素：计算机体系结构与编程方法学](./language-design-factors)
- [x] [语言的实现方法：编译、解释与混合实现](./language-implementation-methods)
- [x] [语言的演化历程：从 Plankalkül 到现代多范式语言](./language-evolution)

### 第二篇 语法描述

- [x] [语法与语义的基本概念](./syntax-semantics-basics)
- [x] [上下文无关文法与推导](./context-free-grammar-and-derivation)
- [x] [BNF 与扩展 BNF（EBNF）](./bnf-and-ebnf)
- [x] [二义性、运算符优先级与结合性](./ambiguity-precedence-associativity)
- [x] [属性文法：静态语义的描述方法](./attribute-grammars)

### 第三篇 语义学基础

- [x] [形式化语义学的动机与分类](./formal-semantics-motivation)
- [x] [操作语义（Operational Semantics）](./operational-semantics)
- [x] [指称语义（Denotational Semantics）初步](./denotational-semantics)
- [x] [公理语义与程序正确性验证](./axiomatic-semantics-verification)
- [x] [λ 演算作为语义描述的元语言](./lambda-calculus-meta-language)

### 第四篇 数据类型

- [x] [数据类型的概念与类型检查](./data-types-and-type-checking)
- [x] [原始数据类型：数值、字符、布尔与字符串](./primitive-data-types)
- [x] [枚举类型与子范围类型](./enumeration-subrange-types)
- [x] [数组类型：下标、绑定与初始化](./array-types)
- [x] [记录类型与元组类型](./record-tuple-types)
- [x] [列表类型与关联数组](./list-associative-arrays)
- [x] [联合类型、判别联合与可选类型（Option/Maybe）](./union-discriminated-union-option)
- [x] [指针与引用类型](./pointer-reference-types)

### 第五篇 类型系统深入

- [x] [强类型与弱类型：类型安全的边界](./strong-weak-typing)
- [x] [静态类型检查与动态类型检查](./static-dynamic-type-checking)
- [x] [类型等价：名字等价与结构等价](./type-equivalence)
- [x] [类型推导与 Hindley-Milner 算法](./type-inference-hindley-milner)
- [x] [参数多态（泛型）、特设多态与子类型多态](./parametric-ad-hoc-subtype-polymorphism)
- [x] [协变与逆变](./covariance-contravariance)

### 第六篇 名称、绑定与作用域

- [x] [名称、变量与值的概念区分](./names-variables-values)
- [x] [绑定的概念与绑定时间](./binding-and-binding-time)
- [x] [静态作用域（词法作用域）与动态作用域](./static-dynamic-scoping)
- [x] [块结构、嵌套子程序与作用域空洞](./block-structure-scope-holes)
- [x] [存储绑定与变量的生存期](./storage-binding-lifetime)

### 第七篇 表达式与赋值语句

- [x] [算术表达式：运算符求值顺序与优先级](./arithmetic-expressions)
- [x] [混合模式表达式与类型转换](./mixed-mode-expressions-coercion)
- [x] [关系表达式、布尔表达式与短路求值](./boolean-expressions-short-circuit)
- [x] [赋值语句的形式与复合赋值](./assignment-statements)
- [x] [表达式中的副作用与函数副作用](./side-effects-expressions)

### 第八篇 语句级控制结构

- [x] [控制结构的选择标准：单入口单出口](./control-structure-criteria)
- [x] [复合语句与条件语句](./compound-conditional-statements)
- [x] [多重选择结构：if-elif 与 switch/match](./multiway-selection)
- [x] [迭代语句：计数循环、逻辑循环与迭代器](./iterative-statements)
- [x] [无条件分支（goto）问题与结构化编程](./goto-structured-programming)
- [x] [卫式命令与非确定性控制结构](./guarded-commands)

### 第九篇 子程序

- [x] [子程序的基本概念与定义](./subprogram-basics)
- [x] [形参与实参：位置参数、关键字参数与默认参数](./formal-actual-parameters)
- [x] [参数传递方式：传值、传引用与传结果](./parameter-passing)
- [x] [协同程序（Coroutine）](./coroutines)
- [x] [嵌套子程序、静态链与非局部变量访问](./nested-subprograms-static-chain)
- [x] [闭包（Closure）与一等函数](./closures-first-class-functions)

### 第十篇 抽象数据类型与封装

- [x] [抽象的概念与数据抽象](./abstraction-data-abstraction)
- [x] [抽象数据类型的设计要求](./adt-design-requirements)
- [x] [Ada、C++ 与 Java 中的数据抽象实现](./data-abstraction-ada-cpp-java)
- [x] [命名封装：命名空间与包](./namespace-packages)
- [x] [参数化抽象数据类型（泛型容器）](./parametric-adt-generic-containers)

### 第十一篇 面向对象程序设计

- [x] [面向对象的基本概念：对象、类与消息传递](./oop-basics-objects-classes)
- [x] [继承的设计问题：单继承与多继承](./inheritance-design)
- [x] [动态绑定与消息的多态性](./dynamic-binding-polymorphism)
- [x] [多重继承的问题：菱形继承及其解决方案](./multiple-inheritance-diamond)
- [x] [Smalltalk 的纯面向对象模型](./smalltalk-pure-oop)
- [x] [接口（Interface）、协议（Protocol）与混入（Mixin）](./interfaces-protocols-mixins)

### 第十二篇 函数式编程

- [x] [函数式编程的基本概念：数学函数与引用透明性](./functional-programming-basics)
- [x] [λ 演算基础：抽象、应用与 Church 编码](./lambda-calculus-basics)
- [x] [高阶函数、柯里化与函数组合](./higher-order-functions-currying)
- [x] [Scheme 基础：函数、表与递归](./scheme-basics)
- [x] [惰性求值（Lazy Evaluation）](./lazy-evaluation)
- [x] [Haskell 简介：类型类与模式匹配](./haskell-intro)
- [x] [单子（Monad）初步：IO 与计算上下文](./monads-intro)
- [x] [函数式语言的实现问题：尾递归与求值策略](./functional-language-implementation)

### 第十三篇 逻辑式编程

- [x] [谓词演算基础与子句形式](./predicate-calculus-clause-form)
- [x] [归结原理（Resolution）](./resolution-principle)
- [x] [Prolog 的基本元素：项、事实、规则与目标](./prolog-basics)
- [x] [合一（Unification）、回溯与 Prolog 的缺陷](./unification-backtracking-prolog)

### 第十四篇 并发程序设计

- [x] [并发的基本概念：任务、同步与竞争](./concurrency-basics)
- [x] [共享内存并发模型：信号量与管程（Monitor）](./shared-memory-concurrency)
- [x] [消息传递模型](./message-passing)
- [x] [Actor 模型](./actor-model)
- [x] [CSP（通信顺序进程）与 Go 的 channel](./csp-go-channels)
- [x] [软件事务内存（STM）与异步编程模型（async/await）](./stm-async-await)

### 第十五篇 内存管理

- [x] [栈分配与堆分配的管理](./stack-heap-allocation)
- [x] [引用计数（Reference Counting）及其循环引用问题](./reference-counting)
- [x] [追踪式垃圾回收：标记-清除、复制式与分代回收](./tracing-garbage-collection)
- [x] [所有权系统与借用检查（Rust 的所有权模型）](./rust-ownership-borrowing)
- [x] [手动内存管理的典型问题：悬垂指针与内存泄漏](./manual-memory-management-bugs)

### 第十六篇 元编程

- [x] [宏系统：文本宏与卫生宏（Hygienic Macro）](./macro-systems)
- [x] [反射（Reflection）与自省（Introspection）](./reflection-introspection)
- [x] [模板元编程与编译期计算](./template-metaprogramming)

### 第十七篇 主流语言机制剖析

- [x] [Python：动态类型、对象模型与 GIL](./python-object-model-gil)
- [x] [Rust：所有权、生命周期与零成本抽象](./rust-ownership-lifetime-zero-cost)
- [x] [Go：接口、goroutine 与极简类型系统](./go-interfaces-goroutines)
- [x] [Java 与 C++：泛型擦除、RAII 与值语义对比](./java-cpp-generics-erasure-raii)
- [x] [五门语言的类型系统横向对比](./type-system-comparison)
- [x] [五门语言的错误处理机制对比：异常、Result 与多返回值](./error-handling-comparison)

### 第十八篇 语言虚拟机与运行时

- [x] [语言虚拟机的概念：从 p-code 到现代 VM](./language-virtual-machine)
- [x] [JVM：类加载、字节码与即时编译（JIT）](./jvm-classloading-bytecode-jit)
- [x] [JVM 内存模型与垃圾回收器体系](./jvm-memory-model-gc)
- [x] [BEAM：Erlang 虚拟机与容错并发模型](./beam-erlang-vm)
- [x] [WebAssembly：面向 Web 的通用虚拟指令集](./webassembly)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
