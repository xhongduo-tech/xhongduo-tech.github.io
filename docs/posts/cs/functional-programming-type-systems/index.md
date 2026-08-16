---
pageClass: plain-doc
---

# 函数式编程与类型系统（TAPL/范畴论基础）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Pierce, "Types and Programming Languages" (2002)
- Bird, "Thinking Functionally with Haskell" (2015)
- Harper, "Practical Foundations for Programming Languages" (2nd ed., 2016)

## 主题规划

<ProgressGrid cat="cs/functional-programming-type-systems" />

### 第1篇

- [x] [λ 演算（丘奇编码、归约策略、不动点组合子）](./lambda-calculus)
- [x] [简单类型 λ 演算（类型安全=进展+保持的证明范式）](./simply-typed-lambda-calculus)
- [x] [多态类型系统（System F、参数多态、ML 类型推断）](./polymorphic-type-systems)
- [x] [Hindley-Milner 类型推断（合一算法、Algorithm W）](./hindley-milner-type-inference)
- [x] [代数数据类型与模式匹配（和类型/积类型、穷尽性检查）](./algebraic-data-types-pattern-matching)
- [x] [高阶抽象（函子/应用函子/单子、范畴论的最小核心）](./higher-order-abstraction-functors-monads)
- [x] [惰性求值（按需计算、无限数据结构、严格性分析）](./lazy-evaluation)
- [x] [效应系统（纯函数与副作用的隔离、IO Monad）](./effect-systems-io-monad)

### 第2篇

- [x] [依赖类型（类型即命题、Agda/Idris 的定理证明）](./dependent-types)
- [x] [线性类型与子结构类型（Rust 所有权/借用检查的理论根源）](./linear-types-substructural)
- [x] [类型系统与编译器（类型引导的优化、擦除与保类型编译）](./types-and-compilation)
- [x] [现代语言实践（Haskell/OCaml/Scala/Rust/Swift 的类型特性对比）](./modern-language-practice)
