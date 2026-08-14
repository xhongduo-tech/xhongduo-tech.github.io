---
pageClass: plain-doc
---

# Rust 编程

Rust 是一门强调内存安全、并发与性能的系统级编程语言，其所有权模型在编译期杜绝空悬指针与数据竞争，无需垃圾回收即可安全地管理系统资源。以《Rust 程序设计语言》（The Rust Book）为纲，从语法基础到高级特性逐章写成博文，学完即掌握这门现代系统编程语言。

## 对标教材

- 《Rust 程序设计语言》(The Rust Book, 第2版)
- 《Programming Rust》(Blandy, Orendorff & Tindall, 第2版)
- 《Rust 实战》(Rust in Action, Tim McNamara)

## 主题规划

<ProgressGrid cat="cs/rust-programming" />

### 第1篇 语言基础与所有权

- [x] [Hello, World 与猜数游戏](./hello-world-guessing-game)
- [x] [变量、数据类型与函数](./variables-data-types-functions)
- [x] [控制流](./control-flow)
- [x] [所有权与内存模型](./ownership-memory-model)
- [x] [引用与借用](./references-borrowing)
- [x] [切片 Slice](./slices)
- [x] [结构体与方法](./structs-methods)
- [x] [枚举与模式匹配](./enums-pattern-matching)

### 第2篇 模块化与核心特性

- [x] [包、Crate 与模块系统](./packages-crates-modules)
- [x] [常用集合：Vec、String 与 HashMap](./common-collections)
- [x] [错误处理：panic 与 Result](./error-handling)
- [x] [泛型、Trait 与生命周期](./generics-traits-lifetimes)
- [x] [编写自动化测试](./automated-tests)
- [x] [命令行程序实战：minigrep](./minigrep)
- [x] [闭包与迭代器](./closures-iterators)
- [x] [Cargo 工作流与 Crates.io](./cargo-crates-io)

### 第3篇 内存安全、并发与高级特性

- [x] [智能指针：Box、Rc 与 RefCell](./smart-pointers)
- [x] [线程与消息传递并发](./threads-message-passing)
- [x] [共享状态并发：Mutex 与 Arc](./shared-state-concurrency)
- [x] [Rust 的面向对象特性](./oop-features)
- [x] [模式与模式匹配进阶](./patterns-matching-advanced)
- [x] [不安全代码与 FFI](./unsafe-ffi)
- [x] [高级 Trait、生命周期与类型](./advanced-traits-lifetimes-types)
- [x] [综合项目：多线程 Web 服务器](./multi-threaded-web-server)
