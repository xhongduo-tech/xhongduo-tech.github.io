---
title: 闭包转换与 lambda 提升
date: 2026-09-08
section: cs
---

# 闭包转换与 lambda 提升

<div class="epigraph">
<p>嵌套函数要变成顶层函数加环境参数。闭包转换把自由变量打成记录；lambda lifting 把自由变量变成额外形参。</p>
<footer>—— 据 Johnsson, Lambda Lifting；Appel, Compiling with Continuations；Peyton Jones 整理</footer>
</div>

上一课[弱引用](/cs/weak-refs-finalizers) 收束堆。语言特性落地：表面有嵌套 λ。[λ 演算](/cs/lambda-calculus) 的自由变量在实现里要有家。缺口是 **closure conversion / lifting**。CPS 下一课。不重写 β。

## 问题

`λx. ... y ...` 的 $y$ 来自外层。机器没有嵌套作用域。转换：`λ(env,x). ... env.y`，调用处传闭包指针。缺口是**环境表示**（扁平记录、链），不是 GC 弱。<span class="marginnote">直觉类比：闭包是「说明书 + 背包」——代码是说明书，环境记录是背包里装着的函数体要用的外部变量。C 函数指针只有说明书、没有背包，所以表达不了「记住调用处的那个 y」。</span>

提升：把 $y$ 变成参数，调用处传递，适合已知调用点；做成闭包值则必须堆记录。<span class="marginnote">术语翻译 lambda lifting：「把嵌套函数拎到顶层」。它不打包环境，而是把每个自由变量 $y$ 升级成正式参数，调用处逐个实参传进去——签名变长是代价，换来完全不需要堆上的环境记录，适合调用点已知的场合。</span>

### 闭包不是 C 函数指针

函数指针无环境。带环境的是胖指针 `{code, env}`。C++ `std::function`、Rust 闭包各有表示。

<span class="marginnote">Johnsson 1985。Appel CWC。Peyton Jones 实现书。与[逃逸](/cs/escape-analysis)：环境未逃逸可栈上。</span>

## 方法

算自由变量。选表示：逃逸则堆闭包。生成顶层函数。调用：间接 `code(env, args)`。已知闭包+内联则可消间接。

```mermaid
flowchart TD
  NEST["嵌套 λ"] --> FV["自由变量"]
  FV --> ENV["环境记录"]
  ENV --> TOP["顶层函数"]
```

与所有权：环境捕获是移动还是借用，接 Rust 课。与 RC：环境字段要计数。

## 机制

链式环境：共享外层，可变捕获麻烦。扁平：复制，可变要盒。不要捕获后还以为外层栈槽活着——悬空，生命周期课已禁。

```mermaid
flowchart TD
  ENV["环境表示二选一"] --> LINK["链式：每层闭包指向父环境"]
  LINK --> LPRO["外层结构共享，不必复制"]
  LINK --> LCON["取远处的变量要沿链走多跳"]
  ENV --> FLAT["扁平：一个记录装齐全部自由变量"]
  FLAT --> FPRO["取任何变量一步到位"]
  FLAT --> FCON["复制了值：可变捕获必须装盒"]
```

<span class="marginnote">这张图回答「环境到底怎么装」。常见误区：以为捕获变量就自动「共享」。链式环境确实共享外层槽位；扁平环境是复制——可变量因此要先装进盒子。循环里给多个闭包捕获同一个计数器时，拿到的究竟是同一个盒子还是各自的拷贝，就取决于选了哪种表示。</span>

优化：已知函数、无自由变量则退化成函数指针。

## 边界

本课不写 CPS。后课默认：嵌套函数降为闭包或提升参数。下一课 CPS 与延续。

也不把闭包当数学闭包算子。

## 小结

- 闭包转换：代码+环境；lifting：自由变量变参数。
- 表示受逃逸与可变捕获约束。
- 无自由变量则退化为函数指针。
- 出处：Johnsson；Appel CWC；Peyton Jones。
