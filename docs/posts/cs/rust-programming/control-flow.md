---
title: 控制流
date: 2026-08-07
---

# 控制流

<div class="epigraph">
<p>程序 = 数据结构 + 算法。而算法，就是控制流在数据上留下的脚印。</p>
<footer>—— 尼古拉斯 · 沃斯（Niklaus Wirth）《Algorithms + Data Structures = Programs》</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从控制流开始

任何程序的本质都是「按条件走不同路径、按次数重复某段工作」。Rust 的控制流结构在表面上与 C/Java 相仿——`if`、`loop`、`while`、`for`——但内里有一条贯穿的哲学：**控制结构都是表达式，都能产出值**。这继承自函数式传统，让 Rust 在命令式外观下拥有表达式的力量。上一课我们学到「函数最后一个表达式即返回值」，控制流正是把这条扩展到「整个 `if` 块、整个循环体都可以作为值」。

## 1 if 表达式

### 基本用法

```rust
let number = 6;

if number % 4 == 0 {
    println!("number 能被 4 整除");
} else if number % 3 == 0 {
    println!("number 能被 3 整除");
} else {
    println!("number 不能被 4 或 3 整除");
}
```

与 C 的三个关键差别：

**条件必须是 `bool` 类型**：`if number { ... }` 是编译错误。Rust 没有 C 那种「非零即真」的隐式转换——条件的真假必须显式写成布尔表达式。这杜绝了 `if (x = 5)` 这类把赋值当比较的著名笔误。<span class="marginnote">这一点让无数 C 漏洞失去温床：`if (x = 5)` 在 C 里是「把 5 赋给 x 然后判断真假」，几乎总是真；在 Rust 里直接编译失败，因为 `x = 5` 的类型是 `()` 而非 `bool`。</span>

**`else if` 不需要括号**：条件后直接跟代码块，`else if` 可以任意链式堆叠。

**多余分支不影响类型**：`if`/`else` 分支里的代码块各自独立，但若作为表达式使用则必须统一类型。

### if 作为表达式

```rust
let condition = true;
let number = if condition { 5 } else { 6 };
println!("number = {number}");
```

`if` 块的值成为 `number` 的值。**两个分支必须返回同一类型**，否则编译错误——编译器不让你写出「有时是整数、有时是字符串」的不确定值。

## 2 loop、while 与 for

### loop：无限循环与 break 携带值

```rust
let mut counter = 0;
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;   // break 带出值，成为 loop 的值
    }
};
println!("result = {result}"); // 20
```

`break counter * 2` 把 `counter * 2` 作为整个 `loop` 表达式的值送出循环。这在「反复尝试直到成功，成功后把结果带走」的场景非常自然——猜数游戏的 `loop` + `break` 正是其雏形。

### while：条件循环

```rust
let mut number = 3;
while number != 0 {
    println!("{number}");
    number -= 1;
}
```

`while` 每次进入循环体前检查条件，条件为 `false` 时退出。适合「循环次数事先不知道，由条件决定」的情形。

### for：遍历集合

Rust 官方最推崇的循环是 `for`，因为它对越界天然免疫：

```rust
let a = [10, 20, 30, 40, 50];

for element in a {
    println!("值是 {element}");
}
```

`for` 遍历的是**迭代器**。数组、切片、`Vec`、`String`、区间都能被转换为迭代器，`for` 逐个取出元素。比起 `while` 加下标手动控制，`for` 不直接碰索引，从根上避免了 off-by-one 与越界问题。<span class="marginnote">`for` 的「不需要索引」是 Rust 借用检查器得以简化的重要原因：元素由迭代器逐个交出所有权或借用，你不需要维护 `i` 与 `len` 的同步。到第13篇《闭包与迭代器》我们会看到 `Iterator` trait 是标准库最庞大的抽象体系。</span>

配合区间语法，`for` 可以复刻「计数器循环」：

```rust
for number in (1..4).rev() {
    println!("{number}");
}
```

`1..4` 是半开区间（1,2,3），`.rev()` 反转迭代方向，输出 3、2、1。**半开区间的语义在这里再次出现**：`1..4` 不含 4，这是 Rust 序列记号的统一约定。

### 循环控制：break 与 continue

`break`：立即退出循环（可携带值）。
`continue`：跳过本次迭代剩余部分，进入下一次迭代。

两者都作用于「最近的循环」。如果存在嵌套循环，可以用**循环标签（loop label）**指定目标：

```rust
'outer: loop {
    loop {
        break 'outer;   // 跳出外层循环
    }
}
```

标签 `'outer` 写在 `loop` 前，`break 'outer` 指名要跳出哪一层。这在深嵌套里比「布尔标志位逐层退出」清晰得多。

## 3 从 if/else 到 match：条件分支的演进

猜数游戏里已经用过 `match`，这里把它与 `if` 对照，理解 Rust 为什么把 `match` 捧为核心：

```rust
let x = 3;

// 用 if-else 链判断
let result = if x == 1 {
    "一"
} else if x == 2 {
    "二"
} else if x == 3 {
    "三"
} else {
    "其他"
};

// 用 match 判断——等价且更清晰
let result = match x {
    1 => "一",
    2 => "二",
    3 => "三",
    _ => "其他",
};
```

`match` 与 `if` 链的差别：

**穷尽性检查**：`match` 必须覆盖所有可能值。漏掉一个分支，编译器报 `non-exhaustive patterns`。`_` 通配符兜底「其余全部」。`if` 链没有这个检查——你忘了 `else`，编译器不会拦。<span class="marginnote">穷尽性是 Rust 模式匹配的灵魂：编译器强制你把「所有情况」都摆到台面上。这在处理网络协议、解析命令行参数、处理 `Option`/`Result` 时是巨大的安全网——漏了一种状态，编译期就能发现，而不是运行期崩溃。</span>

**模式而非布尔**：`match` 的分支条件是**模式**，可以是字面量、变量、结构体字段、通配符——远比 `if` 的布尔表达式丰富。`1 => ...` 里的 `1` 就是一个字面量模式。

`match` 会在第6篇《枚举与模式匹配》与第18篇《模式进阶》被完整展开，这里只需确立它的位置：**`if` 处理「是/否」型分支，`match` 处理「值可能有好几种形态」的分支**。

## 4 公式解析：`for` 循环的迭代次数

把区间与次数写成数学式子，边界的直觉立刻清晰：

$$
\underbrace{0..n}_{n \text{ 个元素}} = \{0, 1, 2, \ldots, n-1\}
$$

拆解：

- **第一步，半开区间 `0..n` 有 $n$ 个元素**：从 0 数到 $n-1$，恰好 $n$ 个。这是「从 0 开始的循环跑 n 次」的标准写法：`for i in 0..n`。
- **第二步，次数公式**：区间 `a..b` 的元素个数恒为 $b - a$。`0..10` 是 10 次，`1..4` 是 3 次——与集合论中「区间长度 = 上界 − 下界」同构。
- **第三步，闭区间 `a..=b` 有 $b-a+1$ 个元素**：含两端，比半开多一个。`1..=100` 是 100 个数，等价于 `1..101`。

记住这张等价表，就掌握了 Rust 里所有「循环次数」问题的答案：

| 写法 | 集合 | 次数 |
| --- | --- | --- |
| `0..n` | $\{0,\ldots,n-1\}$ | $n$ |
| `a..b` | $\{a,\ldots,b-1\}$ | $b-a$ |
| `a..=b` | $\{a,\ldots,b\}$ | $b-a+1$ |
| `a..`（无上界） | $\{a, a+1, \ldots\}$