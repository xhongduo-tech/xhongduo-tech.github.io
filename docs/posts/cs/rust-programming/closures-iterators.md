---
title: 闭包与迭代器
date: 2026-08-07
---

# 闭包与迭代器

<div class="epigraph">
<p>给序列一个管道的形状，代码就顺着管道流走了。</p>
<footer>—— 对 Rust 迭代器适配器风格的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第13章 ｜ 2026-08-07</p>
</div>

## 为什么从闭包与迭代器开始

Rust 有两副面孔：命令式（`for` 循环、`mut` 变量）与函数式（闭包、迭代器）。这一章介绍后者——**闭包（closure）**是可以捕获环境变量的匿名函数，**迭代器（iterator）**是「惰性序列」的抽象。两者结合，可以把「遍历 → 过滤 → 转换 → 求和」写成一条表达式链，替代层层嵌套的循环。

关键点在于：迭代器不是炫技。Rust 的迭代器适配器（`filter`、`map`、`fold` 等）经过单态化后，性能与手写循环**等价甚至更快**——这是「零成本抽象」在函数式风格上的又一次兑现。Rust 官方甚至建议优先用迭代器而非手写循环，除非代码可读性受损。

## 1 闭包：捕获环境的匿名函数

### 基本形态

闭包是可以简写的匿名函数，用 `|参数| 表达式` 定义：

```rust
let add_one = |x: i32| x + 1;
let add_one = |x| x + 1;        // 类型可推断时省略标注

println!("{}", add_one(5));      // 6
```

闭包与普通函数 `fn` 的最大区别：**闭包可以捕获定义它的作用域里的变量**：

```rust
fn main() {
    let x = 4;
    let equal_to_x = |z| z == x;   // 捕获了 x
    println!("{}", equal_to_x(4)); // true
}
```

`equal_to_x` 用到了外部变量 `x`——普通函数做不到这件事。闭包把环境「捕获」进来，相当于携带了一个隐藏的状态。

### 三种捕获方式

闭包怎么捕获环境，取决于闭包内部如何使用它。Rust 按需选择，也可以强制：

| 特征 | 捕获方式 | 对应 trait |
| --- | --- | --- |
| 只读使用 | `&T`（借用） | `Fn` |
| 修改使用 | `&mut T`（可变借用） | `FnMut` |
| 取走所有权 | `T`（移动） | `FnOnce` |

```rust
let s = String::from("hello");

let print_it = || println!("{s}");          // Fn：借用 s

let add_suffix = || {                       // FnMut：可变借用
    s.push_str(" world");
};

let take_it = || {                          // FnOnce：移动 s
    drop(s);
};
```

**`Fn`** 只读借用，可多次调用；**`FnMut`** 可变借用，可多次调用；**`FnOnce`** 消费捕获的值，只能调用一次。编译器的规则很直接：捕获的东西怎么用，就选对应的 trait。<span class="marginnote">`move` 关键字可以强制闭包获取所有权：`let c = move |z| z == x;`。它在「把闭包交给另一个线程」时必不可少——线程的闭包必须完全拥有其捕获的数据，不能借主线程的变量（第16篇会详述）。</span>

## 2 迭代器：惰性序列

### 什么是迭代器

**迭代器（iterator）**实现 `Iterator` trait，核心方法只有一个：`next()`，返回 `Option<Self::Item>`——有下一个元素返回 `Some(元素)`，耗尽返回 `None`：

```rust
let v1 = vec![1, 2, 3];
let mut v1_iter = v1.iter();

assert_eq!(v1_iter.next(), Some(&1));
assert_eq!(v1_iter.next(), Some(&2));
assert_eq!(v1_iter.next(), Some(&3));
assert_eq!(v1_iter.next(), None);   // 耗尽
```

`iter()` 在 `Vec` 上产生不可变引用的迭代器（`Item = &T`）；`into_iter()` 产生拥有所有权的迭代器（`Item = T`）；`iter_mut()` 产生可变引用迭代器（`Item = &mut T`）。三种对应三种所有权姿势。

### 惰性：不消费就没有代价

迭代器是**惰性（lazy）**的：创建迭代器不执行任何遍历，直到你调用 `next()` 或消费方法：

```rust
let v1 = vec![1, 2, 3];
let v1_iter = v1.iter();   // 这里什么都不发生

for val in v1_iter {       // for 循环里才逐个取
    println!("{val}");
}
```

`for` 循环本质上就是把迭代器的 `next()` 逐个调出来，直到 `None`。<span class="marginnote">惰性意味着「构造一条迭代器链」本身零成本——真正的工作在最终消费时才发生。这让你可以先描述「要什么数据」，再决定「什么时候取」，与数据库查询的惰性求值一脉相承（第三级《数据库》课程的查询优化同理）。</span>

## 3 消费适配器与迭代适配器

迭代器分两类方法：**消费适配器（consuming adaptors）**把迭代器用掉，**迭代适配器（iterator adaptors）**把迭代器变成另一个迭代器。

### 消费适配器

```rust
let v1 = vec![1, 2, 3];

let total: i32 = v1.iter().sum();       // 求和，消费整个迭代器
let count = v1.iter().count();          // 计数
let collected: Vec<i32> = v1.iter().map(|x| x + 1).collect();  // 收集成 Vec
```

`sum()`、`count()`、`collect()` 都是消费适配器——调用它们会耗尽迭代器。`collect()` 尤其强大：它能收集到任何 `FromIterator` 类型（`Vec`、`String`、`HashMap` 等），所以它的目标类型通常要显式标注。

### 迭代适配器

```rust
let v1 = vec![1, 2, 3, 4, 5];

let doubled: Vec<i32> = v1.iter()
    .map(|x| x * 2)              // 每个元素 ×2
    .collect();                  // 消费，得到 [2, 4, 6, 8, 10]

let evens: Vec<&i32> = v1.iter()
    .filter(|x| **x % 2 == 0)    // 只留偶数
    .collect();
```

`map` 对每个元素做转换，`filter` 按条件筛掉不满足的元素。注意 `filter` 的闭包参数是 `&&i32`（引用套引用）：`v1.iter()` 产生 `&i32`，`filter` 传给闭包的又是它的引用。`**x` 解两层引用才得到 `i32`。

### 方法链：一次遍历完成多步处理

迭代器的真正威力是把适配器串成**方法链**：

```rust
let text = "hello world hello rust";
let word_count = text
    .split_whitespace()                     // 按空白切词
    .map(|w| w.to_lowercase())              // 转小写
    .filter(|w| w.len() > 3)                // 只留长词
    .count();                               // 计数
```

一条链做完「切词 → 转小写 → 过滤 → 计数」，全程一次遍历，没有中间 `Vec` 分配。`split_whitespace`、`map`、`filter` 都返回新的迭代器，`count` 在链尾消费。

### 用迭代器实现 minigrep 的搜索

第12篇的 `search` 用 `for` 循环手写，这里用迭代器重写：

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    contents
        .lines()                      // 按行
        .filter(|line| line.contains(query))  // 过滤
        .collect()                    // 收集成 Vec
}
```

三行替代原来的 `for` 循环。逻辑完全一致，但更贴近「描述数据流」而非「描述循环步骤」——这是函数式风格的核心价值：**代码说「要什么」，而不是「怎么一步步做」**。

## 4 公式解析：迭代器链的零成本原理

为什么 `contents.lines().filter(...).collect()` 与手写 `for` 循环性能等价？关键在**单态化 + 内联**：

$$
\text{lines()} \ \to \ \text{filter()} \ \to \ \text{collect()}
\quad \text{内联后} \quad
\text{一次遍历，无中间分配}
$$