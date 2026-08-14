---
title: 枚举与模式匹配
date: 2026-08-07
---

# 枚举与模式匹配

<div class="epigraph">
<p>把「可能没有值」这一事实，变成类型系统里一个诚实的公民，而不是运行期的灾难。</p>
<footer>—— 对 Rust `Option` 设计哲学的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从枚举与模式匹配开始

上一课的结构体解决「同时装下多个字段」，这一课的**枚举（enum）**解决另一个正交问题：**一个值在若干种形态中择一**。IP 地址要么是 IPv4 要么是 IPv6；一条消息可能是文本、图片或退出信号；一个函数可能成功返回数字，也可能失败——这些都是「择一」的模型。

Rust 的枚举比 C 的枚举强大得多：C 枚举只是整数常量，Rust 枚举的每个变体可以携带任意数据。而**模式匹配（match）**是消费枚举的标准方式——它把「这个值到底是哪种形态」拆开，并对每一种情况显式处理。这一章同时是 Rust 处理「可空性」的核心：`Option<T>` 枚举让「没有值」成为类型系统的一部分，消灭了空指针异常这一整类错误。

## 1 定义枚举与携带数据

### 基本枚举

```rust
enum IpAddrKind {
    V4,
    V6,
}

let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
```

`enum` 定义枚举，变体用 `::` 访问：`IpAddrKind::V4`。值只能是枚举的某个变体——不存在「既不是 V4 也不是 V6」的 `IpAddrKind`。

### 变体携带数据

Rust 枚举的威力在于：每个变体可以携带**不同种类**的数据：

```rust
enum IpAddr {
    V4(String),
    V6(String),
}

enum Message {
    Quit,                       // 无数据
    Move { x: i32, y: i32 },    // 命名结构体字段
    Write(String),              // 单个字符串
    ChangeColor(i32, i32, i32), // 三个整数
}

let home = IpAddr::V4(String::from("127.0.0.1"));
let msg = Message::Move { x: 1, y: 2 };
```

`Message` 的四个变体形态各异：`Quit` 没有数据、`Move` 带两个命名字段、`Write` 带一个 `String`、`ChangeColor` 带三个 `i32`。它们都是 `Message` 类型的合法值。

这与结构体形成对照：**结构体是「所有字段同时存在」，枚举是「多选一的变体各自携带自己的数据」**。Rust 用枚举替代了许多语言里「继承 + 空接口」才能表达的「异构集合」——`Vec<Message>` 可以装下四种消息，类型系统知道它们都是 `Message`。

### 枚举的方法

枚举也能有 `impl` 块与 `self` 方法，与结构体完全一致：

```rust
impl Message {
    fn call(&self) {
        // 根据 self 是哪种变体做不同处理
    }
}
```

## 2 Option：把「没有值」写进类型

### 标准库的 `Option<T>`

Rust **没有空值（null）**。在 C/Java 里，指针可以是 `NULL`/`null`，访问空指针就是崩溃——空指针是亿万 bug 的源头。Rust 用标准库枚举 `Option<T>` 表达「可能有值，也可能没有」：

```rust
enum Option<T> {
    None,      // 没有值
    Some(T),   // 有一个 T 类型的值
}
```

`T` 是泛型参数，可以代指任意类型：

```rust
let some_number = Some(5);          // Option<i32>
let some_string = Some("hello");    // Option<&str>
let absent: Option<i32> = None;     // 必须标注类型，编译器无法推断
```

**`Option<T>` 与普通值 `T` 是不同类型**。`i32` 可以直接参与加法，`Option<i32>` 不行——你必须先把 `Some(5)` 里的 `5` 取出来。这一强制分离正是消灭空指针的原理：**编译器禁止你把「可能有值也可能没有」的东西当普通值用**。<span class="marginnote">这就是著名的「空指针问题被移到类型系统里」：Java 的 `String` 可以为 null，使用时可能抛 `NullPointerException`；Rust 的 `Option<String>` 明确标记「可能没有」，编译器逼你在使用前处理 `None` 的情况。空值不再是隐形的，而是显式的类型。</span>

### 与 null 的对比

| 维度 | 其他语言 `null` | Rust `Option<T>` |
| --- | --- | --- |
| 是否在类型里可见 | 否（隐式） | 是（`Option<T>` 区别于 `T`） |
| 使用前是否强制处理 | 否（可忘） | 是（编译期要求穷尽） |
| 常见的灾难 | 空指针异常 | 不存在（类型检查已拦住） |
| 语义是否显式 | 否 | `Some`/`None` 一目了然 |

`Option<T>` 不引入新语法，它就是普通枚举 + 模式匹配的产物。标准库提供了大量便捷方法（`unwrap`、`expect`、`map`、`unwrap_or`），但核心心智只有一条：**`None` 必须被显式面对**。

## 3 match：穷尽所有可能

### 用 match 消费 Option

处理 `Option<T>` 的标准姿势是 `match`：

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

let five = Some(5);
let six = plus_one(five);   // Some(6)
let none = plus_one(None);  // None
```

`match` 按分支逐个尝试：`x` 是 `None` 走 `None => None`，是 `Some(i)` 则把内部值绑定到 `i`，算完包回 `Some(i+1)`。<span class="marginnote">模式里的 `Some(i)` 是个「解构模式」：它不仅匹配 `Some` 变体，还把内部数据绑定到新变量 `i`。这是模式匹配的核心能力——匹配的同时取出数据。</span>

**穷尽性（exhaustiveness）是 match 的宪法**：分支必须覆盖所有变体。漏掉 `None` 分支，编译器报 `non-exhaustive patterns: `None` not covered`。这条规则保证：`Option` 的「没有值」情况永远不会被悄悄忽略。

### 绑定与通配符

`match` 模式可以是字面量、通配符、绑定变量、解构：

```rust
let dice = 9;
match dice {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    other => move_player(other),   // 绑定其余所有值
}
```

`other` 绑定了「不是 3 也不是 7」的所有值。若不在乎具体值，用 `_`：

```rust
let coin = Coin::Penny;
match coin {
    Coin::Penny => 1,
    Coin::Nickel => 5,
    _ => 0,   // 其余硬币统一处理
}
```

`_` 通配符「匹配任意值但不绑定」。它在「其余情况统一处理」时最常用。

### if let：只关心一种情况的甜语法

当只关心 `match` 的一个分支时，`match` 显得冗长：

```rust
let config_max = Some(3u8);
if let Some(max) = config_max {
    println!("最大值是 {max}");
}
```

`if let` 是「只匹配一个模式，其余忽略」的简写，等价于 `match config_max { Some(max) => {...}, _ => () }`。它比完整 `match` 少一层缩进，也更贴合「我就想看这一种情况」的意图。<span class="marginnote">`if let` 可以带 `else`：`if let Some(x) = v { ... } else { ... }`，等价于 `match` 的 `_` 分支。判断用哪种：需要处理全部分支用 `match`，只关心一种用 `if let`。</span>

## 4 公式解析：Option 的值域

把 `Option<T>` 写成集合，它的「额外一格」立刻清晰：

$$
\text{Option}(T) = \{\text{None}\} \cup \{\text{Some}(t) \mid t \in T\}
$$

拆解三步：

- **第一步，值域多一**：`Option<T>` 包含普通类型 `T` 的全部值（包在 `Some` 里），外加一个 `None`。所以它的可能值数量是 $|T| + 1$