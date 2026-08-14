---
title: 泛型、Trait 与生命周期
date: 2026-08-07
---

# 泛型、Trait 与生命周期

<div class="epigraph">
<p>抽象不是为了让代码少几行，而是为了让错误不可能被写出来。</p>
<footer>—— 对 Rust 抽象机制目的性的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从泛型、trait 与生命周期开始

前几章我们一直在写「具体类型」的代码：`Vec<i32>`、`Option<String>`。但真实程序想要的是「对任意类型都适用的逻辑」：找最大值不该只在 `i32` 上成立，比较相等不该只为某个结构体写。Rust 用三大抽象支柱回答这个问题：

- **泛型（generic）**：用类型参数 `T` 写「对任意类型成立」的代码。
- **Trait**：定义「一组类型必须满足的行为契约」，让泛型能约束「什么样的类型才能用这段代码」。
- **生命周期（lifetime）**：描述「引用有效的时间范围」，让借用检查器能验证「引用不会悬空」。

三者合在一起，构成了 Rust「**零成本抽象**」的完整图景——抽象在编译期被抹平，运行时不付出额外代价。

## 1 泛型：类型参数

### 泛型函数

最经典的例子是「找最大值」：

```rust
fn largest<T>(list: &[T]) -> T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}
```

`<T>` 声明类型参数 `T`：函数可作用于任意类型 `T` 的切片。但这段代码**编译不过**——`item > largest` 需要 `T` 支持 `>` 比较，不是所有类型都支持。这就是 trait 约束登场的时刻（见第 2 节）。

泛型结构体同理：

```rust
struct Point<T> {
    x: T,
    y: T,
}

let p = Point { x: 5, y: 10 };        // Point<i32>
let p = Point { x: 5.0, y: 10.0 };    // Point<f64>
```

`Point<T>` 的两个字段必须同类型。想要不同类型，用两个参数 `Point<T, U>`。

### 单态化：零成本的原因

泛型不是运行期的动态分发，而是**编译期的单态化（monomorphization）**：编译器为每个用到的具体类型生成一份专用代码。<span class="marginnote">`Option<i32>` 与 `Option<String>` 在编译后是两份不同的代码，各自的 `match`、`unwrap` 都是针对具体类型优化的。这就是「零成本抽象」的含义——泛型的运行期性能与手写具体类型完全一致，因为编译后就一样了。</span>

```rust
let integer = Some(5);              // 编译出 Option<i32> 专用代码
let float = Some(5.0);              // 编译出 Option<f64> 专用代码
```

单态化与 C++ 模板同源，与 Java 的泛型（运行期类型擦除、装箱）形成对照：Rust 泛型没有装箱开销，但代价是二进制体积可能增大（每实例化一个具体类型就多一份代码）。

## 2 Trait：行为契约

### 定义与实现

**Trait** 定义一组方法签名，任何实现它的类型都必须提供这些方法：

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {}", self.headline, self.location)
    }
}
```

`impl Summary for NewsArticle` 表示「为 `NewsArticle` 实现 `Summary`」。实现后，`NewsArticle` 就有了 `summarize` 方法。<span class="marginnote">Trait 是 Rust 的「接口」，但它比 Java 的 interface 更彻底：可以为<strong>已有类型</strong>实现已有 trait（只要两者至少一个本地），这就是「孤儿规则」——这允许你给第三方类型补上自己的行为，而不需要改那个类型的代码。</span>

**孤儿规则（orphan rule）**：实现 trait 时，`trait` 与 `类型` 必须至少有一个是本 crate 定义的。否则两个 crate 都能为同一个外部类型实现同一个外部 trait，会产生歧义——编译器用这条规则堵死。

### trait 作为泛型约束

回到找最大值的例子，用 trait 约束修复它：

```rust
fn largest<T: PartialOrd>(list: &[T]) -> T {
    // ... 现在可以比较了，因为 T 被约束为「可比较大小」
}
```

`T: PartialOrd` 读作「`T` 必须实现 `PartialOrd` trait」，意味着 `>` 比较可用。这就是**trait 约束（trait bound）**：泛型 + 约束，既能写通用代码，又能保证类型具备所需能力。

**trait 对象（trait object）**是另一条路线：`dyn Trait` 表示「一个实现了某 trait 的、类型未知的值」：

```rust
fn notify(item: &dyn Summary) {
    println!("{}", item.summarize());
}
```

`&dyn Summary` 允许传入任何实现了 `Summary` 的类型，运行时通过**动态分发**（虚表）调用方法——与泛型单态化不同，这里付出一次间接调用的代价，换来「容器里能装多种具体类型」。`Box<dyn Error>`（第9篇见过）就是 trait 对象的经典用途。

### 默认实现

trait 可以给方法提供**默认实现**，实现者可以选择覆盖或继承：

```rust
pub trait Summary {
    fn summarize(&self) -> String {
        String::from("(阅读更多...)")
    }
}
```

## 3 生命周期：引用有效的时间范围

### 悬空引用问题

泛型处理「任意类型」，生命周期处理「任意引用有效时长」。看这个函数：

```rust
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

`longest` 返回 `x` 或 `y` 中较长的那个。但编译器不知道返回的引用到底借自 `x` 还是 `y`——如果返回 `x`，`x` 必须活得比返回的引用久；如果返回 `y` 同理。编译器无法确定，于是要求显式标注：

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

`<'a>` 声明一个**生命周期参数**，`&'a str` 表示「引用 `a` 命名的这段有效时间」。`'a` 在这里意味着：`x`、`y`、返回值共享同一个生命周期，**返回的引用必须与两个参数活得一样久**——这保证了返回的引用不会悬空。<span class="marginnote">生命周期参数不是程序运行的「实际时间」，而是编译期的<strong>关系约束</strong>：它说的是「这三个引用的有效范围之间必须满足这样的包含关系」。编译器只在编译期检查这些关系，不生成任何运行时代码——生命周期是零成本的。</span>

### 生命周期省略规则

为什么之前写的函数没标生命周期？编译器有几条**省略规则（elision rules）**，能自动推断的就不要求显式：

- 每个引用参数有自己的生命周期（`&T` → `&'a T`，`&mut T` → `&'a mut T`）。
- 只有一个引用参数时，它的生命周期被赋予所有输出。
- 有多个参数但包含 `&self`/`&mut self` 时，`self` 的生命周期赋予输出。

于是 `fn first_word(s: &str) -> &str` 被自动推断为 `fn first_word<'a>(s: &'a str) -> &'a str`，无需手写。`longest` 有两个引用参数，规则用不上，才需要显式标注。

### 结构体里的生命周期

结构体可以持有引用，但要声明引用的生命周期：

```rust
struct Excerpt<'a> {
    part: &'a str,
}

let novel = String::from("从前有座山。山里有座庙。");
let first_sentence = novel.split('.').next().unwrap();
let excerpt = Excerpt { part: first_sentence };
```

`Excerpt<'a>` 的生命周期参数说明：`part` 借用的数据必须比 `Excerpt` 实例活得久。`novel` 在 `excerpt` 之后仍存在，所以编译通过。

## 4 公式解析：生命周期约束的关系语义

把生命周期写成集合包含关系，语义就精确了。记 `'a` 表示「引用有效的时段」，`'b: 'a` 表示「`'b` 覆盖 `'a`」（`'b` 至少活得跟 `'a` 一样长）：

$$
\text{函数 } \text{longest} \text{ 的约束：} \quad \text{ret}: 'a, \quad x: 'a, \quad y: 'a
$$