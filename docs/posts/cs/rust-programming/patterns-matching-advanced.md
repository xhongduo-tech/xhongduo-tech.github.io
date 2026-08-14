---
title: 模式与模式匹配进阶
date: 2026-08-07
---

# 模式与模式匹配进阶

<div class="epigraph">
<p>模式是把数据拆开的方式——拆得越细，越不可能犯错。</p>
<footer>—— 对 Rust 模式匹配的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第18章 ｜ 2026-08-07</p>
</div>

## 为什么从模式进阶开始

第6篇《枚举与模式匹配》介绍了 `match` 与 `if let` 的基础。这一章把**模式（pattern）**本身系统化：模式不只是 `match` 的分支条件，它是一种独立的语法——「描述一个值应该长什么样，并从中取出数据」。模式出现在 `let` 解构、函数参数、`for` 循环、`while let` 等许多位置。

理解模式的关键是分清两类位置：**不可反驳模式（irrefutable）**——必定匹配（如 `let x = 5`）；**可反驳模式（refutable）**——可能不匹配（如 `if let Some(x) = ...`）。哪些位置允许哪种模式，是这一章反复出现的主线。

## 1 模式可以出现在哪里

### let 解构

`let` 语句左侧就是模式，`let (a, b) = ...` 是最常见的解构：

```rust
let x = 5;                 // 模式：字面量绑定 x
let (a, b, c) = (1, 2, 3); // 模式：解构元组
let (x, y, _) = (1, 2, 3); // _ 忽略第三个
```

`let (a, b, c) = (1, 2, 3)` 把元组拆开，三个变量各得一个值。右侧是 `(1,2,3)`，左侧模式 `(a,b,c)` 精确描述了「元组三元素」的形状。

### match 与 if let

```rust
match value {
    Some(x) => x,
    None => 0,
}

if let Some(x) = value {
    println!("{x}");
}
```

`match` 分支的模式可以是任意可反驳模式；`if let` 只匹配一个模式。

### while let：循环直到不匹配

`while let` 反复匹配直到模式失败：

```rust
let mut stack = Vec::new();
stack.push(1);
stack.push(2);
stack.push(3);

while let Some(top) = stack.pop() {
    println!("{top}");   // 3, 2, 1
}
```

`stack.pop()` 返回 `Option`：有元素返回 `Some`，空了返回 `None`。`while let` 在有元素时循环弹出并打印，空栈时 `None` 不匹配、循环结束。这是「处理队列/栈直到清空」的惯用写法。

### for 循环与函数参数

`for` 循环的 `pattern in` 也是模式：

```rust
let v = vec!['a', 'b', 'c'];
for (index, value) in v.iter().enumerate() {
    println!("{index}: {value}");
}
```

`enumerate()` 产生 `(usize, &T)` 元组序列，`for (index, value) in ...` 解构每个元组。

函数参数同样可以是模式：

```rust
fn print_coordinates(&(x, y): &(i32, i32)) {
    println!("当前位置：({x}, {y})");
}

let point = (3, 5);
print_coordinates(&point);
```

`&(x, y)` 模式匹配「一个元组的引用」，同时解构出 `x`、`y`——参数解构让「取出字段」发生在签名处。

## 2 可反驳与不可反驳模式

### 两种模式的性质

**可反驳性（refutability）**是模式最重要的属性：

- **不可反驳（irrefutable）**：必然匹配。`let x = 5`、`let (a, b) = (1, 2)`——任何值都匹配。
- **可反驳（refutable）**：可能不匹配。`if let Some(x) = ...`——值可能是 `None`。

**使用规则**：接受不可反驳模式的位置（`let`、`for`、函数参数）**不允许**可反驳模式；接受可反驳模式的位置（`match` 分支、`if let`）**允许**两者。

```rust
let Some(x) = maybe_value;   // 错误：let 需要不可反驳模式，Some 可能不匹配
```

这段代码编译失败——`let` 位置要求「必定成功」，而 `Some(x)` 遇到 `None` 就失败了。正确写法是用 `if let` 处理可反驳的情况。反过来，`match` 里用不可反驳模式（如 `match x { y => ... }`）虽然合法，但没有意义——`y` 匹配一切，`match` 退化成赋值。

### 为什么编译器强制这条规则

这条规则的意义是**防止「必定失败或必定忽略」的代码**：`let Some(x) = v` 在 `v` 是 `None` 时程序直接崩溃，这不是 `let` 的语义；而 `match x { y => ... }` 永远走第一个分支，是死代码。编译器在这两种情况下都给出警告/错误，把你从「以为在分支、其实没分支」的陷阱里拉出来。<span class="marginnote">可反驳性检查是「穷尽性检查」的姊妹：穷尽性确保 `match` 覆盖所有情况（第6篇），可反驳性确保「必定匹配」的位置真的必定匹配。两者合起来，模式系统才不会让你写出「必然出错」的代码。</span>

## 3 模式语法详解

### 匹配字面量、命名变量与 `_`

```rust
let x = 1;

match x {
    1 => println!("一"),
    2 => println!("二"),
    other => println!("其他值：{other}"),   // 命名变量匹配一切
}

match x {
    1 => println!("一"),
    _ => println!("其他"),   // 通配符：不绑定
}
```

`other` 命名变量会**绑定**被匹配的值；`_` 通配符匹配但不绑定。两者都匹配「其余一切」，区别是命名变量把值存下来。

### 多重模式与范围模式

```rust
let x = 1;

match x {
    1 | 2 => println!("一或二"),    // | 或
    3..=5 => println!("三到五"),    // ..= 范围
    _ => println!("其他"),
}
```

`1 | 2` 用**或**匹配多个字面量；`3..=5` 用**范围**匹配一个区间。范围模式对数字与 `char` 都有效：`'a'..='z'` 匹配小写字母。

### 解构结构体与枚举

模式可以嵌套解构结构体、枚举、元组：

```rust
struct Point {
    x: i32,
    y: i32,
}

let p = Point { x: 0, y: 7 };

let Point { x: a, y: b } = p;   // 解构：a = 0, b = 7
let Point { x, y } = p;          // 简写：x, y 同名绑定

match p {
    Point { x: 0, y } => println!("在 y 轴，y = {y}"),
    Point { x, y: 0 } => println!("在 x 轴，x = {x}"),
    Point { x, y } => println!("其他点：({x}, {y})"),
}
```

`Point { x: 0, y }` 是「x 必须为 0，y 绑定」的混合模式——**字面量匹配 + 变量绑定可以出现在同一模式里**。这让你能表达「结构体满足某些条件时取出其他字段」。

### 忽略值的进阶语法

```rust
let (x, ..) = (1, 2, 3, 4);        // .. 忽略其余全部
let (.., y) = (1, 2, 3, 4);        // 只取最后一个

struct Point { x: i32, y: i32 }
let Point { x: _, .. } = point;    // 忽略 y 字段
```

`..` 忽略**剩余全部**字段/元素（区别于 `_` 只忽略一个）。`(x, ..)` 取第一个、`(.., y)` 取最后一个、`Point { x: _, .. }` 只关心 x。

### 匹配守卫：模式 + 条件

**匹配守卫（match guard）**是在 `match` 分支上加条件，模式匹配后再检查 `if`：

```rust
let num = Some(4);

match num {
    Some(x) if x % 2 == 0 => println!("偶数 {x}"),
    Some(x) => println!("奇数 {x}"),
    None => (),
}
```

`Some(x) if x % 2 == 0` 表示「匹配 `Some` 且内部值是偶数」。守卫让「模式 + 条件」结合，比在分支体内再写 `if` 更清晰。守卫可以与 `|` 或模式组合：`Some(x) | None if x < 5` 里的条件作用于整个组合。

### @ 绑定

`@` 运算符在测试模式的同时把值绑定到变量：

```rust
let msg = Message::Move { x: 5, y: 6 };

match msg {
    Message::Move { x, y } @ (0..=9, 0..=9) => {
        println!("小范围内移动：({x}, {y})");
    }
    Message::Move { x, y } => println!("大范围移动：({x}, {y})"),
}
```

等等——`@` 绑定的标准用法是绑定到范围/字面量模式（那些模式本身不创建绑定）：

```rust
let num = 5;

match num {
    x @ 1..=10 => println!("1 到 10 之间：{x}"),   // 同时绑定 x
    x @ 11..=20 => println!("11 到 20 之间：{x}"),
    _ => (),
}
```

`x @ 1..=10` 把「落在范围内的值」绑定到 `x`——范围模式本身不产出绑定，`@` 补上这个需求。

## 4 公式解析：模式的匹配关系

把模式看作「值的形状谓词」，匹配是「值是否符合形状」的判断：

$$
\text{模式 } P \text{ 匹配值 } v \iff v \in \text{shape}(P)
$$

拆解：

- **第一步，模式定义形状集合**：字面量模式 `1` 对应集合 $\{1\}$；范围模式 `1..=5` 对应 $\{1,2,3,4,5\}$；绑定模式 `x` 对应全集（任何值）；通配 `_` 也是全集但丢弃值。
- **第二步，复合模式是集合运算**：`1 | 2` 是并集 $\{1\} \cup \{2\}$