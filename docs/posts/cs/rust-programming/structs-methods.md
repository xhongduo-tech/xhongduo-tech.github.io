---
title: 结构体与方法
date: 2026-08-07
---

# 结构体与方法

<div class="epigraph">
<p>先把数据建模清楚，算法自然会变得清晰。</p>
<footer>—— 改写自 The Rust Book 第5章主题</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从结构体与方法开始

前面学的元组可以把几个值捆在一起，但元组的字段只能用 `tup.0`、`tup.1` 这样晦涩的序号访问。真实程序需要的是**给每个字段起名字**的数据类型：用户有用户名与邮箱，矩形有宽与高，日志有条目与时间戳。**结构体（struct）**正是「命名字段的复合类型」，是 Rust 面向数据建模的第一块积木。配合**方法（method）**——绑定在类型上的函数——结构体就具备了「数据 + 行为」的对象雏形，这也是第17篇《Rust 的面向对象特性》的起点。

## 1 定义与实例化

### 定义结构体

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
```

`struct` 关键字定义一个新类型，大括号里是命名字段：`字段名: 类型`。字段默认私有，同模块内可直接访问。

### 实例化

创建实例时给每个字段赋值，字段名不可省略：

```rust
let user1 = User {
    active: true,
    username: String::from("alice"),
    email: String::from("alice@example.com"),
    sign_in_count: 1,
};
```

访问字段用点号：`user1.email`。实例本身默认不可变，要修改字段得让实例可写：

```rust
let mut user1 = User { /* ... */ };
user1.email = String::from("new@example.com");
```

**注意一个所有权细节**：`username` 与 `email` 是 `String`，它们的堆内存归 `user1` 所有。把 `user1` 移动给别的变量，这两个字符串的所有权也跟着走——结构体的移动就是「整体搬家」。

### 字段初始化简写与结构体更新语法

变量名与字段名相同时，可以简写：

```rust
fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username,       // 等价于 username: username
        email,          // 等价于 email: email
        sign_in_count: 1,
    }
}
```

从已有实例创建新实例时，用 `..user1` 继承其余字段：

```rust
let user2 = User {
    email: String::from("bob@example.com"),
    ..user1
};
```

`user2` 的 `email` 是新的，`active`、`username`、`sign_in_count` 从 `user1` 复制/移动过来。这里有个所有权陷阱：`username` 是 `String`（非 `Copy`），它被**移动**进了 `user2`，所以 `user1` 之后不能再访问 `user1.username`（但 `active` 是 `bool`，是 `Copy`，`user1.active` 仍可用）。<span class="marginnote">`..user1` 的语义是「其余字段按所有权规则处理」：`Copy` 字段复制，非 `Copy` 字段移动。这提醒我们：结构体更新语法不是深拷贝，它遵守所有权的每一条规定。</span>

### 元组结构体与单元结构体

还有两种简化形态：

```rust
struct Color(i32, i32, i32);   // 元组结构体：字段无名字
struct UnitLike;                // 单元结构体：无字段

let black = Color(0, 0, 0);
let _unit = UnitLike;
```

**元组结构体（tuple struct）**给元组起个类型名，`Color(255,0,0)` 与 `Point(1,2)` 即使字段类型相同也是不同类型——这避免了「把颜色当坐标」的混用。**单元结构体（unit-like struct）**没有字段，常用来实现 trait（见第10篇）。

## 2 方法：给类型绑定行为

### 方法定义

**方法（method）**是定义在 `impl` 块里的函数，第一个参数是 `self`（当前实例）：

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}
```

调用方法用点语法：`rect.area()`。第一个参数 `&self` 是 `self: &Self` 的简写——**不可变借用**当前实例。方法可以根据需要选择 `self` 的形式：

| 参数形式 | 简写 | 含义 |
| --- | --- | --- |
| `self: Self` | `self` | 拿走所有权（消费实例） |
| `self: &Self` | `&self` | 不可变借用（只读） |
| `self: &mut Self` | `&mut self` | 可变借用（可修改） |

选哪种取决于方法要不要「消费」实例：`area` 只是读宽高，用 `&self`；要改宽高，用 `&mut self`；要「把矩形转换成正方形然后扔掉原矩形」，用 `self`。<span class="marginnote">方法参数 `self` 的所有权形式是 Rust 的常见考点：`self` 消费、`&self` 只读、`&mut self` 可变。选择错误时编译器会提示——比如 `&self` 方法里想修改字段，会报「不可变借用中不能可变修改」。</span>

### 自动引用与解引用

`rect.area()` 这个点语法有个贴心行为：**自动引用/解引用**。Rust 会根据方法签名自动把 `rect` 变成 `&rect` 或 `&mut rect` 再调用。写 `rect.area()` 不用手动 `(&rect).area()`，编译器替你选择正确的借用形式。这是少数「编译器帮你加 `&`」的地方，也是方法调用比自由函数更顺手的原因。

### 关联函数

`impl` 块里不带 `self` 参数的函数是**关联函数（associated function）**，是「属于类型本身」的函数，常用来构造实例：

```rust
impl Rectangle {
    fn square(size: u32) -> Self {
        Self { width: size, height: size }
    }
}

let sq = Rectangle::square(3);   // 用 :: 调用，而不是点号
```

`Rectangle::square(3)` 用**双冒号**调用——关联函数不作用于实例，作用于类型。`String::from`、`Vec::new` 都是关联函数，它们的 `Self` 指向类型本身。

## 3 多个 impl 块与调试输出

Rust 允许一个类型有**多个 `impl` 块**，编译器会把它们合并：

```rust
impl Rectangle {
    fn area(&self) -> u32 { self.width * self.height }
}

impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}
```

多个 `impl` 块并非必要，但为「按关注点组织代码」和「无法修改类型的 trait 实现」提供了可能——后者在《泛型与 trait》一章会用到。

**调试输出 `#[derive(Debug)]`**：结构体默认不能直接用 `println!("{:?}", rect)` 打印，因为编译器不知道如何格式化它。在结构体上方加 `#[derive(Debug)]`，它就能用 `{:?}`（紧凑）或 `{:#?}`（美化多行）输出。这个 derive 宏自动生成 `Debug` 实现，是调试期最常用的工具。<span class="marginnote">`#[derive(...)]` 是 Rust 的派生宏：编译器帮你自动实现某个 trait。`Debug`、`Clone`、`Copy`、`PartialEq` 是最常见的几个。到第19篇《高级 trait 与类型》我们会看到如何手写 trait，而 derive 正是「常见 trait 的样板由编译器代劳」。</span>

## 4 公式解析：`&self` 借用与所有权在方法中的流动

方法调用 `rect.area()` 底层发生了什么？把它拆成内存视角：

$$
\underbrace{\text{rect}}_{\text{栈上 } \{\text{w,h}\}} \quad \xrightarrow{\text{方法调用}} \quad \underbrace{\&\text{rect}}_{\text{借用 8 字节数据}}
$$