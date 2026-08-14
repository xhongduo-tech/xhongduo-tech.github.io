---
title: Rust 的面向对象特性
date: 2026-08-07
---

# Rust 的面向对象特性

<div class="epigraph">
<p>对象是状态与行为的结合；而结合的方式，每个语言都有自己的答案。</p>
<footer>—— 对「Rust 算不算面向对象语言」这一争论的温和收束</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第17章 ｜ 2026-08-07</p>
</div>

## 为什么从面向对象特性开始

「Rust 是不是面向对象语言？」是社区里经典的辩论题。答案取决于你如何定义**面向对象（OOP）**。如果 OOP 指「封装 + 继承 + 多态」三件套，Rust 在**封装**与**多态**上完全支持，唯独**继承**是刻意不提供的——Rust 用 **trait** 实现「共享行为」而不引入「父子类型」的继承链。

这一章先把 Rust 的 OOP 三件套逐个检验，然后实现一个经典的状态模式例子（博客文章的状态机），最后解释 Rust 为什么选择**组合优于继承**，以及 trait 对象（`dyn Trait`）如何充当「接口」的角色。

## 1 封装：默认私有

### 什么是封装

**封装（encapsulation）**指「数据对外隐藏，只通过公开接口访问」。Rust 的实现机制是**默认私有**：结构体字段与方法的可见性默认只限于定义它的模块，用 `pub` 显式公开。

```rust
pub struct AveragedCollection {
    list: Vec<i32>,
    average: f64,     // 私有字段，外部不可直接访问
}

impl AveragedCollection {
    pub fn add(&mut self, value: i32) {
        self.list.push(value);
        self.update_average();
    }

    pub fn average(&self) -> f64 {
        self.average
    }

    fn update_average(&mut self) {
        let total: i32 = self.list.iter().sum();
        self.average = total as f64 / self.list.len() as f64;
    }
}
```

`AveragedCollection` 保证了一个**不变量（invariant）**：`average` 字段永远等于 `list` 的平均值。外部代码无法直接修改 `average`（字段私有），只能通过 `add` 添加元素——每次 `add` 都会自动重算平均值。如果 `list` 和 `average` 都公开，使用者可能改了 `list` 忘了重算平均值，破坏不变量。<span class="marginnote">封装的本质是「把不变量藏进实现里」：使用者只看到安全的接口（`add` 保证平均值同步），看不到内部的 `list` 与 `average` 如何配合。这正是《软件工程》课程「信息隐藏」原则在 Rust 的实现。</span>

**实现细节**：`update_average` 是私有方法，只有内部 `impl` 块能调用。外部代码访问不到它——它纯粹是实现细节，可以随时重构而不影响使用者。

## 2 继承：Rust 刻意不要

### 继承的问题

**继承（inheritance）**让子类获得父类的字段与方法。Java 的 `class Dog extends Animal`。继承能实现两件事：

1. **代码复用**：子类自动拥有父类实现。
2. **多态**：父类引用可以指向子类实例。

Rust 对这两件事给出了不同的方案：代码复用用 **默认 trait 实现**，多态用 **trait 对象**。Rust **不提供继承**，理由很直白：<span class="marginnote">继承有两个著名的问题：深继承链让代码难以理解与维护；「继承耦合」让父类修改波及所有子类。Go 与 Rust 的选择一致——用组合与接口代替继承，这正是「组合优于继承」（composition over inheritance）的设计原则。</span>

继承让父类与子类**紧耦合**：改父类可能弄坏子类。
继承把**非必要的共享**强加给子类：子类继承了一切，哪怕它只需要一部分。
- Rust 的类型系统（所有权、移动语义）与继承天然冲突：子类会「免费得到」父类的字段，这干扰了布局与所有权推断。

### Rust 的替代方案

| 继承能做的 | Rust 的替代 | 区别 |
| --- | --- | --- |
| 代码复用 | trait 默认实现 | 只共享行为，不共享数据 |
| 多态 | trait 对象 `dyn Trait` | 显式的动态分发 |
| 类型层级 | 无（`enum` 表达「多形态」） | 穷尽匹配替代向下转型 |

「代码复用」在 Rust 里用 trait 默认实现完成（第10篇讲过：`Summary` 提供默认 `summarize`），「多态」用 `dyn Trait` 完成。

## 3 多态：trait 对象

### 定义共同接口

**多态（polymorphism）**是「同一接口，多种实现」。Rust 的多态通过 trait 实现。经典的例子——图形面积：

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}
```

`Vec<Box<dyn Draw>>` 是一个**trait 对象数组**：每个元素是「一个实现了 `Draw` 的、类型未知的值」。`Button`、`TextField` 各实现 `Draw`，都能放进 `Screen` 的组件列表。调用 `component.draw()` 时，运行期通过虚表找到具体实现——这就是**动态分发（dynamic dispatch）**。

### 泛型 vs trait 对象

trait 对象与泛型（第10篇）是两条路线：

```rust
// 泛型：编译期单态化，每个类型一份专用代码
pub struct Screen<T: Draw> {
    pub components: Vec<T>,
}

// trait 对象：运行期虚表分发，一个容器装多种类型
pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
}
```

| 维度 | 泛型 `<T: Draw>` | trait 对象 `Box<dyn Draw>` |
| --- | --- | --- |
| 分发时机 | 编译期（单态化） | 运行期（虚表） |
| 容器元素 | 同一具体类型 | 多种具体类型 |
| 性能 | 无间接调用 | 每次调用经虚表 |
| 大小 | 编译期确定 | 需指针（`Box`/`&`） |

**泛型容器只能装一种类型**（`Vec<Button>`），**trait 对象容器能混装**（`Vec<Box<dyn Draw>>` 里 `Button` 和 `TextField` 可以并存）。代价是动态分发的一次间接调用。选择标准：需要「多种类型混装」用 trait 对象，否则用泛型。

## 4 状态模式：用 trait 对象实现状态机

The Rust Book 用一个「博客文章状态机」演示 trait 对象的经典用途。一篇博客文章有三种状态：**草稿（Draft）→ 待审（PendingReview）→ 已发布（Published）**，不同状态下 `content` 的行为不同。

```rust
pub struct Post {
    state: Option<Box<dyn State>>,   // 当前状态：trait 对象
    content: String,
}

trait State {
    fn request_review(self: Box<Self>) -> Box<dyn State>;
    fn approve(self: Box<Self>) -> Box<dyn State>;
    fn content<'a>(&self, post: &'a Post) -> &'a str {
        ""
    }
}

struct Draft {}
struct PendingReview {}
struct Published {}
```

每个状态实现 `State` trait，`request_review`/`approve` 返回**下一个状态**——这是「状态转移」的封装：

```rust
impl State for Draft {
    fn request_review(self: Box<Self>) -> Box<dyn State> {
        Box::new(PendingReview {})
    }
    fn approve(self: Box<Self>) -> Box<dyn State> {
        self   // 草稿不能直接通过，仍返回自己
    }
}

impl State for Published {
    fn request_review(self: Box<Self>) -> Box<dyn State> {
        self
    }
    fn approve(self: Box<Self>) -> Box<dyn State> {
        self
    }
    fn content<'a>(&self, post: &'a Post) -> &'a str {
        &post.content   // 只有已发布状态才返回正文
    }
}
```

`Post::content` 委托给当前状态：

```rust
impl Post {
    pub fn content(&self) -> &str {
        self.state.as_ref().unwrap().content(self)
    }
}
```

**状态模式的价值**：状态转移的合法性被类型封装在每个状态的实现里。`Draft` 的 `approve` 返回自己，意味着「草稿不能直接通过」——调用者无法绕过这个规则，因为转移必须经 `request_review`/`approve`。<span class="marginnote">`self: Box<Self>` 这个特殊签名表示「消费掉自己（Box），返回新状态」。状态转移会消耗旧状态、产出新状态——旧状态被 drop，新的 `Box<dyn State>` 替换 `state` 字段。这是「转移即消费」的所有权风格。</span>

这个模式也展示了 trait 对象的「伪装继承」能力：`Box<dyn State>` 可以装 `Draft`/`PendingReview`/`Published` 任意一个，并调用各自的实现——**多态有了，但没有继承链**。

## 5 公式解析：动态分发的一次间接调用

trait 对象为什么比泛型多一次调用？看 `component.draw()` 的底层：

$$
\text{component: } \underbrace{\text{ptr}}_{\text{数据指针}} + \underbrace{\text{vtable ptr}}_{\text{虚表指针}}
$$