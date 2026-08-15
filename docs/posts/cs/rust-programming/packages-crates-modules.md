---
title: 包、Crate 与模块系统
date: 2026-08-07
---

# 包、Crate 与模块系统

<div class="epigraph">
<p>简单性是可靠性的前提。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从模块系统开始

前面的八篇，我们都在讲**一句话以内的 Rust**：一个变量、一个函数、一个 `match` 分支。但真实程序动辄几万行——把全部代码塞进 `main.rs` 一潭死水，没人读得懂，也没人改得动。Rust 给的答案是**三层组织单元**：包（package）、crate 与模块（module）。<span class="marginnote">这几乎是一切现代语言的共同难题：Java 用包与 `import`，Python 用模块与 `from … import`，C++ 用头文件与命名空间。Rust 的独特之处在于把「可见性」直接做进了编译器的隐私检查——不暴露的项，调用方连名字都取不到。</span>

这一篇讲清楚三件事：**crate 是什么、模块树怎么长、`use` 怎么把东西搬进作用域**。学会它，你才能读懂你装进 `Cargo.toml` 的每一个第三方库，也才写得出能给别人用的库。它与第三级《C 语言编程》里的「头文件 + 静态库」、以及《Java 编程》里的「包 + 类」构成同题对照——组织代码的法则，比具体的语法更长寿。

## 1 包与 crate：一次编译的最小单位

**crate** 是 Rust 编译器的一次编译输入，也是一棵**模块树**的根：它可以编译成一个**可执行文件**（binary crate），也可以编译成一个**库**（library crate），供别的程序链接使用。<span class="marginnote">「crate」就是箱子的意思——Rust 团队用它比喻「打包好的代码单元」。你之前 `cargo new` 出来的项目里那个 `src/main.rs`，其实是一个二进制 crate 的根，编译器从它开始向下展开整棵模块树。</span>

**包（package）** 则是一个更高层的概念：它由一份 `Cargo.toml` 描述，里面可能装着**一个或多个** crate。Cargo 的约定很简单：

- `src/main.rs` 存在 → 这是一个**二进制 crate**，名字与包名相同；
- `src/lib.rs` 存在 → 这是一个**库 crate**，名字也与包名相同；
- 两者都存在 → 一个包同时包含二进制与库两种 crate。

```text
my_package/
├── Cargo.toml          # 描述如何构建这个包
└── src/
    ├── main.rs         # 二进制 crate 的根
    └── lib.rs          # 库 crate 的根
```

二进制 crate 靠 `fn main()` 作为程序入口；库 crate 没有 `main`，它只导出供他人调用的项。**一个包至多有一个库 crate，但可以有任意多个二进制 crate**（把更多二进制放进 `src/bin/` 目录即可）。这一区分贯穿全章：库的职责是「提供能力」，二进制的职责是「把能力变成命令」。

## 2 模块树：作用域与隐私的地图

**模块（module）** 让我们把 crate 内部的代码分组、分层，并逐层决定哪些项对外可见。声明一个模块用 `mod` 关键字，模块可以无限嵌套，形成一棵**模块树**：

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
    mod serving {
        fn take_order() {}
    }
}
```

这棵树的根是 crate 自身。模块里的一切项（函数、结构体、常量、子模块……）**默认私有**——只有本模块及其子孙模块能看到。要给外部使用，就得用 `pub` 打开一层可见性。<span class="marginnote">默认私有的设计是 Rust 对「封装」最彻底的贯彻：C/C++ 的 `extern` 与 `public` 头文件需要程序员自觉，Java 的 `default` 包可见性也常被忽略，而 Rust 在编译期就拒绝越权访问——不写 `pub`，外面就是看不见。</span>

`take_order` 上面的例子没有 `pub`，所以 `front_of_house` 之外的代码无法调用它。可见性沿模块树**向下**传递：`pub mod hosting` 让 `hosting` 模块公开，但 `hosting` 内部那些没标 `pub` 的项依旧私有——**每一层都得显式开锁**。这与真实世界一致：大门开了，不等于房间的门也开了。

### 用路径定位项

要在代码里指到某个项，用**路径（path）**。路径分两种：

- **绝对路径**：从 crate 根开始，以 `crate` 开头；
- **相对路径**：从当前模块开始，以 `self`、`super` 或某个模块名开头。

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // 绝对路径：crate:: → front_of_house → hosting → 函数
    crate::front_of_house::hosting::add_to_waitlist();

    // 相对路径：从当前模块出发，逐级下行
    front_of_house::hosting::add_to_waitlist();
}
```

`super` 相当于文件系统里的 `..`：它指回**父模块**。当模块变深时，`super` 比写一长串绝对路径更抗重构——模块整体搬家，相对路径的语义不变。<span class="marginnote">把模块树想成文件目录树是最高效的心智模型：`crate::` 对应 `/`，`super::` 对应 `..`，`use` 对应「把某条路径的捷径贴到当前目录」。Rust 官方文档自己也是这样打比方的。</span>

## 3 use：把路径搬进作用域

路径写全名又长又啰嗦。**`use` 关键字**能把一条路径的末端**带入当前作用域**，此后直接写短名即可：

```rust
use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();   // 不再写全路径
}
```

`use` 只引入到**当前作用域**，不产生「全局可见」——不同模块各自 `use`，互不干扰。三个常用变体值得单列：

- **`as` 重命名**：`use std::io::Result as IoResult;`——解决同名冲突；
- **`pub use` 再导出**：`pub use crate::front_of_house::hosting;`——把内部路径以新位置对外公开，外部使用者不必知道内部结构，这是库作者的「门面（facade）」手法；
- **嵌套路径与通配**：`use std::{cmp::Ordering, io};` 合并多条；`use std::collections::*;` 引入该模块全部公开项（glob），应谨慎使用，因为语义可能被遮蔽。

| 语法 | 作用 | 使用场合 |
| --- | --- | --- |
| `use path;` | 把路径末端引入当前作用域 | 日常导入 |
| `use path as alias;` | 引入并重命名 | 解决同名冲突 |
| `pub use path;` | 引入并对外再导出 | 库的门面设计 |
| `use p::{a, b};` | 一次引入多条 | 同源多项 |
| `use p::*;` | 引入全部公开项 | 原型与宏场景 |

`pub use` 尤其重要：很多成熟库（如 `serde`）把内部实现藏在深层模块，再用 `pub use` 在 crate 根重新导出成 `serde::Serialize`——使用者只需要记顶层路径，库内部怎么重组都不影响下游。<span class="marginnote">这正是「面向接口而非面向实现」的 Rust 版：把内部路径与公开 API 解耦。你在第12篇《minigrep》里会看到 `pub use` 怎样把库的对外入口收拢成一行；在第17篇《Rust 的面向对象特性》里，它又是「封装」的三大支柱之一。</span>

## 4 把模块拆成独立文件

模块树不必写在一个文件里。给模块单独建文件，树形不变：

```text
src/
├── main.rs
└── front_of_house/
    ├── hosting.rs
    └── serving.rs
```

`main.rs` 里用 `mod front_of_house;` 声明，编译器会去 `src/front_of_house.rs`（或 `src/front_of_house/mod.rs`，后者是旧写法）找模块体；`front_of_house/hosting.rs` 同理。<span class="marginnote">拆文件只是把模块树<strong>分片存储</strong>，模块的嵌套层级、可见性规则、`use` 的作用域完全不变——文件是物理组织，模块树是逻辑组织。把「磁盘布局」和「逻辑结构」分开想，是读任何大型 Rust 项目的钥匙。</span>Rust 的模块文件无需 `#include` 式的头文件，也无需像 Python 那样维护 `__init__.py`——`mod` 声明就是全部接线。

## 5 公式解析：模块树的可见性判定

整章只有一条硬规则，可以写成「公式」。设项 $x$ 定义于模块树中的某个位置，问它在模块 $M$ 里能否被直接引用：

$$
\text{可见}(x, M) \iff x \text{ 定义于 } M \text{ 或 } M \text{ 的子孙模块} \lor \left( x \text{ 到 } M \text{ 路径上每一层均声明 } \text{pub} \right)
$$

拆解三步：

- **第一步，同一模块内无门槛**：$x$ 与引用者同处一个模块，天然可见——`front_of_house` 里的函数彼此随意调用，无需 `pub`。
- **第二步，跨模块要靠 pub 逐层放行**：引用者在 $M$ 之外时，从 $M$ 走向 $x$ 的**每一条边**上，目标模块或目标项都必须是 `pub`，缺一层就编译报错 `private`。
- **第三步，符号化**：把 `pub` 想象成「边的通行证」，可见性 = 是否存在一条全部开放的路径。这个「图可达性」模型能解释一切隐私报错：报错信息 `E0603` 说「项是私有的」，本质是「这条路径上有未开放的边」。

## 6 小结

- **包（package）** 由 `Cargo.toml` 描述，可包含多个 crate；`src/main.rs` 是二进制 crate 根，`src/lib.rs` 是库 crate 根。
- **crate** 是编译与链接的最小单位，其内部是一棵**模块树**；模块用 `mod` 声明，可嵌套、可拆成独立文件。
- 项**默认私有**，只有本模块及子孙模块可见；`pub` 沿路径逐层放行，可见性可用「图可达性」理解。
- **路径**分绝对（`crate::`）与相对（`self`/`super`）；`use` 把路径末端带入作用域，`as` 重命名，`pub use` 再导出。

在下一节，我们将离开「怎么组织代码」，走进「用什么装数据」——**常用集合：Vec、String 与 HashMap**，看看标准库如何把最频繁的数据结构做成类型。
