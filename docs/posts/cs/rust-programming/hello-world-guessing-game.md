---
title: Hello, World 与猜数游戏
date: 2026-08-07
---

# Hello, World 与猜数游戏

<div class="epigraph">
<p>一切有生命的、或能运载生命的事物，都具有一种确定的、可被描述的、可预测的行为。</p>
<footer>—— 杰弗里 · 乔叟（Geoffrey Chaucer）《坎特伯雷故事集》，Rust 官方教程借此说明「程序行为可预测」</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第1-2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hello, World 与猜数游戏开始

Rust 是一门口号极其鲜明的语言：**内存安全、无数据竞争、零成本抽象**。但任何语言的第一课都该是一小段能跑起来的代码——先建立「写代码→编译→运行」的体感，再谈所有权与借用。Rust 官方教程刻意把**猜数游戏**放在第二章，因为它一趟走完了整门语言的骨架：变量绑定、类型推断、`match` 模式匹配、`Result` 错误处理、`loop` 循环、标准库输入输出、以及用 Cargo 引入第三方 crate。<span class="marginnote">这一课也是我们在本专题的主线入口：学完 Rust，你将能用同一套内存安全的心智模型去看 C 语言的裸指针、Java 的垃圾回收与 Python 的引用计数——它们在第三级《C 语言编程》《Java 编程》中各有专篇。</span>

这一章我们会先写最朴素的 `Hello, world`，然后把它升级为「程序与用户对话」的猜数游戏。两段程序都不长，但几乎每个语法点都会在后续章节被反复深化。

## 1 安装 Rust 与第一个程序

### 用 rustup 安装工具链

Rust 的官方安装工具是 **rustup**，它会同时装好三个组件：`rustc`（编译器）、`cargo`（构建与包管理）、`rustup`（工具链管理）。安装完成后，第一件事是用 `rustc --version` 验证编译器就位。

传统上，一门语言的第一段程序长这样：

```rust
fn main() {
    println!("Hello, world!");
}
```

这里已经藏着 Rust 的三个第一印象：

**`fn main()`**：程序的入口函数。Rust 程序的执行从 `main` 函数开始，它不接收参数、不返回 `Result` 时返回单元类型 `()`。<span class="marginnote">单元类型 `()` 是「什么都没有」的类型，等价于 C 的 `void`。Rust 里一切函数都必须有返回类型，`()` 就是那个「不返回值」的返回值。</span>

**`println!` 带感叹号**：说明它不是函数，而是**宏（macro）**。宏可以接受可变数量的参数，并能在编译期做文本层面的展开，这是 Rust 中「没有固定参数个数」的打印能力来源。

**语句以分号结尾**：Rust 是表达式语言，绝大多数行都是带分号的语句；去掉分号则变成表达式——这一区分是后面理解「函数的最后一个表达式就是返回值」的伏笔。

### 编译与运行的完整链条

用 `rustc main.rs` 编译会得到可执行文件，这与 C 的 `gcc main.c` 心智完全一致：源码 → 编译 → 二进制。但对真实项目，我们不用 `rustc` 直接编译，而是用 **Cargo**。

| 命令 | 作用 | 对应心智 |
| --- | --- | --- |
| `cargo new hello` | 新建一个名为 `hello` 的二进制项目 | `mkdir` + 模板脚手架 |
| `cargo build` | 编译并生成可执行文件（调试模式） | `make` / `gcc` |
| `cargo run` | 编译并立即运行 | `make run` |
| `cargo check` | 只做类型检查，不生成二进制 | 快速体检 |
| `cargo build --release` | 以优化模式编译（零成本抽象在此兑现） | `-O2` 级别的优化 |

`cargo new` 生成的目录结构有两个关键文件：**`Cargo.toml`** 是项目的清单，声明包名、版本与依赖；**`src/main.rs`** 是源码入口。<span class="marginnote">把源码放在 `src/` 而清单放在根目录，是为了让 Cargo 的约定大于配置：它规定 `src/main.rs` 是二进制入口、`src/lib.rs` 是库入口，你不需要告诉它「入口在哪」，遵循目录即可。</span>

## 2 猜数游戏：程序如何与人对话

### 读取用户输入

Rust 的标准库不把输入输出隐式注入全局，一切都要显式 `use`。读取一行键盘输入的最简形式是：

```rust
use std::io;

fn main() {
    println!("猜一个数字！");

    let mut guess = String::new();
    io::stdin()
        .read_line(&mut guess)
        .expect("读取输入失败");

    println!("你猜的数是：{guess}");
}
```

这里有三个核心概念第一次登场：

**`let mut guess = String::new();`**：`let` 声明一个**不可变绑定（immutable binding）**。默认不可变是 Rust 的设计宣言——可变性是需要显式声明的特例，所以用 `mut` 关键字申请可变。`String::new()` 在堆上创建一个空字符串。<span class="marginnote">`String` 与字符串字面量 `&str` 是 Rust 里最常见的「双字符串」区分：前者是可增长的堆字符串，后者是借用的一段不可变文本。这个区分会在《常用集合》一章系统讲透。</span>

**`io::stdin().read_line(&mut guess)`**：`read_line` 把读到的内容追加进 `guess`，因此需要传入**可变引用** `&mut guess`——`&` 是借用符号。借用与所有权是 Rust 最核心的模型，本专题第4、5篇会彻底拆解，这里先形成直觉：`&mut` 表示「允许被修改的临时访问权」。

**`.expect("...")`**：`read_line` 返回一个 `Result` 枚举，可能成功也可能失败。`expect` 方法在失败时直接终止程序并打印错误信息——这是错误处理的「快速失败」风格，是第9篇《错误处理》的前菜。

### 用 match 处理结果

上面用 `.expect()` 快速处理了错误，但猜数游戏里我们要显式地看到 `Result` 的真实面目，因为它正是 Rust 没有异常机制、改用返回值传播错误的根基：

```rust
let guess: u32 = match guess.trim().parse() {
    Ok(num) => num,
    Err(_) => {
        println!("请输入数字！");
        continue;
    }
};
```

**`match` 是表达式**：它把 `guess.trim().parse()` 这个 `Result<u32, ParseIntError>` 的两种可能分支展开——`Ok(num)` 时取出数字，`Err(_)` 时打印提示并 `continue` 进入下一轮循环。<span class="marginnote">注意 `_` 通配符：它匹配「任何 Err 里的具体值」，我们不在乎错误长什么样，只在乎它发生了。通配符在后续《模式匹配》两篇里会成为常客。</span>

**变量遮蔽（shadowing）**：第二行 `let guess: u32` 与第一行 `let guess = String::new()` 同名。Rust 允许新绑定**遮蔽（shadow）**旧绑定，于是字符串类型的 `guess` 被数字类型的 `guess` 取代，而无需另起名字。这不仅省事，还暗示了「变量不可变但可被新声明替换」的哲学——同一名字在不同作用域下指向不同值。

**`.trim()`**：去掉首尾空白。用户输入一行后自带换行符 `\n`，`parse()` 不认识它，所以先 `trim` 再解析。

### 循环与随机数

完整的猜数游戏还需要「猜错继续猜」与「随机目标数」两件事：

```rust
use rand::Rng;
use std::cmp::Ordering;

fn main() {
    let secret = rand::thread_rng().gen_range(1..=100);

    loop {
        // ...读取与解析输入（如上）...
        match guess.cmp(&secret) {
            Ordering::Less => println!("太小了！"),
            Ordering::Greater => println!("太大了！"),
            Ordering::Equal => {
                println!("你赢了！");
                break;
            }
        }
    }
}
```

**`loop` 无限循环**：Rust 有三种循环——`loop`（无条件循环）、`while`（条件循环）、`for`（遍历）。`loop` 的 `break` 可以携带一个值作为循环的结果，这是它比 C 的 `while(1)` 更「表达式化」的地方。

**`rand::thread_rng().gen_range(1..=100)`**：`rand` 不是标准库，而是 **crate**（第三方库）。`gen_range(1..=100)` 用**区间语法** `1..=100`（闭区间，含两端）生成目标数。第一次使用第三方库，就要在 `Cargo.toml` 里声明依赖：

```toml
[dependencies]
rand = "0.8.5"
```

**`guess.cmp(&secret)`**：`cmp` 比较两个数并返回 `Ordering` 枚举（`Less`/`Greater`/`Equal`），`match` 再根据它走三个分支。这里 `secret` 以 `&secret` 借用传入，因为 `cmp` 不需要拿走所有权，只需要读它。

## 3 核心对比：猜数游戏这一课到底教了什么

猜数游戏被称为 Rust 的「迷你全景」，是因为它用最少代码覆盖了后续所有章节的引子。把每句话对应的未来主题列成表：

| 代码片段 | 概念 | 后续专篇 |
| --- | --- | --- |
| `let mut guess = String::new()` | 变量绑定、可变性、堆字符串 | 第3章《变量、数据类型与函数》、第8章《常用集合》 |
| `&mut guess` | 可变借用 | 第5篇《引用与借用》 |
| `.expect(...)` 与 `match ... Ok/Err` | `Result` 错误处理 | 第9篇《错误处理》 |
| `let guess: u32 = ...`（同名覆盖） | 变量遮蔽 | 第3篇 |
| `match guess.cmp(&secret)` | `match` 穷尽匹配、`Ordering` 枚举 | 第6篇《枚举与模式匹配》、第18篇《模式进阶》 |
| `loop ... break` | 循环与流程控制 | 第3篇《控制流》 |
| `use rand::Rng` + `[dependencies]` | crate 生态与依赖管理 | 第14篇《Cargo 工作流》 |

## 4 公式解析：`gen_range(1..=100)` 的边界语义

随机数生成是猜数游戏里最容易「差一」的地方，值得像解一道数学题一样拆开：

$$
\text{target} \in \{ n \in \mathbb{Z} \mid 1 \le n \le 100 \}
$$