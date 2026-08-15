---
title: Cargo 工作流与 Crates.io
date: 2026-08-07
---

# Cargo 工作流与 Crates.io

<div class="epigraph">
<p>独行快，众行远。</p>
<footer>—— 中国谚语</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第14章 ｜ 2026-08-07</p>
</div>

## 为什么从 Cargo 工作流开始

前十三篇，Cargo 一直在背后默默工作：`cargo new` 建项目、`cargo build` 编译、`cargo test` 跑测试、`cargo run` 跑程序。这一篇把 Cargo 从「幕后」请到「台前」，讲清三件让 Rust 成为**工程语言**的能力：**构建配置**（release profile）、**文档即测试**（doc comment）与**共享生态**（Crates.io）。<span class="marginnote">语言与人一样，单靠自身走不快，要靠社区——Crates.io 至今托管着数十万个 crate，从命令行解析库 `clap` 到异步运行时 `tokio` 再到序列化 `serde`，几乎任何轮子都已有人造好。会用生态，等于站在整个社区的肩上。</span>

这一篇也把第11篇欠的「文档测试」补上：Rust 的文档注释里的代码示例会被 `cargo test` 编译执行，所以文档**永远与代码同步**——这是许多语言梦寐以求而不得的保证。它与第三级《Java 编程》里的 Javadoc、Maven 仓库形成直接对照：同样的问题，Rust 用不同的答案解决。

## 1 构建配置：dev 与 release profile

Cargo 内置两套**构建画像（profile）**：

- **dev**（默认）：`cargo build` / `cargo run` / `cargo test` 使用，优化级别低、编译快、带调试信息；
- **release**：`cargo build --release` 使用，开启优化、产物小而快，但编译时间显著变长。

| 配置项 | dev 默认 | release 默认 | 含义 |
| --- | --- | --- | --- |
| `opt-level` | 0 | 3 | 优化级别，0–3 递增 |
| `debug` | 2 | 0 | 是否携带调试信息 |
| `panic` | unwind | unwind | panic 策略 |
| `lto` | false | false | 链接时优化（跨函数内联） |
| `codegen-units` | 256 | 16 | 并行代码生成单元数 |

默认值足够绝大多数场景，但可用 `[profile.release]` 段在 `Cargo.toml` 里覆盖。比如追求极致性能的库，常开 LTO 并调高优化：

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

`codegen-units = 1` 让编译器把整个 crate 当一块来优化，往往再挤出几个百分点，代价是更长的编译时间。<span class="marginnote">「默认慢、显式快」是 Cargo 的刻意取舍：开发期你最缺的是「改动—验证」的反馈速度，release 才需要极致性能。Rust 因此既不是「总是慢」，也不是「总是牺牲可调试性」，而是把选择权交给你。</span>profile 不是二元的——你可以在 `[profile.dev]` 里微调开发构建，也可以自定义第三个 profile（如 `[profile.bench]`），机制完全一样。

## 2 文档注释：cargo doc 与文档测试

Rust 的文档注释用**三个**斜杠，写在被注释项的上方，支持 Markdown：

```rust
/// 将两数相加。
///
/// # Examples
///
/// ```
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

`cargo doc` 把全部文档注释渲染成 HTML 文档网站，`cargo doc --open` 直接在浏览器打开。三个要点让它与众不同：

- **`///` 文档项，`//!` 文档模块/crate**：`//!` 写在文件开头，描述整个 crate 或模块的用途；
- **代码示例会跑**：上面 ``` 代码块里的例子，`cargo test` 会把它编译并执行——文档示例失败，测试就失败；
- **约定俗成的章节**：`# Examples`（示例）、`# Panics`（何时 panic）、`# Errors`（何时返回 `Err`）、`# Safety`（不安全函数的前提）。<span class="marginnote">官方约定 `# Panics` 与 `# Errors` 尤其重要：调用者最想知道「什么情况下这函数会炸」，而这正是错误处理那一章反复强调的「把失败写进类型」的文档版。</span>

「文档示例即测试」的价值怎么强调都不为过。别的语言文档常因代码演进而腐化，Rust 让编译器替你盯着——示例一失真，`cargo test` 立刻报错。这也是 crates.io 上成熟库文档质量普遍偏高的制度性原因。

## 3 发布到 Crates.io：一次 cargo publish

要发布库给别人用，`[package]` 段得先把元信息补全——`name`、`version`、`description`、`license` 缺一不可，且**名字必须在整个 Crates.io 全局唯一**（像域名一样先到先得）：

```toml
[package]
name = "my_package"
version = "0.1.0"
edition = "2021"
description = "A useful helper crate"
license = "MIT"
```

`cargo publish` 会把当前 crate 上传到 Crates.io 并打包归档。几个关键纪律：

- **发布前先 `cargo publish --dry-run`**：本地模拟打包，检查元信息与文件是否齐全；
- **版本不可复用**：Crates.io 不允许「删掉 1.0.0 再传一次 1.0.0」——语义版本是永久的；
- **`cargo yank` 撤回版本**：如果发布了坏版本，`cargo yank --version 1.0.0` 把它标记为不可用——已依赖它的项目继续用，新项目不能再用它，这避免了「静默换货」破坏下游构建。

**语义化版本（SemVer）** 是这一切的基石：主版本号不兼容时递增、次版本号向后兼容地加功能时递增、修订号向后兼容地修 bug 时递增。依赖方写 `"^1.4.0"` 表达「兼容 1.4.0 的一切版本」，Cargo 据此解析出具体锁定版本。<span class="marginnote">版本号是承诺：发布 0.1.0 时你要知道，下游成千上万个 `^0.1` 依赖正等着你的 API 保持稳定。SemVer 把「改 API 的成本」从不可见变成可计算——这是大规模协作能进行的隐秘前提。</span>

## 4 Cargo workspace：把多个 crate 放一个仓库

大型项目往往拆成多个 crate：一个核心库 + 若干二进制 + 测试工具。**工作区（workspace）** 让这些 crate 共享一个 `Cargo.lock` 与一个 `target/` 构建目录：

```toml
[workspace]
members = ["add_one", "adder"]
resolver = "2"
```

工作区内的 crate 之间用**路径依赖**互相引用，不用发布到 crates.io 也能协作：

```toml
[dependencies]
add_one = { path = "../add_one" }
```

工作区的价值在一致性：所有成员锁定同一组依赖版本、共享一次编译结果，`cargo build` 在根目录一次构建全部成员。组织层面的收益更隐蔽——核心库与业务代码同仓演进，提交历史完整可追溯。另一个命令 **`cargo install`** 则从 Crates.io 拉取并安装**二进制** crate（如 `cargo install cargo-expand`），装进 `~/.cargo/bin`，让开发者工具也走「发布—安装」的公开通道。

## 5 公式解析：Caret 版本约束的语义

`Cargo.toml` 里最常见的版本写法是 caret `^`，它的语义是一条区间：

$$
v \text{ 满足 } \hat{}\, 1.4.0 \iff 1.4.0 \le v \lt 2.0.0
$$

$$
v \text{ 满足 } \hat{}\, 0.4.0 \iff 0.4.0 \le v \lt 0.5.0
$$

拆解三步：

- **第一步，向后兼容的区间**：`^1.4.0` 的意思是「任何兼容 1.4.0 的版本」——因为次版本号与修订号的增长都承诺不破坏 `1.4.0` 的 API，所以区间一直延伸到主版本 2 之前。
- **第二步，0.x 的特殊性**：主版本为 0 时，SemVer 允许任何破坏性改动，因此 caret 把区间收窄到**次版本号不变**——`^0.4.0` 只覆盖 `0.4.x`，到 `0.5.0` 就被视为可能不兼容。
- **第三步，解析为具体版本**：Cargo 在区间内选择**最新的**版本写入 `Cargo.lock`；发布依赖时若想保持高度可控，可以用 `=1.4.0` 精确锁定，或用 `1.4.0`（等价 `^1.4.0`）给出版本范围。

## 6 小结

- **profile 控制构建**：dev 默认编译快、带调试信息；release 默认深度优化；`[profile.release]` 可覆盖优化级别、LTO 与 `codegen-units`。
- **文档注释**：`///` 文档项、`//!` 文档模块；`cargo doc` 生成文档站，代码示例会被 `cargo test` 编译执行。
- **发布**：`cargo publish` 上传到 Crates.io；名字唯一、版本不可复用、`cargo yank` 可撤回；SemVer 是版本承诺的契约。
- **工作区**：多个 crate 共享 `Cargo.lock` 与 `target/`，用路径依赖互引；`cargo install` 安装二进制工具。

在下一节，我们将告别「工具与组织」，回到语言本身最有魅力的部分——**智能指针：Box、Rc 与 RefCell**，看 Rust 如何在零成本抽象下管理堆、共享与可变性。
