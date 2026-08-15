---
title: 编写自动化测试
date: 2026-08-07
---

# 编写自动化测试

<div class="epigraph">
<p>先让它能跑，再让它正确，最后让它变快。</p>
<footer>—— 肯特 · 贝克（Kent Beck）</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从自动化测试开始

前九篇我们都在「写代码」，这一篇第一次谈「证明代码没写错」。手工测试靠眼睛，改一处逻辑就要把所有路径重走一遍——程序一长大，手工回归的成本就指数上涨。**自动化测试**把「检查正确性」这件事本身变成代码：写一次，每次 `cargo test` 自动重跑。<span class="marginnote">一个残酷的数字：业界公认修复 bug 的成本随发现时间呈数量级增长——编译期发现、单元测试发现、上线后用户发现，三者成本差距可达百倍。测试是把「缺陷发现时刻」尽量左移的最廉价工具。</span>

Rust 把测试**内建**进语言与工具链：`#[test]` 属性标记测试函数，`cargo test` 一条命令跑完所有测试。更重要的是，Rust 的测试哲学与它的类型系统一脉相承：**失败就是 panic，成功就是不 panic**——没有「测试框架的黑魔法」，只有你早已熟悉的机制。这一篇是第12篇《minigrep》的直接铺垫：那一章我们会边写真实命令行程序边写测试，用「测试驱动」的方式把 `search` 函数打磨正确。

## 1 测试函数：三兄弟宏

把 `#[test]` 标在一个 `fn` 前面，它就成了测试函数。跑 `cargo test`，编译器把项目以**测试模式**编译，逐个执行带 `#[test]` 的函数，报告通过/失败。

```rust
#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
```

判断「对错」靠标准库三兄弟：

- **`assert!`**：参数必须是 `bool`，为 `false` 就 panic；可以加格式化参数输出提示信息；
- **`assert_eq!`**：断言两个值**相等**，失败时把左右两个值都打印出来；
- **`assert_ne!`**：断言两个值**不等**，常用于「确认代码改变了什么」的场景。

```rust
#[test]
fn greeting_contains_name() {
    let result = greeting("Carol");
    assert!(
        result.contains("Carol"),
        "greeting 没有包含名字，结果为 `{result}`"
    );
}
```

`assert_eq!` 要求比较的值实现 `PartialEq` 与 `Debug`，因为失败时要打印两个值。<span class="marginnote">这正是 Rust「一切皆 trait」的一个实例：比较的能力来自 `PartialEq`，打印的能力来自 `Debug`。给结构体加 `#[derive(PartialEq, Debug)]`，它立刻就能进断言宏——第10篇讲过的 derive 机制在这里兑现了价值。</span>

| 宏 | 断言的命题 | 失败时的输出 |
| --- | --- | --- |
| `assert!(cond)` | `cond == true` | 无（可加自定义信息） |
| `assert_eq!(a, b)` | `a == b` | 打印 `a` 与 `b` 两个值 |
| `assert_ne!(a, b)` | `a != b` | 打印 `a` 与 `b` 两个值 |

### should_panic：测试「应该失败」的情况

有些测试要验证的恰恰是「代码会 panic」——比如越界、违反不变量。用 `#[should_panic]` 声明预期：

```rust
#[test]
#[should_panic]
fn greater_than_100_panics() {
    Guess::new(200);   // 内部会 panic!
}
```

`should_panic` 可以带 `expected = "..."` 参数，断言 panic 消息**包含**某个子串，从而精确匹配「是我们想要的那种 panic」，而不是任何 panic 都算过。

### 用 Result 写测试

测试函数也可以返回 `Result<(), E>`：成功路径返回 `Ok(())`，失败路径返回 `Err(...)`。好处是能用 `?` 运算符，把「预期会失败的中间步骤」写得干净：

```rust
#[test]
fn it_works() -> Result<(), String> {
    let result = add(2, 2)?;
    Ok(())
}
```

注意：返回 `Result` 的测试**不能**再配 `#[should_panic]`——一个函数要么以 panic 宣告失败，要么以 `Err` 宣告失败，两者只能选一。

## 2 控制测试运行：cargo test 的选项

`cargo test` 默认并发、静默地跑完所有测试。真实场景往往需要精确控制，几个高频开关：

```text
cargo test                 # 跑全部测试
cargo test 过滤词           # 只跑名字包含「过滤词」的测试
cargo test -- --nocapture  # 打印测试里的 println! 输出
cargo test -- --ignored    # 只跑标了 #[ignore] 的测试
cargo test -- --test-threads=1  # 单线程跑，避免共享资源竞争
cargo test -- --exact      # 测试名必须完全匹配，而非子串匹配
```

**`#[ignore]`** 标注「平时不跑」的测试——比如耗时很长、或依赖外部环境的测试，提交代码前 `cargo test` 保持快速，需要时显式补跑。<span class="marginnote">`-- --` 之后的内容直接传给测试执行器，前面才是 `cargo test` 自身的选项。这个「两段式」接口初看反直觉，但它是 Cargo 统一「构建选项」与「运行选项」的干净切分——与第14篇 Cargo 工作流的哲学一致。</span>

并发默认就是**每个测试一个线程**。这本是好事，但测试之间一旦共享全局状态（环境变量、临时文件），并发就会造成偶发失败——`--test-threads=1` 是排查这类「偶发测试」的第一步。

## 3 单元测试与集成测试：两层防线

Rust 把测试按「放哪儿」分成两类，职责不同。

### 单元测试：与实现同居

单元测试写在**源码文件内部**，用 `#[cfg(test)]` 标注的模块包裹：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_can_hold_smaller() {
        let larger = Rectangle { width: 8, height: 7 };
        let smaller = Rectangle { width: 5, height: 1 };
        assert!(larger.can_hold(&smaller));
    }
}
```

`#[cfg(test)]` 的意思是：**仅测试模式编译此模块**——发布构建时它整体消失，不拖累产物体积。<span class="marginnote">`cfg` 是「配置（configuration）」的缩写，编译期条件。`#[cfg(test)]` 是最常用的一个，但同样的机制还能做「平台条件编译」（`#[cfg(target_os = "linux")]`）、「特性开关」，它们都是同一套编译期分支。</span>`use super::*;` 把被测模块的私有项一并引入——**单元测试可以访问私有项**，这正是「白盒测试」的权力来源。

### 集成测试：从外部用库

集成测试放在项目根的 `tests/` 目录，**每个文件是一个独立 crate**，只能通过库的公开 API 调用它——相当于「外部使用者」的视角，验证的是公开接口的稳定性：

```text
project/
├── src/lib.rs
└── tests/
    └── integration_test.rs
```

```rust
use adder;   // 通过库 crate 的名字引用

#[test]
fn it_adds_two() {
    assert_eq!(adder::add_two(3), 5);
}
```

集成测试文件之间互相独立，公共辅助代码可以放 `tests/common/mod.rs`——放在 `mod.rs` 里而不是直接放 `common.rs`，Cargo 才不会把它当成独立测试文件执行。<span class="marginnote">`cargo test` 会按「库单元测试 → 集成测试文件 → 文档测试」的顺序分段跑。第14篇我们会看到：文档注释里的代码示例其实也是测试——`cargo test` 会编译并执行它们，形成「文档永不撒谎」的保证。</span>

## 4 公式解析：测试通过性的判定

整章的语义可以压缩成一条「全称量词」公式。设某次 `cargo test` 收集到的测试集合为 $T$：

$$
\text{全部通过} \iff \forall\, t \in T:\ \neg\, \text{panic}(t)
$$

加上 `#[should_panic]` 与 `expected` 后，判定变为「预期的失败才叫通过」：

$$
\text{should\_panic}(t) \text{ 通过} \iff \text{panic}(t) \land \text{msg}(t) \supseteq \text{expected}
$$

拆解三步：

- **第一步，失败唯一化**：任何测试函数，只要 panic 就判失败；不 panic 就判通过——`assert!` 的 `false`、越界、`unwrap` 的 `None`，最终都以 panic 收场，所以「失败」在 Rust 里只有一个词。
- **第二步，加否定**：`should_panic` 把「panic = 失败」翻转成「不 panic = 失败」，所以是 $\neg$ 与 $\forall$ 的组合——你要的是「确实发生了错误」，而不是「没发生」。
- **第三步，收窄匹配**：`expected` 子串条件把「发生了 panic」进一步收窄为「发生了**指定** panic」，保证测试在「错误的错误」上照样报失败。

## 5 小结

- 用 `#[test]` 标记测试函数，`cargo test` 执行；**失败 = panic，成功 = 不 panic**。
- 断言三兄弟：`assert!`（布尔）、`assert_eq!` / `assert_ne!`（比较值）；`#[should_panic(expected = "...")]` 验证「应该失败」的路径。
- 测试可返回 `Result<(), E>` 以使用 `?`；返回 `Result` 的测试不能再配 `should_panic`。
- 运行控制：`-- --nocapture`、`-- --ignored`、`--test-threads=1`、名字过滤与 `-- --exact`。
- **单元测试**藏在 `#[cfg(test)]` 模块里、能访问私有项；**集成测试**放在 `tests/` 目录、只走公开 API。

在下一节，我们将把前十一篇的一切熔于一炉——**命令行程序实战：minigrep**，用「测试驱动」的方式编写一个真实的文件搜索工具。
