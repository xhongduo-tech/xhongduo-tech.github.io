---
title: 错误处理：panic 与 Result
date: 2026-08-07
---

# 错误处理：panic 与 Result

<div class="epigraph">
<p>错误不是程序的例外，而是程序的常态——好的语言把「可能失败」写进类型。</p>
<footer>—— 对 Rust 错误处理哲学的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从错误处理开始

程序总会失败：文件不存在、网络断开、用户输入了非法数字。**如何处理失败**是语言设计的分水岭。Java/Python 用**异常（exception）**，错误从函数抛出、沿调用栈向上传播；C 用返回码，函数返回 `-1` 表示失败，但没人强制你检查。Rust 选择第三条路：把失败分成**两类**，用**类型系统**表达，编译器强制你面对。

- **可恢复错误（recoverable）**：比如「文件不存在」，程序可以提示用户换一个文件继续跑——用 `Result<T, E>`。
- **不可恢复错误（unrecoverable）**：比如「数组越界」，程序已不可能安全继续——用 `panic!`。

这一章先把这两种机制讲清楚，再重点拆解 `Result` 的组合器与 `?` 运算符——那是 Rust 日常编码里出现频率最高的错误处理姿势。

## 1 panic：不可恢复错误的急停

### panic 的两种触发

**`panic!` 宏**手动触发急停：

```rust
panic!("crash and burn");   // 打印错误信息，展开/放弃调用栈，退出
```

很多库函数在检测到不可能恢复的状态时也会 panic——比如越界索引 `&v[100]`、`unwrap()` 碰到 `None`。<span class="marginnote">panic 默认会「展开栈（unwind）」：逐层运行析构函数清理资源，然后打印回溯。也可以在 `Cargo.toml` 里设 `panic = "abort"` 直接终止，适合嵌入式和追求体积的场景——见第22篇《不安全代码与 FFI》。</span>

### panic 与错误处理的边界

官方建议的边界很清晰：**panic 用于「程序员的 bug」或「不可能发生的状态」**——比如违反不变量、索引越界、`unwrap` 一个逻辑上必定为 `Some` 的值。而「可能因外部环境失败」的操作（文件、网络、解析用户输入）应该用 `Result`。

## 2 Result：可恢复错误的类型

### 定义与使用

`Result<T, E>` 是标准库枚举：

```rust
enum Result<T, E> {
    Ok(T),    // 成功，携带结果
    Err(E),   // 失败，携带错误
}
```

`T` 是成功值的类型，`E` 是错误值的类型。比如打开文件：

```rust
use std::fs::File;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => panic!("打开文件失败：{error:?}"),
    };
}
```

`File::open` 返回 `Result<File, io::Error>`——它**不抛异常**，而是把「成功或失败」打包成返回值。调用者用 `match` 决定如何处理：继续用文件，还是 panic、重试、返回错误。<span class="marginnote">`Result` 与 `Option` 的区别：`Option` 是「有值或没值」，`Result` 是「有值或有一个具体的错误」。错误被携带在 `Err(E)` 里，可以被打印、传播、转换——不像异常那样「飞」过调用栈。</span>

### unwrap 与 expect：快速解包

`unwrap` 是「成功就取出值，失败就 panic」的简写：

```rust
let file = File::open("hello.txt").unwrap();
let file = File::open("hello.txt").expect("hello.txt 应该存在");
```

`expect` 比 `unwrap` 多一个自定义错误信息，失败时打印它。两者都适合「开发期原型」或「逻辑上不会失败」的场合——生产代码里应谨慎使用，因为一行 `unwrap` 就把可恢复错误变成了 panic。

### 传播错误：`?` 运算符

不想就地处理错误时，把它**返回给调用者**——这是 `?` 运算符的工作：

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username_file = File::open("hello.txt")?;
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}
```

`File::open(...)?` 展开为：成功则取出 `File`，失败则 `return Err(error)`——把错误返回给 `read_username_from_file` 的调用者。函数返回类型必须是 `Result`（或 `Option`），`?` 才能工作。

`?` 的完整展开等价于：

```rust
let mut username_file = match File::open("hello.txt") {
    Ok(file) => file,
    Err(e) => return Err(e),
};
```

### 错误类型转换

`?` 不止传播错误，还负责**类型转换**：如果函数返回 `Result<T, MyError>`，而 `?` 遇到 `io::Error`，会调用 `From` trait 把 `io::Error` 转成 `MyError`。这让多层库的错误能汇聚成统一的错误类型，是 `?` 比手动 `match` + `return` 强大之处。标准库的 `main` 甚至可以返回 `Result`：

```rust
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("config.txt")?;   // 任何错误都会被转换
    Ok(())
}
```

`Box<dyn Error>` 是「任意错误类型」的 trait 对象（第15篇《智能指针》会讲），让 `main` 里所有 `?` 都能直接传播。

## 3 让错误处理更精致：自定义错误与组合器

### 何时用 panic、何时用 Result

| 场景 | 机制 | 理由 |
| --- | --- | --- |
| 越界、类型转换失败（程序员 bug） | `panic!` | 无法恢复，越早暴露越好 |
| 打开文件、网络请求（外部失败） | `Result` | 可重试、可提示用户、可传播 |
| 库函数的内部不变量被破坏 | `panic!` | 调用者违反了契约 |
| 用户输入解析 | `Result` | 输入不可控，失败是常态 |

### 自定义错误类型

复杂项目里，常定义一个枚举描述自己可能的所有错误：

```rust
#[derive(Debug)]
enum ConfigError {
    Io(io::Error),
    Parse(std::num::ParseIntError),
}
```

配合 `impl From<...> for ConfigError` 实现转换，`?` 就能自动把底层错误变成 `ConfigError` 的变体。这是把「错误类型」做进类型系统的标准姿势：**错误不再是字符串，而是有结构的枚举**。

### 组合器：map、unwrap_or、ok_or

`Result` 和 `Option` 提供大量**组合器（combinator）**，让错误处理不用手写 `match`：

```rust
let v = opt_value.unwrap_or(default);      // None 时用默认值
let v = result.unwrap_or_else(|e| fallback(e)); // Err 时用闭包计算

// Option 转 Result，附带错误说明
let r = opt.ok_or("配置缺失")?;

// 链式处理：解析失败给默认值
let port: u16 = env::var("PORT")
    .ok()               // Result<String,_> → Option<String>
    .and_then(|s| s.parse().ok())   // 解析成数字
    .unwrap_or(8080);   // 默认端口
```

`ok_or` 把 `Option` 转成 `Result`（`None` 变成 `Err(消息)`）；`unwrap_or` 给出兜底值。组合器把「可能失败的流水线」写成链式表达式，可读性优于层层嵌套的 `match`。

## 4 公式解析：`?` 运算符的类型变换

`?` 的本质是一个类型变换，可以写成类型层面的「式子」：

$$
x: \text{Result}(T, E) \quad \xrightarrow{?} \quad T \ \ \text{或} \ \ \text{return Err}(E')
$$

拆解：

- **第一步，成功路径取 $T$