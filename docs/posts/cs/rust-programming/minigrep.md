---
title: 命令行程序实战：minigrep
date: 2026-08-07
---

# 命令行程序实战：minigrep

<div class="epigraph">
<p>把每一件小事都做对，大事就会自己涌现。</p>
<footer>—— The Rust Book 第12章构建 minigrep 的工程思路</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第12章 ｜ 2026-08-07</p>
</div>

## 为什么从 minigrep 开始

前几章是语言零件，这一章是第一次**把零件组装成一个完整程序**。minigrep 是一个简化版 `grep` 命令行工具：接收一个查询串与一个文件名，在文件内容里搜索匹配的行并打印。它刻意选得很小，却恰好覆盖了一个真实命令行工具的全部工程问题：

- **解析命令行参数**（`std::env::args`）
- **读取文件**（`fs::read_to_string`）与**错误处理**（`?`、`Box<dyn Error>`）
- **分离「配置」与「逻辑」**（结构体 + 构造函数 + 单元测试）
- **支持环境变量开关**（`MINIGREP_IGNORE_CASE`）
- **把错误写进标准错误**（`eprintln!`）

The Rust Book 用这一章演示「关注点分离」的渐进重构：先写一个能跑的坏版本，再一步步拆成可测试的函数。这正是软件工程里「先能跑，再重构」的真实节奏。

## 1 解析命令行参数

### 读取参数

`std::env::args()` 返回命令行参数的迭代器，第一个是程序名，后面是用户参数：

```rust
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let query = &args[1];       // 查询串
    let file_path = &args[2];   // 文件名

    println!("搜索 {query}，目标文件 {file_path}");
}
```

`cargo run -- 查询串 文件名` 运行时，`--` 之后的部分会作为程序参数传入。`env::args()` 遇到非法 Unicode 参数会 panic；需要容忍非法参数时用 `env::args_os()`（返回 `OsString`）。

### 把配置收进结构体

直接在下标里取 `args[1]`、`args[2]` 脆弱且不可测。第一步重构：把「查询串 + 文件名」装进结构体，用构造函数解析：

```rust
pub struct Config {
    pub query: String,
    pub file_path: String,
}

impl Config {
    pub fn build(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 3 {
            return Err("参数不足：需要 查询串 与 文件名");
        }
        let query = args[1].clone();
        let file_path = args[2].clone();

        Ok(Config { query, file_path })
    }
}
```

`Config::build` 返回 `Result`——参数不足是**可恢复错误**，应提示用户而不是 panic。`args[1].clone()` 复制字符串：`args` 是借用，而 `Config` 需要拥有数据。<span class="marginnote">`build`（而非 `new`）的命名是有意的：它可能失败，返回 `Result`，`new` 的惯例是「不会失败」。参数校验失败返回 `Err`，`main` 里再决定打印用法并退出——「构建失败」被当作正常流程而非崩溃。</span>

## 2 读取文件与运行逻辑分离

### 读取文件

```rust
use std::fs;

fn main() {
    // ...解析 Config...
    let contents = fs::read_to_string(&config.file_path)
        .expect("读取文件失败");
}
```

`fs::read_to_string` 读整个文件为 `String`，失败返回 `io::Error`。这里用 `expect` 是「暂时能跑」的写法，稍后换成 `?` 传播。

### run 函数：核心逻辑独立成函数

把「读文件 + 搜索 + 打印」封装成 `run` 函数，返回 `Result<(), Box<dyn Error>>`：

```rust
use std::error::Error;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(&config.file_path)?;
    let results = if config.ignore_case {
        search_case_insensitive(&config.query, &contents)
    } else {
        search(&config.query, &contents)
    };

    for line in results {
        println!("{line}");
    }

    Ok(())
}
```

`Box<dyn Error>` 让任何错误都能经 `?` 传播（第9篇讲过）。`run` 返回 `Result` 而非直接 panic，意味着主函数可以决定错误如何呈现——这层「逻辑与 UI 分离」正是可测试性的关键。<span class="marginnote">把 `run` 做成 `Result<(), Box<dyn Error>>` 的另一个好处是：`main` 里 `if let Err(e) = run(config) { ... }` 可以捕获错误、打印友好信息、再决定退出码——错误处理与业务逻辑彻底分开。</span>

### main 的最终形态

```rust
fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::build(&args).unwrap_or_else(|err| {
        eprintln!("解析参数出错：{err}");
        process::exit(1);
    });

    if let Err(e) = run(config) {
        eprintln!("程序出错：{e}");
        process::exit(1);
    }
}
```

两个工程细节值得注意：**`unwrap_or_else`** 处理 `Config::build` 的失败（打印错误到 `stderr` 并退出码 1）；**`eprintln!`** 把错误写进标准错误流而非标准输出——这样 `cargo run -- 词 文件 2>/dev/null` 能把错误与正常输出分离。

## 3 搜索逻辑与测试

### 核心搜索函数

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let mut results = Vec::new();
    for line in contents.lines() {
        if line.contains(query) {
            results.push(line);
        }
    }
    results
}
```

`search` 返回「包含查询串的所有行」——`Vec<&'a str>` 的元素借用 `contents`（生命周期 `'a` 是省略规则推断的：一个参数，输出继承它的生命周期）。`contents.lines()` 按行分割，`line.contains(query)` 检查包含。

### 单元测试驱动

搜索逻辑是纯函数，最好测：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_result() {
        let query = "duct";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.";

        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }

    #[test]
    fn case_sensitive() { /* ... */ }

    #[test]
    fn case_insensitive() { /* ... */ }
}
```

`search` 不碰文件、不碰输出，只做「查询串 × 文本 → 匹配行」的纯函数——因此可以用固定字符串做测试，无需临时文件。**把逻辑写成纯函数，是让程序可测试的第一步**：文件 I/O 留在 `run`，搜索逻辑留在 `search`。

### 忽略大小写：环境变量

支持 `MINIGREP_IGNORE_CASE` 环境变量控制是否忽略大小写：

```rust
use std::env;

impl Config {
    pub fn build(args: &[String]) -> Result<Config, &'static str> {
        // ...
        let ignore_case = env::var("MINIGREP_IGNORE_CASE").is_ok();
        Ok(Config { query, file_path, ignore_case })
    }
}

pub fn search_case_insensitive<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let query = query.to_lowercase();
    let mut results = Vec::new();
    for line in contents.lines() {
        if line.to_lowercase().contains(&query) {
            results.push(line);
        }
    }
    results
}
```

`env::var(...).is_ok()` 判断环境变量是否存在（存在即忽略大小写）。不区分大小写的实现先把查询串与每一行都转小写再比较——注意 `query.to_lowercase()` 产生新 `String`，`contains(&query)` 用 `&query` 而不是 `query`。<span class="marginnote">`to_lowercase()` 返回新的 `String` 而不是 `&str`：Unicode 大小写转换可能改变长度（比如德语 `ß`），无法原地借用。这是「为什么字符串处理常常要产生新分配」的典型例子——与第8篇的 UTF-8 话题再次相逢。</span>

## 4 公式解析：生命周期在 `search` 里的自动推断

`search` 的签名里藏着一个生命周期，编译器自动补全：

$$
\text{写} \ \ fn\ \text{search}(query: \&str,\ contents: \&'a str) \to Vec\lt \&'a str>
$$