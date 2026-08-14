---
title: 不安全代码与 FFI
date: 2026-08-07
---

# 不安全代码与 FFI

<div class="epigraph">
<p>安全不是让你永不进入雷区，而是让你在进入时，手里有一张标出雷区的地图。</p>
<footer>—— 对 Rust `unsafe` 语义的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第19章 ｜ 2026-08-07</p>
</div>

## 为什么从 unsafe 与 FFI 开始

Rust 的内存安全来自编译器，但世界上有两类现实让「纯安全代码」不够用：

1. **需要绕过安全检查**：裸指针操作、调用系统接口、实现 `Send`/`Sync`——这些无法被借用检查器静态证明。
2. **需要调用其他语言**：C 库（操作系统 API、SQLite、OpenSSL）数量庞大，Rust 不可能重写一切。

Rust 的回答是 **`unsafe`**：一个显式的逃生舱。**`unsafe` 不是「关闭安全检查」**——它只是允许你做五类「安全代码做不了」的操作，而安全代码的保证仍然有效。这一章讲清楚 `unsafe` 的五种能力、`unsafe` 与安全代码的正确边界，以及如何用 **FFI**（Foreign Function Interface）调用 C 函数。

## 1 unsafe 的五种超能力

`unsafe` 关键字开启一个块或函数，允许五种操作：

### 解引用裸指针

裸指针是 `*const T`（不可变）与 `*mut T`（可变），不遵循借用规则：

```rust
let mut num = 5;

let r1 = &num as *const i32;   // 不可变裸指针
let r2 = &mut num as *mut i32; // 可变裸指针
```

创建裸指针是安全的，**解引用裸指针**必须在 `unsafe` 里：

```rust
unsafe {
    println!("r1 指向：{}", *r1);   // 解引用
}
```

裸指针可能悬空、可能为空、可能未对齐——解引用它们是未定义行为（UB）的来源。Rust 要求你**显式**承认这一点。

### 调用不安全函数

`unsafe fn` 是「调用它的人必须保证安全条件成立」的函数：

```rust
unsafe fn dangerous() {}

unsafe {
    dangerous();
}
```

`dangerous` 的定义声明了「调用者需自行保证安全」。标准库里就有：`slice::from_raw_parts`、`std::mem::transmute` 等。

### 访问/修改可变静态变量

静态变量（`static`）是全局的，可变静态变量（`static mut`）跨线程访问不安全：

```rust
static mut COUNTER: u32 = 0;

fn add_to_counter(inc: u32) {
    unsafe {
        COUNTER += inc;   // 访问可变静态变量需 unsafe
    }
}
```

Rust 2018 之前曾用 `static mut` 表达全局可变状态，如今推荐用 `Mutex` 等安全容器。

### 实现不安全 trait

`Send` 与 `Sync`（第17篇）是 `unsafe trait`——实现它们等于向编译器承诺「这个类型线程安全」：

```rust
unsafe impl Send for MyType {}
```

`unsafe impl Send` 意味着你在**手动断言**：`MyType` 跨线程转移是安全的。这个承诺错了就是数据竞争——编译器信任你，也让你承担后果。

### 访问联合体字段

C 语言的**联合体（union）**字段访问是 `unsafe` 的（Rust 的 `union` 字段类型必须 `Copy`，且访问哪个字段由你负责）。

## 2 unsafe 的正确姿势：封装成安全抽象

### 核心原则：unsafe 要小、要包进安全接口

Rust 官方与社区的铁律：**`unsafe` 代码应该尽可能小，并且被安全的函数封装**——`unsafe` 内部的错误被隔离，对外只暴露安全接口。看一个标准例子——把裸指针封装成「安全的分片访问」：

```rust
use std::slice;

fn split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();

    assert!(mid <= len);   // 前置条件：mid 不能越界

    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}
```

`split_at_mut` 把切片从 `mid` 处切成两半。直接用安全代码做不到——借用检查器不允许同时把 `values` 可变借用给两个切片。但用裸指针 + `from_raw_parts_mut` 可以，前提是 `mid <= len` 保证了两半不会重叠。<span class="marginnote">关键在 `assert!(mid <= len)`：它把「指针操作安全」的前提变成运行时检查，然后 `unsafe` 只负责「相信这个前提成立」。这就是「把 unsafe 关进安全接口」——调用者永远接触不到 `unsafe`，他们只看到两个不重叠的可变切片。</span>

**设计原则**：`unsafe` 块里假设了什么（如「指针有效」「不重叠」），就用安全代码（`assert`、前置检查）把它变成可验证的条件。

### 什么情况不该用 unsafe

能用安全代码解决的问题不用 unsafe。很多新手把「借用检查器报错」当成「需要用 unsafe」的信号——这是误解。借用检查器报错通常是「设计要改」，而不是「上 unsafe 绕过」。unsafe 是最后的工具，不是第一个。

## 3 FFI：与 C 语言互操作

### 声明外部函数：extern 块

**FFI** 让 Rust 调用 C 函数。用 `extern "C"` 块声明外部函数：

```rust
extern "C" {
    fn abs(input: i32) -> i32;   // 来自 C 标准库
}

fn main() {
    unsafe {
        println!("绝对值 -3 是 {}", abs(-3));
    }
}
```

`extern "C"` 块声明了「存在一个外部 `abs` 函数，遵循 C 调用约定」。调用外部函数是 unsafe 的——Rust 不知道 C 函数会做什么，必须由调用者保证安全（比如传入合法参数）。

`"C"` 是**调用约定（ABI）**：约定参数如何传、返回值如何回。Rust 也有自己的 ABI（`extern "Rust"`，默认），C ABI 是与 C 库互操作的标准。

### 与 C 字符串交互：`CString` 与 `CStr`

C 字符串是 `\0` 结尾的字节序列，与 Rust 的 `String`（UTF-8、带长度）不同。跨 FFI 时需转换：

```rust
use std::ffi::CString;

let c_string = CString::new("hello").expect("不能包含内部 NUL");
// c_string 传给 C 函数，用 .as_ptr() 得到 *const c_char
```

`CString::new` 拒绝包含 `\0` 的输入（C 字符串以此为结尾），失败返回 `Err`。`CString` 拥有数据，`.as_ptr()` 取裸指针传给 C。

### 调用 C 库的完整示例

以调用 C 的 `printf` 为例：

```rust
use std::ffi::CString;

extern "C" {
    fn printf(format: *const i8, ...) -> i32;   // 可变参数函数
}

fn main() {
    let msg = CString::new("来自 Rust 的问候\n").unwrap();
    unsafe {
        printf(msg.as_ptr());
    }
}
```

`printf` 的签名带 `...`（C 变参）。`msg.as_ptr()` 把 `CString` 转成 `*const i8` 传给 C。<span class="marginnote">`CString` 保证传出去的指针在函数调用期间有效（`msg` 还活着）；若 C 函数把指针存起来留作后用，就会产生悬空——FFI 的安全性由调用者负责，这正是 `extern` 调用必须在 `unsafe` 里的原因。</span>

### 从 Rust 导出函数给 C：#[no_mangle] 与 extern

反向的 FFI——让 C 调用 Rust：

```rust
#[no_mangle]
pub extern "C" fn call_from_c(x: i32) -> i32 {
    x * 2
}
```

`#[no_mangle]` 让编译器不改名（Rust 默认会给符号加哈希后缀），C 才能按 `call_from_c` 找到它；`extern "C"` 指定 C 调用约定。编译成静态库（`crate-type = ["staticlib"]`）后就能被 C 链接。

## 4 公式解析：安全保证的范围

unsafe 的边界可以形式化地画出来。设程序的所有操作为集合 $\mathcal{O}$，借用检查器能证明安全的是子集 $\mathcal{S}$：

$$
\mathcal{S} \subseteq \mathcal{O}, \qquad \mathcal{O} \setminus \mathcal{S} = \{\text{裸指针解引用, 不安全函数, 可变静态, unsafe trait, union}\}
$$

拆解：

- **第一步，安全子集 $\mathcal{S}$**：借用检查器证明安全的所有操作——引用、所有权移动、`match` 等。编译器保证这些操作无内存错误。
- **第二步，unsafe 补集**：五类操作不在 $\mathcal{S}$