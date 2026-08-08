---
title: Rust：所有权、生命周期与零成本抽象
date: 2026-08-07
---

# Rust：所有权、生命周期与零成本抽象

<div class="epigraph">
<p>Rust 的承诺是：让 C 的速度、Java 的安全、和现代抽象三者兼得——只要你愿意让编译器管得足够严。</p>
<footer>—— 佚名（Rust 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ PLT 综合专题 ｜ 2026-08-07</p>
</div>

## 为什么从 Rust 开始

上一节 Python 是「动态 + 运行时」的极致；Rust 是另一极——**静态 + 编译期**的极致。Rust 把内存安全（所有权/借用）、并发安全（Send/Sync）、零成本抽象（trait + 泛型 + 单态化）全部做进**编译期**：运行时几乎无隐藏开销，同时没有悬垂、没有竞态。这一节整合前面讲过的所有权、生命周期、trait、泛型，看它们如何在 Rust 中合成一个完整系统，以及「零成本抽象」的具体含义——它是系统编程（操作系统、浏览器引擎、AI 基础设施）的新一代语言。<span class="marginnote">Rust 的口号「零成本抽象（zero-cost abstraction）」：<strong>抽象不引入运行时开销</strong>——Vec 与手写数组一样快、Box 与裸指针一样紧凑、trait 泛型与手写专用代码一样快。代价是「编译期复杂度」：编译器（借用检查器）替你做那些 C 里靠人肉保证的事。</span>

## 1 所有权 + 生命周期：内存安全编译期保证

回顾（第十五篇详讲）——Rust 的内存安全由三条编译期规则保证：

**所有权**：一值一所有者；离开作用域即 drop。
**移动语义**：let s2 = s1 转移所有权，s1 失效——无双重释放。
**借用检查**：&T（多读）/\&mut T（单写）互斥；引用不超过被引用值寿命。

``rust
fn main() {
    let s = String::from("hello");   // 所有权：s 拥有这块堆内存
    let s2 = s;                      // 移动：所有权转移，s 从此失效
    println!("{}", s2);

    let v = vec![1, 2, 3];
    for x in &v {                    // 借用：&v 只读共享，v 仍然有效
        println!("{}", x);
    }
}                                    // s2、v 离开作用域，自动 drop
``

**生命周期** 'a 显式标注「引用间寿命关系」——保证返回的引用不悬垂。编译器（借用检查器）逐函数验证这些规则，**编译通过 = 内存安全**。<span class="marginnote">Rust 的「内存安全」是<strong>无运行时开销的：没有 GC、没有引用计数的「+1/-1」开销（Rc/Arc 是显式选择的例外）。「安全」由编译期检查保证，运行时「零负担」——这是它区别于 Java（GC 停顿）与 Swift（ARC 计数开销）的根本。</strong></span>

## 2 零成本抽象：trait 与泛型

Rust 的抽象工具：

**trait**（类型类）：定义「能力契约」——Display、Clone。
**泛型 + 单态化**：`Vec<T>`、`fn max<T: Ord>`(...) 编译期为每个 T 生成专用代码。
**迭代器链**：iter().map().filter().collect()——每个组合零抽象层，编译后等同手写循环。

``rust
// trait：定义「能力契约」
trait Greet {
    fn greet(&self) -> String;
}

// 泛型 + 单态化：编译期为每个 T 生成专用代码
fn max`<T: Ord>``(a: T, b: T) -> T { if a > b { a } else { b } }

// 迭代器链：每层都是泛型方法，内联合并成手写循环
let sum: i32 = (1..=10)
    .map(|x| x * x)
    .filter(|&x| x % 2 == 0)
    .sum();
``

「零成本」的具体保证：**单态化**让泛型调用与直接调用等价；**内联**让迭代器链合并成手写循环；trait 无虚表（静态分派）或显式 dyn（动态分派）。<span class="marginnote">「零成本抽象」的关键是<strong>静态分派（单态化）：单态化后的 `max::<i32>` 是专用代码，无虚表、无间接调用——与手写 i32 版等价。Rust 把「抽象」（trait、泛型、迭代器）全部编译成最优代码。「抽象不花运行时，花编译时」——代价是编译慢、二进制大。</strong></span>

## 3 无 GC 的资源管理：RAII

**RAII（Resource Acquisition Is Initialization）**：资源（内存、文件、锁、网络连接）的获取在初始化时、释放在析构时（离开作用域）——**自动、确定、无泄漏**：

``rust
struct File { name: String }

impl Drop for File {
    fn drop(&mut self) {
        println!("closing {}", self.name);   // 离开作用域时自动调用
    }
}

fn main() {
    let f = File { name: String::from("data.txt") };
    // 离开 main 时 f 自动 drop，无需手动关闭
}
``

对比：C 要手动 close（忘关即泄漏）；Java 要 try-with-resources（仍需写）；Rust 的 RAII 让「资源随作用域自动清理」——**无需显式管理**。<span class="marginnote">RAII 的意义：资源释放<strong>确定性发生（作用域结束的瞬间），不像 GC「不确定何时」。这对文件句柄、锁、GPU 内存等「数量有限、需及时释放」的资源至关重要。Rust 的所有权 + drop 是 RAII 的类型系统化——「资源管理的正确性由编译器保证」。</strong></span>

## 4 公式解析：Rust 的零成本模型

Rust 的「零成本」可以形式化：抽象代码与等价的手写代码**生成相同机器码**。设抽象版本 $A$（trait 泛型）与手写版本 $H$：

$$
\text{cost}(A) = \text{cost}(H) \quad \text{（单态化 + 内联后机器码等价）}
$$

trait 分派的对比：

$$
\text{静态分派（泛型）}: \text{call}(f, T) = \text{call}(\text{monomorphized}(f, T)) \quad \text{无间接跳转}
$$

$$
\text{动态分派（dyn）}: \text{call}(f) = \text{vtable lookup} \quad \text{一次间接寻址}
$$

三步拆解：

- **第一步，单态化**：fn max`<T: Ord>`` 展开为专用代码——与手写 i32 版逐指令等价。
- **第二步，静态 vs 动态分派**：泛型（静态）无虚表、无间接跳转；dyn Trait（动态）走虚表（一次间接寻址）。**Rust 让你显式选择**——默认静态（零成本），需要时才 dyn。
- **第三步，内联**：迭代器链的每层 map/filter 都是泛型方法，编译器内联合并——最终是单个循环。**「抽象在编译期被『摊平』成最优代码」**——这就是零成本抽象的严格含义。

**辨析｜易错点：** 零成本 ≠ 零开销语言。Rust 的**运行时**零开销（无 GC、无虚表默认），但**编译时**开销巨大（单态化、借用检查、生命周期推断）——「Rust 慢在编译，快在运行」。且 String/Vec/Box（堆分配）仍有堆开销——「零成本」指「抽象机制不额外收费」，不是「所有操作免费」。

## 5 Rust 的工程位置

- **系统软件**：Linux 内核模块、Windows 组件、嵌入式——替代 C/C++ 的安全系统语言。
- **浏览器与工具**：Firefox 的 Servo、ripgrep、Alacritty——性能敏感工具。
- **AI 基础设施**：PyTorch 的绑定、推理引擎（vLLM 的部分组件）、数据管线——「安全 + 性能」兼顾。
- **WebAssembly**：Rust 是 WASM 的一等公民——前端高性能计算。<span class="marginnote">Rust 填补了「C 太危险、Java 太慢、Python 太慢」的空白：需要<strong>确定性性能 + 内存安全 + 现代抽象的系统软件，正是 Rust 的主场。它在 AI 基础设施（tokenizer、KV cache、推理内核）越来越常见——「大模型时代的高性能组件用 Rust 重写」是清晰趋势。</strong></span>


## 6 零成本抽象：trait 与泛型 |
| 泛型 + 单态化 | - 泛型 + 单态化：`Vec<T>`、`fn max<T: Ord>`(...) 编译期为每个 T 生成专用代码。 |
| 迭代器链 | - 迭代器链：iter().map().filter().collect()——每个组合零抽象层，编译后等同手写循环。 |
| RAII（Resource Acquisition Is Initialization） | RAII（Resource Acquisition Is Initialization）：资源（内存、文件、锁、网络连接）的获取在初始化时、释放在析构时（离 |
| 自动、确定、无泄漏 | RAII（Resource Acquisition Is Initialization）：资源（内存、文件、锁、网络连接）的获取在初始化时、释放在析构时（离 |
| 生成相同机器码 | Rust 的「零成本」可以形式化：抽象代码与等价的手写代码生成相同机器码。设抽象版本 A（trait 泛型）与手写版本 H： |
| 第一步，单态化 | - 第一步，单态化：fn max`<T: Ord>`` 展开为专用代码——与手写 i32 版逐指令等价。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。


## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 所有权 | # Rust：所有权、生命周期与零成本抽象 |
| 移动语义 | 移动语义：let s2 = s1 转移所有权，s1 失效——无双重释放。 |
| 借用检查 | 借用检查：&T（多读）/&mut T（单写）互斥；引用不超过被引用值寿命。 |
| trait | ## 2 零成本抽象：trait 与泛型 |
| 泛型 + 单态化 | 泛型 + 单态化：`Vec<T>`、`fn max<T: Ord>`(...) 编译期为每个 T 生成专用代码。 |
| 迭代器链 | 迭代器链：iter().map().filter().collect()——每个组合零抽象层，编译后等同手写循环。 |
| RAII（Resource Acquisition Is Initialization） | RAII（Resource Acquisition Is Initialization）：资源（内存、文件、锁、网络连接）的获取在初始化时、释放在析构时 |
| 自动、确定、无泄漏 | RAII（Resource Acquisition Is Initialization）：资源（内存、文件、锁、网络连接）的获取在初始化时、释放在析构时 |
| 生成相同机器码 | Rust 的「零成本」可以形式化：抽象代码与等价的手写代码生成相同机器码。设抽象版本 A（trait 泛型）与手写版本 H： |
| Rust 让你显式选择 | 第二步，静态 vs 动态分派：泛型（静态）无虚表、无间接跳转；dyn Trait（动态）走虚表（一次间接寻址）。Rust 让你显式选择——默认静态（零成 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 7 小结

- **所有权 + 生命周期**：内存安全编译期保证——无 GC、无运行时开销，编译通过即安全。
- **零成本抽象**：trait + 泛型 + 单态化——抽象编译成最优代码，静态分派默认、dyn 显式选择。
- **RAII**：资源随作用域自动释放——确定、自动、无泄漏。
- Rust = C 的性能 + 安全语言的保证 + 现代抽象；代价是编译期复杂度；主场是系统软件与 AI 基础设施。

在下一节，我们将看 Go——**Go：接口、goroutine 与极简类型系统**。
