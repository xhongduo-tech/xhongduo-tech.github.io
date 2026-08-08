---
title: JVM：类加载、字节码与即时编译（JIT）
date: 2026-08-07
---

# JVM：类加载、字节码与即时编译（JIT）

<div class="epigraph">
<p>JVM 是虚拟机工程的珠穆朗玛峰：类加载、字节码验证、即时编译——每一层都打磨了三十年。</p>
<footer>—— 佚名（JVM 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ PLT 综合专题 ｜ 2026-08-07</p>
</div>

## 为什么从 JVM 开始

上一节看了 VM 的通用概念；**JVM（Java Virtual Machine）**是其中最成功、最成熟的代表——支撑了 Java、Kotlin、Scala、Groovy 等一整个生态。JVM 的三个机制值得深挖：**类加载（class loading）**——字节码如何被按需加载与验证；**字节码（bytecode）**——`.class` 文件的中间指令；**JIT（即时编译）**——如何把热点字节码编译成机器码，让 Java 从「慢的解释语言」进化到「接近原生」的性能。理解 JVM，就理解了「一个 VM 如何做到安全 + 跨平台 + 高性能」。<span class="marginnote">JVM 的核心理念：<strong>「平台无关的字节码 + 平台相关的 JIT」</strong>。`.class` 文件在任何平台都一样；JVM 在每平台上解释/编译它。Java 程序 = 一堆 class 文件，JVM = 执行它们的虚拟 CPU。「字节码是 Java 的汇编」——`javap` 可以看到它。</span>

## 1 类加载：按需装载字节码

**类加载（class loading）**：JVM 在**首次使用**类时，把它从 `.class` 文件加载进内存——**延迟加载（lazy loading）**，用到才载。

**类加载器（class loader）**的三步：

1. **加载（loading）**：读 `.class` 字节码，构造 `Class` 对象。
2. **链接（linking）**：验证字节码合法性（**字节码验证器**检查类型安全、栈溢出等）、准备（分配静态字段）、解析（符号引用转直接引用）。
3. **初始化（initialization）**：执行静态初始化器。

**双亲委派模型（parent delegation）**：类加载请求先给父加载器——保证「核心类（`java.lang.String`）只被引导加载器加载一次」，防类冲突与恶意替换。<span class="marginnote">「字节码验证器」是 JVM 安全的第一道门：加载时检查字节码不越界、类型正确、栈操作合法——<strong>在运行前就拒绝恶意/损坏的字节码</strong>。「双亲委派」保证核心 API 不被篡改：请求加载 `java.lang.String` 时，总是优先给引导加载器——防止「自定义的 String」混进来。「加载 + 验证 + 委派」构成 JVM 的安全基座。</span>

## 2 字节码：class 文件的指令集

**字节码（bytecode）**：`.class` 文件的指令序列——JVM 的「汇编语言」。它是**面向栈**的指令集（上一节讲过）：

```
// 方法 int add(int a, int b) 的字节码
iload_1      // 把第 1 个局部变量（a）压入操作数栈
iload_2      // 把第 2 个局部变量（b）压栈
iadd         // 弹出两个 int，相加后结果压栈
ireturn      // 返回栈顶的 int
```

JVM 字节码的特点：

**指令紧凑**：多数指令 1 字节——`iload_0`、`iconst_1` 是单字节操作码。
**类型化**：`iadd`（int 加）与 `fadd`（float 加）分开——类型信息在字节码里。
**可验证**：字节码验证器靠「类型化的栈图」检查安全性。<span class="marginnote">字节码的「类型化」是 JVM 安全的底气：`iadd` 明确「这是 int 加法」——验证器能检查栈上类型匹配。`javap` 反汇编 class 文件，你会看到这些指令。JVM 字节码有约 200 条指令——比真实 CPU 指令集小，但足够表达 Java 的一切计算。</span>

## 3 JIT：从解释到机器码

**JIT（Just-In-Time，即时编译）**：JVM 把**热点**字节码在**运行期**编译成机器码——绕过解释循环，大幅提速。

HotSpot JVM 的两级策略：

**解释执行**：启动时解释字节码——**启动快**（不用先编译）。
**C1 编译**：轻量 JIT——快速编译热点，优化有限。
**C2 编译**：重型 JIT——深度优化（内联、逃逸分析），慢编译但极快执行。

**分层编译（tiered compilation）**：先解释 → C1 → C2——「启动快 + 峰值高」兼得。<span class="marginnote">JIT 的「热点检测（profiling）」：JVM 运行时统计哪些方法被频繁调用，只编译热点的——<strong>「编译成本花在最值得的地方」</strong>。C2 的优化深度惊人：方法内联、死代码消除、逃逸分析（栈上分配不逃逸的对象）、锁消除。「Java 慢」已是过去式——现代 JVM 的峰值性能接近 C++ 的 80%+。</span>

## 4 公式解析：JIT 的性能模型

JIT 的收益可以量化。设解释执行每条字节码成本 $c_i$、编译后机器码执行成本 $c_m$，方法执行 $N$ 次：

$$
\text{总成本}_{解释} = N \times c_i, \qquad \text{总成本}_{JIT} = c_{\text{compile}} + N \times c_m
$$

JIT 优于解释当且仅当：

$$
c_{\text{compile}} + N \times c_m < N \times c_i \;\Longrightarrow\; N > \frac{c_{\text{compile}}}{c_i - c_m}
$$

三步拆解：

- **第一步，两本账**：解释成本 = 每次执行 × 次数；JIT 成本 = 编译一次 + 每次执行 × 次数（编译后快）。
- **第二步，找阈值**：JIT 不划算当且仅当执行次数太少（编译成本没摊薄）——所以只编译**热点**。
- **第三步，看摊薄**：`N`（执行次数）越大，编译成本占比越小——**「热点方法执行百万次，编译一次的成本微不足道」**。这就是 JIT「只编热点」的依据，也是「先解释后 JIT」分层的原因。

**辨析｜易错点：** JIT 的「预热（warm-up）」：程序启动后的**前几万次调用**是解释执行（慢），热点被编译后才快——**「Java 程序要跑一会才达到峰值性能」**。对「冷启动 + 短任务」（命令行工具、Serverless），JIT 收益小、甚至不如 AOT（提前编译，GraalVM Native Image 的做法）。「JIT 适合长跑服务，AOT 适合快速启动」——这是现代 Java 的两条路线。

## 5 JVM 的现代生态

- **多语言**：Kotlin（Android 官方）、Scala、Groovy、Clojure——都编译到 JVM 字节码，共享 JVM 的 GC/JIT/工具链。
- **GraalVM**：把 JVM 扩展为多语言运行时（Truffle 框架解释 JS/Python/Ruby），并提供 Native Image（AOT 编译成原生二进制）。
- **JVM 在 AI**：大数据（Hadoop/Spark）、分布式系统（Kafka）、以及部分 AI 服务（Java 高并发后端）——「JVM 生态 + 高性能 GC」让它仍是企业后端主力。<span class="marginnote">JVM 的成功不只是「语言」的成功，是「平台」的成功：字节码标准 + 运行时服务 + 庞大工具链（IDE、profiler、监控）——「写一次在 JVM 上，跑到任何平台，任何语言都能上 JVM」。这印证了 VM 的核心价值：<strong>语言编译到「通用运行时」而非「具体硬件」，生态由此聚集</strong>。</span>

## 6 亲手看看字节码

理解 JVM 字节码最好的方式，是**亲手反汇编一个 class 文件**。写一个简单的 Java 类：

```java
public class Add {
    public int add(int a, int b) {
        return a + b;
    }
}
```

编译后运行 `javap -c Add`，看到字节码：

```
$ javap -c Add
Compiled from "Add.java"
public class Add {
  public int add(int, int);
    Code:
       0: iload_1
       1: iload_2
       2: iadd
       3: ireturn
}
```

观察三个特征：

- **面向栈**：`iload_1`/`iload_2` 都在操作数栈上进行——`iadd` 不需要写「哪两个寄存器」，栈顶就是操作数。
- **类型化**：`i` 前缀 `iadd` 表示 int——对应 `iload` 有 `iload_1`、`fload` 有 `fload_1`，类型信息编码在指令里。
- **紧凑**：多数指令一字节——class 文件小而解析快。

**辨析｜易错点：** 字节码 ≠ 机器码：**字节码是「平台无关的中间表示」，机器码是「平台相关的最终代码」**。`iload_1` 在任何平台的 JVM 上都是「压第一个局部变量」；JIT 把它编译成 x86/ARM 的具体加载指令。**「读字节码 = 看 Java 程序的『编译中间形态』」**——它帮你理解「Java 程序实际执行的是什么」，也是理解「为什么 Java 能跨平台」的最直观方式。


## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 类加载（class loading） | 类加载（class loading）：JVM 在首次使用类时，把它从 .class 文件加载进内存——延迟加载（lazy loading），用到才载。 |
| 首次使用 | 类加载（class loading）：JVM 在首次使用类时，把它从 .class 文件加载进内存——延迟加载（lazy loading），用到才载。 |
| 延迟加载（lazy loading） | 类加载（class loading）：JVM 在首次使用类时，把它从 .class 文件加载进内存——延迟加载（lazy loading），用到才载。 |
| 类加载器（class loader） | 类加载器（class loader）的三步： |
| 加载（loading） | 1. 加载（loading）：读 .class 字节码，构造 Class 对象。 |
| 链接（linking） | 2. 链接（linking）：验证字节码合法性（字节码验证器检查类型安全、栈溢出等）、准备（分配静态字段）、解析（符号引用转直接引用）。 |
| 字节码验证器 | 2. 链接（linking）：验证字节码合法性（字节码验证器检查类型安全、栈溢出等）、准备（分配静态字段）、解析（符号引用转直接引用）。 |
| 初始化（initialization） | 3. 初始化（initialization）：执行静态初始化器。 |
| 字节码（bytecode） | 字节码（bytecode）：class 文件的指令序列——JVM 的「汇编语言」。它是面向栈的指令集（上一节讲过）： |
| 面向栈 | 字节码（bytecode）：class 文件的指令序列——JVM 的「汇编语言」。它是面向栈的指令集（上一节讲过）： |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 7 小结

- **类加载**：按需加载 + 字节码验证 + 双亲委派——安全基座（核心类不被篡改、恶意字节码被拒）。
- **字节码**：面向栈的 `.class` 文件指令集——紧凑、类型化、可验证。
- **JIT**：热点字节码运行期编译成机器码——分层（解释 → C1 → C2）兼顾启动快与峰值高。
- JIT 只编热点（编译成本摊薄）；「预热」让长跑服务受益；GraalVM Native Image 用 AOT 补冷启动场景。

在下一节，我们将深入 JVM 的运行时——**JVM 内存模型与垃圾回收器体系**。
