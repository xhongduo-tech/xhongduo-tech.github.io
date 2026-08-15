---
title: Java 语言概述与开发环境搭建
date: 2026-08-07
---

# Java 语言概述与开发环境搭建

<div class="epigraph">
<p>一次编写，处处运行。</p>
<footer>—— 太阳微系统公司（Sun Microsystems）Java 口号</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第1-2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Java 的诞生讲起

这个网站叫「从极限到大模型」，主线是从数学极限走到当代大模型。在第三级·计算机基础这条线上，**Java 是理解「企业级软件如何被规模化建造」的第一语言**。<span class="marginnote">后端服务、大数据（Hadoop/Spark/Kafka）、Android 应用、以及大量云原生中间件都以 Java 为母语——它像数学里的实数一样，是个绕不开的「完备数域」。</span>你学完 C 或 Python 后转入 Java，会发现它的语法吸收了 C 的骨架，却在内存管理、跨平台与类型系统上做了三件截然不同的设计决策。这一篇讲清楚 Java「从何而来、怎么跑起来、如何搭好环境」，是后续所有语法细节的地基。

## 1 Java 的语言定位与设计目标

Java 诞生于 1991 年，最初是 Sun 公司为消费电子设计的、代号 **Oak** 的语言。1995 年正式改名 Java 发布，靠三个设计目标杀出重围：

**平台无关**：Java 源代码被编译成**字节码（bytecode）**，字节码不针对任何特定 CPU，而是交给**Java 虚拟机（JVM, Java Virtual Machine）**解释执行。同一个 `.class` 文件在 Windows、Linux、macOS 上行为一致，这就是「一次编写，处处运行」的技术内核。<span class="marginnote">对比 C/C++：源码直接编译成面向 x86 或 ARM 的机器码，换平台必须重新编译。Java 用「中间层 JVM」换来了跨平台，代价是启动略慢、执行多一层翻译——这个权衡直到 JIT 即时编译成熟后才被基本抹平。</span>

**自动内存管理**：Java 没有 `malloc/free`，对象不再被引用时由**垃圾收集器（GC, Garbage Collector）**自动回收。程序员从「手动管理内存」的深渊中被解放，换来的是偶尔的 GC 停顿。

**面向对象**：Java 强制一切数据与行为都封装在类里（除了八个基本类型），类是模块化与复用的最小单元。

**性能**：早期 JVM 靠解释执行，慢；1998 年引入**JIT（Just-In-Time）编译器**，把热点字节码在运行时编译成机器码，性能逼近 C++。此后 Java 每隔数年做一次大版本升级，从 Java 8 的 lambda 到 Java 17 的密封类、Java 21 的虚拟线程，语言仍在持续演进。

## 2 核心概念：JVM、JRE 与 JDK

搭建环境前，必须分清三个层层包含的缩写——这是初学者第一个高频混淆点：

| 缩写 | 全称 | 包含内容 | 谁需要它 |
| --- | --- | --- | --- |
| JVM | Java Virtual Machine | 执行字节码的虚拟机 | 运行 Java 程序 |
| JRE | Java Runtime Environment | JVM + 核心类库 | 只运行不开发的人 |
| JDK | Java Development Kit | JRE + 编译器 `javac` + 工具链（`jar`、`javap` 等） | 开发者 |

**JDK  ⊃ JRE ⊃ JVM**：装了 JDK，开发、运行、调试一应俱全；只装 JRE 也能跑 `.jar` 程序，但无法编译源码。

一条 Java 程序的完整旅程是：

$$
\text{源码 } \text{Hello.java} \xrightarrow{\text{javac}} \text{字节码 } \text{Hello.class} \xrightarrow{\text{java}} \text{JVM 执行} \to \text{输出}
$$

`javac` 是编译器，`java` 是启动器。**编译一次、到处运行**的秘密就在中间那步：`.class` 是平台无关的。

## 3 开发环境搭建与第一个程序

搭建现代 Java 环境的推荐路径分四步：

**第一步，安装 JDK**。OpenJDK 是官方参考实现，免费开源；Oracle JDK 是商业发行版。到 2026 年，**长期支持版本（LTS）**是 Java 17 与 Java 21，初学者直接装 JDK 21 即可。安装后在终端执行 `java -version` 验证：

```bash
$ java -version
openjdk version "21.0.2" 2024-01-16
```

**第二步，配置环境变量**。`JAVA_HOME` 指向 JDK 安装目录，`PATH` 加入 `$JAVA_HOME/bin`。IDE 会自动探测，但命令行工具链依赖它们。<span class="marginnote">macOS 上用 Homebrew 装 `openjdk@21`，Linux 上用发行版包管理器，Windows 上用官方安装包——细节各异，验证标准只有一个：终端能敲出 `java -version`。</span>

**第三步，写第一个程序**。文件主名必须与公共类名一致，这是 Java 的硬性规定：

```java
public class Hello {
    public static void main(String[] args) {
        System.out.println("你好，Java");
    }
}
```

**第四步，编译并运行**：

```bash
$ javac Hello.java     # 产出 Hello.class
$ java Hello           # 用 JVM 运行主类，输出「你好，Java」
你好，Java
```

`javac` 产出的 `.class` 文件与你写代码的机器无关——把它拷到任何装有 JVM 的机器上，`java Hello` 都能跑。这就是「一次编写、处处运行」的完整闭环。<span class="marginnote">IDE（IntelliJ IDEA 是最主流的 Java IDE）把上面四步封装成了「一键运行」，但你仍要理解背后发生了什么：IDE 只是替你调用 `javac` 与 `java`。排障时（编译错、类找不到）回到底层命令行，往往一眼看穿。</span>

## 4 公式解析：从源码到执行的流水线

把「Java 程序怎么跑起来」用一个流水线公式钉死：

$$
\text{源码 .java} \xrightarrow{\text{javac 编译}} \text{字节码 .class} \xrightarrow{\text{类加载器}} \text{JVM} \xrightarrow{\text{解释 + JIT}} \text{机器码} \to \text{执行}
$$

对这条流水线做三步拆解：

- **第一步，编译**：`javac` 把 `.java` 编译成**平台无关**的字节码 `.class`。字节码不是任何 CPU 的机器码，而是一套 JVM 自己的「中间指令集」。
- **第二步，加载**：`java` 启动 JVM，由**类加载器（ClassLoader）**把 `.class` 按需读进内存——这是「懒加载」，用到哪个类才加载哪个。
- **第三步，执行**：JVM 先**解释执行**字节码；对反复执行的热点代码，**JIT（Just-In-Time）编译器**在运行期把它编译成**当前机器的原生机器码**——所以同一份字节码在 x86 上变成 x86 指令、在 ARM 上变成 ARM 指令，性能逼近 C++。

**重点结论：平台的差异被「编译 + 解释/JIT」两步消化掉了。** 你只编译一次，JVM 针对每台机器再「适配」一次。这正是「一次编写、处处运行」的技术本质——与 Python 的「解释器直译」、C 的「每平台编译一次」构成三种不同的跨平台策略，而 Java 选了「编译到中间层 + 运行时适配」的折中。

## 5 常用工具与开发流程

JDK 附带一整套命令行工具，是日常排障的武器库：

| 工具 | 用途 | 典型场景 |
| --- | --- | --- |
| `javac` | 编译源码 | 把 `.java` 编成 `.class` |
| `java` | 运行程序 | `java Hello` 启动主类 |
| `jar` | 打包/解包 | 把一堆 `.class` 打成可执行 jar |
| `javap` | 反汇编字节码 | 查看 `.class` 的方法签名（排错用） |
| `jps` | 列出 JVM 进程 | 看后台 Java 进程 |
| `jstack` | 打印线程转储 | 排查死锁、卡死 |

**面向对象工程的构建工具**：真实项目不用命令行逐个 `javac`，而是用 **Maven / Gradle** 管理依赖、编译、打包、测试——它们把「下载依赖 jar、编译、打包、跑测试」编排成一条命令（`mvn package`）。这属于第三级《软件工程与构建》的内容，但你现在就该知道：**JDK 是地基，构建工具是脚手架。**

## 6 小结

- Java 靠**字节码 + JVM** 实现平台无关；JDK ⊃ JRE ⊃ JVM，开发装 JDK。
- 流程四步：**写源码 → `javac` 编译 → `java` 加载运行 → JVM 解释 + JIT 执行**。
- 安装 JDK 21（LTS），配置 `JAVA_HOME` 与 `PATH`，能敲出 `java -version` 即环境就绪。
- 文件名主名必须与公共类名一致；`main` 是程序入口。
- 排障武器库：`javac`/`java`/`jar`/`javap`/`jstack`；大型项目用 Maven/Gradle。

在下一节，我们进入 Java 语法的第一个正式章节——**基本程序设计结构：变量、数据类型与运算符**。