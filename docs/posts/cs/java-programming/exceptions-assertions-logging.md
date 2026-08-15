---
title: 异常处理、断言与日志
date: 2026-08-07
---

# 异常处理、断言与日志

<div class="epigraph">
<p>程序最诚实的时刻，是它出错的时候。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第7章 ｜ 2026-08-07</p>
</div>

## 为什么从异常处理开始

至今为止的代码都假设「一切顺利」——文件一定存在、网络一定连通、整数一定不溢出。但真实世界不是这样：磁盘会满、用户会输入乱码、第三方接口会超时。**异常处理（exception handling）**是 Java 对「程序出错时怎么办」的体系化回答：它把「错误信号」与「正常返回值」分开，让错误沿着调用栈向上传递，直到有人愿意处理它。配合**断言**（开发期检查「不可能发生」的条件）与**日志**（记录运行时发生了什么），三件套构成了 Java 程序的「容错与可观测性」基础设施。

## 1 异常层次：Throwable 家族

Java 的所有错误都以**对象**形式存在，它们的根是 `java.lang.Throwable`，下分两支：

$$

\text{Throwable} \begin{cases} \text{Error} & \text{—— 虚拟机层面的严重问题（OOM、StackOverflow）} \\ \text{Exception} & \begin{cases} \text{RuntimeException} & \text{—— 运行时异常（不受检查）} \\ \text{其他受检异常} & \text{—— 编译器强制处理} \end{cases} \end{cases}

$$

对这条家族树做三步拆解：

- **`Error`**：JVM 或系统级的严重问题——`OutOfMemoryError`（内存耗尽）、`StackOverflowError`（栈溢出）。**正常程序不该捕获它们**，捕获了也多半救不回来，让 JVM 崩掉反而干净。
- **`RuntimeException` 及其子类**：**非受检异常（unchecked）**，编译器不强制处理——`NullPointerException`、`ArrayIndexOutOfBoundsException`、`IllegalArgumentException`、`ClassCastException`。它们代表**编程错误**：是你写错了，不是调用方可以「恢复」的意外。
- **其他 `Exception` 子类**：**受检异常（checked）**，编译器强制「处理或向上声明」——`IOException`、`SQLException`、`InterruptedException`。它们代表**可以合理恢复的外部意外**：文件没了、网络断了、超时了。

**重点结论：`Error` 不捕获，`RuntimeException` 是编程错误，受检异常是可恢复的意外。** 这三条决定了 `catch` 该写谁、不写谁。

## 1 捕获与抛出：try-catch-finally

**`try` 块**放可能出错的代码，**`catch` 块**处理错误，**`finally` 块**无论是否异常都执行：

```java
try {
    int result = risky();          // 可能抛 IOException
} catch (IOException e) {          // 只捕获受检异常
    System.out.println("IO 出错：" + e.getMessage());
} finally {                        // 无论成败都执行
    cleanup();                     // 典型用途：释放资源
}
```

**三个要点：**

- **`catch` 可以按异常类型分支**：`catch (IOException e)` 与 `catch (SQLException e)` 分别处理——捕获的异常类型**必须存在继承关系，且子类在前**，否则编译报错「不可达的 catch 块」。
- **`finally` 是「一定会执行」的保证**——即使 `catch` 里 return、即使又抛了异常。它是释放资源的传统手段，但更优雅的方式是 try-with-resources（见下文）。
- **向上抛**：方法不处理就 `throws IOException` 声明「我会把这异常交出去」——受检异常要么 catch 要么 throws，二选一。

**捕获多个异常**（Java 7 起）用竖线：`catch (IOException | SQLException e)`——异常变量 `e` 自动是 final，不能重新赋值，语义更清晰。

**辨析｜易错点：`finally` 里 return 会覆盖 `try` 里的 return。** 如果 `try` 和 `finally` 都有 `return`，`finally` 的返回值**胜出**——这是经典陷阱。**`finally` 里别写 return**，它只该做清理。

## 2 受检异常 vs 非受检异常：核心对比表

纯概念主题用**核心对比表**替代公式解析的展开，把两种异常的性质摆开：

| 维度 | 受检异常（checked） | 非受检异常（unchecked） |
| --- | --- | --- |
| 父类 | `Exception`（非 `RuntimeException`） | `RuntimeException` |
| 编译器强制 | 必须 catch 或 throws | 不强制 |
| 代表 | 可恢复的外部意外 | 编程错误 |
| 例子 | `IOException`、`SQLException` | `NPE`、越界、`IllegalArgumentException` |
| 处理义务 | 调用方被逼着考虑 | 靠写对代码避免 |

**重点结论：受检异常是「编译器逼你负责任」——它把「可能失败」写进方法签名，让调用方无法假装不会失败。** 非受检异常则把「处理」留给写代码的人：空指针不该靠 catch 兜底，而该靠判空杜绝。两种异常各管一摊：**该恢复的用受检，该避免的用非受检。**<span class="marginnote">初学最纠结「受检异常好烦」——确实，Java 是唯一把「异常必须在签名里声明」做成编译期强制的主流语言。代价是样板代码，收益是「读方法签名就知道它会不会失败」。这个取舍也是 Effective Java 第 70 条反复讨论的主题。</span>

## 3 try-with-resources：自动释放资源

Java 7 引入 **try-with-resources**，让「自动关闭资源」成为语言特性。语法：在 `try` 括号里声明资源，**无论正常还是异常结束，资源都会被自动关闭**：

```java
try (BufferedReader reader = new BufferedReader(
        new FileReader("config.txt"))) {
    String line = reader.readLine();
    // 用 reader……
}   // 编译器自动插入 close()，包括异常路径
```

**为什么它比手写 `finally` 更好**：手写 `finally` 时，如果主体抛异常、`close()` 也抛异常，**`close` 的异常会覆盖主体异常**，原始错误被掩盖；try-with-resources 会把 `close` 的异常记为**抑制异常（suppressed）**，主体异常保留——排障信息更完整。<span class="marginnote">任何实现了 <strong>`AutoCloseable`</strong> 接口的对象都能写进 `try (...)` 里。`InputStream`、`Connection`、`Statement` 都实现了它。需要「用完必关」的自定义资源，就实现 `AutoCloseable` 的 `close()` 方法——这是 Java 资源管理的现代标准。</span>

## 4 断言：开发期的「不可能发生」

**断言（assertion）**是「开发期检查不变量」的轻量工具：

```java
assert x >= 0 : "x 不能为负，收到：" + x;
```

- 冒号后是失败时的消息（可省）。
- **断言默认关闭**：JVM 参数 `-ea` 才启用，发布时默认关闭——所以**别把断言当作参数校验**（生产环境关掉就失效）。
- 断言失败抛 `AssertionError`（`Error` 家族，正常不该被捕获）。

**重点结论：断言用于「不可能发生」的内部不变量，参数校验用「一定发生」的显式 `throw`。** 断言关了不影响正确代码，参数校验关了会放过非法输入——两者定位不同。把「用户输入校验」写进 `assert`，是发布后「悄悄失效」的经典 bug。

## 5 日志：运行期的「黑匣子」

**日志（logging）**记录「运行时发生了什么」，是排障的最后一根稻草。Java 自带 `java.util.logging`（JUL），生产环境更常用 **Log4j2 / SLF4J / Logback**。核心是分级过滤：

| 级别 | 含义 | 典型用途 |
| --- | --- | --- |
| `SEVERE` / ERROR | 严重错误 | 服务崩溃、关键操作失败 |
| `WARNING` | 可疑但可继续 | 重试后成功、配置缺失 |
| `INFO` | 正常关键事件 | 启动、请求处理完成 |
| `FINE` / DEBUG | 调试细节 | 变量值、中间结果 |

**日志四律**：

- **用占位符而不是字符串拼接**：`log.info("用户 {} 登录", userId)`——不拼接字符串，未启用 DEBUG 时不浪费。
- **别在循环里记日志**：循环体内 INFO 会刷爆磁盘，高频路径用计数聚合。
- **记「上下文」**：`log.error("处理订单 {} 失败", orderId, e)`——带订单号、异常栈，不看代码也能定位。
- **级别过滤在配置里**：开发开 DEBUG、生产开 INFO，改配置不动代码。

**辨析｜易错点：日志 ≠ `System.out`。** `System.out.println` 只能进控制台、无级别、无格式化、无滚动——它不是日志。正式系统用日志框架：支持级别过滤、写文件、滚动、远程收集。**`System.out` 留给「教学程序」，日志留给「生产系统」。**

## 6 小结

- `Throwable` 分两支：**`Error` 不捕获，受检异常（可恢复）必须处理，`RuntimeException`（编程错误）靠写对代码避免**。
- 处理结构：`try-catch-finally`；**`finally` 里别写 return**。
- 受检异常「该恢复」，非受检异常「该避免」；捕获多异常用 `catch (A | B e)`。
- **try-with-resources** 自动关资源、保留原始异常，替代手写 `finally`。
- 断言查「不可能发生」（默认关闭），日志记「运行时发生了什么」（分级过滤）。

在下一节，我们让「容器与算法脱离具体类型」——**泛型程序设计**。