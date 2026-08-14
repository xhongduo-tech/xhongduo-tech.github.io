---
title: 输入输出流与文件处理
date: 2026-08-07
---

# 输入输出流与文件处理

<div class="epigraph">
<p>程序的一半是内存里的计算，另一半是内存与外部世界的搬运。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第2卷第2章 ｜ 2026-08-07</p>
</div>

## 为什么从输入输出流开始

至此的程序都在内存里自嗨——数据没进过程序，结果也没出过程序。真实程序必须与外部世界打交道：读配置文件、写日志、上传下载文件、从网络取数据。Java 用**流（stream）**这个统一抽象处理所有 IO：**流是「字节/字符的有序序列」**，无论是文件、网络套接字还是内存缓冲区，读写的方式都一样。这一篇先立起流的体系（字节流与字符流），再落到**文件**这个最重要的外部世界入口——重点掌握 `Files` 类（Java 7 的 NIO.2）这个现代文件操作的瑞士军刀。

## 1 流的两大家族：字节流与字符流

Java 的 IO 体系以四个抽象类为根，分成两族：

$$

\text{字节流：} \text{InputStream} \;\text{(读)} \quad \text{OutputStream} \;\text{(写)}

$$

$$

\text{字符流：} \text{Reader} \;\text{(读)} \quad \text{Writer} \;\text{(写)}

$$

**字节流**以 `byte` 为单位，处理一切二进制数据（图片、视频、压缩包）；**字符流**以 `char`/字符串为单位，处理文本，内部负责**编码解码**（把字符转成字节、把字节转回字符）。

| 家族 | 读基类 | 写基类 | 处理单位 | 适用数据 |
| --- | --- | --- | --- | --- |
| 字节流 | `InputStream` | `OutputStream` | `byte` | 二进制 |
| 字符流 | `Reader` | `Writer` | `char`/`String` | 文本 |

**为什么需要两族？** 文本文件有**字符编码**问题——同样一个「中」字，UTF-8 是 3 字节，GBK 是 2 字节。用字节流读文本，你要自己管编码；用字符流（`InputStreamReader` 指定字符集），它替你完成字节↔字符的转换。<span class="marginnote">字符编码是 IO 世界的第一个「易错点」：读文件必须与文件实际编码一致。现代规范是<strong>统一 UTF-8</strong>——写文件时显式指定 `StandardCharsets.UTF_8`，读文件时同样指定，两边对齐才不会出乱码。</span>

**常用的具体流**：

字节：`FileInputStream`/`FileOutputStream`（文件）、`ByteArrayInputStream`（内存）、`BufferedInputStream`（加缓冲）。
字符：`FileReader`/`FileWriter`（文件，注意默认编码）、`BufferedReader`/`BufferedWriter`（加缓冲）、`InputStreamReader`（字节→字符桥）。

**装饰器模式**：流可以**层层包裹**——`new BufferedReader(new FileReader(...))`。内层负责「从文件读字节/字符」，外层负责「加缓冲提升性能、加 `readLine` 提供行读取」。这种「功能叠加」正是设计模式里的装饰器，也是为什么 Java 的流类数量庞大——它们是积木，不是各管一摊的独立类。

## 2 用 Files 读写文件：现代姿势

手写 `FileInputStream` + 循环读字节是上个世纪的写法。**Java 7 的 `java.nio.file.Files`** 提供了一行式文件操作，日常 90% 的文件读写用它就够了：

```java
// 读：整个文件读成字符串 / 按行读成 List
String content = Files.readString(Path.of("config.txt"), StandardCharsets.UTF_8);
List<String> lines = Files.readAllLines(Path.of("data.txt"));

// 写：整个字符串写出 / 追加
Files.writeString(Path.of("out.txt"), "内容", StandardCharsets.UTF_8);
Files.write(Path.of("out.txt"), lines);

// 复制 / 移动 / 删除
Files.copy(Path.of("a.txt"), Path.of("b.txt"), StandardCopyOption.REPLACE_EXISTING);
Files.move(Path.of("a.txt"), Path.of("archive/a.txt"));
Files.deleteIfExists(Path.of("tmp.txt"));
```

`Path` 是「路径」的现代抽象（替代老旧的 `File`），`Path.of("dir", "sub", "file.txt")` 还能按平台自动拼分隔符。

**重点结论：能用 `Files` 就别手写流。** `readString`/`writeString`/`readAllLines` 内部封装了打开、读取、关闭、异常处理的全过程，代码量少一个量级。**读大文件（几百 MB 以上）才需要回落到 `BufferedReader` 逐行流式读**，避免一次性把整个文件装进内存。

**注意**：`Files` 的这些便捷方法默认**一次性装载**，适合中小文件；而且**所有文件 IO 都可能抛 `IOException`（受检异常）**——调用处要 try-catch 或 `throws`，或用 try-with-resources 包住流式操作。

## 3 逐行处理大文件：BufferedReader

处理超大文件时不能整读，要**逐行流式**处理。经典模式：

```java
try (BufferedReader reader = Files.newBufferedReader(
        Path.of("big.log"), StandardCharsets.UTF_8)) {
    String line;
    while ((line = reader.readLine()) != null) {   // 读一行处理一行
        process(line);
    }
}   // try-with-resources 自动关闭
```

**为什么用 `BufferedReader`**：`FileReader` 每次 `read()` 都触一次系统调用，慢；`BufferedReader` 一次性从文件读一大块到**内存缓冲区**，`readLine` 从缓冲区取——IO 次数从「每字符一次」降到「每几千字符一次」，性能天壤之别。<span class="marginnote">缓冲的原理是「以空间换 IO」：磁盘/网络的单次 IO 开销远大于内存拷贝，把多次小 IO 合并成一次大 IO，是 IO 优化的第一课。这个思想在第三级《操作系统》的块缓存、以及大数据系统的「批量刷写」里反复出现。</span>

**辨析｜易错点：`readLine()` 返回 `null` 表示读完**——`while ((line = reader.readLine()) != null)` 这种「边读边判」是标准写法，但新手容易写成 `while (reader.readLine() != null)`（把一行读丢了）或忘记判 null（读到末尾抛 `NullPointerException`）。

**写大文件**同理：`BufferedWriter` 的 `write` + `newLine()`，或用 `Files.newBufferedWriter`。批量写入时**留意缓冲区**——`write` 的内容先进缓冲区，`flush()` 或关闭时才真正落盘；想立即落盘就调 `flush()`。

## 4 公式解析：字符流的编码换算

字符流的核心工作是编码换算。**「字符」与「字节」之间的换算率由字符集决定**：

$$

\text{UTF-8：} \quad \text{字符数} \times (1 \sim 4 \text{ 字节}) \qquad \text{GBK：} \quad \text{汉字} = 2 \text{ 字节，ASCII} = 1 \text{ 字节}

$$