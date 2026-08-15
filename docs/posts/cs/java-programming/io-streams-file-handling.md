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

对这条公式做三步拆解：

- **第一步，字符 → 字节的换算率由字符集决定**：UTF-8 里 ASCII 字符（拉丁字母、数字）占 1 字节、常用汉字 3 字节、生僻字可达 4 字节；GBK 里汉字固定 2 字节、ASCII 1 字节。
- **第二步，「读文件」其实是「读字节 + 解码」**：磁盘上存的永远是**字节**；字符流把字节按指定字符集**解码**成 `char`/字符串。字符集选错，同一串字节会解码成乱码——**读文件必须与写入时的编码一致**。
- **第三步，统一 UTF-8 消灭问题**：现代实践是「全链路 UTF-8」——写时显式指定 `StandardCharsets.UTF_8`，读时同样指定，两边对齐，乱码从根上消失。遗留系统里 GBK 文件才需要按它解码。

**辨析｜易错点：`FileReader` 用「平台默认编码」，不可靠。** `new FileReader("a.txt")` 用的默认编码随系统变化（Windows 上是 GBK、macOS/Linux 是 UTF-8）——同一段代码在不同机器上行为不同。**用 `Files.newBufferedReader(path, StandardCharsets.UTF_8)` 显式指定编码**，比依赖默认值稳得多。

## 4 目录与文件管理：Files 的完整能力

`Files` 类不只读写文件，它还覆盖了「文件与目录管理」的全部日常操作：

```java
// 目录遍历：列出目录内容（Java 8 起）
try (Stream<Path> entries = Files.list(Path.of("."))) {
    entries.forEach(System.out::println);
}

// 递归遍历整棵目录树（含子目录）
try (Stream<Path> walk = Files.walk(Path.of("src"))) {
    walk.filter(Files::isRegularFile).forEach(System.out::println);
}

// 判断与属性
Files.exists(path);
Files.isDirectory(path);
Files.isRegularFile(path);
Files.size(path);                    // 字节数

// 创建目录（含父目录）
Files.createDirectories(Path.of("a/b/c"));
```

**重点结论：目录遍历用 `Files.list` / `Files.walk` + Stream**——这正是《Lambda 与 Stream 流式编程》的用武之地：过滤、映射、统计一行流式搞定。`Files.walk` 返回的 `Stream` 要放进 try-with-resources（它持有打开的目录句柄）。

**大文件的最优策略**：几个数量级的内存账——`Files.readAllBytes` 一次性装载（适合小文件）、`Files.readAllLines` 按行装进 `List`（中等）、`BufferedReader.readLine` 流式逐行（大文件）。**选型依据是文件大小与内存预算**：处理 2 GB 日志还用 `readAllLines`，内存直接爆。

## 5 小结

- IO 两族：**字节流**（`InputStream`/`OutputStream`）管二进制，**字符流**（`Reader`/`Writer`）管文本并负责编码解码。
- 流的**装饰器**：`BufferedReader(FileReader(...))` 层层包裹、功能叠加。
- 现代文件操作用 **`Files`**：`readString`/`writeString`/`readAllLines`/`walk`/`createDirectories` 一行式。
- 读大文件用 **`BufferedReader` 逐行流式**；**读文件显式指定 `StandardCharsets.UTF_8`**，别依赖平台默认编码。
- 所有文件 IO 都抛受检 `IOException`；try-with-resources 自动关闭。

在下一节，我们把对象「送到文件里、再从文件里取回」——**序列化的最佳实践**。