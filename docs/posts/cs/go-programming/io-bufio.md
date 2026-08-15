---
title: io 与 bufio：输入输出抽象
date: 2026-08-07
---

# io 与 bufio：输入输出抽象

<div class="epigraph">
<p>接口描述行为而不是实现——这就是为什么 io.Reader 能同时代表文件、网络与内存。</p>
<footer>—— Go I/O 设计哲学（Readers and Writers）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第7章 + io/bufio 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 I/O 抽象开始

程序与外界交换数据的每一种方式——文件、网络连接、标准输入、内存缓冲区、压缩流——本质都是「读」与「写」。Go 用两个**小而关键**的接口 `io.Reader` 与 `io.Writer` 把它们统一起来：只要实现了 `Read`/`Write` 方法，任何数据源都能被同一套代码处理。<span class="marginnote">对标《Go语言圣经》第7.5 节「接口值」与 io 包：`io.Reader`/`io.Writer` 是 Go 生态最重要的两个接口——它们只有一个方法，却支撑起整个标准库的 I/O。这正是《接口》篇「接口越小越强」的最有力证明。</span>

I/O 在本专题的定位：它是接口思想的「毕业设计」——把《接口》篇讲的理论应用到全生态最核心的抽象上。同时它也是《net/http》篇（网络读写）、`encoding/json`（从流中解码）的基础。

## 1 io.Reader 与 io.Writer：最小抽象

**`io.Reader`** 表示「可以读数据的东西」，**`io.Writer`** 表示「可以写数据的东西」。各只有一个方法：

```go
type Reader interface {
	Read(p []byte) (n int, err error)
}

type Writer interface {
	Write(p []byte) (n int, err error)
}
```

`Read` 把最多 `len(p)` 字节读入 `p`，返回实际读到的字节数 `n` 与可能的错误。<span class="marginnote">Read 的语义细节：`n > 0` 且 `err == nil` 表示「读了但还没到 EOF」；`err == io.EOF` 表示「已到文件末尾」；还可能 `n == 0` 且 `err != nil`（如网络超时）。用 `for` 循环 + `io.EOF` 判断读完，是标准姿势。</span>

**为什么用「接口」而不是「具体的 File 类型」？** 因为「能读」的东西远超文件：

| 类型 | 实现了 | 代表什么 |
| --- | --- | --- |
| `*os.File` | Reader + Writer | 文件 |
| `*bytes.Buffer` | Reader + Writer | 内存缓冲区 |
| `*strings.Reader` | Reader | 内存字符串 |
| `net.Conn` | Reader + Writer | 网络连接 |
| `*gzip.Reader` | Reader | 解压流 |

一个接受 `io.Reader` 的函数，可以喂给它文件、字符串、网络数据——**同一份代码，无数种数据源**。这就是「面向接口编程」的威力。

## 2 io 包的常用工具

`io` 包提供一组基于 Reader/Writer 的**组合函数**，让常见操作一行搞定：<span class="marginnote">`io.Copy` 是最常用的一个：它从 Reader 循环读、写进 Writer，直到 EOF——复制文件、转发网络流都靠它。`io.ReadAll` 读完整输入到内存。这些工具把「循环读、判断 EOF、处理短读」的样板代码收进标准库。</span>

```go
// 把文件内容复制到标准输出
f, err := os.Open("data.txt")
if err != nil { log.Fatal(err) }
defer f.Close()

io.Copy(os.Stdout, f)          // 整文件流式复制

// 读全部到内存
data, err := io.ReadAll(f)     // []byte

// 有限读取
buf := make([]byte, 8)
n, err := io.ReadFull(f, buf)  // 恰好读 8 字节，不足则报错
```

**核心对比：读数据的三种姿势**

| 函数 | 行为 | 适用 |
| --- | --- | --- |
| `io.ReadAll(r)` | 读全部到内存 | 小文件、配置 |
| `io.Copy(dst, src)` | 流式复制 | 大文件、网络转发 |
| `io.ReadFull(r, buf)` | 恰好读满 buf | 定长结构 |

**易错点：** 不要用 `io.ReadAll` 读大文件——它把整个内容放进内存，几 GB 的文件会撑爆内存。大文件的正确姿势是 `io.Copy` 流式处理，或 `bufio.Reader` 分块读。

## 3 bufio：缓冲 I/O 的威力

**`bufio`** 在 Reader/Writer 之上加一层**缓冲区**，减少底层系统调用次数：<span class="marginnote">直接 `file.Read(buf)` 每次调用都要进入系统调用（内核），几百字节一次会非常慢。`bufio.Reader` 一次性从底层读一大块（默认 4096 字节）进缓冲区，后续的「读一行」「读一个字节」都从缓冲区拿，只有缓冲区耗尽才再次系统调用——次数从「每读一次一次」降到「每 4096 字节一次」。</span>

```go
import "bufio"

f, _ := os.Open("data.txt")
defer f.Close()

r := bufio.NewReader(f)
line, err := r.ReadString('\n')   // 读到换行符（含）为止
scanner := bufio.NewScanner(f)
for scanner.Scan() {              // 逐行扫描
	fmt.Println(scanner.Text())
}
if err := scanner.Err(); err != nil { log.Fatal(err) }
```

**`bufio.Scanner`** 是「逐行/逐 token 处理文本」的默认工具——Go 处理日志、CSV、标准输入的第一选择：

```go
sc := bufio.NewScanner(os.Stdin)
for sc.Scan() {
	fmt.Println("输入:", sc.Text())
}
```

**要点：** Scanner 默认以换行符切分，`bufio.ScanLines` 是其分词器；也内置了 `ScanWords`（按单词）、`ScanBytes`（按字节）等。`Scanner` 的最大 token 长度默认 64KB，超长行需调 `Buffer` 上限。

**易错点：** 用完 Scanner 必须检查 `sc.Err()`——输入中途的 IO 错误（磁盘坏块、网络断开）只有在扫描结束后才能被发现。忘记检查 = 吞掉错误。

## 4 公式解析：系统调用次数的压缩

**缓冲 I/O 的性能收益可以用「系统调用次数」量化。** 设要读 $B$ 字节、底层一次系统调用读 $S$ 字节、`bufio` 缓冲块为 $K$ 字节（默认 4096），则

$$
\text{直接读：调用次数} = \lceil B/S \rceil, \qquad
\text{bufio：调用次数} = \lceil B/K \rceil \cdot (\text{每次只从缓冲取所需})
$$

以逐字节读一个 1MB（$B = 10^6$）文件为例：

- **第一步，直接读**：每次 `Read(buf)` 若读 4096 字节，需 $\lceil 10^6 / 4096 \rceil \approx 244$ 次系统调用。
- **第二步，逐字节直接读**：`Read` 一次读 1 字节，需 $10^6$ 次系统调用——灾难。
- **第三步，bufio 逐字节**：底层仍按 4096 块读入（约 244 次系统调用），但「逐字节」全从缓冲区取，**系统调用次数仍约 244**。
- **第四步，结论**：`bufio` 让「逐字节处理」也能享受「块级系统调用」的成本——这正是 `bufio.Reader.ReadByte`、`Scanner` 高效的原因。

这条公式揭示了缓冲的本质：**把高频的小请求，聚合成低频的大请求**——减少上下文切换与内核开销。同理可解释 `bufio.Writer` 写侧（攒满才 flush）与 `strings.Builder`（《基准测试》篇）的加速原理。

## 5 自实现 Reader：接口的实践

理解了 `io.Reader`，就能**自己实现一个**——这正是接口精神的体现：<span class="marginnote">实现 `io.Reader` 只需写一个 `Read` 方法。下面这个 `Counter` 输出 `0, 1, 2, ...` 的无限流，既可用于 `io.Copy`，也可接 `bufio`。因为「能读」的抽象足够小，任何数据源都能参与 I/O 生态。</span>

```go
type Counter struct{ n int }

func (c *Counter) Read(p []byte) (int, error) {
	c.n++
	s := fmt.Sprintf("%d, ", c.n)
	return copy(p, s), nil   // copy 到 p，返回写入字节数
}

c := &Counter{}
io.CopyN(os.Stdout, c, 20)   // 输出 "1, 2, 3, ..."
```

**要点：** `Read` 的协议是「尽力而为」——可以少读、可以返回 `n < len(p)`、用 `io.EOF` 表示结束。实现者要遵守这些约定，调用者（`io.Copy` 等）按约定消费。**接口约定是双边的：实现方保证语义，调用方按语义使用**。

## 6 小结

- **`io.Reader`/`io.Writer`** 各一个方法，统一文件、网络、内存、压缩流——「接口越小越强」的范本。
- `io.Copy` 流式复制、`io.ReadAll` 读全部、`io.ReadFull` 读定长——按场景选。
- **`bufio`** 加缓冲层：`bufio.Reader` 读行、`bufio.Scanner` 逐行扫描（日志、CSV 标配）。
- Scanner 用完要查 `sc.Err()`，否则 IO 错误被吞。
- 缓冲的本质：把小请求聚合成大请求，`bufio` 让逐字节处理也享受块级系统调用。
- 自实现 `Read` 就能接入整个 I/O 生态——接口约定是双边契约。

在下一节，我们把 I/O 与网络结合：**net/http 与 Web 服务开发**——用 Go 写 HTTP 服务，从 `http.Handler` 到路由与中间件。
