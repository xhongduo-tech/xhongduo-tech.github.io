---
title: CSP（通信顺序进程）与 Go 的 channel
date: 2026-08-07
---

# CSP（通信顺序进程）与 Go 的 channel

<div class="epigraph">
<p>不要通过共享内存来通信；相反，要通过通信来共享内存。</p>
<footer>—— 罗勃 · 派克（Rob Pike，Go 语言格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第13章 + Go CSP 模型 ｜ 2026-08-07</p>
</div>

## 为什么从 CSP 开始

消息传递有两个大流派：**Actor**（上一节，异步邮箱、自治实体）与 **CSP（通信顺序进程，Communicating Sequential Processes）**——由霍尔（C. A. R. Hoare）1978 年提出。CSP 与 Actor 的根本区别：**通信通过命名的通道（channel）进行，且默认同步（握手）**。Go 语言把 CSP 带进主流——goroutine + channel 成了 Go 并发编程的标志。「通过通信共享内存」这句格言，正是 Go 对 CSP 哲学的凝练。这一节看 CSP 的思想、channel 的机制、以及它与 Actor 的对比。<span class="marginnote">CSP vs Actor 一句话：CSP 里「通道是显式的、收发是同步的」，Actor 里「邮箱是隐式的、收发是异步的」。CSP 的同步握手让「数据流动」可预测；Actor 的异步让「吞吐」最大化。Go 选了 CSP（同步 channel），Erlang 选了 Actor（异步邮箱）。</span>

## 1 CSP 的基本思想

**CSP（通信顺序进程）**：并发程序 = 一组**顺序进程**（sequential processes），进程间通过**命名的通道**同步通信。

三个要素：

- **进程（process）**：独立的顺序执行单元（Go 里是 goroutine）。
- **通道（channel）**：命名的通信管道——进程经它发送/接收值。
- **同步通信**：`send` 与 `receive` 必须**同时发生**（握手）——发送方等接收方、接收方等发送方。

```
P1 ──channel──▶ P2    （P1 发、P2 收，同步握手）
```

## 2 Go 的 goroutine 与 channel

**goroutine**：Go 的轻量线程——由 Go 运行时调度，栈初始极小（几 KB），可并发成千上万。

**channel**：goroutine 间通信的管道——`make(chan T)` 创建，`<-` 发送/接收。

```go
ch := make(chan int)          // 无缓冲通道

go func() {                    // goroutine
    ch <- 42                   // 发送：阻塞直到有人接收
}()

x := <-ch                      // 接收：阻塞直到有人发送
fmt.Println(x)                 // 42
```

无缓冲 channel 是**同步**的：`ch <- 42` 阻塞，直到 `<-ch` 执行——握手完成双方继续。<span class="marginnote">无缓冲 channel 的同步语义是 Go 的「裸 CSP」：发送与接收必须同时就绪，否则一方阻塞。这是 Go 并发的「通信即同步」——数据流动本身就完成了同步，无需额外的锁。「通过通信共享内存」：数据经 channel 传递（所有权转移），而非多线程抢同一内存。</span>

## 3 缓冲 channel 与 select

**缓冲 channel（buffered channel）**：`make(chan T, N)` 带容量 N 的队列——发送方在队列未满时**不阻塞**（异步）。这打破了「纯同步」，成为「有界异步」：缓冲满则阻塞发送，空则阻塞接收。

**select**：Go 的多通道选择——从多个就绪的 channel 操作中任选一个执行：

```go
select {
case v := <-ch1:
    // ch1 有数据
case ch2 <- v:
    // ch2 可写
case <-time.After(1 * time.Second):
    // 超时
}
```

`select` 是「非确定性选择」的经典实现——与卫式命令的 `do...od` 一脉相承：多个就绪分支任选其一。<span class="marginnote">`select` 是 CSP 的「选择原语」：当多个通道都可用时，Go 随机选一个（公平性）。它让「等待多个事件」成为一等操作——超时、取消、多路复用都靠它。这直接继承了霍尔 CSP 的「交替（alternative）」构造，也呼应迪杰斯特拉卫式命令的非确定性选择。</span>

## 4 公式解析：通道握手的语义

CSP 通道的同步语义可以用「值传递 + 同时性」刻画。设通道 $c$，发送方 $P$、接收方 $Q$：

$$
\text{send}(c, v) \parallel \text{receive}(c) \;\Rightarrow\; Q \text{ 得到 } v，P \text{ 与 } Q \text{ 同时解除阻塞}
$$

$$
\text{无缓冲}: t_{\text{send}} = t_{\text{receive}} \quad \text{（同一时刻握手）}
$$

三步拆解：

- **第一步，配对**：send 与 receive 配对才发生——单独一个会阻塞（发送方等接收、接收方等发送）。
- **第二步，值传递**：配对的瞬间，值 $v$ 从发送方传给接收方——数据在握手时流动。
- **第三步，同时解阻**：握手后双方**同时**继续——不存在「一方先走」的不对称。**「通信即同步」**：通道不仅传数据，还完成了两个 goroutine 的同步（它们在同一时刻对齐）。这是 CSP 与「锁 + 共享」最根本的不同。

**辨析｜易错点：** 无缓冲 channel 的阻塞是**双向**的：发送阻塞等接收，接收阻塞等发送。若只发不收（或只收不发），goroutine 永久阻塞——这是「goroutine 泄漏」的常见来源。**「channel 操作必须成对」**（就像 P/V 成对），否则死锁或泄漏。

## 5 CSP 与 Actor 的对比与工程实践

| 维度 | CSP（Go） | Actor（Erlang） |
| --- | --- | --- |
| 通信媒介 | 显式通道 | 隐式邮箱 |
| 同步方式 | 默认同步（握手） | 默认异步 |
| 缓冲 | 可选（有界/无界） | 邮箱天然缓冲 |
| 消息选择 | `select` 多路 | 模式匹配选择性接收 |
| 进程模型 | goroutine（运行时调度） | 进程/actor（轻量） |
| 适用 | 高吞吐并发服务 | 高容错分布式系统 |

<span class="marginnote">工程经验：Go 的 CSP 风格适合「单机内高并发」（goroutine + channel 轻量高效），Erlang 的 Actor 适合「分布式容错」（进程隔离 + 监督树）。现代框架常在两者间融合——「通道传递消息 + 进程监督恢复」各取所长。无论哪种，「不共享数据」都是共同底线。</span>



## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| CSP（通信顺序进程） | # CSP（通信顺序进程）与 Go 的 channel |
| 顺序进程 | # CSP（通信顺序进程）与 Go 的 channel |
| 命名的通道 | CSP（通信顺序进程）：并发程序 = 一组顺序进程（sequential processes），进程间通过命名的通道同步通信。 |
| 进程（process） | 进程（process）：独立的顺序执行单元（Go 里是 goroutine）。 |
| 通道（channel） | 通道（channel）：命名的通信管道——进程经它发送/接收值。 |
| 同步通信 | CSP（通信顺序进程）：并发程序 = 一组顺序进程（sequential processes），进程间通过命名的通道同步通信。 |
| 同时发生 | 同步通信：send 与 receive 必须同时发生（握手）——发送方等接收方、接收方等发送方。 |
| goroutine | 进程（process）：独立的顺序执行单元（Go 里是 goroutine）。 |
| channel | # CSP（通信顺序进程）与 Go 的 channel |
| 缓冲 channel（buffered channel） | 缓冲 channel（buffered channel）：make(chan T, N) 带容量 N 的队列——发送方在队列未满时不阻塞（异步）。这打破 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **CSP** = 顺序进程 + 命名通道 + 同步通信——「通信即同步」。
- **Go** 的 goroutine（轻量线程）+ channel（通道）+ `select`（多路选择）把 CSP 带入主流。
- 无缓冲 channel 同步握手；缓冲 channel 有界异步；`select` 是非确定性选择（呼应卫式命令）。
- CSP vs Actor：显式通道 vs 隐式邮箱、同步 vs 异步——Go 适合单机高并发，Erlang 适合分布式容错。

在下一节，我们将看并发抽象的另一条路线——**软件事务内存（STM）与异步编程模型（async/await）**。
