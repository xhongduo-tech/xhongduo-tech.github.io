---
title: 数据竞争与竞态检测
date: 2026-08-07
---

# 数据竞争与竞态检测

<div class="epigraph">
<p>并发不是并行，而是关于结构的问题；并行只是关于执行的问题。</p>
<footer>—— 罗勃 · 派克（Rob Pike，Concurrency is not Parallelism）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从数据竞争开始

并发程序最大的敌人不是「死锁」而是**数据竞争（data race）**——两个 goroutine 同时读写同一个变量，结果取决于调度顺序，错误时有时无、难以复现。Go 用两件武器对抗它：**内存模型**定义什么行为是合法的，**竞态检测器（race detector）**把隐藏的竞争在测试阶段揪出来。这一章解决「为什么我的并发程序时对时错」这个最折磨人的问题。<span class="marginnote">对照第二级《操作系统》与第三级《并发与并行编程》课程：数据竞争是「访问同一内存位置时，至少有一个是写，且无同步关系」。它不是 bug 的一种，而是一整类 bug 的共同根源。Herlihy 的《多处理器编程的艺术》里对「安全性」的要求，第一句话就是「禁止数据竞争」。</span>

## 1 什么是数据竞争

**数据竞争（data race）** 的定义：两个或多个 goroutine 并发访问同一个变量，**至少一个是写操作，且没有任何同步机制**保证它们的顺序。

```go
var counter int

func inc() {
	counter++          // 读-改-写三步，不是原子的！
}

func main() {
	for i := 0; i < 1000; i++ {
		go inc()       // 1000 个 goroutine 同时改 counter
	}
	time.Sleep(time.Second)
	fmt.Println(counter)   // 几乎不可能等于 1000
}
```

`counter++` 看似一行，实际是三步：读旧值、加一、写回。两个 goroutine 可能同时读到同一个旧值、各自加一、先后写回——结果只加了一次。最终 `counter` 是小于等于 1000 的某个不确定数。

**辨析｜易错点：** 数据竞争**不总是**表现为错误结果。它可能只在特定调度下出错，可能仅在特定硬件上出错，还可能由于编译器重排而「偶然正确」。这就是它难以排查的原因——**没出错的测试不能证明没有竞争**。

## 2 Go 内存模型：happens-before

Go 内存模型定义了「哪些内存访问顺序是保证的」，核心概念是 **happens-before（先行发生）**：

若操作 A **happens-before** 操作 B，则 A 对内存的写入在 B 读取时**必然可见**。
同一 goroutine 内，代码按书写顺序 happens-before。
- **channel 通信建立 happens-before**：无缓冲通道的「发送完成」happens-before「接收开始」。
- 有缓冲通道：向容量为 N 的通道发送第 K 个值，happens-before 第 K+N 个接收完成。
- `sync.Mutex` 的 Lock 之前的所有写入，happens-before 下一个 goroutine 的 Unlock 之后的读取。

这些规则给出了「安全的并发代码」的判定标准：**只要两个访问之间能建立 happens-before 链，就不算竞争**。反过来，若无同步关系，编译器与硬件都允许重排，结果不确定。<span class="marginnote">happens-before 是并发正确性的通用语言：Java 内存模型、C++ 内存模型都有等价概念。Go 的模型刻意保持最小——只承认 channel、sync 包、`sync/atomic` 等少量同步原语，让程序员更容易判断「我的代码安全吗」。</span>

## 3 竞态检测器：go test -race

Go 自带**竞态检测器（race detector）**，一行参数即可启用：

```bash
$ go test -race ./...
$