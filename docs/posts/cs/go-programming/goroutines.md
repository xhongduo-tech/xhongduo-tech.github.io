---
title: goroutine 与并发基础
date: 2026-08-07
---

# goroutine 与并发基础

<div class="epigraph">
<p>不要通过共享内存来通信，而要通过通信来共享内存。</p>
<footer>—— Go 语言格言（Do not communicate by sharing memory; instead, share memory by communicating）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从 goroutine 开始

前面的章节都假设程序是单线程的：一条路走到黑。但真实世界——Web 服务的成千上万个请求、分布式系统的节点间消息、大数据的并行计算——天然是「同时发生」的。Go 用 **goroutine** 让并发成为语言的一等公民：一个 `go` 关键字就能启动一个轻量并发单位。这一篇从 goroutine 的机制讲起，为后面的 channel、select、数据竞争与 sync 包铺设地基。<span class="marginnote">对标《Go语言圣经》第8章：这是全书的「并发之章」，也是 Go 与 C、Java 相比最革命性的部分。第8.1节从「并发的两种模型——多线程与消息传递」讲起，而 goroutine 正是 Go 对「轻量并发单位」的回答。</span>

goroutine 在课程坐标上连接着多个层面：往下，它建立在《计算机组成原理》的线程/进程之上；往上，它是《操作系统》课程「并发与调度」的 Go 具体化；而云原生世界——Kubernetes、etcd——正是靠海量 goroutine 支撑「同时处理无数请求」。

## 1 进程、线程与 goroutine

回顾一下并发的基本单位：

- **进程（process）**：操作系统分配资源的最小单位，拥有独立内存空间。
- **线程（thread）**：CPU 调度的最小单位，共享进程内存。
- **goroutine**：Go 运行时的**用户态并发单位**，跑在线程之上，由 Go 调度器（scheduler）管理。

**核心对比：goroutine 与线程的差别**

| 维度 | goroutine | 系统线程 |
| --- | --- | --- |
| 初始栈 | 约 2KB，可动态增长 | 约 1–8MB，固定 |
| 创建成本 | 极低（微秒级） | 高（内核切换） |
| 调度 | Go 运行时调度器（M:N） | 操作系统内核调度 |
| 数量级 | 可轻易开数十万个 | 上千个就吃力 |

<span class="marginnote">goroutine 的栈从 2KB 起步、按需增长到至多 1GB，这让「开十万个 goroutine」成为可能——而同样数量的线程会耗尽内存。Go 采用 <strong>M:N 调度</strong>：M 个 goroutine 被调度到 N 个系统线程上，由运行时在多个线程之间分配，程序员无需关心线程细节。</span>

启动一个 goroutine 只需要 `go` 关键字：

```go
func main() {
	go fmt.Println("我在 goroutine 里")
	fmt.Println("我在 main 里")
}
```

程序可能打印「main 在 goroutine 里」，也可能只打印 main 那一行——**`main` 返回时，程序就结束，其它 goroutine 被直接杀死**。这是初学最容易踩的坑。

## 2 并发 vs 并行

Rob Pike 的著名区分：**并发（concurrency）是结构，并行（parallelism）是执行**。并发是「如何设计同时进行的事」，并行是「多核真的同时算」。

- **并发**：多个任务以「交织」的方式推进，单核也能并发（时间片切换）。
- **并行**：多个任务在同一时刻真正同时执行，需要多核。

goroutine 让并发**结构化**：你不关心「现在真在跑几个」，只关心「有这些独立的任务」。调度器决定它们是否并行——在有多个核的机器上，运行时自动把 goroutine 分布到多个线程上并行执行。对照《操作系统》课程的时间片轮转：goroutine 是「语言级的任务」，线程是「内核级的任务」，Go 调度器在两者之间架桥。

**辨析｜易错点：** 并发程序**不保证**更快。若任务是串行依赖的、或受限于单一瓶颈（如磁盘 IO），并发只会增加调度开销。并发解决的是「结构清晰与吞吐」，不是「单点加速」——这一点在《基准测试与 pprof》篇会用数据验证。

## 3 goroutine 的生命周期

goroutine 的生命由**它自己**决定，没有「外部终止」的原语（`kill`）。它会在三种情况结束：

- 函数执行完毕返回。
- 通过 channel 收到关闭信号（第8章后续）。
- 程序整体退出（`main` 返回或 `os.Exit`）。

一个 goroutine **会一直运行到被阻塞或返回**，哪怕它在等待一个永远不会来的事件——这就是「goroutine 泄漏」：

```go
func main() {
	ch := make(chan int)
	go func() {
		// 永远阻塞：没有生产者往 ch 发数据
		<-ch
	}()
	// main 很快返回，程序退出——但若这是长期服务，该 goroutine 永久泄漏
}
```

**要点：** 泄漏的 goroutine 占用栈内存与调度器资源，长期运行的服务会逐渐退化。标准做法是给 goroutine 一个「退出信号」通道，或用 `context` 取消——后者在《context 上下文与并发取消》篇专门讲解。

## 4 调度与 GOMAXPROCS

Go 运行时的调度是 **M:N** 模型，由调度器在系统线程之间分配 goroutine。`GOMAXPROCS` 控制「同时运行用户代码的系统线程数」（默认等于 CPU 核数）：<span class="marginnote">`runtime.GOMAXPROCS(n)` 可以查询或设置。它不限制 goroutine 总数——goroutine 是「就绪任务」，`GOMAXPROCS` 决定「同时有几个真在跑」。理解这个区别，就理解了并发与并行的分工。</span>

```go
fmt.Println(runtime.GOMAXPROCS(0))   // 打印当前值，如 8（8 核）

runtime.GOMAXPROCS(4)                // 设为 4
```

**关键机制：** 当一个 goroutine 阻塞在系统调用（如磁盘读取）或 channel 操作上时，Go 调度器会**让出线程**给其它 goroutine——所以即使只有一个线程，goroutine 也能「看起来同时推进」。这就是「并发 ≠ 并行」的具体呈现：单核照样能并发多个 goroutine，只是不并行。

**易错点：** 不要手动调大 `GOMAXPROCS` 试图「用满多核」——默认值已按核数设置。改动它通常无益，反而引入不可预测性。真正影响吞吐的是「任务如何划分」，而不是「几个线程在跑」。

## 5 用 goroutine 组织并发任务

goroutine 最常见的用途是「并行执行独立任务，最后汇总」。一个简单的「并发下载多个 URL 并打印长度」的例子：

```go
func main() {
	urls := []string{"https://go.dev", "https://golang.org"}
	done := make(chan struct{})

	for _, u := range urls {
		go func(u string) {
			resp, err := http.Get(u)
			if err != nil {
				fmt.Println(u, err)
			} else {
				fmt.Println(u, resp.StatusCode)
				resp.Body.Close()
			}
			done <- struct{}{}   // 通知完成
		}(u)
	}

	// 等所有 goroutine 完成
	for range urls {
		<-done
	}
	fmt.Println("全部完成")
}
```

这里用了 `done` 通道让 main 等待所有 goroutine——这是「等待组」的最简原型，正式版本是 `sync.WaitGroup`（第9章）。模式是清晰的：**每个 goroutine 干一件事，干完发信号，main 收齐所有信号再继续**。

**重点：** 循环变量传参 `go func(u string){...}(u)` 是必须的——直接引用循环变量 `u` 会共享同一个变量，goroutine 可能读到被改写的值（Go 1.22 之前尤其如此）。这与《函数》篇的闭包捕获陷阱同源。

## 6 小结

- **goroutine** 是 Go 运行时的轻量并发单位：2KB 起步的栈、M:N 调度、可开数十万。
- `go f()` 启动并发任务；**`main` 返回即程序结束**，其它 goroutine 被杀死。
- **并发是结构、并行是执行**：单核可并发，并行才需要多核。
- goroutine 无法被外部强杀，靠函数返回、channel 信号或 context 取消结束。
- **`GOMAXPROCS`** 控制并行线程数，默认即最优，一般不要手动改。
- 用 channel 发信号组织「启动多个任务 → 全部完成」的并发模式。

在下一节，我们深入并发通信的主干：**channel——无缓冲与有缓冲、通道方向**，看 goroutine 之间如何优雅地传数据。
