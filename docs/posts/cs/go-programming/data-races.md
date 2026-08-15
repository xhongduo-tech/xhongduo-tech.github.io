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
WARNING: DATA RACE
Read at 0x00c000012300 by goroutine 7:
  main.inc()
      data.go:6 +0x5e
  ...
```

竞态检测器在 `go test`、`go run`、`go build` 都能用，跑出的程序带**插桩**——每次内存访问都会记录「谁在何时读了/写了哪块内存」，若发现「两个 goroutine 未同步地访问同一地址且至少一个写」，立刻报 `WARNING: DATA RACE` 并给出完整的访问栈。

**要点：** 竞态检测器**不查找死锁**，只查**数据竞争**；它需要程序**实际运行到**竞争路径才触发。所以：

1. 测试要**覆盖到并发的实际执行**——一个从未并行跑的测试，检测器无事可做。
2. 检测器会**放大内存开销**（约 10 倍）与运行时间（约 2–20 倍），通常只在 CI 或测试阶段开。
3. **没有报告的测试不能证明没有竞争**——只是「这次没跑到」。

**辨析｜易错点：** 竞态检测器依赖**happens-before 关系**判定竞争。若两段访问之间通过 channel、Mutex、WaitGroup 建立了同步，就不算竞争；反之即使「看起来不相关」，只要没有同步关系就是竞争。因此「加锁保护」或「channel 传递」是消灭竞争的唯一正道，`time.Sleep` 之类的「碰运气等待」**不算同步**，检测器依然会报。

## 4 内存模型的规则与应用

《Go 内存模型》文档给出了判定「是否竞争」的完整规则，核心是 happens-before 的**传递性**与**同步原语的建立**：

**同步原语建立的 happens-before 规则：**

| 原语 | 规则 |
| --- | --- |
| 无缓冲 channel | 发送完成 happens-before 接收开始 |
| 有缓冲 channel | 第 $k$ 次发送 happens-before 第 $k+\text{cap}$ 次接收 |
| `sync.Mutex` | 第 $n$ 次 `Unlock` happens-before 第 $n+1$ 次 `Lock` |
| `sync.WaitGroup` | `Done` happens-before 对应的 `Wait` 返回 |
| `sync/atomic` | 原子读 happens-before 原子写（`atomic` 提供最小同步） |

**重点：** 这些规则就是「安全并发代码」的检查清单——两个访问之间能建立任意一条 happens-before 链，就安全；否则就是竞争。用「锁 + 共享变量」或「channel + 值传递」组织并发，本质都是「建立 happens-before」。

**辨析｜易错点：** `sync/atomic` 提供的是**原子性**（不会撕裂），但不保证 happens-before 的**可见性**（缓存/重排）。只用 `atomic.LoadInt64`/`atomic.StoreInt64` 做标志位是安全的，但「先写普通变量、再用 atomic 发信号」可能仍存在可见性问题——需要时用 `sync/atomic` 的完整原语或 fall back 到 Mutex。

## 5 公式解析：竞争窗口的概率

**数据竞争的危害可以用「竞争窗口（race window）」建模**：两个 goroutine 对同一变量的「读-改-写」若在时间上交叠，就发生竞争。设每个 goroutine 的临界操作耗时为 $\Delta$，调度周期为 $T$（$\Delta \ll T$），则两次操作在窗口内交叠的概率

$$
P_{\text{race}} \approx \frac{2\Delta}{T}
$$

以 `counter++` 为例（$\Delta$ 为读-改-写三步的纳秒级时长）：

- **第一步，窗口分析**：两个 `inc()` 若读操作交错在对方「写回之前」——即第二步的写落在第一步的读-写之间——就丢失一次更新。
- **第二步，概率表达**：交叠窗口约 $2\Delta$（两个 goroutine 各持一个窗口），调度周期 $T$ 越大，交叠概率越小——**但这只是概率，永远不为零**。
- **第三步，放大器**：1000 个 goroutine 并发时，交叠机会成百上千倍放大，出错「几乎必然」。
- **第四步，结论**：竞争是**概率性**的——时对时错、难以复现，正是它比死锁更可怕的原因。检测器 + 内存模型规则是「把概率降为零」的工程手段。

## 6 小结

- **数据竞争** = 两个 goroutine 未同步访问同一变量且至少一个写；结果取决于调度，时对时错。
- **happens-before** 是判定核心：能建立同步链就安全，否则竞争。
- **竞态检测器** `go test -race` 在运行时插桩检测，报告竞争处的完整访问栈。
- 检测器需程序**实际运行**到竞争路径；无报告 ≠ 无竞争；`time.Sleep` 不算同步。
- channel、Mutex、WaitGroup、atomic 是建立 happens-before 的四类原语。
- 竞争是**概率性**的，窗口越大越易触发——靠检测器与内存模型把它降为零。

在下一节，我们用工具系统地解决共享状态：**sync 包——Mutex、RWMutex、WaitGroup 与 Once**。