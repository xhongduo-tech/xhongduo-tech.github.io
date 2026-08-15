---
title: sync 包：Mutex、RWMutex、WaitGroup 与 Once
date: 2026-08-07
---

# sync 包：Mutex、RWMutex、WaitGroup 与 Once

<div class="epigraph">
<p>互斥锁是并发共享变量的护栏：一次只让一个 goroutine 通过。</p>
<footer>—— 并发编程共识（Locks guard shared state）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从 sync 包开始

channel 适合「goroutine 之间传值」，但还有一类常见需求：**多个 goroutine 共享同一个变量**（计数器、配置、缓存），需要保证读写安全。Go 的答案是 `sync` 包——一组**同步原语**：`Mutex`（互斥锁）、`RWMutex`（读写锁）、`WaitGroup`（等待组）、`Once`（只执行一次）。它们直接服务于「并发共享内存」这个最危险的领域。<span class="marginnote">对标《Go语言圣经》第9章「使用共享变量实现并发」：这一章专门处理「必须共享内存」的场景。Go 虽然推崇 channel，但 `sync.Mutex` 是处理共享状态的标准工具——官方格言是「channel 与锁各有用武之地，选择适合场景的那个」。</span>

sync 包在专题中与《数据竞争与竞态检测》互为表里：数据竞争篇讲「问题与检测」，本篇讲「解决工具」。理解它们，才能写出真正线程安全的代码。

## 1 Mutex：互斥锁

**`sync.Mutex`** 保证「同一时刻只有一个 goroutine 持有锁」。用 `Lock`/`Unlock` 围住临界区（critical section）：<span class="marginnote">对照《操作系统》课程的「临界区」概念：多个 goroutine 都要读写的共享变量所在区域，必须互斥进入。Mutex 是最经典的互斥实现——谁能 `Lock` 成功谁就进入，其余 goroutine 阻塞等待。</span>

```go
var (
	mu      sync.Mutex
	counter int
)

func inc() {
	mu.Lock()
	counter++        // 临界区：读-改-写，现在安全了
	mu.Unlock()
}
```

**重点：** 没加锁时 `counter++` 是「读-改-写」三步，可能被并发打断导致丢失更新（数据竞争）；加锁后，三步要么整体执行、要么整体不执行。锁把「必须原子化」的代码块保护了起来。

**惯用法：立即 `defer` 解锁**——在拿到锁后紧跟 `defer mu.Unlock()`，保证无论函数怎么返回（包括 panic）锁都会被释放：

```go
func withLock() {
	mu.Lock()
	defer mu.Unlock()
	// 临界区
	// 后面无论怎样 return，defer 都会解锁
}
```

**辨析｜易错点：** 忘记解锁会导致**死锁**（其它 goroutine 永远等不到锁）。**用 defer 解锁是安全默认**。另外，`sync.Mutex` **不可复制**——把含锁的结构体按值拷贝，会复制锁状态，是隐患（`go vet` 能检查出来，见《go 工具链》篇）。

## 2 RWMutex：读写锁

**`sync.RWMutex`** 区分读者与写者：多个读者可以**同时**持有读锁，写者必须独占。

```go
var (
	mu   sync.RWMutex
	conf map[string]string
)

func get(key string) string {
	mu.RLock()          // 读锁：多个读者可同时进入
	defer mu.RUnlock()
	return conf[key]
}

func set(key, val string) {
	mu.Lock()           // 写锁：独占
	defer mu.Unlock()
	conf[key] = val
}
```

**核心对比：Mutex vs RWMutex**

| 维度 | Mutex | RWMutex |
| --- | --- | --- |
| 写者独占 | 是 | 是 |
| 读者互斥 | 是（写者读者都独占） | 否（读者可共存） |
| 适用场景 | 读写均衡 | 读多写少 |
| 性能 | 简单、开销小 | 读并发高、写独占 |

**要点：** 「读多写少」是 RWMutex 的甜区——比如缓存配置，绝大多数请求是读，偶尔更新。多读者可并发进入，吞吐远高于 Mutex。但若写操作频繁，RWMutex 的维护开销反而可能超过 Mutex 的简单实现——**选型要看实际读写比例**。

**易错点：** `RLock` 必须配 `RUnlock`，`Lock` 必须配 `Unlock`，**不能混用**。一个 goroutine 拿着读锁又想升级成写锁会死锁。

## 3 WaitGroup：等待一组 goroutine

**`sync.WaitGroup`** 等待「一组 goroutine 全部完成」——这是「并发启动多个任务，全部结束后继续」的标准工具：

```go
func main() {
	var wg sync.WaitGroup

	for i := 1; i <= 5; i++ {
		wg.Add(1)          // 计数器 +1
		go func(n int) {
			defer wg.Done()   // 计数器 -1（相当于 wg.Add(-1)）
			fmt.Println("任务", n, "完成")
		}(i)
	}

	wg.Wait()              // 阻塞，直到计数器归零
	fmt.Println("全部完成")
}
```

**三个方法：** `Add(delta)` 增加计数器、`Done()` 减少（内部即 `Add(-1)`）、`Wait()` 阻塞到归零。

**易错点：** `Add` 必须在 `go` 启动**之前**调用——若在 goroutine 内部 `Add`，可能发生「`Wait` 已经检查了计数器却还没加到」的竞争。惯用法是启动前 `Add(1)`、goroutine 内 `defer Done()`。

**辨析｜易错点：** WaitGroup 的计数器不能为负：`Done` 超过 `Add` 会 panic。也不要在 `Wait` 之后复用同一个 WaitGroup 而不重新 `Add`——若在 `Wait` 返回前再次 `Add`，行为未定义。

## 4 Once：只执行一次

**`sync.Once`** 保证一个函数「整个程序生命周期内只执行一次」——哪怕被多个 goroutine 并发调用：<span class="marginnote">Once 是实现「惰性单例」的标准工具：昂贵的初始化（加载配置、建立连接池）只在第一次真正需要时执行，且并发调用也安全。它内部用 mutex + 状态位保证「恰好一次」，是「线程安全的 double-checked locking」的 Go 化。</span>

```go
var (
	once   sync.Once
	config *Config
)

func getConfig() *Config {
	once.Do(func() {
		config = loadConfig()   // 只执行一次
	})
	return config
}
```

**要点：** 无论多少 goroutine 同时调用 `getConfig`，`loadConfig` 只运行一次，其余 goroutine 等待这次执行完成后再读取。这比「自己写 `if config == nil { ... }` 加锁」更简洁、更不易错。

## 5 核心对比：sync 原语选型

| 原语 | 解决的问题 | 典型场景 |
| --- | --- | --- |
| `Mutex` | 共享变量互斥写 | 计数器、缓存更新 |
| `RWMutex` | 读多写少共享 | 配置、词典、路由表 |
| `WaitGroup` | 等待一组任务完成 | 并发任务聚合、并行计算 |
| `Once` | 初始化恰好一次 | 单例、惰性加载配置 |

**选型直觉：**

- 需要「传值 + 同步」→ 用 channel。
- 需要「共享同一个变量、互斥修改」→ 用 Mutex。
- 读远多于写 → 用 RWMutex。
- 只是「等一组 goroutine 干完」→ 用 WaitGroup。
- 只是「某个初始化只做一次」→ 用 Once。

**易错点：** 锁与 channel 不是二选一的对立，而是互补：channel 适合「数据流经」，锁适合「状态驻留」。一个并发程序常常两者并用——channel 协调任务分发，锁保护共享状态。

## 6 小结

- **Mutex** 互斥锁：`Lock`/`Unlock` 围住临界区，立即 `defer Unlock` 是安全默认。
- **RWMutex** 读写锁：多读者可共存、写者独占，适合读多写少。
- **WaitGroup** 等待组：`Add` 在启动前、`Done` 用 defer、`Wait` 阻塞到归零。
- **Once** 只执行一次：惰性单例的标准工具，并发调用安全。
- `sync` 原语**不可复制**；go vet 会检查复制锁的隐患。
- 选型：channel 传值、锁护状态、WaitGroup 聚合、Once 初始化。

在下一节，我们把并发积木拼成真实模式：**并发模式——工作池、扇出扇入与并发 Web 爬虫**。
