---
title: 并发模式：工作池、扇出扇入与并发 Web 爬虫
date: 2026-08-07
---

# 并发模式：工作池、扇出扇入与并发 Web 爬虫

<div class="epigraph">
<p>不要等待；与其等待，不如组织。</p>
<footer>—— 罗勃 · 派克（Rob Pike，并发编程格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从并发模式开始

goroutine 与 channel 是积木，但「用积木搭出什么」才是工程问题。真实系统里反复出现的并发结构——**工作池（worker pool）**、**扇出扇入（fan-out/fan-in）**、**并发 Web 爬虫**——就是 Go 并发编程的「设计模式」。这一篇把它们逐个拆开，讲清「为什么这样组织」而不是「背一套模板」。<span class="marginnote">对标《Go语言圣经》第8章：第8.7 节「并发爬虫」是全书最完整的并发案例，综合了 channel、WaitGroup、限速器与递归遍历。学习并发模式的正确方式是「先看它解决什么问题，再看它怎么用语言原语组装」。</span>

这些模式在本专题的用途：它们把前几篇的并发原语（goroutine、channel、select、WaitGroup）真正用起来，也是理解 `net/http` 服务器「同时服务百万请求」的钥匙——服务器本质就是一个巨大的扇出扇入系统。

## 1 工作池：限流与复用

**工作池（worker pool）** 固定创建 N 个 worker goroutine，通过一个任务 channel 分发工作，worker 处理完再从 channel 取下一个。它解决两个问题：**限制并发数量**（防止开十万个 goroutine 打爆资源）与**复用 worker**（避免反复创建销毁的开销）。<span class="marginnote">工作池的本质是「信号量限流」：`make(chan Task, N)` 里的 N 就是同时进行任务的硬上限。对照《操作系统》的「线程池」与《数据库》的「连接池」，Go 用 channel 天然实现了这一经典结构。</span>

```go
func worker(id int, jobs <-chan int, results chan<- int) {
	for j := range jobs {          // 从任务通道取任务，通道关闭则退出
		results <- j * 2
	}
}

func main() {
	const numJobs = 10
	const numWorkers = 3

	jobs := make(chan int, numJobs)
	results := make(chan int, numJobs)

	for w := 1; w <= numWorkers; w++ {
		go worker(w, jobs, results)   // 启动固定数量的 worker
	}

	for j := 1; j <= numJobs; j++ {
		jobs <- j                     // 派发任务
	}
	close(jobs)                       // 通知 worker：没有更多任务了

	for r := 1; r <= numJobs; r++ {
		<-results                     // 收集结果
	}
}
```

**关键点：** 3 个 worker 并发处理 10 个任务，无论任务有多少，**同时只有 3 个在跑**。`jobs` 通道关闭后，`for range` 循环自然结束，worker 退出。

## 2 扇出扇入：任务分裂与结果汇聚

**扇出（fan-out）**：一个任务源把工作分发给多个处理者（一个 channel 被多个 goroutine 消费）。**扇入（fan-in）**：多个处理者的结果合并到一条输出（多个 channel 汇入一个）。<span class="marginnote">扇出扇入是「分治 + 聚合」的并发表达：把大任务切成小片并行处理，再把结果合并。它与《算法》课程的「归并排序」「分治思想」一一对应，只是把「递归分治」换成了「并发分派」。</span>

```go
// 扇入：把两个输入 channel 合并成一个输出 channel
func fanIn(ch1, ch2 <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for {
			select {
			case v := <-ch1:
				out <- v
			case v := <-ch2:
				out <- v
			}
		}
	}()
	return out
}
```

**要点：** 扇入用 select「同时监听多个来源」，这正是《select 多路复用》篇的核心应用。扇出则相反——多个 goroutine 共享一个输入 channel 消费任务，`range` 循环自动保证「每个任务只被一个 worker 处理」。

**核心对比：扇出 vs 扇入**

| 维度 | 扇出 | 扇入 |
| --- | --- | --- |
| 方向 | 一源 → 多消费 | 多源 → 一汇合 |
| 工具 | 一个 channel + 多 goroutine | 多个 channel + select |
| 语义 | 任务分发 | 结果合并 |
| 类比 | 主管分派工作 | 秘书汇总报表 |

## 3 并发 Web 爬虫：综合案例

《Go语言圣经》第8.6 节给出了一个经典案例：并发抓取网页、递归提取链接、用**限速（throttling）**控制请求频率，避免打爆服务器：<span class="marginnote">这个例子把几乎全部并发原语用了一遍：`tokens` 带缓冲 channel 当信号量限速、`seen` map 去重、`worklist` channel 传递待爬链接。读它时注意：每部分都是前几篇讲过的原语，组合起来却是一台完整的并发机器。</span>

```go
// 限速信号量：同时最多 20 个 HTTP 请求
var tokens = make(chan struct{}, 20)

func crawl(url string) []string {
	tokens <- struct{}{}      // 获取令牌：满了就阻塞（限流）
	links, err := extract(url)
	<-tokens                  // 释放令牌
	if err != nil {
		return nil
	}
	return links
}

func main() {
	worklist := make(chan []string)
	go func() { worklist <- []string{"https://golang.org"} }()

	seen := make(map[string]bool)
	for list := range worklist {
		for _, link := range list {
			if !seen[link] {
				seen[link] = true
				go func(link string) {
					worklist <- crawl(link)
				}(link)
			}
		}
	}
}
```

**三个设计要点：**

- **限速**：`tokens` 缓冲为 20 的 channel 充当信号量——拿令牌才发请求，令牌有限所以并发请求有上限。这是「带缓冲 channel 当计数器」的经典用法。
- **去重**：`seen` map 记录已爬链接，避免重复抓取。注意它只在主 goroutine 里读写，无竞争。
- **递归分发**：每个新链接启动一个 goroutine 抓取，结果通过 `worklist` channel 传回主循环，主循环再派发——天然的「广度优先 + 并发」。

**辨析｜易错点：** 这个模式有一个**计数陷阱**：主循环 `for list := range worklist` 什么时候结束？答案在《Go语言圣经》中用「当前处于处理中的 URL 计数」解决——当计数归零且没有新任务时关闭 worklist。若没有这个终止条件，程序会永远等待，这是并发程序最常见的「泄漏 + 死等」。

## 4 公式解析：工作池的并发度

**工作池的「并发度」等于 worker 数量，而吞吐由「任务处理时间」与「worker 数」共同决定。** 设单个任务处理时间为 $T$，worker 数为 $N$，任务总数为 $M$，则完成全部任务的时间

$$
T_{\text{total}} = \left\lceil \frac{M}{N} \right\rceil \cdot T
$$

以 `numJobs=10`、`numWorkers=3`、`T=1ms` 为例：

- **第一步，理论下限**：$M/N = 10/3 \approx 3.33$，取整为 4 轮。
- **第二步，总耗时**：$4 \times 1\text{ms} = 4\text{ms}$（理想化、无调度开销）。
- **第三步，与串行对比**：串行需 $10 \times 1\text{ms} = 10\text{ms}$，并发约提速 2.5 倍。
- **第四步，边际效应**：worker 从 3 增到 10，理想时间降到 1ms，但**调度与竞争开销随 worker 增多而上升**——这就是「并发不总是更快」的量化来源，也是 Amdahl 定律在并发编程中的体现。

这条公式的启示：**worker 数不是越大越好**。它受限于任务间的依赖、共享资源的竞争（本例中磁盘/网络 IO）与 CPU 核数。选择 worker 数要用《基准测试与 pprof》篇的实测数据说话，而不是拍脑袋。

## 5 小结

- **工作池**：固定 N 个 worker 消费任务通道，`close(jobs)` 通知结束，天然限流与复用。
- **扇出**：一源多消费；**扇入**：多源用 select 合并，是「分治 + 聚合」的并发表达。
- **并发爬虫**综合案例：`tokens` 信号量限速、`seen` map 去重、worklist 递归分发。
- **终止条件是关键**：并发程序必须有明确的「何时结束」，否则会永远等待或泄漏。
- worker 数 = 并发度；「worker 越多越快」不成立，受依赖、竞争与核数约束。
- 读并发案例的正确姿势：先看「解决什么问题」，再看「用哪些原语组装」。

在下一节，我们从并发回到工程组织：**包与模块——go mod 依赖管理与版本语义**，学会如何让代码被别人复用、如何复用别人的代码。
