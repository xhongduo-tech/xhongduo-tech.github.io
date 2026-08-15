---
title: context 上下文与并发取消
date: 2026-08-07
---

# context 上下文与并发取消

<div class="epigraph">
<p>context 让「取消」与「截止时间」成为请求生命周期的第一等公民。</p>
<footer>—— Go 上下文设计（The context package）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第8章 + context 官方博客 ｜ 2026-08-07</p>
</div>

## 为什么从 context 开始

goroutine 一旦启动就很难「礼貌地停下」——没有 `kill` 原语，只能靠「信号」。真实系统里这种需求无处不在：HTTP 客户端超时了要停掉慢请求、用户点了取消要停掉整个操作、服务关停要把正在处理的请求优雅收尾。**`context` 包**正是 Go 官方的答案：一个贯穿调用链的「取消信号 + 截止时间 + 请求值」载体。<span class="marginnote">对标 Go 官方博客《Using contexts correctly》与《The context package》：context 最早由 Sameer Ajmani 为 Google 内部需求设计，2014 年随 `net/http` 一起进入标准库。此后「第一个参数是 context」成为 Go API 的惯例，`net/http`、数据库驱动、gRPC 全面采用。</span>

context 在本专题的定位：它是并发篇章的收官——channel 解决「通信」、select 解决「多路等待」、sync 解决「共享状态」，而 context 解决「**级联取消**」：一个请求被取消，其下所有 goroutine 都要收到信号并停止。

## 1 context 是什么：一个贯穿调用链的对象

**`context.Context`** 是一个接口，携带三样东西：**取消信号**（`Done`）、**截止时间**（`Deadline`）、**请求级值**（`Value`）。

```go
type Context interface {
	Deadline() (deadline time.Time, ok bool)   // 截止时间
	Done() <-chan struct{}                     // 取消信号：关闭即取消
	Err() error                                // 取消原因
	Value(key any) any                         // 请求级键值
}
```

**核心概念：** `Done()` 返回一个**只读 channel**——当 context 被取消时，这个 channel 会被关闭。因此「监听 context 取消」就是「从这个 channel 上接收」：

```go
select {
case <-ctx.Done():
	// 被取消了，清理并退出
case <-someWork:
	// 正常工作
}
```

`<-ctx.Done()` 在取消发生时立即返回，等价于「等到取消信号」——这是所有 context 集成的基础模式。<span class="marginnote">`Done()` 返回 `nil` 的 context（如 `context.Background()`）永远不取消——`<-nil` 永久阻塞。所以「不该取消」的根 context 可以直接当作永不触发的信号源。</span>

**规范：** `context.Context` 应作为**函数第一个参数**传递，不放进结构体字段（少数例外）。`context.Background()` 是根上下文（永不取消），`context.TODO()` 表示「还没想好用什么」。

## 2 WithCancel：手动取消

**`context.WithCancel(parent)`** 派生出子 context，并返回一个**取消函数**。调用取消函数，子 context 的 `Done()` 被关闭：

```go
func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()   // 重要：defer 调用，防泄漏

	go worker(ctx)
	time.Sleep(100 * time.Millisecond)
	cancel()         // 主动取消
	fmt.Println("已取消")
}

func worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println("worker 收到取消，退出")
			return
		default:
			fmt.Println("工作中...")
			time.Sleep(20 * time.Millisecond)
		}
	}
}
```

**关键纪律：**

- **`cancel` 必须被调用**——`WithCancel` 创建的 context 若不被取消，它的 `Done` channel 与计时器会**泄漏**。用 `defer cancel()` 保证无论函数怎么返回都会取消。
- 取消是**级联的**：父 context 取消，所有派生子的 context 一并取消。

**辨析｜易错点：** 谁创建 context，谁就有责任取消它。派生它的 goroutine 可能早已退出，但取消信号仍要发出——这正是 `defer cancel()` 存在的意义。

## 3 WithTimeout 与 WithDeadline：限时执行

**`context.WithTimeout`** 与 **`context.WithDeadline`** 给任务加「截止时间」，到点自动取消：

```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

resp, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
// 若 2 秒内没完成，请求被自动取消
```

`NewRequestWithContext` 让 HTTP 请求**自动监听 ctx**：超时或取消时，底层连接被关闭，请求立即返回错误。<span class="marginnote">`WithTimeout` 是 `WithDeadline(time.Now().Add(d))` 的语法糖：一个按「时长」、一个按「时刻」。它们常用于「外部调用限时」——数据库查询、HTTP 调用、RPC 的每个调用都该有超时，否则一个慢依赖会拖垮整个服务。这就是「超时传递」的工程价值。</span>

**核心对比：三种派生方式**

| 函数 | 语义 | 取消时机 |
| --- | --- | --- |
| `WithCancel` | 手动取消 | 调用 `cancel()` |
| `WithTimeout(d)` | 限时长 | `cancel()` 或 `d` 后 |
| `WithDeadline(t)` | 限时刻 | `cancel()` 或到 `t` |

**易错点：** 超时后 `ctx.Err()` 返回 `context.DeadlineExceeded`，可用于区分「主动取消」与「超时」。判断取消原因在日志与指标里很重要——它是诊断「为什么任务没完成」的关键。

## 4 WithValue：请求级数据传递

**`context.WithValue`** 在 context 上携带请求级数据，随调用链传递：

```go
type traceIDKey struct{}

func withTraceID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, traceIDKey{}, id)
}

func traceID(ctx context.Context) string {
	if id, ok := ctx.Value(traceIDKey{}).(string); ok {
		return id
	}
	return ""
}
```

**用途：** 追踪 ID、用户身份、请求 ID 等「沿调用链每个函数都可能用到」的元数据。在 `net/http` 里，`r.Context()` 拿到当前请求的 context，中间件可以向其中注入值，后续 Handler 用 `ctx.Value` 取出。

**关键纪律：**

- **键必须是自定义类型**（`traceIDKey{}`），不能是内置类型（`string`）——避免不同包之间键冲突。
- `Value` 返回 `any`，取出时需**类型断言**。
- **不要用 Value 传「可选参数」**——那是函数参数的职责；Value 只传「横切关注点」的请求级数据。

**辨析｜易错点：** `WithValue` 不适合存「会影响控制流的配置」——那会让调用链隐式依赖，难以阅读。Go 官方建议 Value 用于「请求级、只读、跨层通用的数据」，用 string 键是反模式。

## 5 公式解析：context 的取消传播

**context 的取消传播可以用「树 + 广播」建模。** 每个 `WithX(parent)` 创建一个节点，`Done` 的关闭时机

$$
\text{Done}(child) \text{ 关闭} \iff \text{cancel}(child) \lor \text{Done}(parent) \text{ 关闭} \lor \text{Deadline}(child) \text{ 到期}
$$

以 `WithTimeout(parent, 2s)` 派生的 `ctx` 为例：

- **第一步，创建节点**：`ctx` 是 `parent` 的子节点，绑定 2 秒定时器。
- **第二步，三条取消路径**：调用 `cancel()`、父 `parent` 取消、2 秒到点——任一发生，`ctx.Done()` 关闭。
- **第三步，广播**：`ctx` 的所有后代节点（再 `WithCancel(ctx)` 派生的）`Done()` 一并关闭——级联传播。
- **第四步，监听方**：每个 `select { case <-ctx.Done(): ... }` 的 goroutine 同时被唤醒，各自清理退出。

这条模型揭示了 context 的**幂等性**：`cancel()` 可以被调用多次，第二次起无效果；多 goroutine 同时监听 `Done()` 都能收到信号。这种「一次广播、多方接收、级联传递」正是「取消一个请求 = 停止它下钻的所有工作」的数学表达——也是 gRPC、数据库、HTTP 三方协作实现「全链路超时」的基石。

## 6 小结

- **`context.Context`** 携带取消信号（`Done`）、截止时间（`Deadline`）与请求值（`Value`）。
- `WithCancel` 手动取消；**`cancel` 必须调用**（`defer cancel()` 防泄漏）。
- `WithTimeout`/`WithDeadline` 限时自动取消；超时后 `ctx.Err()` 返回 `DeadlineExceeded`。
- 取消是**级联的**：父取消，全部子 context 一起取消。
- `WithValue` 传请求级元数据；键用自定义类型、取出要类型断言。
- context 作为**函数第一个参数**传递，`Background()` 是永不取消的根。
- 取消传播 = 树 + 广播：一次取消，级联唤醒所有监听方。

在下一节，我们回到数据处理：**encoding/json 与数据序列化**——把「结构体 ↔ JSON」讲深，覆盖标签、流式编解码与自定义编码。
