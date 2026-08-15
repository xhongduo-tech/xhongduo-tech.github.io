---
title: 错误处理与 errors 包
date: 2026-08-07
---

# 错误处理与 errors 包

<div class="epigraph">
<p>错误处理是程序的一部分，而不是程序出问题时才发生的意外。</p>
<footer>—— Go 错误处理哲学（Error handling in Go）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第5章 + 官方 Error Handling 博客 ｜ 2026-08-07</p>
</div>

## 为什么从错误处理开始

每一门语言都要回答「出错了怎么办」：C 用错误码、Java/Python 用异常、Go 用「错误即值」——错误是普通返回值，必须显式处理。这套设计让「哪一行可能出错、出错了怎么办」在代码里一目了然，代价是代码里 `if err != nil` 满天飞。本篇系统讲透 Go 的错误模型：**error 接口、哨兵错误、错误包装、`errors.Is`/`As`、panic 与 recover**。<span class="marginnote">对标《Go语言圣经》第5.4 节「错误」与 Go 官方博客《Error handling and Go》：Go 1.13 引入的 `errors.Is`/`errors.As`/`%w` 把错误处理从「比较字符串」升级为「错误链的语义匹配」。这是每个 Go 工程都要用到的核心机制。</span>

错误处理在本专题是「工程化」的分水岭：前面学的语法如何组合成可靠系统，第一步就是「错误怎么不被吞掉、怎么被正确归类」。它直接服务《net/http》篇的 `http.Error`、《io》篇的错误，以及一切「打开文件、调用外部服务」的代码。

## 1 error 接口与错误值

Go 的错误就是实现了 `error` 接口的值——一个接口，一个方法：<span class="marginnote">`error` 接口只有 `Error() string` 一个方法，任何实现它的类型都是错误。这意味着错误可以携带任意结构化信息——`*os.PathError` 带 `Path`、`Err`、`Op` 三个字段，`*url.Error` 带 URL 与具体错误。</span>

```go
type error interface {
	Error() string
}
```

**创建错误的两种方式：**

```go
// 简单错误
err := errors.New("user not found")

// 带格式化的错误
err2 := fmt.Errorf("user %q not found", name)
```

`fmt.Errorf` 是 `errors.New(fmt.Sprintf(...))` 的语法糖，两者都返回 `*errorString`——一个携带字符串、实现 `Error()` 的类型。

**要点：** 错误值是**一等公民**——可以比较、可以存进变量、可以放 map、可以传参。`if err != nil` 检查「是否出错」；`err == nil` 表示成功。一个函数约定「成功时错误为 nil」，是 Go 全生态的隐形契约。

**辨析｜易错点：** 返回**带 nil 的非 nil 接口**是经典陷阱。若函数声明返回 `error`，却在某分支返回 `(*MyError)(nil)`（一个非 nil 的接口，动态类型为 `*MyError`，动态值为 nil），调用方的 `err != nil` 为 true——因为接口本身非 nil。**判断「是否出错」看的是接口值，不是动态值**。

## 2 哨兵错误：定义错误

**哨兵错误（sentinel error）** 是预定义的、代表「特定情况」的错误变量，用 `errors.New` 定义，包级导出：<span class="marginnote">「哨兵」原指代码里标记特殊位置的变量。标准库的 `io.EOF`、`os.ErrNotExist`、`sql.ErrNoRows` 都是哨兵错误。它们给「错误是什么」一个可比较的名字，让调用方不必依赖错误文案。</span>

```go
var ErrNotFound = errors.New("not found")
var ErrPermission = errors.New("permission denied")

func Get(id int) (*Item, error) {
	if id < 0 {
		return nil, ErrNotFound
	}
	// ...
}
```

**为什么要哨兵而不是字符串？** 因为调用方需要「程序化」判断错误种类：

```go
item, err := Get(-1)
if err == ErrNotFound {
	fmt.Println("没找到，走兜底逻辑")
}
```

`err == ErrNotFound` 依赖哨兵错误是**同一个实例**——`errors.New` 每次创建不同的实例，所以必须包级共享同一个变量，而不能每次 `errors.New("not found")`。

**易错点：** 哨兵错误要**导出**给调用方用（`ErrNotFound` 大写），且**不可变**——不要在包内修改哨兵错误的内容，否则 `==` 判定全乱。

## 3 错误包装与 %w

现实中错误总是「层层经过」：`Get` 里文件打开失败，错误要告诉上层「Get 失败，原因是打开文件失败，根因是文件不存在」。**错误包装（error wrapping）** 解决这个问题——用 `%w` 把下层错误包进新错误：<span class="marginnote">`%w` 是 Go 1.13 引入的格式动词：`fmt.Errorf("...: %w", err)` 会「记住」被包装的底层错误，形成错误链。配合 `errors.Is`/`errors.As` 沿链查找，既保留「在哪一层失败」的上下文，又不丢失「根因是什么」的类型信息。</span>

```go
func LoadConfig(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("load config %s: %w", path, err)
	}
	// ...
}
```

调用方可以**解链**：

```go
_, err := LoadConfig("/etc/app.yaml")
if errors.Is(err, os.ErrNotExist) {
	fmt.Println("配置文件不存在，用默认配置")
}
```

`errors.Is` 沿错误链查找「链上是否有等于 `os.ErrNotExist` 的错误」——不管中间包了几层 `%w`。

**核心对比：`%w` vs `%v` vs 字符串拼接**

| 写法 | 保留错误链 | 上层的 `Is`/`As` 能解出 |
| --- | --- | --- |
| `fmt.Errorf("x: %w", err)` | 是 | 是 |
| `fmt.Errorf("x: %v", err)` | 否 | 否（只剩字符串） |
| `fmt.Errorf("x: " + err.Error())` | 否 | 否 |

**要点：** 需要「保持根因可查」用 `%w`；想「抹掉细节」用 `%v`。现代 Go 项目默认 `%w`——错误链是「可读的诊断信息 + 可程序化判定的根因」的合体。

## 4 errors.Is 与 errors.As

**`errors.Is`** 判断错误链上是否存在某个错误（比较「等于」）：

```go
if errors.Is(err, os.ErrNotExist) { ... }
```

**`errors.As`** 把错误链上的错误「提取」到某个具体类型的变量（比较「类型」）：

```go
var pathErr *os.PathError
if errors.As(err, &pathErr) {
	fmt.Println("出错的文件:", pathErr.Path)
	fmt.Println("底层错误:", pathErr.Err)
}
```

**核心对比：Is vs As**

| 维度 | `errors.Is` | `errors.As` |
| --- | --- | --- |
| 比较目标 | 特定错误值（哨兵） | 特定错误类型 |
| 典型场景 | `err == io.EOF` | 提取 `*net.DNSError` 看字段 |
| 链上匹配 | 有即返回 true | 匹配则填充目标变量 |

**易错点：** `errors.As` 的目标必须是**指向「实现 error 的类型」的指针**（`&pathErr`，其中 `pathErr` 是 `*os.PathError`）——写错会 panic。它找到**第一个匹配类型**的错误，不是全部。

## 5 panic 与 recover：异常的正确边界

Go 没有异常，但有 `panic` 与 `recover`——它们是「程序员不该让 bug 发生的」的最后防线：<span class="marginnote">`panic` 是「不可恢复的错误」：数组越界、空指针解引用、断言失败都会触发。它终止当前 goroutine 的执行，沿途执行 defer，若无人 recover 则程序崩溃。官方立场：<strong>panic 用于「不该发生的事」与「程序员 bug」，可预期的错误一律用返回值</strong>。</span>

```go
func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("恢复:", r)
		}
	}()
	panic("something broke")   // 触发，但被上面的 recover 接住
	fmt.Println("不会执行到这")
}
```

**要点：**

- `panic(v)` 终止当前函数与 goroutine，沿调用栈执行所有 defer。
- `recover()` 只能在 **defer 函数里**有效——直接调用 `recover()` 返回 nil。
- **不要在业务逻辑里用 panic 做错误处理**——那是 `(value, err)` 的职责。panic 是「程序已处于不可能状态」的信号。

**易错点：** panic 会跳过它后面的 defer 吗？不会——defer 在 panic 时**依然执行**，这正是「defer 保证资源释放」在异常路径也成立的原因。`recover` 接住后，函数的返回值仍是「恐慌发生时的零值」，调用方无法知道处理不完整——所以 recover 通常只在「最高层」兜底，不让整个进程崩溃。

## 6 公式解析：错误链的判定

**错误链可以看成一个「链表」**：每个错误 `err` 可能通过 `Unwrap()` 指向下一个错误。`errors.Is` 的判定

$$
\text{Is}(e, \text{target}) \iff e = \text{target} \ \text{或} \ \exists\, e' \in \text{chain}(e):\ e' = \text{target}
$$

以 `LoadConfig` 返回的错误链为例（`errors.Is(err, os.ErrNotExist)`）：

- **第一步，取链头**：`err` = `fmt.Errorf("load config ...: %w", pathErr)`。
- **第二步，比较链头**：`err` 不等于 `os.ErrNotExist`（类型不同）。
- **第三步，沿 `Unwrap` 下探**：`err.Unwrap()` 得到 `pathErr`，其 `Err` 字段为 `os.ErrNotExist`。
- **第四步，命中**：链上存在等于 `os.ErrNotExist` 的错误，返回 true。

这条模型的启示：**`%w` 包装得越完整，错误链越长，`Is`/`As` 能匹配到越深层的根因**。而 `%v` 或字符串拼接「截断」了链，根因从此不可程序化判定。错误处理的质量，取决于你是否完整保留这条链。

## 7 小结

- **错误是值**：`error` 接口只有一个 `Error()` 方法；`if err != nil` 是显式契约。
- **哨兵错误**用 `errors.New` 包级定义，`err == ErrXxx` 判定，不比较字符串。
- **错误包装**用 `%w` 保留错误链：`fmt.Errorf("ctx: %w", err)`。
- **`errors.Is`** 沿链查「是否有等于某哨兵的错误」；**`errors.As`** 沿链提取「某种类型的错误」。
- **panic/recover** 是「不可恢复错误」的最后防线，业务错误一律用返回值。
- 错误链 = 链表，`%w` 保留链、`%v` 截断链——处理质量取决于链的完整度。

在下一节，我们进入 Go 1.18 最大的语言革新：**泛型——类型参数、类型集合与约束**。
