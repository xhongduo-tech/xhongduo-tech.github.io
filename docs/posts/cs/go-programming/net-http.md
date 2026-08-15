---
title: net/http 与 Web 服务开发
date: 2026-08-07
---

# net/http 与 Web 服务开发

<div class="epigraph">
<p>Go 的 Web 服务是最诚实的网络编程：一个 Handler 函数，收到请求，返回响应。</p>
<footer>—— Go 网络编程共识（Web servers in Go）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第7章 + net/http 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 net/http 开始

Go 被称为「云原生语言」，很大程度上因为 `net/http` 标准库把「写一个 Web 服务」压缩到十几行：内置 HTTP/1.1 与 HTTP/2、并发连接管理、路由与静态文件、客户端与服务器一应俱全。一个 `http.Handler` 接口（只有 `ServeHTTP` 一个方法）就定义了「如何处理一个请求」——这正是《接口》篇思想的又一次胜利。<span class="marginnote">对标《Go语言圣经》第7.7 节「基于网络的应用」：书中用 `http.HandlerFunc`、`http.ServeMux` 演示了最小 Web 服务器，而完整的服务端/客户端设计在《The Go Programming Language》与官方 `net/http` 文档中系统展开。</span>

net/http 在本专题是「工程实战」的入口：它是 `io.Reader/Writer` 的实战场景（请求体、响应体都是流）、是 `context` 取消机制的主战场（每个请求一个 context）、也是 goroutine 并发模型的自然呈现（每个连接一个 goroutine）。

## 1 最小 Web 服务器

一个能响应的 HTTP 服务器只需要 `http.Handler` 接口与 `ListenAndServe`：

```go
package main

import (
	"fmt"
	"log"
	"net/http"
)

type helloHandler struct{}

func (h helloHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, 世界")
}

func main() {
	mux := http.NewServeMux()
	mux.Handle("/hello", helloHandler{})
	log.Fatal(http.ListenAndServe(":8080", mux))
}
```

**`http.Handler`** 接口：

```go
type Handler interface {
	ServeHTTP(w http.ResponseWriter, r *http.Request)
}
```

**核心概念：** 服务器收到请求后调用 Handler 的 `ServeHTTP`。`w` 用于写响应，`r` 携带请求信息（URL、方法、Header、Body）。**整个 Go Web 生态——路由、中间件、框架——都建立在这个接口之上**。<span class="marginnote">`http.ServeMux`（多路复用器）是内置路由器：`mux.Handle("/path", handler)` 把路径映射到 Handler，`ListenAndServe` 接受任意 Handler。第三方框架（gin、echo）本质上也是「实现了 Handler 接口 + 更聪明的路由」。</span>

## 2 HandlerFunc 与函数式路由

绝大多数 Handler 是一个普通函数，`http.HandlerFunc` 把函数适配成 Handler：

```go
func hello(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if name == "" {
		name = "world"
	}
	fmt.Fprintf(w, "Hello, %s!\n", name)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/hello", hello)
	http.ListenAndServe(":8080", mux)
}
```

`mux.HandleFunc` 用 `http.HandlerFunc(hello)` 把 `func(w, r)` 包装成实现了 `ServeHTTP` 的类型——**函数即 Handler**。这比「定义一个 struct + 实现方法」简洁得多，是 Go Web 开发的默认姿势。

**要点：** 请求方法（GET/POST）由 `r.Method` 判断；路径参数用 `r.URL.Path`、`r.URL.Query().Get` 获取。Go 1.22 起 `ServeMux` 支持 `{id}` 路径通配符，`r.PathValue("id")` 取参数——内置路由首次支持 REST 风格路径。

**核心对比：Handler 的三种实现方式**

| 方式 | 写法 | 适用 |
| --- | --- | --- |
| 自定义类型 | `struct` + `ServeHTTP` 方法 | 有状态、需要复用 |
| `HandlerFunc` | 普通函数 | 无状态、简单路由 |
| 闭包 Handler | `http.HandlerFunc(func(...){...})` | 捕获外层变量 |

## 3 读取请求与写响应

Handler 里常见的「读请求、写响应」操作：

```go
func echo(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		q := r.URL.Query().Get("q")
		fmt.Fprintf(w, "GET q=%s\n", q)
	case http.MethodPost:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body failed", http.StatusBadRequest)
			return
		}
		fmt.Fprintf(w, "POST body=%s\n", body)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}
```

**关键操作：**

- `r.Method` 请求方法；`r.URL` 路径与查询；`r.Header` 请求头。
- `r.Body` 是 `io.ReadCloser`——用 `io.ReadAll` 读全部（小 body）或 `json.NewDecoder(r.Body)` 流式解码。
- `w.Header().Set` 设置响应头；`w.WriteHeader` 设置状态码；`fmt.Fprintf(w, ...)` 写响应体。
- `http.Error(w, msg, code)` 快速返回错误响应。<span class="marginnote">`r.Body` 实现了 `io.Reader`，`w` 实现了 `io.Writer`——所以上一节学的 `io.Copy`、`json.Decoder`、`bufio` 全部能直接用于 HTTP。这就是「I/O 抽象」与「接口」在真实系统中的闭环：学过的抽象在这里无缝复用。</span>

**易错点：** `w.WriteHeader` 一旦调用，就不能再写 Header——所以「设置 Header 再 WriteHeader 再写 body」的顺序不能错。`http.Error` 内部先设 Header、再 WriteHeader、再写 body，帮你把顺序包好。

## 4 JSON API 与内容类型

Web 服务最常见的形态是 JSON API——配合《encoding/json》篇的知识：<span class="marginnote">JSON API 的标准姿势：`json.NewEncoder(w)` 直接向 `w`（io.Writer）编码——不用先 `json.Marshal` 成 `[]byte` 再 `w.Write`。解码同理用 `json.NewDecoder(r.Body)` 从请求体流式解码。流式编解码省掉一次内存拷贝，且天然处理「编码出错」。</span>

```go
type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

func getUser(w http.ResponseWriter, r *http.Request) {
	u := User{ID: 1, Name: "Alice"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(u)
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var u User
	if err := json.NewDecoder(r.Body).Decode(&u); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	fmt.Fprintf(w, "created %+v\n", u)
}
```

**要点：** `json.NewDecoder(r.Body).Decode(&u)` 从请求体解码 JSON 到结构体——若 JSON 格式错误或字段类型不匹配，返回 `err`，应回 `400`。设置 `Content-Type: application/json` 让客户端正确解析。

**易错点：** `Decode` 之后要**检查错误**（上面的 `http.Error`）。解码失败时 `u` 部分填充，直接使用会读到零值——这再次印证《错误处理》篇「必须检查错误」的纪律。

## 5 中间件与优雅关闭

**中间件（middleware）** 是「包一层 Handler」的模式：在 Handler 前后注入逻辑（日志、鉴权、超时）。它本质是函数式组合：<span class="marginnote">中间件 = 高阶函数：接收 Handler，返回新 Handler。`logMiddleware(handler)` 返回的新 Handler 在调用原 Handler 前打日志。多个中间件层层嵌套，构成「洋葱模型」——请求穿过外层进入内层，响应再反向穿出。这与《函数》篇的高阶函数、与《并发模式》篇的组合思想一脉相承。</span>

```go
func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)   // 调用被包裹的 Handler
		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/hello", hello)
	http.ListenAndServe(":8080", logMiddleware(mux))
}
```

**优雅关闭**：生产环境的服务器需要「收到停止信号后，先把正在处理的请求完成」——用 `http.Server` + `Shutdown`：

```go
srv := &http.Server{Addr: ":8080", Handler: logMiddleware(mux)}
go srv.ListenAndServe()

// 收到中断信号后
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()
srv.Shutdown(ctx)   // 等待现有请求完成，最多 5 秒
```

**易错点：** 用 `http.ListenAndServe` 返回的错误直接 `log.Fatal`——因为服务器错误（端口占用、Shutdown）就是进程应该终止的信号。生产代码常用 `log.Fatal(http.ListenAndServe(...))`。

## 6 公式解析：并发连接模型

**net/http 服务器为每个连接分配一个 goroutine**，因此并发处理能力受 goroutine 规模约束。设同时有 $C$ 个连接、每个连接一个 goroutine，则活跃 goroutine 数

$$
G = C \cdot \frac{T_{\text{活跃}}}{\text{总时间}}
$$

对于一个处理耗时 $T$ 的 API，每秒请求量 $Q$ 与并发连接的关系：

- **第一步，单连接吞吐**：每个连接上 $Q_{\text{单}} = 1/T$ 请求/秒。
- **第二步，多连接**：$C$ 个连接给 $C \cdot (1/T)$ 请求/秒（理想）。
- **第三步，goroutine 约束**：$G \approx C$，因为 goroutine 栈小，$C$ 可达数万——这就是 Go 服务器「轻松扛住上万并发」的根源。
- **第四步，瓶颈转移**：真正的瓶颈往往不在 goroutine 数，而在**下游**——数据库连接、外部 API 限流、磁盘 IO。这些瓶颈需要用《并发模式》篇的工作池/信号量来控制。

这条模型解释了 Go 在 Web 领域的核心优势：**每个请求一个 goroutine，goroutine 便宜到可以「连接数 = goroutine 数」**。Java 的「一请求一线程」受线程内存限制，Go 则把并发成本压到极低——云原生服务因此「默认就能并发」。

## 7 小结

- **`http.Handler`** 是 Web 生态的心脏：`ServeHTTP(w, r)` 定义「怎么处理一个请求」。
- `http.ServeMux` 内置路由；`mux.HandleFunc` 把函数当 Handler（Go 1.22 支持 `{id}` 路径参数）。
- 请求信息在 `r`（Method/URL/Header/Body），响应写 `w`（Header/WriteHeader/body）。
- JSON API：`json.NewEncoder(w)` 编码、`json.NewDecoder(r.Body)` 解码，检查错误必不省略。
- **中间件**是高阶函数包 Handler：日志、鉴权、超时层层嵌套（洋葱模型）。
- 每连接一个 goroutine，Go 服务器天然高并发；用 `Shutdown` 优雅关闭。

在下一节，我们处理 Web 服务的关键配角：**context 上下文与并发取消**——如何优雅地停止一个正在运行的任务。
