---
title: 综合项目：多线程 Web 服务器
date: 2026-08-07
---

# 综合项目：多线程 Web 服务器

<div class="epigraph">
<p>理论终将落幕，而工程才是语言的最终裁判。</p>
<footer>—— The Rust Book 第20章收官项目的主题</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第20章 ｜ 2026-08-07</p>
</div>

## 为什么从多线程 Web 服务器开始

这是 The Rust Book 的收官项目：从零写一个多线程 Web 服务器。它把本专题前 19 章的所有零件一次性组装起来——**TCP 监听**（网络）、**HTTP 解析**（字符串与模式匹配）、**线程池**（并发）、**优雅关闭**（所有权与 `Drop`）、**`Arc`/`Mutex`/`mpsc`**（共享状态与消息传递）。没有外部框架，只用标准库。

更重要的是它示范了一条工程路线：**渐进重构**——先写一个能跑的单一循环服务器，再逐步引入线程池、优雅关闭。每一步都保持「可编译、可运行」，这正是真实软件演进的节奏。而「线程池」的构建过程，是把「一堆概念」落成「一个类型」的完整范例：从需求分析到接口设计到并发实现。

## 1 单线程服务器：先让它跑起来

### 监听 TCP 连接

HTTP 基于 TCP。第一步：监听本地 7878 端口，接受连接：

```rust
use std::io::prelude::*;
use std::net::TcpListener;
use std::net::TcpStream;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        handle_connection(stream);
    }
}
```

`TcpListener::bind` 绑定端口，`.incoming()` 产生「每有一个新连接就 yield 一个 `TcpStream`」的迭代器。`handle_connection` 处理单个连接。

### 读取 HTTP 请求

```rust
fn handle_connection(mut stream: TcpStream) {
    let mut buffer = [0; 1024];     // 1024 字节缓冲区
    stream.read(&mut buffer).unwrap();

    let get = b"GET / HTTP/1.1\r\n";   // 请求行

    let (status_line, filename) = if buffer.starts_with(get) {
        ("HTTP/1.1 200 OK", "hello.html")
    } else {
        ("HTTP/1.1 404 NOT FOUND", "404.html")
    };

    let contents = fs::read_to_string(filename).unwrap();
    let response = format!(
        "{status_line}\r\nContent-Length: {}\r\n\r\n{}",
        contents.len(),
        contents
    );

    stream.write_all(response.as_bytes()).unwrap();
    stream.flush().unwrap();
}
```

`buffer` 是 `[0; 1024]` 字节数组，`stream.read` 把请求读进来。`b"GET / HTTP/1.1\r\n"` 是字节字符串字面量，`buffer.starts_with(get)` 判断请求是否为根路径。根据请求类型返回 200 或 404，响应按 HTTP 格式拼装：状态行、`Content-Length` 头、空行、正文。<span class="marginnote">HTTP 响应格式是「状态行 + 头部 + 空行 + 正文」，`Content-Length` 告诉浏览器正文有多少字节。这里手工拼响应字符串，是理解 HTTP 协议最直接的方式——与第三级《计算机网络》的 HTTP 章节完全对应。</span>

这个版本的问题很明显：**单线程**。它处理完一个连接才处理下一个——一个慢客户端会阻塞所有后续请求。要支持并发，需要线程池。

## 2 线程池：需求与接口设计

### 为什么需要线程池

**线程池（thread pool）**预先创建一组线程，任务到达时分配给空闲线程。相比「每来一个请求新建一个线程」，线程池避免频繁创建/销毁线程的开销，也限制并发数量防止资源耗尽。

需求拆解：

1. 程序启动时创建固定数量的工作线程（比如 4 个）。
2. 每个请求是一个「任务」，交给池子处理。
3. 任务执行完，工作线程回来接下一个任务。

### 接口设计：先写想要的 API

The Rust Book 教一个「先写想要的接口，再实现」的方法：

```rust
fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    let pool = ThreadPool::new(4);       // 创建 4 线程的池

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        pool.execute(|| {                // 提交任务
            handle_connection(stream);
        });
    }
}
```

`ThreadPool::new(4)` 创建池子，`pool.execute(闭包)` 提交任务。这个接口想清楚了，实现才有方向——`ThreadPool` 是自定义类型，`execute` 接收闭包。<span class="marginnote">「先写使用方代码，再实现类型」是测试驱动开发的变体：接口先于实现，且接口的形状由「好用」决定而非「好实现」决定。`ThreadPool::new` 与 `execute` 的设计，处处参考标准库的 `thread::spawn` 与 `mpsc::Sender`。</span>

## 3 实现线程池：从闭包到工作线程

### execute 的签名

`execute` 接收闭包，用 `FnOnce`（任务只执行一次）：

```rust
impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool { ... }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // 把 f 发送给某个工作线程
    }
}
```

`F: FnOnce() + Send + 'static` 的三重约束：

**`FnOnce()`**：任务被执行一次。
**`Send`**：闭包要跨线程发送给工作线程。
- **`'static`**：闭包不借用任何短期数据（工作线程可能比调用者活得久）。

### 线程池的结构

```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}
```

- `Job` 是任务类型：`Box<dyn FnOnce() + Send + 'static>`（第19篇的 `type` 别名用在了这里）。
- `workers` 是工作线程列表，每个 `Worker` 有编号与线程句柄。
- `sender` 是 `mpsc` 发送端——任务经通道发给工作线程。

### 工作线程的循环

每个 `Worker` 的线程跑一个循环：**接收任务 → 执行 → 再接下一个**。`Arc<Mutex<mpsc::Receiver<Job>>>` 是关键——多个工作线程共享同一个接收端，`Mutex` 保证同一时刻只有一个线程 `recv`：

```rust
impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv().unwrap();
            println!("Worker {id} 执行任务");
            job();
        });

        Worker { id, thread }
    }
}
```

`receiver.lock().unwrap().recv()` 是「加锁取消息」：`lock()` 拿 `MutexGuard`，`recv()` 阻塞等待任务，拿到 `Job` 后执行 `job()`。<span class="marginnote">`Arc<Mutex<Receiver>>` 是第17篇组合的实战应用：`Arc` 让每个 `Worker` 共享同一个 `Receiver`，`Mutex` 让同一时刻只有一个线程在 `recv`——通道本身是线程安全的，但标准 `Receiver` 不是 `Sync`，包一层 `Mutex` 才能多线程共享。</span>

### execute 发送任务

```rust
impl ThreadPool {
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}
```

`execute` 把闭包装箱成 `Job`，经 `sender` 发送。某个空闲 `Worker` 的 `recv` 会收到它并执行。

## 4 优雅关闭：Drop 与 JoinHandle

### 问题：主线程退出，工作线程还活着

当前实现有个缺陷：主线程处理完所有连接退出后，工作线程还在 `recv` 阻塞等待。程序不会干净退出。需要**优雅关闭**——通知工作线程「没有更多任务了」，等它们完成当前任务后退出。

### Drop：让池子优雅停机

`recv()` 在**所有发送端 drop 后**返回 `Err`。因此让 `ThreadPool` 的 `Drop` 实现中 drop 掉 `sender`，工作线程的 `recv().unwrap()` 就会 `Err`，循环退出：

```rust
impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender);   // 关闭所有发送端

        for worker in &mut self.workers {
            println!("关闭 Worker {}", worker.id);
            worker.thread.join().unwrap();   // 等待线程结束
        }
    }
}
```

`drop(self.sender)` 让通道「没有更多发送者」，`Worker` 循环里的 `recv().unwrap()` 返回 `Err`，`unwrap` panic，线程退出。<span class="marginnote">这个实现依赖 `recv()` 的语义：发送端全部关闭时返回 `Err`。用 `drop` 关通道是「通知停止」的惯用手法——比引入专门的「停止信号」更简单，且 `Drop` 保证无论 `main` 怎么退出（正常返回或提前 `return`），池子都会尝试优雅关闭。</span>

### 更好的实现：让 recv 返回 Result 而非 unwrap

更健壮的写法是显式处理 `Err` 而非 `unwrap`：

```rust
let thread = thread::spawn(move || loop {
    let job = match receiver.lock().unwrap().recv() {
        Ok(job) => job,
        Err(_) => break,   // 发送端关闭，退出循环
    };
    job();
});
```

`Err(_) => break` 让工作线程**主动退出**而不是靠 `unwrap` panic——语义更清晰，也不会有 panic 回溯的噪音。

## 5 公式解析：线程池的资源守恒

线程池的并发模型可以量化。设池有 $W$ 个工作线程，每时刻在执行的请求数为 $A$，待处理请求数为 $Q$：

$$
A \le W, \qquad \text{等待中的请求} = Q
$$

拆解：

- **第一步，并发上限 $A \le W$**：同时执行的请求不超过工作线程数。线程池把「并发度」钉死在 $W$，防止每个请求建线程导致的资源耗尽。
- **第二步，任务排队 $Q$**：超出 $W$ 的请求在通道里排队，`recv` 依次取出。队列是缓冲区，平滑突发流量。
- **第三步，与「每请求一线程」对比**：后者并发度无上限，线程创建/销毁开销大且可能拖垮系统；线程池以「最多 $W$ 并发 + 队列缓冲」换取「可控的资源使用与可预测的延迟」。

## 6 核心对比：三种并发服务器形态

| 形态 | 并发度 | 资源开销 | 实现复杂度 |
| --- | --- | --- | --- |
| 单线程循环 | 1 | 最低 | 最低 |
| 每请求一线程 | 无上限 | 高（频繁建线程） | 低 |
| 线程池 | 固定 $W$ | 中等（复用线程） | 较高 |

## 7 小结

- **单线程服务器**：`TcpListener::bind` + `incoming()` 循环，`read`/`write_all` 处理 HTTP 请求，手工拼响应。
- **线程池**：预先创建 $W$ 个工作线程，任务经 `mpsc` 通道分发，`execute(闭包)` 提交任务。
- **任务类型** `Box<dyn FnOnce() + Send + 'static>`：闭包跨线程、只执行一次、不借用短期数据。
- 工作线程循环「`lock().recv()` → 执行任务」，`Arc<Mutex<Receiver>>` 让多线程共享接收端。
- **优雅关闭**：`Drop` 中 drop `sender`，`recv()` 返回 `Err`，工作线程退出，`join()` 等待收尾。
- 线程池把并发度钉在 $W$