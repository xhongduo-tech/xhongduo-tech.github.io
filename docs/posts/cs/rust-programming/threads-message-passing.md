---
title: 线程与消息传递并发
date: 2026-08-07
---

# 线程与消息传递并发

<div class="epigraph">
<p>不要通过共享内存来通信；相反，通过通信来共享内存。</p>
<footer>—— 戈夫曼等（Effective Go 名言，Rust 消息传递并发同此哲学）</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第16章 ｜ 2026-08-07</p>
</div>

## 为什么从线程与消息传递开始

现代 CPU 是多核的，程序要变快就必须**并行**。但并行带来的数据竞争是并发 bug 的头号来源。Rust 的并发方案有两条主线：**消息传递**（这一章）与**共享状态**（下一章）。

消息传递的哲学是「**不要共享内存，通过发消息协作**」：每个线程拥有自己的数据，需要协作时把数据作为消息发给别的线程。这天然回避了数据竞争——因为数据要么在发送者手里，要么在接收者手里，从不同时被两个线程写。Rust 用 `std::thread` 创建线程，用 **`mpsc` 通道**（channel）在线程间传递消息，而所有权保证「发出去的数据再也不能被原线程使用」——这一条在编译期就杜绝了「两个线程同时碰同一数据」。

## 1 创建线程

### 最基本的线程

```rust
use std::thread;
use std::time::Duration;

fn main() {
    thread::spawn(|| {
        for i in 1..10 {
            println!("子线程：{i}");
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("主线程：{i}");
        thread::sleep(Duration::from_millis(1));
    }
}
```

`thread::spawn` 接收一个闭包，在新线程里运行。但这里有个陷阱：**主线程结束，程序就结束**——即使子线程还没跑完。上面的代码很可能只打印到主线程 `4` 就退出了，子线程未必跑完。

### join：等待线程结束

用 `join` 句柄等待子线程完成：

```rust
let handle = thread::spawn(|| {
    for i in 1..10 {
        println!("子线程：{i}");
    }
});

handle.join().unwrap();   // 阻塞直到子线程结束
```

`thread::spawn` 返回 `JoinHandle`，`join()` 等待该线程完成并返回它的结果（`Result`）。`handle.join()` 确保子线程跑完，主线程才继续。

## 2 用 move 闭包传递数据

### move：把所有权交给线程

子线程闭包要使用主线程的数据，必须**拥有**它——因为闭包可能比主线程变量活得更久。用 `move` 强制闭包取得所有权：

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];

    let handle = thread::spawn(move || {
        println!("这里是 {v:?}");   // v 被移动进线程
    });

    // println!("{v:?}");  // 错误：v 已被移动

    handle.join().unwrap();
}
```

`move ||` 把 `v` 的所有权移入线程闭包。之后主线程再访问 `v` 是编译错误——**这正是消息传递哲学的根基**：数据被移交后，原线程就不再拥有它，不可能出现「两个线程同时写」。
<span class="marginnote">如果不用 `move`，编译器会报错「闭包可能比捕获的变量活得久」（`E0373`）——因为线程闭包可能在线程里运行很久，而借用的 `v` 可能已在主线程被 drop。`move` 让闭包完全拥有 `v`，生命周期问题消失。</span>

### 所有权如何保证线程安全

这套设计的关键洞察：**数据竞争的前提是「同一数据被多个线程同时访问」**。Rust 的所有权让这个前提无法成立——要么数据还在主线程（子线程拿不到），要么数据已被 `move` 进子线程（主线程不能再碰）。数据在任何时刻只有一个「所有者线程」，竞争在编译期就消失了。

## 3 消息传递：mpsc 通道

### 创建通道与发送

**`mpsc`**（multiple producer, single consumer）是多生产者单消费者的通道。用 `std::sync::mpsc`：

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = String::from("你好");
        tx.send(val).unwrap();   // 发送
        // println!("{val}");    // 错误：val 已被 send 移动
    });

    let received = rx.recv().unwrap();   // 接收（阻塞）
    println!("收到：{received}");
}
```

`mpsc::channel()` 返回一对 `(tx, rx)`：**`tx`（transmitter）发送端**、**`rx`（receiver）接收端**。`tx.send(val)` 把 `val` 发进通道——注意 `val` 的所有权被 **send 移走**，发送后原变量不可再用。这再次是「通过通信共享内存」：数据通过通道移交，发送者不再拥有。<span class="marginnote">`send` 转移所有权是一个极其聪明的设计：接收端拿到的字符串与发送端的 `val` 是同一个堆数据，只是「搬家」进了通道。整个传输过程中不存在两份拷贝，也不存在两个线程同时访问——所有权让通道传输既安全又高效。</span>

`rx.recv()` 阻塞直到收到消息，返回 `Result`（发送端全部关闭时返回 `Err`）。不阻塞的版本是 `try_recv()`（立即返回）。

### 发送多条消息与迭代接收

`rx` 可以作为迭代器使用——`for` 循环直到发送端关闭：

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];
        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });

    for received in rx {   // 迭代接收，直到发送端 drop
        println!("收到：{received}");
    }
}
```

`for received in rx` 逐个接收消息，发送端（`tx` 及它的 `move` 闭包）全部 drop 后循环结束。这是「从通道里流水线消费」的标准姿势。

### 多生产者：克隆发送端

`mpsc` 允许多个发送端、一个接收端。克隆 `tx` 得到多个发送端：

```rust
let (tx, rx) = mpsc::channel();
let tx1 = tx.clone();    // 第二个发送端

thread::spawn(move || {
    tx1.send(String::from("来自线程1")).unwrap();
});

thread::spawn(move || {
    tx.send(String::from("来自线程2")).unwrap();
});

for msg in rx {
    println!("{msg}");
}
```

`tx.clone()` 创建新的发送端，两个线程各持一个，都往同一个 `rx` 发送。接收端看到的是两个发送端消息的**交错顺序**（由调度决定，不可预测）。所有发送端 drop 后，`rx` 迭代结束。

## 4 公式解析：通道语义与所有权转移

通道的语义可以用「消息从哪个所有权区域流向哪个」描述：

$$
\underbrace{\text{发送端 tx}}_{\text{持有 val 的所有权}} \quad \xrightarrow{\text{send(val)}} \quad \underbrace{\text{通道}} \quad \xrightarrow{\text{recv()}} \quad \underbrace{\text{接收端 rx}}_{\text{获得 val 的所有权}}
$$