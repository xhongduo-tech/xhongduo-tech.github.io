---
title: 并发基础：线程、同步与 volatile
date: 2026-08-07
---

# 并发基础：线程、同步与 volatile

<div class="epigraph">
<p>并发是把双刃剑：多核并行是性能的礼物，共享可变状态是正确性的诅咒。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第14章 ｜ 2026-08-07</p>
</div>

## 为什么从并发基础开始

现代 CPU 有 8 核、16 核，若程序始终单线程，大部分核心在围观。**并发编程（concurrency）**让程序同时做多件事——处理请求、下载文件、后台计算——充分利用多核。但并发也引入了单线程世界不存在的两类问题：**竞态**（两个线程同时改一个变量，结果取决于时序）与**可见性**（一个线程改了值，另一个线程看不见）。Java 的并发是「共享内存模型」：线程之间通过共享的堆内存通信，所以必须用**同步机制**管住共享数据。这一篇从线程本身讲起，到 `synchronized` 与 `volatile`，建立并发正确性的第一块基石。

## 1 线程：程序里的并行执行单元

**线程（thread）**是进程内的独立执行流。JVM 启动后，`main` 方法运行在**主线程**里；你可以创建新线程让代码并行跑：

```java
// 方式一：继承 Thread（不推荐，Java 单继承被占）
// 方式二：实现 Runnable（推荐）
Runnable task = () -> System.out.println("子线程：" + Thread.currentThread().getName());
Thread t = new Thread(task);
t.start();              // 启动线程（注意不是 run()）
```

**辨析｜易错点：`start()` 与 `run()` 天差地别。** `t.start()` 会**新起一个线程**并让它执行 `run()`；直接调 `t.run()` 只是在当前线程里**同步调用**一个普通方法——不会并行。新手常把 `run()` 当成启动方式，结果程序「看似能用」但根本没并发。

**线程的生命周期**五态：

$$

\text{NEW} \to \text{RUNNABLE} \leftrightarrow \text{BLOCKED / WAITING / TIMED\_WAITING} \to \text{TERMINATED}

$$

- `NEW`：创建了 `Thread` 对象，还没 `start`。
- `RUNNABLE`：可运行（可能在跑，也可能在等 CPU）。
- `BLOCKED`：等锁（被 `synchronized` 挡住）。
- `WAITING` / `TIMED_WAITING`：等通知/等一段时间（`wait()`、`sleep()`、`join()`）。
- `TERMINATED`：执行完。

**常用的线程控制**：`Thread.sleep(ms)` 让当前线程睡指定毫秒（`TIMED_WAITING`）；`t.join()` 让当前线程**等待 t 结束**；`Thread.yield()` 让出 CPU。注意 `sleep` 与 `join` 都抛 `InterruptedException`（受检异常）——它在「中断」机制里扮演信号角色。

## 2 竞态条件：并发错误的源头

**竞态条件（race condition）**：多个线程**无协调地**读写同一块共享数据，最终结果取决于线程的调度顺序。看经典的计数器：

```java
public class Counter {
    private int count = 0;
    public void increment() { count++; }   // 不是原子的！
}
```

`count++` 在字节码层面是**三步**：读 count、加 1、写回。两个线程同时执行，可能都读到 `count=5`，各自加 1 都写回 `6`——**丢了一次更新**。这就是竞态。

**公式解析：为什么 `count++` 不是原子的**：

$$

\text{count++} \;\Rightarrow\; \text{LOAD } \text{count} \to \text{ADD } 1 \to \text{STORE } \text{count}

$$