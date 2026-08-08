---
title: 线程库：Pthreads、Java 线程与 Windows 线程
date: 2026-08-07
---

# 线程库：Pthreads、Java 线程与 Windows 线程

<div class="epigraph">
<p>接口是承诺：只要接口不变，背后的实现可以翻天地覆。</p>
<footer>—— 佚名，软件工程箴言</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《操作系统概念》§4.4 线程库 ｜ 2026-08-07</p>
</div>

## 为什么从线程库开始

上一节讲了线程模型，但程序员并不直接跟内核线程打交道——他们调用的是**线程库（thread library）**提供的 API。线程库把「创建线程、等待线程、给线程加锁」包装成一套友好的函数接口，底层则隐藏了线程模型的实现细节。<span class="marginnote">回顾多线程模型：库是「用户态的门面」，模型是「库与内核的配合」。同一个 pthread API，在 Linux 上走一对一模型，在其他平台上可能走不同模型——程序员通常无需关心。</span>

学线程库，重点是看**创建、等待、退出**这三个最基本动作在不同库里的表达，一通百通。

## 1 Pthreads：POSIX 标准线程库

**Pthreads（POSIX threads）**：POSIX 标准定义的 C 语言线程 API，是 Unix/Linux 世界的标准。编译链接时加 `-lpthread`。核心函数：

```c
#include <pthread.h>

void *worker(void *arg) {
    printf("hello from thread\n");
    return NULL;
}

int main(void) {
    pthread_t t;
    pthread_create(&t, NULL, worker, NULL);  // 创建线程
    pthread_join(t, NULL);                    // 等待线程结束
    return 0;
}
```

四个最常用的 Pthreads 函数：

| 函数 | 作用 |
| --- | --- |
| `pthread_create` | 创建线程，指定线程函数与参数 |
| `pthread_exit` | 线程主动退出，可传返回值 |
| `pthread_join` | 等待指定线程结束并回收其资源 |
| `pthread_self` | 获取当前线程 ID |

Pthreads 的同步原语（`pthread_mutex_t`、`pthread_cond_t` 等）会在第六篇《互斥锁与信号量》展开。这里记住：**Pthreads 是「裸」的线程 API，一切都要自己管理**——没有线程池、没有锁的自动释放，这正是 C 语言的风格。<span class="marginnote">Pthreads 不是内核，是库；Linux 上它调用 `clone` 系统调用创建线程（见 Linux 篇《NPTL 与 pthread》）。接口与实现分离在此体现得淋漓尽致。</span>

## 2 Java 线程：语言内建的线程

Java 把线程做进了语言里，创建线程有两种方式：

```java
// 方式一：继承 Thread 类
class Worker extends Thread {
    public void run() { System.out.println("hello"); }
}
new Worker().start();

// 方式二：实现 Runnable（推荐）
Thread t = new Thread(() -> System.out.println("hello"));
t.start();
```

Java 线程的关键点：

**`start()` 才真正创建线程**，`run()` 只是普通方法调用。
**线程状态机内置在语言里**：`NEW`、`RUNNABLE`、`BLOCKED`、`WAITING`、`TIMED_WAITING`、`TERMINATED`，对应操作系统状态模型。
**同步机制是语言关键字**：`synchronized`、`volatile`，无需显式锁对象。
Java 线程映射到宿主操作系统的原生线程（一对一），JVM 之下调用平台线程 API。<span class="marginnote">JVM 的线程模型随时代演进：早期「绿线程」是多对一，现代 HotSpot JVM 用原生线程一对一。Java 19+ 的虚拟线程（Virtual Threads）则把大量用户级线程映射到少量内核线程，这是多对多思想的现代回归。</span>

## 3 Windows 线程：Win32 线程 API

Windows 的线程 API（Win32/`CreateThread`）与 Pthreads 思路相近：

```c
DWORD WINAPI worker(LPVOID arg) {
    return 0;
}

HANDLE h = CreateThread(NULL, 0, worker, NULL, 0, NULL); // 创建线程
WaitForSingleObject(h, INFINITE);                        // 等待线程结束
CloseHandle(h);
```

Windows 与 Pthreads 的对照：

| 概念 | Pthreads | Windows |
| --- | --- | --- |
| 创建 | `pthread_create` | `CreateThread` |
| 等待 | `pthread_join` | `WaitForSingleObject` |
| 退出 | `pthread_exit` | `ExitThread`/返回 |
| 当前 ID | `pthread_self` | `GetCurrentThreadId` |

**三大线程库的共性**：无论哪个平台，「创建 → 执行 → 等待 → 退出」的骨架完全一致。学会一个，另两个半小时上手——这就是接口抽象的威力。<span class="marginnote">Rust 的 `std::thread`、Go 的 goroutine 底层也分别封装了 pthread 或平台线程。语言提供的是「更顺手的皮」，底下仍是这三个库中的一个在扛活。</span>

## 4 核心对比表：三大线程库

| 维度 | Pthreads | Java 线程 | Windows 线程 |
| --- | --- | --- | --- |
| 语言 | C | Java（语言内建） | C/C++（Win32） |
| 平台 | POSIX（Linux/Unix/macOS） | 跨平台（JVM 抽象） | Windows |
| 线程模型 | 一对一（Linux） | JVM 原生线程，虚拟线程多对多 | 一对一 |
| 同步方式 | 库函数（mutex/cond） | 语言关键字 `synchronized` | API（`WaitForSingleObject` 等） |
| 资源管理 | 手动 join | JVM 自动管理 | 手动句柄 |

**辨析｜易错点：** 「线程库 = 线程实现」是常见误解。线程库是**接口**，线程模型是**实现**。同一个 Pthreads 接口，在不同系统上可能映射到不同的模型；Go 的 goroutine 甚至把「库」与「实现」都搬进了语言运行时。**写代码时你依赖的是接口，性能时你才需要关心实现。**

## 5 小结

- **线程库**是线程编程的接口层，底层隐藏线程模型实现。
- **Pthreads**（C/POSIX）：`pthread_create`、`pthread_join`、`pthread_exit` 三件套，手动管理一切。
- **Java 线程**：语言内建，`start()` 创建、`synchronized` 同步，状态机内置。
- **Windows 线程**：`CreateThread`/`WaitForSingleObject`，与 Pthreads 概念一一对应。
- 三大库骨架一致（创建→执行→等待→退出），接口与实现的分离是通用设计智慧。

在下一节，我们不再手动管线程，而是让框架替我们管——**隐式线程：线程池、Fork-Join 与 OpenMP**。
