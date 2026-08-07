---
title: 信号处理函数与可重入问题
date: 2026-08-07
---

# 信号处理函数与可重入问题

<div class="epigraph">
<p>信号处理函数像不速之客——它可能在你做任何事的时候闯进来，而你永远不知道屋里当时是什么状态。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与 APUE §10 ｜ 2026-08-07</p>
</div>

## 为什么从可重入问题开始

信号处理函数在**任意时刻**打断主程序——这意味着处理函数执行时，主程序可能正处于**任何中间状态**。若处理函数访问主程序正在使用的数据或函数，就会出乱子——这就是**可重入（reentrancy）**问题。它是信号处理最经典也最危险的陷阱，也是理解并发安全的基础。<span class="marginnote">回顾《Linux 信号机制》：处理函数在「从内核返回用户态」时被调用，打断主程序的任意执行点。<strong>可重入问题 = 处理函数与主程序「重入」同一份数据/函数时的安全</strong>——这是并发安全最朴素的形式。</span>

## 1 什么是可重入

**可重入（reentrant）**：一个函数被**多次调用且调用可能重叠**时，行为仍然正确——即函数被「中断后再次进入」不会出问题。

- **可重入函数**：不依赖全局/静态可变状态，或每次进入用独立的状态（栈、寄存器）。如 `read`、`write`、`_exit`。
- **不可重入函数**：依赖全局可变状态——`malloc`（管理堆）、`printf`（用全局缓冲）、`strtok`（用静态指针）。

**信号处理的场景**：主程序正在调 `printf`（写全局缓冲），信号处理函数也调 `printf`——**两个 `printf` 重入同一份全局缓冲** → 数据错乱、甚至死锁。

**为什么信号处理尤其危险**：

- 处理函数**在任意指令处**打断主程序——主程序可能正处于「改了数据一半」的状态。
- 处理函数**和主程序共享同一进程的地址空间**——没有隔离。
- 处理函数**可能再次被同一信号打断**（除非屏蔽）——多层重入。

## 2 典型的不可重入灾难

**灾难一：malloc/printf 重入**。

```c
void handler(int sig) {
    printf("signal!\n");    // 用 malloc + 全局缓冲
}
int main() {
    signal(SIGINT, handler);
    while (1) printf("hi\n");  // 主程序也在用 printf
}
```

主程序调 `printf` → 进入 malloc 分配缓冲 → 信号打断 → handler 调 `printf`/`malloc` → **重入 malloc 的内部状态** → 堆损坏或死锁。

**灾难二：全局变量竞争**。

```c
int counter = 0;
void handler(int sig) { counter++; }      // 读改写
int main() { while (1) { counter++; } }   // 读改写
```

`counter++` 是「读-改-写」三段（回顾竞态条件）——主程序与处理函数并发改 `counter`，**丢更新**。

**灾难三：非原子操作被二次进入**。

处理函数修改了一个链表/标志，主程序刚检查完正要使用——**状态不一致**。

## 3 异步信号安全（async-signal-safe）函数

**异步信号安全（async-signal-safe）**：可以在信号处理函数中**安全调用**的函数集合。POSIX 定义了这个清单：

**安全（可在 handler 中调用）**：

- `write`、`read`、`open`、`close`（不依赖进程全局可变缓冲）。
- `_exit`（立即退出）。
- `sigaction`、`signal`。
- `getpid`、`getuid` 等纯查询。

**不安全（禁止在 handler 中调用）**：

- `printf`/`sprintf`（用全局缓冲）。
- `malloc`/`free`（管理全局堆）。
- `strtok`（静态指针）。
- 任何锁操作（`pthread_mutex_lock`——可能死锁）。

**工程实践**：

- handler 里**只调 async-signal-safe 函数**——通常只做「置标志位」：`handler` 置一个 `volatile sig_atomic_t` 标志，主程序循环检查标志再处理。

```c
volatile sig_atomic_t flag = 0;
void handler(int sig) { flag = 1; }   // 安全：只写标志
int main() {
    signal(SIGINT, handler);
    while (1) { if (flag) { flag = 0; do_work(); } }
}
```

**公式解析：安全标志的模式**

$$\text{handler 只做} \quad flag \leftarrow 1; \quad \text{主程序} \quad \text{if } flag \text{ then handle}$$

- **`volatile sig_atomic_t`**：保证读写是原子的（int 级）且不被编译器优化掉。
- handler 只**置位**，主程序**轮询并处理**——**把「复杂工作」移出 handler，放进主循环**。
- 这是信号处理的黄金模式：**handler 极简（置标志），工作在主程序做**。

**直觉**：信号处理的安全铁律是「**handler 里不做复杂的事**」——它的工作应该只是「记个号」，剩下的主程序慢慢做。这避免了重入的一切风险。

**辨析｜易错点：** 「printf 在信号处理里偶尔能跑，所以能用」是极其危险的侥幸。**printf 能否工作取决于当时的竞态**——大多数时候碰巧没事，但**一旦与主程序的 printf/malloc 重入，就是堆损坏或死锁**，且 bug 极难复现。**「碰巧能跑」≠「安全」**——这正是竞态条件的经典教训（回顾《竞态条件》）。

## 4 核心对比表：可重入 vs 不可重入函数

| 函数 | 依赖状态 | 可在 handler 用？ |
| --- | --- | --- |
| `write` | 内核（无进程全局态） | **是** |
| `_exit` | 无 | **是** |
| `getpid` | 只读 | **是** |
| `printf` | 全局 stdio 缓冲 | 否 |
| `malloc` | 全局堆 | 否 |
| `strtok` | 静态指针 | 否 |

## 5 小结

- **可重入**：函数被重叠调用时仍正确——不依赖全局可变状态。
- 信号处理危险：处理函数在**任意执行点**打断主程序，共享同一地址空间。
- 典型灾难：**printf/malloc 重入、全局变量竞争、非原子操作**。
- **async-signal-safe** 函数（write/_exit/getpid）才可在 handler 用。
- 黄金模式：**handler 只置 `volatile sig_atomic_t` 标志，主程序轮询处理**。

在下一节，我们看进程间通信的经典实现——**管道与 FIFO 的实现与使用**。
