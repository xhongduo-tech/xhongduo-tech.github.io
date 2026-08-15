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

**常用的线程控制**：`Thread.sleep(ms)` 让当前线程睡指定毫秒（`TIMED_WAITING`）；`t.join()` 让当前线程**等待 t 结束**；`Thread.yield()` 让出 CPU。注意 `sleep` 与 `join` 都抛 `InterruptedException`（受检异常）——它在「中断」机制里扮演信号角色。<span class="marginnote">中断（interrupt）是 Java 的「优雅停线程」机制：`t.interrupt()` 给目标线程设一个中断标志，被阻塞的目标线程抛 `InterruptedException` 醒来，自行决定收尾退出。直接 `t.stop()` 已被废弃——它可能让线程死在一半状态。</span>

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

对这条公式做三步拆解：

- **第一步，LOAD**：把 `count` 的当前值从内存读到寄存器/CPU 缓存。
- **第二步，ADD**：在寄存器里加 1。
- **第三步，STORE**：把新值写回内存。

**关键点：三步之间随时可能被线程调度打断。** 线程 A 做完 LOAD（读到 5）被切走，线程 B 也 LOAD（也读到 5）、ADD、STORE（写回 6），A 回来后再 ADD、STORE（也写回 6）——两次 `count++` 只让 `count` 从 5 变成 6，**丢了一次更新**。`count++` 不是原子操作（atomic operation），这是竞态的根源。

**原子性（atomicity）**：一个操作「要么完整执行、要么完全不执行，中间不可被打断」。`count++` 不原子，所以必须用同步机制把它「包起来」——这正是下一节 `synchronized` 要做的。

## 3 synchronized：互斥锁让临界区原子化

**`synchronized`** 是 Java 的**内置锁（intrinsic lock）**。它保证：**同一时刻只有一个线程能进入被 `synchronized` 修饰的代码块**——其他线程必须等它出来。

```java
public class Counter {
    private int count = 0;
    public synchronized void increment() {   // 整个方法上锁
        count++;                             // LOAD-ADD-STORE 现在不可被打断
    }
    public synchronized int get() { return count; }   // 读也要同步！
}
```

**两个线程同时调 `increment()` 时**：第一个拿到锁进入方法，第二个在方法门口**阻塞（BLOCKED）**等待；第一个执行完释放锁，第二个才进入。于是 `count++` 的三步成为一个不可分割的整体——这就是「互斥（mutual exclusion）」：临界区同一时刻只许一个线程进。

**同步的传递性**：方法上 `synchronized`，等价于把方法体包进 `synchronized (this) { ... }`。锁的粒度可大（整个方法）可小（代码块），**锁的粒度越小，并发度越高**——只锁必须保护的那几行。<span class="marginnote">`synchronized` 是可重入的：同一线程可多次获取同一把锁（方法 A 调方法 B、两者都上锁也不会死锁），JVM 用「持有计数」记录进入次数，退出一次减一。重入让组合调用不再担心「自己锁自己」。</span>

**辨析｜易错点：读也要加锁。** 只给写加 `synchronized`、读不加，读者可能读到**过期的值**（写线程改了还没刷回内存，或改了但读者读到旧值）。同步必须**对称**——共享字段的每次读、每次写都要在锁的保护下。

## 4 volatile：轻量级的可见性保证

**可见性（visibility）问题**与竞态不同：竞态是「读改写被打断」，可见性是「改了，但别人看不见」。Java 内存模型允许线程把变量缓存进自己的 CPU 缓存/寄存器——线程 A 改了 `flag`，线程 B 可能一直读着自己缓存里的旧值，**无限循环**：

```java
// 反例：flag 不 volatile，B 可能永远看不见 A 的修改
boolean flag = false;    // 应声明为 volatile boolean flag
// 线程 A：flag = true;
// 线程 B：while (!flag) { }   // 可能死循环！
```

**`volatile` 修饰的变量**告诉 JVM：**每次读都从主内存读，每次写都立即写回主内存，并且禁止指令重排**——保证「一个线程的写，其他线程立刻可见」。

```java
volatile boolean flag = false;
// 线程 A：flag = true;       // 立即对 B 可见
// 线程 B：while (!flag) { }  // 正常退出循环
```

**`volatile` 与 `synchronized` 的区别**：`volatile` 只解决**可见性**，不解决**原子性**——`volatile int count; count++` 依然不是原子的（还是三步）。`synchronized` 同时解决**原子性 + 可见性 + 互斥**（进锁刷新缓存、出锁写回内存）。

| 维度 | `volatile` | `synchronized` |
| --- | --- | --- |
| 原子性 | 无 | 有 |
| 可见性 | 有 | 有（进锁刷缓存） |
| 互斥 | 无 | 有 |
| 开销 | 轻 | 重（锁竞争） |
| 适用 | 一个写、多个读的标志位 | 读改写、复合操作 |

**重点结论：`volatile` 适合「一个线程写、其他线程读」的简单标志**（状态开关、发布引用）；一旦涉及「读改写」（计数、累加、CAS 之外的复合操作），必须 `synchronized` 或 `AtomicInteger`。把 `volatile` 当万能钥匙、对 `count++` 也用 volatile，是经典误区。

**公式解析：volatile 可见性的本质**

`volatile` 解决可见性的机制，本质是「绕过线程本地缓存」：

$$
\text{写} \;\text{volatile:} \quad \text{CPU 缓存} \to \text{主内存（立即刷回）}
$$

$$
\text{读} \;\text{volatile:} \quad \text{主内存} \to \text{CPU 缓存（每次重新读）}
$$

配合「禁止重排」，`volatile` 保证了：**先写后读的跨线程顺序确定**。这正是 Java 内存模型（JMM）里 happens-before 规则的一种——想深入，第三级《并发与分布式》会展开 JMM 的全貌。

## 5 小结

- **线程**是进程内的并行执行流；用 `start()` 启动、别直接 `run()`。
- **竞态**源于 `count++` 不原子（LOAD-ADD-STORE 可被打断），需同步保护。
- **`synchronized`** 提供互斥 + 原子性 + 可见性；**读写必须对称加锁**。
- **`volatile`** 只保证可见性、不保证原子性，适合「一写多读」的标志位。
- 死锁、线程池、并发集合是进阶主题，下一节用 JUC 的「自动挡」接管并发。

在下一节，我们将把「手动挡」换成「自动挡」——**并发集合、执行器与 CompletableFuture**。