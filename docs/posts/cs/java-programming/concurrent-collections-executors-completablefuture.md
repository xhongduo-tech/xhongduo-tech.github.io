---
title: 并发集合、执行器与 CompletableFuture
date: 2026-08-07
---

# 并发集合、执行器与 CompletableFuture

<div class="epigraph">
<p>别重复造轮子：线程池替你管线程，并发集合替你管同步，CompletableFuture 替你管编排。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第14章 ｜ 2026-08-07</p>
</div>

## 为什么从并发集合与执行器开始

上一章的 `synchronized` 是**手动挡**——你亲自给每个临界区上锁。手动挡的问题在于：锁的粒度、锁的顺序、死锁风险都要人脑维护。`java.util.concurrent` 包（简称 **JUC**）是 Java 并发的「**自动挡**」：**并发集合**帮你把共享容器做成线程安全的，**执行器（Executor）**帮你把线程池化、复用，**CompletableFuture** 帮你把异步任务编排成流水线。这一篇把这三件套讲透——它们覆盖了日常并发编程 90% 的场景，绝大多数时候你根本不需要手写 `synchronized`。

## 1 并发集合：线程安全的容器

普通的 `HashMap`、`ArrayList` 在多线程下共享是**危险的**：扩容、`put` 时的数据竞争会损坏内部结构，甚至死循环。JUC 提供了一组**专为并发设计**的集合：

**`ConcurrentHashMap`**——并发的 `HashMap`。它在 Java 8 后采用**细粒度锁/无锁 CAS**：不同桶可以同时读写，`putIfAbsent`、`compute`、`merge` 等复合操作也是原子的。

```java
ConcurrentHashMap<String, Integer> counts = new ConcurrentHashMap<>();
counts.merge(word, 1, Integer::sum);   // 原子地「不存在就放 1，存在就 +1」
```

**`CopyOnWriteArrayList`**——「写时复制」的并发 List。读操作完全无锁（直接读快照），写操作复制整个底层数组再改。**适合「读多写极少」**的场景（如配置监听器列表），写多时复制开销大。

**`BlockingQueue`**——带阻塞语义的队列：`put` 在队列满时阻塞，`take` 在队列空时阻塞。它是**生产者-消费者模式**的天然实现——生产线程往队列放任务，消费线程从队列取任务，两端不用互相等待轮询。<span class="marginnote">`ArrayBlockingQueue`（有界）、`LinkedBlockingQueue`（无界或指定界）、`PriorityBlockingQueue`（按优先级出队）是三个常用实现。有界队列 + 阻塞正是「背压（backpressure）」的实现：生产者太快时被挡住，不会把内存撑爆。</span>

**辨析｜易错点：`Hashtable` 与 `Collections.synchronizedMap` 是「老的并发方案」**——它们给**整个 Map 加一把大锁**，任何读写都串行，并发度极低。**新代码一律用 `ConcurrentHashMap`**，它按桶加锁，并发度接近读多写少的理想值。

## 2 执行器与线程池：别裸 new 线程

**为什么不要手动 `new Thread` 跑任务？** 创建线程很贵（要分配栈、与 OS 交互），频繁创建销毁浪费巨大；线程数失控还会拖垮系统。**线程池（thread pool）**提前创建一批线程，任务来了就分配、做完就复用，从根上解决这两个问题。

**`ExecutorService`** 是线程池的统一抽象，用 `Executors` 工厂创建：

```java
// 固定大小线程池：最常用
ExecutorService pool = Executors.newFixedThreadPool(10);
// 单线程池：保证任务顺序执行
ExecutorService single = Executors.newSingleThreadExecutor();
// 缓存线程池：任务多则新线程，闲则回收（注意：任务量不可控时慎用）
ExecutorService cached = Executors.newCachedThreadPool();

pool.execute(() -> System.out.println("提交一个 Runnable"));
Future<Integer> f = pool.submit(() -> 1 + 2);   // 提交 Callable，返回 Future
int result = f.get();                            // 阻塞等待结果
```

**`execute` 与 `submit` 的区别**：`execute` 接收 `Runnable`（无返回值），`submit` 接收 `Runnable` 或 `Callable`（有返回值），返回 `Future` 用于取结果。

**重点结论：线程池的核心理念是「池化复用 + 有界资源」**——固定大小线程池把并发度钉死在预设值，防止无节制的线程创建。**务必记得关闭**：`pool.shutdown()` 优雅关闭（等已提交任务完成）；不关，非守护线程会让 JVM 无法退出。

**线程池的三种饱和策略**（队列满时怎么办）：`AbortPolicy`（默认，抛异常）、`CallerRunsPolicy`（让提交方自己跑）、`DiscardPolicy`（静默丢弃）。选哪种取决于「丢任务 vs 降速」的取舍。<span class="marginnote">`CallerRunsPolicy` 是「背压」的妙用：队列满时让提交线程自己执行任务，提交方被「拖住」，自然放慢生产速度——任务不丢、系统不被冲垮。Java 并发工具里这类「以退为进」的设计非常多。</span>

**生产级线程池最好直接 `new ThreadPoolExecutor`**：`Executors` 工厂的便捷实现默认使用无界队列（`newFixedThreadPool` 用无界 `LinkedBlockingQueue`），任务积压时队列无限膨胀、内存告急。显式指定**有界队列 + 饱和策略 + 自定义线程工厂**（给线程起有含义的名字，排障时能从线程转储认出它）才是可运维的配置：

```java
new ThreadPoolExecutor(
    10, 20, 60L, TimeUnit.SECONDS,
    new ArrayBlockingQueue<>(100),     // 有界队列，背压生效
    new ThreadFactoryBuilder().setNamePrefix("order-pool-").build(),
    new ThreadPoolExecutor.CallerRunsPolicy()
);
```

这条经验与你下一章学的「并发编程最佳实践」一脉相承：**默认选项往往牺牲可观测性，生产环境要显式、要有界、要可命名**。

## 3 公式解析：CompletableFuture 的异步编排

`Future.get()` 是**阻塞等待**——你调它，当前线程就停在那等结果。而 `CompletableFuture` 是**非阻塞编排**：任务完成后自动触发后续动作，不用干等。它把异步逻辑写成「**回调流水线**」：

$$

\text{task} \to \text{thenApply(变换)} \to \text{thenCompose(连接)} \to \text{exceptionally(兜底)}

$$

对这条流水线做三步拆解：

- **第一步，启动异步**：用 `CompletableFuture.supplyAsync(() -> 计算, pool)` 把任务丢进线程池异步执行，返回一个「未来的结果」。
- **第二步，串联变换**：`thenApply(结果 -> 新结果)` 在前一个任务**完成后**自动把结果喂给变换函数，产出新的 `CompletableFuture`——多个 `thenApply` 串成「先算 A、再算 B、再算 C」的流水线，全程不用阻塞等待。
- **第三步，兜底异常**：`exceptionally(e -> 默认值)` 捕获链上任何一步的异常并给出回退结果，让流水线「即使出错也有出口」。

```java
CompletableFuture<String> cf =
    CompletableFuture.supplyAsync(() -> fetchUser(id), pool)
        .thenApply(user -> user.getDept())          // 拿到用户后取部门
        .thenApply(dept -> dept.getManager())       // 再取部门经理
        .exceptionally(e -> Manager.DEFAULT);       // 任一步出错就兜底
```

**`thenApply` 与 `thenCompose` 的区别**：`thenApply` 的变换函数返回**普通值**（拆开），`thenCompose` 的变换函数返回**另一个 `CompletableFuture`**（不拆开、直接衔接）——需要「异步后再异步」时用 `thenCompose`，否则回调里嵌套 `CompletableFuture` 会越套越深。

**并行汇聚**：`allOf(...)` 等所有任务完成、`anyOf(...)` 任意一个先完成即可——这是「扇出/扇入（fork-join）」的声明式写法，对应后面大模型推理里的「多个候选打分后再聚合」场景。

**关键收获**：`CompletableFuture` 把「等待结果」从**阻塞**（`Future.get()`）变成了**回调编排**（`thenApply` 等）——代码不卡线程，线程池的线程得以服务更多请求。这正是「异步非阻塞」编程的范式：从命令式「我要等」，到声明式「完成时做什么」。

## 4 核心对比表：三件套的分工

纯概念主题用**核心对比表**替代公式解析的展开，把 JUC 三件套的职责钉死：

| 组件 | 解决的问题 | 心智模型 | 何时用 |
| --- | --- | --- | --- |
| 并发集合 | 共享容器的线程安全 | 「线程安全的 HashMap/List」 | 多线程共享容器 |
| 执行器 | 线程的创建与复用 | 「线程池：任务队列 + 工作线程」 | 异步执行任务 |
| CompletableFuture | 异步结果的编排 | 「回调流水线」 | 多步异步、依赖组合 |

**重点结论：三件套是「自动挡」，但不是「免检」。** 并发集合管住了容器，执行器管住了线程，CompletableFuture 管住了编排——但**共享可变状态的正确性**仍需你理解（见《并发基础》与《并发编程最佳实践》）。JUC 消灭的是「重复造轮子」，不是「并发正确性思考」。

## 5 小结

- **`ConcurrentHashMap`** 按桶并发，替代 `Hashtable`/`synchronizedMap`；`CopyOnWriteArrayList` 适合读多写少。
- **`BlockingQueue`** 的 `put`/`take` 阻塞语义天然实现生产者-消费者与背压。
- **线程池**用 `ThreadPoolExecutor` 显式配置：有界队列 + 饱和策略 + 可命名线程工厂。
- **`CompletableFuture`** 用 `thenApply`/`thenCompose`/`exceptionally` 编排异步流水线，非阻塞。
- `execute` 无返回值，`submit` 返回 `Future`；记得 `shutdown()` 关线程池。

在下一节，我们把并发的「武器与纪律」合二为一——**并发编程最佳实践**。