---
title: OpenMP 基础（并行区/工作共享）
date: 2026-08-07
---

# OpenMP 基础（并行区/工作共享）

<div class="epigraph">
<p>最好的并行，是不让程序员看见的并行。</p>
<footer>—— OpenMP 设计哲学</footer>
</div>

<div class="article-byline">
<p>第三级 · 高性能计算 ｜ 陈国良《并行计算》 第五章 §5.1 ｜ 2026-08-07</p>
</div>

## 为什么从 OpenMP 开始

MPI 处理的是「多台机器」，但在**一台机器内部**，还有一层并行没利用：

多核 CPU 的多个线程。

这就是 **OpenMP（Open Multi-Processing）** 的战场。

它是一组**编译指导指令（compiler directives）**：

在 C/C++ 里以 `#pragma omp` 开头，程序员只需告诉编译器「这段代码可以并行」，剩下的线程创建、调度、同步全由编译器与运行时搞定。

<span class="marginnote">OpenMP 的杀手锏是<strong>增量并行</strong>：从串行程序出发，逐段加 pragma，边加边验证正确性——比 MPI 从零写并行要温和得多。</span>

本节的纲：

**Fork-Join 执行模型**、并行区、工作共享、以及最容易踩的 shared/private 坑。

## 1 Fork-Join 模型

OpenMP 的执行模型叫 **Fork-Join（分叉-合并）**：

程序开始时只有一个**主线程（master thread）**，串行执行；
遇到 `parallel` 指令时，主线程**分叉（fork）**出一组工作线程；
并行区结束后，工作线程合并（join）回主线程，继续串行。

这个模型意味着：

**并行只发生在并行区内部，区外全是单线程。**

线程的个数由环境变量 `OMP_NUM_THREADS` 或子句 `num_threads` 控制，运行前就定好。

**核心直觉：** OpenMP 程序 = 串行骨架 + 若干并行区。

并行区可以嵌套、可以循环，但每次进入都是一次「fork，干活，join」。

## 2 并行区：parallel

最基础的指令是开启一个并行区：

```c
#pragma omp parallel
{
    printf("Hello from thread %d\n", omp_get_thread_num());
}
```

每个工作线程都会执行大括号里的代码——这与 SPMD 的「多进程跑同一份代码」同构，只是线程共享内存。

两个常用查询函数：

`omp_get_thread_num()`：我是第几个线程；
`omp_get_num_threads()`：一共几个线程。

`num_threads` 子句指定线程数：

```c
#pragma omp parallel num_threads(4)
{
    // 4 个线程并行执行
}
```

<span class="marginnote">并行区内每个线程默认能看到<strong>所有共享变量</strong>——这既是 OpenMP 好写的来源，也是数据竞争频发的根源。</span>

**辨析｜易错点：** 并行区内的输出（打印）顺序不保证，与 MPI 一样。

区分「谁干的」要靠线程号，不能靠输出顺序。

## 3 工作共享：for 与 sections

一个裸的 `parallel` 区里，所有线程做**同一份**工作——这往往不是我们想要的。

我们要的是**把大循环拆开，每人算一段**。

于是有了**工作共享指令（work-sharing constructs）**：

**for：把循环分给各线程。**

```c
#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i < n; ++i)
        a[i] = b[i] * c[i];
}
```

编译器自动把 `for` 的迭代切块分给线程，谁算哪几轮由调度子句决定。

**sections：把不同的代码段分给不同线程。**

```c
#pragma omp parallel
{
    #pragma omp sections
    {
        #pragma omp section
        { /* 任务 A：由某个线程执行 */ }
        #pragma omp section
        { /* 任务 B：由另一个线程执行 */ }
    }
}
```

`for` 适合**数据并行**（同样操作、不同数据），`sections` 适合**功能并行**（不同操作）。

<span class="marginnote">`for` 有<strong>隐式屏障</strong>：循环结束后所有线程自动对齐。想跳过对齐，用 `nowait` 子句。</span>

**辨析｜易错点：** `for` 与 `parallel for` 是两回事。

`for` 必须待在某个 `parallel` 区里才生效；`parallel for` 是「开并行区 + 分循环」的合体写法，两个可以互相替换。

## 4 数据环境：shared 与 private

线程共享内存，意味着变量默认**对所有线程可见**——这叫**共享（shared）**。

但循环里的临时变量必须是**私有的（private）**：

每个线程一份自己的副本，互不干扰。

```c
int x = 0;
#pragma omp parallel private(x)
{
    x = omp_get_thread_num();   // 每个线程各自的 x
    printf("%d\n", x);
}
```

数据属性的子句：

`shared(x)`：x 所有人共享（默认行为）；
`private(x)`：x 每人一份副本，进入并行区时未初始化；
`firstprivate(x)`：x 每人一份，但副本用进入时的值初始化；
`lastprivate(x)`：最后一个迭代的 x 值在区后写回共享变量。

<span class="marginnote">循环索引 `i` 在 `for` 里<strong>自动私有</strong>——这是编译器替你做的第一件事，也是新手最容易忘记但编译器最勤快的一件。</span>

**辨析｜易错点：** OpenMP 的默认是 **shared**，与直觉相反。

C 语言里「每个函数栈上的变量本来该是独立的」，但 OpenMP 把并行区外的变量默认共享——**忘记声明 private 是数据竞争的第一来源**。

## 5 代码解析：向量加法

把上面的知识拼成一个完整例子：

```c
#pragma omp parallel for
for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];       // 向量加法：每个线程算自己那几轮
```

三步读懂：

- **第一步，`#pragma omp parallel for`**：开启并行区并把循环分给各线程；
- **第二步，索引 `i` 自动私有**：每个线程算自己的那几轮，互不踩踏；
- **第三步，数组 `a`、`b`、`c` 默认共享**：大家读写同一块内存——因为各自只碰自己那段的元素，所以安全。

如果迭代间存在依赖（第 $i$ 轮要用第 $i-1$ 轮的结果），这个并行就有错——**判断循环能否并行，先看依赖，再看语法**。

## 6 核心对比：MPI 与 OpenMP

| 维度 | MPI | OpenMP |
| --- | --- | --- |
| 内存模型 | 分布式（进程） | 共享（线程） |
| 并行单位 | 进程 | 线程 |
| 通信 | 显式消息 | 共享变量 + 同步 |
| 编程方式 | 函数库调用 | 编译指令 pragma |
| 适用范围 | 单机到超算 | 通常单机多核 |
| 增量并行 | 难（需重构） | 易（逐段加指令） |

**核心结论：** MPI 管「机间」，OpenMP 管「机内」。

现代超算几乎都是「MPI 分节点 + OpenMP 分核」的混合模式——这正是后面《混合并行》一节的内容。

## 7 小结

- **OpenMP** 是共享内存的指令式并行，执行模型是 **Fork-Join**。
- **并行区** `parallel` 让所有线程执行同一段代码。
- **工作共享**：`for` 分循环、`sections` 分功能段。
- **数据环境**默认 shared，临时变量要 `private`——**默认共享是竞态第一来源**。
- 铁律：**并行前先查循环依赖，依赖存在就不能并行。**

在下一节，我们处理共享内存特有的难题：**同步与归约**——当多个线程要更新同一个变量时怎么办。
