---
title: CUDA Stream 与异步执行、事件计时
date: 2026-08-07
---

# CUDA Stream 与异步执行、事件计时

<div class="epigraph">
<p>如果你不能测量它，你就不能改进它。</p>
<footer>—— 开尔文勋爵（Lord Kelvin）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Stream 开始

前几课我们把单个 kernel 的性能逼到极限：合并访存、bank conflict、Shared Memory、occupancy。但真实程序不是「一个 kernel 跑完再跑下一个」，而是**一批 kernel 连同数据搬运挤在一起跑**。这一课回答两个工程上每天都在问的问题：**host 什么时候真的「等」GPU？如何在数据搬运与计算之间把时间重叠起来？** 答案是 CUDA Stream（流）与事件（event）。<span class="marginnote">回到第一篇开头的吞吐/延迟之争：单 kernel 优化是在压「单个请求的延迟」，而 Stream 重叠是在把整条流水线的「吞吐」顶上去——后者才是 GPU 真正的主场，也是后续《集合通信》里 NCCL 通信-计算重叠的思想原型。</span>

理解 Stream 的关键是先把「host 与 device 异步」这个默认事实焊进脑子里：**CUDA 里几乎一切 launch 都是异步的**，真正的同步边界要靠你显式声明。

## 1 CUDA 的异步执行模型：别把 host 当成「等 GPU 的人」

一个最常见的误解是「`kernel<<<...>>>()` 调用会等 kernel 跑完才返回」。**事实正相反：kernel 启动是异步的，launch 语句把命令提交给驱动后立即返回，host 线程不等 GPU 干完活。** 这背后的机制是**命令队列（command queue）**：驱动把 kernel、拷贝等指令依次写入一块 CPU 与 GPU 共享的命令缓冲，GPU 自己按队列取指令执行。

由此派生出三条你必须记住的规则：

- **kernel launch 默认异步**：`cudaMemcpy`（同步版）是个例外，它会把 host 阻塞到拷贝完成。
- **`cudaDeviceSynchronize()`**：唯一的「全设备屏障」，host 阻塞直到**所有**此前提交的命令完成。
- **同一队列内严格有序（in-order）**：后提交的命令不会先于前一条执行。

于是「host 什么时候等」完全由你决定的同步点控制。同步点给多了，异步白搭；给少了，你读到的结果可能还是旧的——这引出了计时与并发两个主题。<span class="marginnote">异步模型在第三级《计算机体系结构》里你其实见过：CPU 的乱序执行与写缓冲（store buffer）都是「提交后立即放行、硬件保证顺序」的同一哲学。GPU 只是把「一条指令的异步」放大成了「一整条命令队列的异步」。</span>

## 2 Stream：把任务装进不同的泳道

单条命令队列的缺点是：所有操作**串行排队**，拷贝与计算之间没有重叠的机会。CUDA 的解法是 **Stream（流）**：每条 stream 是一条独立的命令队列，**stream 内部按提交顺序执行，不同 stream 之间互不保证顺序、可以并发执行**。

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1);          // 创建两条泳道
cudaStreamCreate(&s2);

my_kernel<<<grid, block, 0, s1>>>(...);   // 第 4 个参数指定 stream
my_kernel<<<grid, block, 0, s2>>>(...);   // 两条流上的 kernel 可能同时跑
```

不指定 stream 时默认用**默认流（default stream，stream 0）**。这里有个阴险的坑：**legacy 默认流与所有其他流存在隐式同步**——向默认流提交的操作会强迫之前的其他流操作完成。很多新手写了多 stream 却看不到加速，十有八九是某个操作漏写了 stream 参数、掉进了默认流的隐式同步。<span class="marginnote">编译器加 `--default-stream per-thread` 后，每个线程拥有独立、行为与普通流一致的默认流，不再有隐式同步。但老代码的行为会变，迁移需谨慎。这是 CUDA 里「看起来改了配置，实际改了内存模型」的典型例子。</span>

要想让「搬下一块数据」和「算上一块数据」重叠，标准手法是**乒乓缓冲（ping-pong / double buffering）**：用两条 stream、两块缓冲轮流干活，让 stream 0 算 chunk $i$ 的同时 stream 1 搬 chunk $i+1$。

```cpp
// 注意：cudaMemcpyAsync 要求 pinned memory（cudaHostAlloc / cudaMallocHost 分配）
cudaMemcpyAsync(d_in, h_in + i * CHUNK, CHUNK_BYTES, cudaMemcpyHostToDevice, s[i % 2]);
my_kernel<<<grid, block, 0, s[i % 2]>>>(d_in, d_out + i * CHUNK);
```

## 3 事件（Event）：GPU 侧的时间戳与正确的计时法

测量一个 kernel 到底跑了多久，用 CPU 时钟 `clock()` 是**不可靠**的——launch 是异步的，`clock()` 量到的是「提交命令那一刻」，而不是 GPU 真正执行的时刻。CUDA 的正确计时工具是 **事件（event）**：一个记录在 stream 上的时间戳。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);  cudaEventCreate(&stop);

cudaEventRecord(start, s);            // 在 stream s 上打点
my_kernel<<<grid, block, 0, s>>>(...);
cudaEventRecord(stop, s);             // 同一条 stream 上的第二个点

cudaEventSynchronize(stop);           // host 等到 stop 事件被记录
float ms = 0.f;
cudaEventElapsedTime(&ms, start, stop); // 两个事件间的毫秒数
```

三个细节值得停下来：

- 事件是**挂在 stream 上的**：`start` 与 `stop` 之间夹着的所有操作（拷贝、kernel）都被量进来，所以事件天然给出「这条流上一段工作」的真实耗时。
- **必须 `cudaEventSynchronize`**：在 GPU 回写时间戳之前，`stop` 可能尚未生效。
- 事件还能做**跨 stream 的依赖**：用 `cudaStreamWaitEvent(stream, event)` 让一条流等另一条流的某个点，这是精细控制依赖的进阶工具。<span class="marginnote">想测「纯 kernel 时间」而排除启动开销？在 kernel 前后各打一个点即可；但如果 kernel 太短（微秒级），测量本身的开销会污染结果——这个问题正是下一课《kernel 启动开销》的主角。</span>

## 4 公式解析：copy-compute 重叠的吞吐模型

设我们有 $K$ 块数据，每块需要拷贝时间 $c$、kernel 计算时间 $k$。

**不重叠（单 stream 串行）**：每块先拷贝后计算，总时间

$$
T_{\text{串行}} = K(c + k)
$$

**乒乓重叠（双 stream）**：第一块拷贝先行，之后每 $\max(c, k)$ 时间完成一块的计算（因为拷贝与计算在两条流上并行，步调由较慢的那一侧决定），最后再排空最后一块：

$$
T_{\text{重叠}} \approx c + (K - 1)\max(c, k) + k
$$

做三步拆解：

- **第一步，看懂 $\max(c, k)$**：重叠状态下，每一个「周期」里拷贝 $c$ 与计算 $k$ 同时进行，这一周期实际耗时是两者中较长的一个。这个 $\max$ 是整条流水线的**周期（cycle time）**。
- **第二步，数项**：第一项 $c$ 是流水线的**填充（fill）**——第一块拷贝时还没有可算的东西；最后一项 $k$ 是**排空（drain）**——最后一块算完时拷贝早已结束；中间 $(K-1)$ 个周期才是稳定的满流水阶段。
- **第三步，求加速比**：$K$ 很大时，常数项可忽略，

$$
S \approx \frac{K(c + k)}{K \cdot \max(c, k)} = \frac{c + k}{\max(c, k)}
$$

于是结论一目了然：**若 $c \approx k$，$S \approx 2$，重叠几乎翻倍吞吐**；**若 $k \gg c$（计算主导），$S \to 1$，重叠收益趋近于零**——瓶颈已经不在拷贝，重叠也救不了。这个模型与《GPU 与 CPU 的设计哲学》里 Little 定律的精神一致：系统吞吐由最慢的环节决定，重叠只是把「串行等待」变成「并行填满」。

## 5 辨析｜易错点

- **「kernel launch 之后立刻读结果是对的」**——错。launch 异步，立刻访问输出会读到旧值。要么 `cudaDeviceSynchronize()`，要么用 `cudaMemcpy` 同步版或 event 同步。
- **「`cudaMemcpyAsync` 任何内存都行」**——错。**异步拷贝要求 pinned（页锁定）内存**，普通 `malloc` 的内存会被驱动降级为同步拷贝，重叠效果消失。用 `cudaHostAlloc` / `cudaMallocHost` 分配 host 侧缓冲。
- **「开了两条 stream 就一定并发」**——不一定。不同 stream 的操作**可以**并发，但能否真的并行受硬件资源约束：同一个 SM 的资源不够时，两个 kernel 仍会串行排布。Stream 提供的是「允许并发」，不是「保证并发」。
- **「用 CPU `clock()` 计时最准」**——错。kernel 异步执行，CPU 时间量不到 GPU 执行区间；正确做法是 event 计时。
- **「默认流没有隐式同步」**——错。legacy 默认流与阻塞式调用、与其他流都存在隐式同步，这是多流程序「莫名串行化」的头号元凶。

## 6 小结

- **kernel launch 默认异步**，命令进入 GPU 的命令队列；同步点由你显式声明（`cudaDeviceSynchronize`、event、同步 `cudaMemcpy`）。
- **Stream 是独立命令队列**：流内有序、流间可并发；默认流（legacy）带隐式同步，容易悄悄串行化你的程序。
- **乒乓缓冲**用两条流让「搬 $i+1$」与「算 $i$」重叠，是 copy-compute overlap 的标准实现。
- **事件计时**用 GPU 侧时间戳，`cudaEventRecord` + `cudaEventSynchronize` + `cudaEventElapsedTime` 三步；`clock()` 量不到异步执行的真实耗时。
- 重叠的收益由 $\max(c, k)$ 决定：**只有瓶颈两侧规模相当时，重叠才接近翻倍吞吐**。

在下一节，我们顺着「测量」往下追一层：**kernel 启动本身到底要花多少时间，为什么短 kernel 会被启动开销吃掉大半性能，以及如何用算子融合（kernel fusion）把这些开销一口气省掉**。
