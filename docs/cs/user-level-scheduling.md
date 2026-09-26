---
title: 用户态调度与 M:N
date: 2026-09-08
section: cs
---

# 用户态调度与 M:N

<div class="epigraph">
<p>M:N 把许多用户纤维映射到较少内核线程：切换可在用户完成，但阻塞系统调用会卡住整根内核线程，除非另做包装。</p>
<footer>—— 据 Anderson et al. 对调度器激活的论述；Go/Erlang 运行时实践；[线程](/cs/thread-shared-addr) 为先修</footer>
</div>

[sched_ext](/cs/sched-ext) 仍决定内核任务。[ULT/KLT](/cs/ult-klt) 主干有分类。缺口是 **用户态 M:N 的 OS 接头**：阻塞、信号、与 1:1 的取舍。

## 问题

1:1：每个 goroutine 若都是内核线程，创建贵。M:N：用户调度器把可运行纤维放到 M 个 pthread 上。缺口：`read` 阻塞会占死一个 M；需要 netpoller 把 fd 变非阻塞 + 用户等待队列；与 [NAPI](/cs/napi) 无关直接，但 eventfd/epoll 是接头。本课不把 Go GMP 细节背完。

<span class="marginnote">scheduler activations 试图让内核通知用户「这根线程阻塞了」。Linux 上实践多是运行时自己包装 syscall。</span>

## 方法

运行时：工作窃取队列，fiber 切换改上下文（不进核）。I/O：epoll + 非阻塞。对照 [POSIX AIO](/cs/posix-aio)：一个内核完成，一个用户把阻塞变事件。对照 [uffd](/cs/userfaultfd)：缺页仍进核。对照 EEVDF：内核只看见 N 个线程的 util。

```mermaid
flowchart TD
  F["用户纤维"] --> U["用户调度器"]
  U --> M["内核线程"]
  BLK["阻塞 syscall"] --> STUCK["该 M 卡住"]
  IO["非阻塞+epoll"] --> U
```

## 机制

M:N 用用户切换换百万级并发，把 OS 调度粒度变粗。正确性取决于运行时包装了所有阻塞点。不要写成语言课的 async 语法。与 [cgroup](/cs/cgroup-cpu-sched)：限额作用在 M 上，纤维数看不见。

一次「被包装的阻塞读」如何在运行时里绕回用户态，而不卡死内核线程：

```mermaid
flowchart TD
  FIB["纤维调用 read(fd)"] --> WRAP["运行时包装层接管"]
  WRAP --> NB["fd 已是非阻塞, 内核立即返回 EWOULDBLOCK"]
  NB --> PARK["纤维挂到该 fd 的用户等待队列"]
  PARK --> STEAL["M 不闲着, 窃取/领取下一根纤维"]
  EP["epoll 线程等待事件"] -->|"fd 可读"| WAKE["把纤维挪回运行队列"]
  WAKE --> RESUME["任意 M 重新执行该纤维"]
```

信号与 TLS：要按纤维还是按 M 投递，是实现地狱，课序只要求意识到裂缝。


实现上：运行时必须包装所有可能阻塞的 libc，包括 getaddrinfo 和锁。信号投到 M 上，要转给当前纤维。cgroup 限额看见的是 M 的 CPU，不是纤维数。 读法上只引用[上一课](/cs/sched-ext)的结论，不把对象换成训练推理或限价簿。

<span class="marginnote">数字实例：一根 pthread 默认预留约 $8$ MB 栈，$10^4$ 根就是 $80$ GB——1:1 模型根本开不出百万并发；Go 纤维初始栈只有约 $2$ KB 且可增长，$10^6$ 根也只占约 2 GB。用户栈按需伸缩，是 M:N 能撑住「百万级」的第一张底牌。</span>

<span class="marginnote">常见误区：以为只要用了 M:N 运行时，阻塞就无害。绕过包装层的调用——比如直接 `syscall.Syscall` 发阻塞读，或经 cgo 调一个自己 `flock` 的 C 库——会让宿主 M 卡死，内核看不见里面的纤维，其它纤维也被这一根 pthread 拖住。运行时的正确性取决于「所有阻塞点都被包装」，漏一处就漏一整根 M。</span>

<span class="marginnote">直觉类比：工作窃取像自助餐补菜——每个 M 手里有一小盘活儿，吃完了就近偷别人盘里的，而不是全体排队找中央调度员。窃取只发生在本地队列空了之后，多数切换仍是几十纳秒级的用户态压栈换栈，不进内核。</span>

本课在操作系统进阶的「调度进阶 / 公平、实时与能耗」课序里，对象是 **用户态调度与 M:N**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入 stackful vs stackless 的全部。不保证与 C 库的取消点。下一课更简单的一端：协作调度。


版本字段会变，课序钉的是机制对象「用户态调度与 M:N」，不是某一主线内核的结构体名。
后课默认：M:N 依赖非阻塞包装。自愿让出的协作式调度，下一课。

## 小结

- M:N：用户调度纤维，内核只见少量线程。
- 阻塞 syscall 是主要裂缝。
- 协作调度是下一课。
- 出处：Anderson 调度器激活；Go 运行时；*OSTEP* 线程。
