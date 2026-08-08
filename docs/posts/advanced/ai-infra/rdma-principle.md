---
title: RDMA 原理：内核旁路、零拷贝与队列对（QP）
date: 2026-08-07
---

# RDMA 原理：内核旁路、零拷贝与队列对（QP）

<div class="epigraph">
<p>The network is the computer.</p>
<p>网络即计算机。</p>
<footer>—— 约翰 · 盖奇（John Gage），Sun Microsystems，1984</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 集合通信·RDMA ｜ 2026-08-07</p>
</div>

## 为什么从 RDMA 开始

上一节我们认识了 PCIe、NVLink、NVSwitch 构成的**机内互连**，但机内互连只解决「一张机器里的事」。大模型训练动辄几百上千张 GPU，**一旦跨机，数据就要走网络**。传统 TCP 网络栈是为「文件下载、网页请求」设计的，它把每一次发送都变成一场「用户态 → 内核态 → 协议栈 → 网卡」的长途旅行：四次内存拷贝、多次系统调用与上下文切换。而分布式训练需要的恰恰相反——**微秒级延迟、接近网卡极限的带宽、并且不能反复打断正在执行的 GPU 核函数**。RDMA（Remote Direct Memory Access，远程直接内存访问）就是为这些要求而生的。它让 NCCL 的跨机通信成为可能，是理解整条分布式训练技术栈的必经一站。<span class="marginnote">这条线会一直延续：本文讲「网卡如何绕过内核」，下一节讲「RDMA 跑在什么网络上」（InfiniBand / RoCE v2），再往后 NCCL 就在这些原语之上搭建集合通信。</span>

## 1 传统网络栈的代价：一次发送要旅行多远

想象进程 A 要把一块数据发给远端进程 B。走 TCP，数据要经历这样的旅程：

1. 应用调用 `send()`/`write()`——这是一次**系统调用**，CPU 陷入内核态（上下文切换）。
2. 数据从**用户态缓冲区**拷贝进**内核的 Socket 发送缓冲**（第 1 次拷贝）。
3. TCP/IP 协议栈在内核里处理分片、校验和、重传管理（纯 CPU 计算）。
4. 网卡驱动把数据从内核缓冲交给网卡（第 2 次拷贝/DMA），网卡把它发到线上。
5. 对端网卡收到后，把数据 DMA 进对端内核的接收缓冲（第 3 次拷贝）。
6. 对端应用调用 `recv()`，数据从**内核接收缓冲**拷贝进**用户态缓冲**（第 4 次拷贝）。

也就是说，**发送一次数据，CPU 至少参与四次拷贝、两次系统调用与若干次上下文切换**。每一次拷贝都占用 CPU、消耗内存带宽；每一次上下文切换都让流水线「停车」。在大模型训练这种「每步都要全量同步梯度」的场景里，这些开销直接变成墙钟时间。下图把这条路径与 RDMA 路径放在一起对比。

![传统 TCP 路径与 RDMA 路径对比](/images/ai-infra/rdma-principle-1.svg)

## 2 RDMA 的三板斧：内核旁路、零拷贝、网卡卸载

**RDMA 的三大核心技术是：内核旁路（kernel bypass）、零拷贝（zero-copy）与网卡卸载（offload）。** 三者分别攻击传统路径里的三处开销：

**内核旁路**：数据不再经过内核网络栈。应用与网卡之间建立一条直接的「私有通道」，所有数据移动都在用户态与网卡之间完成，没有系统调用、没有上下文切换。
**零拷贝**：发送端不再把用户缓冲拷贝进内核缓冲，接收端也不再从内核缓冲拷回用户缓冲。数据在**注册内存**与网卡之间直接 DMA，一次拷贝都不发生。
**网卡卸载**：分片、重组、确认、重传、拥塞控制这些协议杂务全部由**RNIC**（RDMA 网卡，全称 RDMA-capable Network Interface Card）硬件完成。CPU 与 GPU 完全从数据面（data plane）退出，只保留控制面（control plane）。

这三件事本质上是同一件事的三个侧面：**让网卡直接访问应用内存，让协议栈从「软件」变成「硬件」，让内核从「数据必经之路」变成「只管初始化」**。把「谁来搬数据」从 CPU 换成网卡，就是 RDMA 的全部秘密。

实现 RDMA 的网络技术有三条路线：**InfiniBand**（IB，原生 RDMA 网络，从设计之初就以 RDMA 为目标）、**RoCE**（RDMA over Converged Ethernet，把 RDMA 封装进以太网）与 **iWARP**（把 RDMA 跑在 TCP 之上，因性能最差而少见）。三者的上层编程接口统一，都是本文后面要讲的 **Verbs API**——这也是它们能无缝切换的底气。<span class="marginnote">iWARP 在 TCP 之上做 RDMA，等于「给一个本来就要拷贝的协议栈打补丁」，延迟很难压下去，工业界主流只剩 IB 与 RoCE 两条路。</span>

## 3 队列对（QP）与完成队列（CQ）：RDMA 的编程模型

RDMA 的编程模型不叫「socket」，而叫 **队列对（Queue Pair，QP）**。每个 QP 由一对工作队列组成：**发送队列（Send Queue，SQ）**与**接收队列（Receive Queue，RQ）**。应用往队列里投递工作请求，网卡从中取走并执行，执行完把结果写进**完成队列（Completion Queue，CQ）**。**这个模型里没有系统调用参与数据面——投递与取完成都是直接在用户态写的寄存器/内存映射。**

**WQE（Work Queue Element，工作队列元素）**：一次收发的最小单位。应用调用 `ibv_post_send`/`ibv_post_recv` 把 WQE 投进 SQ/RQ。
**CQE（Completion Queue Entry，完成队列元素）**：WQE 执行完后的回执。应用调用 `ibv_poll_cq` 从 CQ 里取出。
**SRQ（Shared Receive Queue，共享接收队列）**：多个 QP 共用一个接收队列，避免为每个 QP 预分配接收缓冲。
**内存注册（memory registration）**：使用 RDMA 前，内存必须通过 `ibv_reg_mr` 注册，把虚拟地址固定（pin）住并换取一把「钥匙」（`lkey`/`rkey`），告诉网卡这段内存的物理位置与访问权限。

一段最核心的发送流程：

```text
# RDMA 发送流程（Verbs API 伪代码）
# 1. 注册内存：把 buf 的物理地址固定下来，换取钥匙 lkey
mr = ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE)

# 2. 组装 WQE：告诉网卡“把 buf 里的 size 字节发出去”
wqe = {
    opcode:   IBV_WR_SEND,
    num_sge:  1,
    sg_list: [{ addr: buf, length: size, lkey: mr.lkey }],
}

# 3. 投递 WQE 到发送队列——只写队列、立即返回，传输由网卡异步完成
ibv_post_send(qp, wqe)

# 4. 需要结果时去完成队列轮询（无中断、无系统调用）
while (cqe = ibv_poll_cq(cq, 1)) is empty:
    continue
# 取到 CQE，本次发送完成
```

注意 `ibv_post_send` **不会阻塞**：它只是把 WQE 写进队列就返回，真正的传输由网卡异步完成。这个「异步 + 轮询」模型是 RDMA 延迟低的另一个来源——没有中断、没有系统调用，CPU 只需要在需要结果时去 CQ 里看一眼。

## 4 两种语义：消息语义与内存语义

RDMA 提供两种截然不同的数据传输语义，它们的分界线是**对端 CPU 是否参与**：

| 语义 | 操作 | 对端 CPU | 典型场景 |
| --- | --- | --- | --- |
| 消息语义（two-sided） | `Send` / `Recv` | 必须参与（投递 Recv WQE） | 像消息传递一样收发，匹配语义 |
| 内存语义（one-sided） | `RDMA Read` / `RDMA Write` / `Atomic` | 完全不参与 | 直接读写对端内存，无需通知对端 |

**消息语义是「打电话」**：双方都要在场，发送端投递 Send，接收端必须预先投递 Recv 才能配对上。**内存语义是「隔空取物」**：本地进程持有对端内存的 `rkey`（远程钥匙），就能直接读走或写入对端的内存，对端 CPU 毫不知情——甚至对端进程此刻根本没在运行也能完成。One-sided 操作是大规模并行程序里最有力的武器，也是实现「隐式同步」与「无接收侧开销」传输的基础。<span class="marginnote">NCCL 的许多内核集合通信在 IB 上就用 one-sided RDMA Write 实现：数据直接被写到目标 GPU 的内存里，目标 GPU 的 SM 全程不用碰网络。</span>

## 5 公式解析：零拷贝到底省了什么

把「拷贝」量化，才能理解 RDMA 的价值。设一次要搬运 $S$ 字节的数据，单次内存拷贝的代价近似为 $T_{\text{copy}} \approx S / b$（$b$ 是内存复制带宽），一次上下文切换的代价为 $T_{\text{ctx}}$，网络往返的固有延迟为 $T_{\text{lat}}$。则传统 TCP 路径的端到端延迟约为：

$$T_{\text{tcp}} \approx \underbrace{4 \cdot \frac{S}{b}}_{\text{四次拷贝}} + \underbrace{K \cdot T_{\text{ctx}}}_{\text{若干次上下文切换}} + T_{\text{lat}}$$

对这条式子做三步拆解：

- **第一步，数清拷贝**：第 1、2、3、4 次拷贝分别对应「用户→内核发送缓冲」「内核→网卡」「网卡→对端内核接收缓冲」「对端内核→对端用户」。四次拷贝每个都要把 $S$ 字节在内存里搬一遍，共搬 $4S$ 字节。
- **第二步，数清切换**：`send`、`recv` 等系统调用各触发一次用户态/内核态切换，接收端还有中断与唤醒开销，计为 $K \cdot T_{\text{ctx}}$。$K$ 通常是个位数，但每次切换都是几十微秒量级之外的「停车」。
- **第三步，对比 RDMA**：RDMA 路径没有拷贝、没有系统调用，只剩下：

$$T_{\text{rdma}} \approx T_{\text{lat}}' + O(S)$$

其中 $T_{\text{lat}}'$ 是网卡硬件处理与链路传播的固有延迟（微秒级）。**四次拷贝与若干次上下文切换，就是 RDMA 与传统 TCP 的全部差距**。当 $S$ 很大（比如梯度张量动辄几十 MB），拷贝项 $4S/b$ 占主导，RDMA 靠省掉四倍的搬运带宽赢；当 $S$ 很小（比如控制消息），切换项 $K \cdot T_{\text{ctx}}$ 占主导，RDMA 靠省掉系统调用赢。

## 6 为什么大模型训练需要 RDMA：GPU Direct RDMA

对 GPU 训练而言，RDMA 的价值还要再放大一层，因为**数据的最初产地与最终归宿是显存，而不是主机内存**。如果数据要「GPU → 主机内存 → 网卡 → 对端」，中途仍免不了跨越 PCIe 的搬运。**GPU Direct RDMA（GDR）** 让网卡直接通过 PCIe 读写 GPU 的显存：梯度在显存里算好之后，网卡直接把显存里的数据搬走，全程不经过主机内存、不占用 CPU。<span class="marginnote">GDR 需要 NVLink/PCIe P2P 之外的另一段「P2P」：NIC 与 GPU 之间通过 PCIe BAR 直通。驱动里通常用 `nvidia-smi topo -m` 检查「HCA P2P Capability」是否 Enabled。</span>

NCCL 正是这么做的：跨机时，NCCL 在支持 GDR 的硬件上把数据从 GPU 显存直接灌进 InfiniBand/RoCE 网卡，再配合上一节讲过的 NVLink 做机内归约，构成「机内 NVLink + 跨机 RDMA」的两级流水。**RDMA 不是可选项，而是大模型跨机通信的物理底座。** 下一节我们会拆开这个底座本身：InfiniBand 与 RoCE v2 用什么机制保证数据不丢、不堵。

## 7 小结

- 传统 TCP 路径的代价是**四次内存拷贝 + 多次系统调用/上下文切换**，CPU 深陷数据面。
- RDMA 三板斧：**内核旁路、零拷贝、网卡卸载**，把「搬数据」从 CPU 交给 RNIC。
- RDMA 的编程对象是**队列对 QP（SQ+RQ）与完成队列 CQ**：投递 WQE、轮询 CQE，全程异步、无系统调用。
- 两种语义：**消息语义（two-sided Send/Recv）**与**内存语义（one-sided RDMA Read/Write/Atomic）**，分界在对端 CPU 是否参与。
- 零拷贝省掉 $4S/b$ 的拷贝时间与 $K \cdot T_{\text{ctx}}$ 的切换时间；**GPU Direct RDMA** 让网卡直读显存，NCCL 跨机通信由此而来。

在下一节，我们将回答「RDMA 跑在什么网络上」：InfiniBand 与 RoCE v2 各自如何做到无损传输，PFC 与 DCQCN 又各自付出怎样的代价。
