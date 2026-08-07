---
title: PyTorch DDP 内部机制：bucket 分桶与梯度 allreduce 调度
date: 2026-08-07
---

# PyTorch DDP 内部机制：bucket 分桶与梯度 allreduce 调度

<div class="epigraph">
<p>同步不是一次性事件，而是与计算交织的流水。</p>
<footer>—— 沈翔（Shen Li），PyTorch 分布式核心维护者</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PyTorch DDP 设计文档与源码 · 训练框架篇 ｜ 2026-08-07</p>
</div>

## 为什么从 DDP 内部机制开始

你可能已经用过 `torch.nn.parallel.DistributedDataParallel`（DDP）——它是 PyTorch 里使用率最高的分布式训练接口。但「会用」和「懂它为什么快」是两回事。DDP 之所以能在几乎不损耗单卡速度的前提下完成跨卡梯度同步，靠的是一套精密的内部机制：**参数分桶（bucket）、反向钩子（autograd hook）与通信计算重叠**。

理解 DDP 的内部，你就掌握了一条普适的分布式优化主线：**把「一次大同步」拆成「多次小同步」，并让同步与计算并行**。这条主线在 FSDP、ZeRO、Megatron 里会反复出现。本篇以 DDP 为第一课讲透它。

## 1 DDP 的基本模型：多进程 + 每进程一份模型

DDP 采用**单进程单 GPU（或单进程多 GPU）**模型：每个 rank 持有一个完整模型副本，处理不同的 batch 分片。

- **模型**：每进程全量复制（无分片）。
- **数据**：每进程一份独立 batch（shuffle 不同）。
- **同步**：后向时梯度 AllReduce（求和后除以世界大小，等价于平均）。

训练循环三件套：

1. **前向**：本地 batch 前向，计算 loss。
2. **后向**：`loss.backward()` 自动求梯度；**梯度算完立即进入 AllReduce 队列**。
3. **更新**：`optimizer.step()` 前，所有梯度已完成同步，各进程用相同梯度更新相同模型。<span class="marginnote">DDP 能保证「模型一致」的前提是：所有进程在同一 step 结束时拿到相同梯度。为此 DDP 在每次 `backward` 结束时自动同步，`optimizer.step()` 只是一个普通的本地操作——一致性由梯度同步保证，而不是由 optimizer 保证。</span>

## 2 bucket 分桶：为什么要分桶

朴素做法：每个参数算出梯度，就立刻对这一个参数做一次 AllReduce。模型有几千个参数，就要发几千次消息——**每次通信都有固定开销（延迟、协议头），小消息的带宽利用率极低**。

DDP 的做法是**分桶（bucketing）**：把若干参数的梯度**拼进一个连续 buffer**，凑满一个 bucket 再一起 AllReduce。

$$\text{bucket}_k = \text{concat}\big[\nabla W_{i_1}, \nabla W_{i_2}, \ldots\big], \quad \text{size} \approx 25\text{MB}$$

- **减少消息数**：几千次小 AllReduce → 几十次大 AllReduce。
- **提高带宽利用率**：大消息占满链路，延迟占比下降。
- **顺序按反向图**：bucket 的划分顺序与反向计算顺序一致，梯度一凑齐就能发。<span class="marginnote">bucket 大小默认 25MB（`bucket_cap_mb`）。太小则消息多、延迟高；太大则要等很久才凑满一个 bucket，重叠效果变差。25MB 是 PyTorch 在「等待时长」与「消息开销」之间实测出的甜点。</span>

## 3 梯度 AllReduce 的调度：autograd hook 与重叠

DDP 真正的精巧之处在**调度**：它不在 `backward()` 结束后才做同步，而是通过**反向自动求导钩子（autograd hook）**在梯度产生的瞬间就触发同步。

当一个参数的梯度算好：

1. DDP 注册的 hook 被调用，把梯度拷贝进对应 bucket。
2. bucket 一旦凑满，立即对这个 bucket 发起**异步 AllReduce**（不阻塞反向继续算）。
3. 反向继续计算其他参数梯度时，这个 AllReduce 正在后台进行——**通信与计算重叠**。

于是总耗时为：

$$T_{\text{total}} \approx T_{\text{backward}} + T_{\text{last AllReduce}}$$

即**同步的大部分时间被反向计算盖住**，只有最后一个 bucket 的通信暴露在外。这正是 DDP「几乎零开销同步」的秘密。<span class="marginnote">「重叠」的实现依赖异步通信 API（`nccl` 的 non-blocking allreduce）与 GPU 上通信算子的并发执行。如果通信与计算串行（比如每算完一个梯度就同步等待），DDP 会比手写同步还慢。这是理解分布式优化的第一性原理：<strong>把同步藏进计算里</strong>。</span>

## 4 其他内部细节：平均、未使用参数与共享

- **梯度平均**：AllReduce 用 SUM，DDP 除以世界大小 $N$ 得到平均；或用 AVERAGE 模式。正确性上二者等价（学习率差异可吸收），但 SUM 的梯度数值更大，fp16 下要注意溢出。
- **`find_unused_parameters`**：默认 False，假定所有参数都参与后向；若模型有未用参数且不开此开关，会报错或静默不同步。
- **参数共享**：DDP 要求模型参数不跨进程共享（每进程独立副本），但**进程内**共享参数（如权值绑定）DDP 能自动去重梯度同步。
- **`no_sync()`**：梯度累积时不触发同步，攒几个 micro-batch 的梯度后再一次同步——这是「梯度累积」的官方姿势。<span class="marginnote">`no_sync()` 与 `bucket_cap_mb` 是 DDP 调优的两个最常用旋钮：前者管「多久同步一次」，后者管「一次同步多大」。很多 DDP 卡顿问题最终都回到这两个参数上。</span>

## 5 公式解析：bucket 大小与重叠收益

设模型梯度总量 $G$ 字节，网络带宽 $B$，单个 AllReduce 的固定延迟 $\alpha$，bucket 大小 $c$。消息数为 $\lceil G / c \rceil$，通信总耗时：

$$T_{\text{comm}} = \left\lceil \frac{G}{c} \right\rceil \cdot \alpha + \frac{2G}{B}$$

- **$\left\lceil \frac{G}{c} \right\rceil \cdot \alpha$（延迟项）**：每个 bucket 一次通信都要付一次 $\alpha$。$c$ 越小，消息数越多，延迟项越大。
- **$\frac{2G}{B}$（带宽项）**：AllReduce 要收发，总数据量为 $2G$（每个字节发出一份、收一份）。带宽项与 $c$ 无关。
- **$\alpha$（固定延迟）**：包含协议握手、GPU 间同步等，$10$–$100$ 微秒量级。

直觉判断：当 $G$ 很大（大模型）时带宽项主导，$c$ 大一点无所谓；当 $G$ 很小（小模型）时延迟项主导，$c$ 太大反而让第一个 bucket 迟迟凑不满，阻塞反向。**bucket 大小本质是在「延迟 × 消息数」与「等待凑桶」之间找平衡**——PyTorch 用 25MB 默认值覆盖大多数场景，手调时看 profiler 里的「等待通信」时间。<span class="marginnote">这也解释了为什么小模型用 DDP 时网络很快但吞吐上不去：模型小 ⇒ 每步通信量小 ⇒ 延迟项占比高 ⇒ 通信没被计算盖住。此时 DDP 的优势就打折扣，可能要考虑梯度累积或干脆减小同步频率。</span>

## 6 辨析｜易错点：DDP 的常见误解

**辨析｜易错点：**
- **DDP 不是模型并行**：它复制模型、切数据，不做任何权重切分。模型放不下时 DDP 无能为力（那是 FSDP/ZeRO 的事）。
- **DDP 的 AllReduce 不是一次性的**：它是分 bucket、按反向顺序渐次触发的多次异步 AllReduce——「一次迭代一次同步」是表象，内部是几十次小同步。
- **`average` 是默认但不必纠结**：SUM+手动除 N 与 AVERAGE 语义等价，别在实现细节上浪费时间。
- **别用手动 `.all_reduce()` 替换 DDP 的梯度同步**：手动同步没有 bucket 与重叠，性能通常更差，还容易忘掉 `no_sync` 的语义。
- **DDP 对模型结构有约束**：`find_unused_parameters`、共享参数、稀疏梯度都要额外处理，否则静默出错。

## 7 小结

- **DDP 本质**：数据并行，每进程全量模型副本，梯度 AllReduce 保证一致性。
- **bucket 分桶**：梯度拼成 ~25MB 的桶再同步，减少消息数、提升带宽利用率。
- **autograd hook 调度**：梯度一凑满就异步 AllReduce，与反向计算重叠，总耗时 ≈ 反向 + 最后一个桶。
- **核心思想**：把一次大同步拆成多次小同步，并让通信藏进计算。
- **调优旋钮**：`bucket_cap_mb`、`no_sync()`、`find_unused_parameters`。

## 8 进阶与延伸

**动手改一下 bucket 大小**：把 `bucket_cap_mb` 从默认 25 调到 5 和 100，各跑 50 步看每步耗时——你会看到「太小消息多、太大凑不满」的两个极端，并找到你的模型的甜点。

**几个值得进一步挖的方向**：

- **DDP 的 `find_unused_parameters` 陷阱**：模型里有未用参数（如条件分支）时，不开这个开关会怎样？了解 autograd 图与 DDP 的梯度同步逻辑，你就明白了。
- **梯度累积与 DDP**：`no_sync()` + 多 micro-batch 累积时，通信被推迟到累积结束——这与 1F1B 的「后向交错」有异曲同工之妙，都靠「推迟同步」换取效率。
- **DDP 与 FSDP 的通信差异**：DDP 一次 AllReduce、FSDP 前后向各一次 AllGather + ReduceScatter——从通信量公式看，什么模型规模下两者吞吐趋同？

**自测题**：为什么 bucket 让「梯度一凑满就异步 AllReduce」能藏进反向计算？如果你能画出「反向算后面、通信同步走」的时间线，就真正理解了重叠的本质。

## 9 动手实践清单

- 把 `bucket_cap_mb` 从 5 调到 100，记录每步耗时，找你的甜点。
- 用 profiler 抓 DDP 的梯度同步事件，数一数一个 step 的 AllReduce 次数。
- 尝试 `no_sync()` + 梯度累积，观察「推迟同步」的吞吐收益。
- 开 `find_unused_parameters`，测试含条件分支的模型能否正确同步。
- 画一张「反向计算与 AllReduce 并行」的时间线，验证重叠。
- 对比 DDP 与手写 `.all_reduce()` 的性能，体会框架的价值。
- 用「延迟 × 消息数 + 带宽」公式估算你的模型的通信时间。

在下一节，我们看 DDP 的「升级版」——**PyTorch FSDP2（`fully_shard`）**如何用 per-parameter sharding 重新设计分片语义。
