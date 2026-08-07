---
title: 弹性训练（elastic training）：TorchElastic 的动态扩缩容
date: 2026-08-07
---

# 弹性训练（elastic training）：TorchElastic 的动态扩缩容

<div class="epigraph">
<p>系统最好的状态，是能在变化中保持不中断。</p>
<footer>—— 路易斯 · 冯 · 阿彭（Luis von Ahn，Duolingo 创始人）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PyTorch TorchElastic 官方文档 · 集群调度篇 ｜ 2026-08-07</p>
</div>

## 为什么从弹性训练开始

传统分布式训练有一个隐含假设：**world size 固定**。任务启动时是 32 个 rank，就一直是 32 个——坏一个 rank，整个训练就得停。而现实集群上节点会故障、会被高优先级任务抢走、也会随时补充进来。**弹性训练（elastic training）** 打破「world size 固定」的假设：训练任务在节点数变化时**自动调整、继续训练**，而不是崩溃。

TorchElastic 是 PyTorch 官方的弹性训练实现。理解它，就理解了「容错训练」从「手动重启」进化到「自动自愈」的最后一环——这也是上一节 Slurm/K8s 调度器与训练框架的衔接点。

## 1 从固定 world size 到弹性 world size

固定 world size 的问题：

- 一个 rank 故障 → 通信域断裂 → 整个训练停止。
- 修复靠「人工重启」，恢复时间分钟级到小时级。
- 集群资源波动（别的任务腾出节点）无法利用。

弹性训练的思路：**把「rank 数量」当作一个可协商的变量**。训练启动时声明「最少 $N_{\min}$ 个、最多 $N_{\max}$ 个 rank」，只要有 $N_{\min}$ 个活着，训练就继续；节点增减时，训练**重新 rendezvous（会合）**，按新 world size 重新切分数据与模型，继续跑。<span class="marginnote">「弹性」的本质是「训练框架自己适应资源」，而不是「调度器保证资源」。坏 2 个节点，训练自动缩容到剩下的节点上继续；资源回来了，自动扩容回去——全程无需人工。这个能力让「故障恢复」从「运维操作」变成「框架行为」。</span>

## 2 TorchElastic 架构：三个角色

TorchElastic 的运行时由三块组成：

- **Agent（agent）**：每个节点一个，负责启动/监控该节点上的训练进程（worker）。它也是「检测者」——worker 崩溃了它知道。
- **Rendezvous（会合）**：各节点的 agent 通过一个协调点（如 etcd 或 Kubernetes API）「会合」，协商出当前有效的 world size 与 rank 分配。会合是弹性的心脏。
- **Monitor（监控）**：agent 持续监控 worker；若 worker 异常退出，触发新一轮 rendezvous（重新会合、重新分配），然后**从 checkpoint 恢复**再拉起 worker。

一次故障的完整自愈流程：worker 崩溃 → agent 检测 → 触发 rendezvous → 剩余 agent 重聚 → 协商新 world size → 各自从 checkpoint 恢复 → 继续训练。<span class="marginnote">Rendezvous 协议的关键性质是「一致性」：所有存活的 agent 必须协商出<strong>同一个</strong>新 world size 与 rank 号，否则训练组就散架了。TorchElastic 用「最小成员数 + 超时」来保证：先到齐最少成员，再统一分发成员列表——这一步没做好，恢复后各 rank 认知不一致，是弹性训练最常见的坑。</span>

## 3 动态扩缩容：min_size 与 max_size

弹性训练通过 `TorchElastic` 的 `min_size` / `max_size` 声明弹性区间：

- **`min_size`**：少于这个数，训练无法继续（会等待/退出）。
- **`max_size`**：多于这个数不接收（或用于动态扩到上限）。

实际运行时，`rendezvous` 的 `current_size` 在 `[min, max]` 之间浮动。world size 变化后，**数据也要重新切分**：

- 数据并行下，batch 被新 world size 重新均分。
- 学习率、梯度累积若依赖 batch 总量，要重新换算。
- TP/PP 维度通常**不可变**（它们对 rank 数敏感），弹性的自由主要在 DP 维。<span class="marginnote">「弹性只在 DP 维」是理解弹性训练边界的关键：TP 把单个算子切开，通信图对 rank 数固定；PP 把层切成固定段。只有 DP 是「加一份副本、减一份副本」都行的维度。所以弹性训练通常配 ZeRO/FSDP（它们沿 DP 分片），而不是 Megatron 式 TP。</span>

## 4 与 checkpoint 的深度配合

弹性训练与 checkpoint 是「孪生」的：

- **恢复必须精确**：重新会合后，各 worker 从「同一个 checkpoint」恢复，步数一致。
- **动态 checkpoint**：world size 变化后，旧的分片布局不再匹配，需要**重新分片**再保存——这就是上一节说的「re-shard」。
- **频率要密**：弹性训练的恢复依赖「最近 checkpoint 够近」，否则缩容一次的损失很大。

TorchElastic 的恢复流程里，checkpoint 的「数据位置」尤其关键：world size 变了，数据 shard 的划分也变，每个 rank 必须从「新划分」里对应位置继续——否则样本重复或漏读。<span class="marginnote">实践上，弹性训练的 checkpoint 频率通常比固定世界训练更密——因为「可能被抢节点」的集群里，每次事件都对应一次恢复，而恢复的损失 ≈ checkpoint 间隔的一半。把间隔从 1000 步降到 200 步，弹性恢复的「税」就小很多。</span>

## 5 公式解析：弹性训练的有效吞吐

设训练需要 $N$ 个 rank，弹性区间 $[N_{\min}, N_{\max}]$，节点故障事件率为 $\lambda$，每次恢复开销 $C$（含 rendezvous + checkpoint 加载 + re-shard），单 rank 贡献算力 $P$。

**有弹性**时，训练「有效算力 × 时间」：

$$\text{Work} = P \cdot \int_0^T n(t)\, dt - \lambda T \cdot C$$

- **$P \int n(t) dt$（累计算力）**：弹性让 rank 数 $n(t)$ 在区间内波动，坏节点时降速而非停摆。
- **$\lambda T \cdot C$（恢复税）**：每次故障事件付一次恢复开销。
- **对比无弹性**：无弹性时一个故障使 $n(t)$ 掉到 0（全停），$P \int n(t)dt$ 归零并等待人工——弹性把「停摆」变成「降速」，把「小时级人工」变成「分钟级自动」。<span class="marginnote">弹性的收益不是「更快」，而是「更稳」：它把故障从「灾难」降级为「小插曲」。在共享集群（节点会被抢占）里，弹性训练甚至是「能否跑完」的决定性因素——固定 world size 的任务被抢占一次就得全停，弹性任务只是缩容继续。</span>

## 6 辨析｜易错点：弹性训练的常见误区

**辨析｜易错点：**
- **「弹性 = 动态 TP/PP」是误解**：弹性主要在 DP 维；TP/PP 通信图对 rank 数固定，不可随意伸缩。
- **「rendezvous 只是重启」不完整**：rendezvous 要协商出**一致**的新 world size，不是简单重启进程。
- **「弹性不需要 checkpoint」是致命误解**：弹性恢复完全依赖 checkpoint，且需要 re-shard 的动态 checkpoint。
- **「min_size 越小越好」不成立**：min 太小会让训练在几乎没资源时还硬跑，吞吐极低、浪费电。
- **别忽略「数据重新切分」**：world size 变化后每个 rank 的数据 shard 都要重算，忘了就样本重复。

## 7 小结

- **弹性训练**：把 world size 变成可协商变量，节点增减时自动调整继续训练。
- **TorchElastic 三件套**：Agent（监控）、Rendezvous（会合协商）、Monitor（触发恢复）。
- **弹性区间**：min_size / max_size，实际 world size 在区间内浮动。
- **边界**：弹性主要在 DP 维，TP/PP 通信图固定不可伸缩。
- **与 checkpoint 孪生**：恢复靠 checkpoint，world size 变化要 re-shard 的动态 checkpoint。
- **核心价值**：把故障从「停摆」降级为「降速」，是共享集群训练的生存技能。

## 8 进阶与延伸

**动手演练一次弹性缩容**：用 TorchElastic 跑一个 4-rank 的训练，中途 kill 掉一个 worker——观察系统自动 rendezvous、以 3 个 rank 继续训练。这就是「故障从停摆变成降速」的实景演示。

**几个值得进一步挖的方向**：

- **rendezvous 后端的选择**：etcd、C10d、Kubernetes 三种 rendezvous 后端各适合什么环境？etcd 的运维成本 vs K8s 的原生集成怎么权衡？
- **re-shard 的开销**：world size 从 8 变 6 时，数据重新切分 + checkpoint 重存要花多久？「缩容省下的等待」 vs 「re-shard 的成本」谁大——这就是弹性的经济账。
- **弹性 + FSDP 的组合**：FSDP 沿 DP 维分片、弹性在 DP 维伸缩——两者天然契合。FSDP 的 `reshard_after_forward` 在 world size 变化后怎么重新分片？

**自测题**：为什么「弹性的自由在 DP 维」？如果你能说清「TP/PP 通信图固定、只有 DP 能加副本」，就抓住了弹性训练的核心边界。

## 9 动手实践清单

- 用 TorchElastic 跑 4-rank 训练，kill 一个 worker 观察自动缩容。
- 配置 `min_size` / `max_size`，验证 world size 的弹性区间。
- 对比 etcd / C10d / K8s 三种 rendezvous 后端。
- 观察 re-shard 的开销，评估「缩容省等待 vs re-shard 成本」。
- 验证 FSDP 沿 DP 维分片与弹性伸缩的契合。
- 记录「故障到恢复」的端到端时间。
- 画「弹性训练自愈」的流程图，标出 rendezvous 的协商点。
- 验证「min_size 太小」时训练在低资源下硬跑的浪费。
- 对比「固定 world size」与「弹性」在故障下的恢复时间。

在下一节，我们继续集群调度的上层建筑——**训练任务的排队、抢占与配额管理实践**。
