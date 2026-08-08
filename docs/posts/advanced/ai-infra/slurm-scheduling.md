---
title: Slurm 作业调度：分区、优先级与多节点任务提交
date: 2026-08-07
---

# Slurm 作业调度：分区、优先级与多节点任务提交

<div class="epigraph">
<p>作业调度器是超算中心的交通警察。</p>
<footer>—— 莫里斯 · 帕里（Morris Parry，HPC 系统管理员）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ Slurm 官方文档与 HPC 实践 · 集群调度篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Slurm 开始

上一节的 Kubernetes 是「云原生」路线的调度器，擅长动态、弹性、容器化。但很多 AI 训练集群——尤其高校超算、企业 HPC 中心——用的是另一套老牌方案：**Slurm**。它诞生于 HPC（高性能计算），管的是「把一整个节点完整地分配给一个作业」，与 GPU 训练任务「独占整机、多节点协同」的需求天然契合。

理解 Slurm 不必懂它的全部内部，抓三条主线即可：**分区（partition）怎么把资源切块、优先级怎么决定谁先跑、多节点任务怎么提交**。这三条正好对应 K8s 那篇里「队列、抢占、PodGroup」的对应物——两条路线殊途同归。

## 1 Slurm 架构：三个核心角色

Slurm 是「中心化调度」架构，三个角色：

- **slurmctld（controller）**：中心调度器，维护集群状态、作业队列，决定「哪个作业上哪个节点」。
- **slurmd（daemon）**：每个计算节点一个，执行 controller 的指令，启动/监控作业。
- **客户端命令**：`sbatch`（提交批作业）、`srun`（交互/并行执行）、`squeue`（看队列）、`scancel`（取消）。

生命周期：`sbatch` 提交作业 → controller 排队 → 资源可用时分配节点 → `srun` 在节点上拉起任务 → 完成后回收。<span class="marginnote">与 K8s 的「Pod 到处跑、动态调度」不同，Slurm 的作业通常「独占整节点」——一个训练作业拿到 4 个节点，这 4 个节点的全部 GPU 都归它，直到作业结束。这种「粗粒度独占」对训练很友好：没有资源争抢，行为可预测。</span>

## 2 分区（Partition）：把集群切成队列

**分区（partition）** 是 Slurm 把资源切块的方式：每个分区是一组节点 + 一组配置（时间限制、优先级、访问权限）。

典型分区设计：

`hpc`：高带宽节点池，供 TP 大作业。
`gpu`：通用 GPU 池，供小作业。
`cpu`：纯 CPU 池，供数据预处理。
`debug`：短时间限，快速试跑。

作业通过 `--partition` 指定目标分区；`--qos`（服务质量）再细分优先级与资源上限。**分区 = 资源切块，QoS = 服务分级**——这是 Slurm 的队列体系。<span class="marginnote">分区是 Slurm 的「硬隔离」：作业只能在分区内调度，不能跨分区。这既是优点（物理隔离、互不干扰）也是缺点（分区空着也借不出去）。工程上分区数要克制——分区越多，碎片化越严重。</span>

## 3 优先级与公平调度：谁先跑

当多个作业竞争同一批资源时，Slurm 用**优先级**排序：

$$\text{Priority} = f(\text{age}, \text{fair-share}, \text{partition factor}, \text{job size}, \text{QoS}, \ldots)$$

- **Age（等待时间）**：等得越久，优先级越高（防止饿死）。
- **Fair-share（公平份额）**：按用户/组的「历史使用量」计算——用得越多，下次优先级越低。
- **QoS**：高 QoS 作业权重更高。
- **Job size / 分区因子**：鼓励或抑制大作业。

这些因子加权求和，**每次资源空出来，最高优先级的作业先上**。Slurm 还有**抢占（preemption）**：高优先级作业可以把低优先级作业「挤下来」，抢走它的节点。<span class="marginnote">Fair-share 是 Slurm 与「先来先服务」的区别：它用「历史用量」调整未来优先级，保证谁都能分到算力。但抢占有代价——被抢的作业要 checkpoint 保存进度，否则白算。这也是为什么「抢占 + 自动 checkpoint」常搭配使用：抢可以，别让人白跑。</span>

## 4 多节点任务提交：一个作业，多个节点

训练任务的典型提交方式：

```bash
#!/bin/bash
#SBATCH --job-name=train-70b
#SBATCH --partition=hpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --qos=high
#SBATCH --time=48:00:00

srun python train.py --model 70B
```

`--nodes=4 --ntasks-per-node=8`：4 节点 × 每节点 8 任务 = 32 个 rank。
`--gpus-per-node=8`：每节点申请 8 块 GPU。
作业内用 `srun` 启动训练脚本，Slurm 通过环境变量 `SLURM_PROCID`、`SLURM_NODEID` 等把「谁是哪号 rank」告诉各进程。

分布式初始化：PyTorch 的 `env://` 初始化方式从这些环境变量读 rank/world_size/主节点地址。**Slurm 负责「把进程摆到节点上」，训练框架负责「从环境变量得知自己是谁」**。<span class="marginnote">Slurm 的多节点语义与 K8s 的 PodGroup 异曲同工：`sbatch` 提交的是一个「作业」，控制器保证它的所有节点<strong>同时就位</strong>后才启动——这就是 HPC 世界的 gang scheduling。一个作业的所有进程要么全起来，要么全等。</span>

## 5 公式解析：资源分配与利用率

设集群有 $P$ 个分区，分区 $k$ 有 $N_k$ 个节点、每节点 $g$ 块 GPU。某作业请求分区 $k$ 的 $n$ 个节点，则它获得的算力：

$$\text{Compute} = n \cdot g \cdot \text{FLOP}_{\text{GPU}}$$

- **$n \cdot g$（GPU 总数）**：作业独占的 GPU 数。Slurm 按节点整数分配，不切分 GPU（除非开 MIG）。
- **FLOP GPU（单卡算力）**：如 A100 的 19.5 TFLOPS（FP32）。
- **集群利用率**：$U = \frac{\sum_{\text{running}} n_i g_i}{\sum_k N_k g}$，即「在跑作业占用」/「总资源」。

**调度的目标**：在「利用率」（资源不空）与「公平性」（谁都能用）之间平衡。Slurm 靠分区与优先级实现——分区保利用率（避免跨区争抢），优先级保公平（fair-share 防垄断）。<span class="marginnote">一个 Slurm 集群的常见矛盾：分区把资源「锁死」导致空置，而优先级把资源「挤满」导致小作业被饿死。成熟的集群用「分区 + QoS + 周期性重排」的组合拳，让大作业有保障、小作业有机会——调度器的终极艺术就是「让每个人都满意地等」。</span>

## 6 辨析｜易错点：Slurm 的常见误区

**辨析｜易错点：**
- **「Slurm 只能整节点」绝对化**：`--gpus-per-node` 可以申请单节点的部分 GPU，但多作业共享单节点会引入资源争抢，训练场景常避开。
- **「Slurm 与 K8s 二选一」不是唯一解**：很多集群两者并存（Slurm 管训练、K8s 管服务），甚至用 K8s 的 Slurm operator 桥接。
- **「srun 与 sbatch 等价」是错的**：`srun` 会同步阻塞地跑并分配终端，`sbatch` 是后台提交；混用要清楚语义。
- **忽略环境变量**：分布式初始化全靠 `SLURM_PROCID` 等环境变量，改脚本时别丢 `SLURM_*` → rank 的映射。
- **别把「分区」当「队列」只用一层**：分区 × QoS 两维配合才能表达复杂的调度策略。

## 7 小结

- **Slurm 架构**：controller（中心调度）+ slurmd（节点代理）+ 客户端命令。
- **分区**：资源切块，配 QoS 细分服务等级，作业只能在分区内调度。
- **优先级**：age + fair-share + QoS 加权；抢占机制保证高优先级作业能抢到资源。
- **多节点提交**：`sbatch` 脚本 + `srun`，作业整体就位后启动，Slurm 环境变量驱动分布式初始化。
- **与 K8s 对比**：Slurm 粗粒度独占、HPC 导向；K8s 细粒度、云原生导向——训练集群常两者共存。

## 8 进阶与延伸

**动手提交一次多节点作业**：用 `sbatch` 提交一个打印 `SLURM_PROCID` 和 `SLURM_NODEID` 的测试脚本——你会看到 Slurm 怎么把环境变量喂给每个 rank，这正是 PyTorch 分布式初始化读的数据。

**几个值得进一步挖的方向**：

- **分区 × QoS 的二维配额**：`--partition` 管「放哪」、`--qos` 管「优先级多高」——两个维度怎么组合出「生产任务优先 + 实验任务兜底」的调度策略？
- **fair-share 的公平性**：一个团队连续跑大作业后，它的 fair-share 降多少？用 `sacct` 观察「等待时间 vs 历史用量」的博弈。
- **backfill 的收益**：Slurm 的 backfill 默认开启——用 `squeue` 看「大作业等待期间，小作业是否插空跑了」，验证回填的利用率提升。

**自测题**：为什么 Slurm 的作业「要么整节点、要么不调度」？如果你能说清「多节点训练必须同步就位」，就理解了 Slurm 与 gang scheduling 的等价性。

## 9 动手实践清单

- 用 `sbatch` 提交多节点作业，打印环境变量。
- 用 `squeue` 观察作业排队与优先级排序。
- 用 `sacct` 查作业历史，验证 backfill 是否插空。
- 设计「分区 × QoS」的二维调度策略，写出配置文件。
- 观察 fair-share 对「历史用量」的响应。
- 验证「作业整体就位才启动」的 gang 语义。
- 把 Slurm 与 K8s 的调度模型画一张对比图。

在下一节，我们把「故障恢复」与「调度」结合——**弹性训练（TorchElastic）**：让训练任务在节点数变化时自动扩缩容。
