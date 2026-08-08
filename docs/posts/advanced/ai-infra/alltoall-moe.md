---
title: AllToAll 在 MoE 场景下的通信模式与优化
date: 2026-08-07
---

# AllToAll 在 MoE 场景下的通信模式与优化

<div class="epigraph">
<p>The whole is greater than the sum of its parts.</p>
<p>整体大于部分之和。</p>
<footer>—— 亚里士多德（Aristotle），常被引述的箴言</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 集合通信·MoE ｜ 2026-08-07</p>
</div>

## 为什么从 AllToAll 开始

前面的文章里，我们看到的通信几乎都是 **AllReduce**——数据并行里人人持有同样的梯度，把它们归约成一份再广播回去。但有一类模型的通信主角不是 AllReduce，而是 **AllToAll**（全到全）：**混合专家模型（Mixture-of-Experts，MoE）**。MoE 把 Transformer 里的 FFN 层替换成一组「专家」，每个 token 只激活其中 top-k 个专家——于是在训练时，**token 必须被搬去「持有它目标专家」的那张卡上**。这就产生了全网「每张卡都给每张卡发数据」的通信模式，也就是 AllToAll。<span class="marginnote">一句话记住两种通信的气质差异：AllReduce 是「人人一份，归约合一」——数据有去有回、量不随规模涨；AllToAll 是「各取所需，换人换货」——纯搬运、量随规模线性涨。MoE 用的是后者，这也是它被称为「通信密集型」的根源。</span>

## 1 MoE 的一步前向：路由与分发

MoE 层把原来的单个 FFN 换成 $E$ 个专家 FFN 与一个**门控网络（gating network）**。前向时，每个 token 先经过门控，选中最相关的 top-k 个专家：

$$g(x) = \operatorname{softmax}\left(x \cdot W_g\right), \qquad k\text{-top experts selected}$$

- $x$：token 的隐藏向量（hidden state）；
- $W_g$：门控权重矩阵，形状 $d \times E$；
- $g(x)$：$E$ 维的概率向量，$g_i(x)$ 是「token $x$ 分给专家 $i$」的分数。

这行式子读作：**先对「token 与每个专家的亲密度」做 softmax，取分数最高的 k 个专家干活。** 经典配置：GShard 用 top-2，Switch Transformer 激进地只用 top-1，Mixtral 8×7B 用 8 个专家 + top-2——虽然总参数量 47B，但每个 token 只激活约 13B，算力成本接近 13B 稠密模型。<span class="marginnote">MoE 的「大模型幻觉」由此而来：总参数多到像超大模型，单次前向的算力却接近小模型。代价就是——参数分散在专家里，token 得四处去找它们，通信随之暴涨。</span>

路由结果决定了通信：**每个 token 都要去它选中专家所在的 rank**。假设 $E$ 个专家被分布到 $P$ 个 rank 上（每个 rank 持有 $E/P$ 个专家），那么 token 的去向分布完全取决于路由，**与数据并行的「固定拓扑」不同，MoE 的通信模式是动态的、由数据决定的**。

这种把专家切开、分到各 rank 的做法叫**专家并行（Expert Parallelism，EP）**。它与数据并行（DP，每 rank 一份完整模型、各算各的数据）、张量并行（TP，把单个算子切开）是正交的：DP 切数据、TP 切算子、EP 切专家。真实训练里三者常叠加（EP + DP + TP），而 MoE 层**必须**用 EP 才能把专家塞进显存——专家总参数量太大了，单卡放不下。EP 付出的代价，就是本文要讲的 AllToAll 通信。<span class="marginnote">EP 与 DP 的一个关键差异：DP 的 AllReduce 之后人人仍持有<strong>完整梯度</strong>；EP 的 AllToAll 之后，每个 rank 只拥有<strong>属于自己那批专家</strong>的梯度——所以 EP 的优化器状态也天然按专家分片，不需要 ZeRO 再切。</span>

## 2 AllToAll 原语回顾与两个阶段

**AllToAll（全到全）**：每个 rank 把**不同的数据**分别发给其他 $P-1$ 个 rank（与发给自己的那部分合起来，正好构成全部数据）。数学上它就是一个**矩阵转置**——把「按发送方组织」的数据重排成「按接收方组织」。MoE 场景下这张「分发矩阵」的行和列分别对应「每 rank 发出多少」与「每 rank 收到多少」，下图给出 $P=4$ 的直观示意。

![MoE 专家并行的 Token AllToAll 分发矩阵（P=4）](/images/ai-infra/alltoall-moe-1.svg)

MoE 层的通信分两个阶段，各做一次 AllToAll：

**分发（dispatch）**：路由决定每个 token 的目标 rank 后，把 token 连同其编号搬过去。本地的输出是「每个目标 rank 一个分桶」。
**收集（combine）**：专家算完 token 的前向后，把结果按来源 rank 送回。通信量与分发完全相同，方向相反。

在 NCCL 里对应 $P$（等长）与 $2(P-1)N/P \approx 2N$（变长——MoE 几乎必然用这个，因为每个 rank 发给各方的 token 数不固定）；在 PyTorch 里是 $P$，输入是「给每个 rank 的 tensor 列表」。<span class="marginnote">对比上一节：AllReduce 每 rank 的数据量 $2(P-1)N/P \approx 2N$ 与 $P$ 无关；而 AllToAll 是<strong>纯搬运</strong>，网络上的总搬移量随 rank 数增长——这是两种原语最本质的区别，也是 MoE 训练「通信重」的结构性原因。</span>

**收集阶段的隐藏细节**：分发时搬走的是 token，收集时要把「专家输出」按来源还回去。为了让 token 回到正确的位置，分发时每个 token 必须随身携带**原始序号**（token id）与位置掩码，接收方在 combine 时按序号把结果插回原序列。这些「元数据」虽然很小，却让 AllToAll 的缓冲管理与索引重建变得繁琐——也是 MoE 实现容易出 bug 的地方。

**实现上的一个对照**：NCCL 的 AllToAll 没有「归约」环节，所以它不能像 Ring AllReduce 那样边传边归约、降低峰值带宽需求；它是一笔纯粹的「搬运账」，每字节都必须实打实过网络。因此 AllToAll 的优化重点是「减少要搬的字节数」与「把搬运藏起来」，而不是「降低归约开销」。

## 3 公式解析：一次 Token AllToAll 到底搬了多少数据

把通信量算清楚，才能明白 MoE 的优化往哪儿使劲。设每个 rank 有 $B$ 个 token，每个 token 是隐藏维 $d$、精度 $\tau$ 字节的向量，记单个 token 的大小为 $s = d \times \tau$。路由均衡时，一个 token 的专家在本 rank 的概率是 $1/P$（本 rank 持有 $E/P$ 个专家、共 $E$ 个），所以每个 rank 发出的**跨机 token 数**为：

$$B \times \frac{P-1}{P}$$

- **第一步，算单个 rank 的发出量**：$B(P-1)/P$ 个 token 要离开本机，每 token $s$ 字节，故单个 rank 的跨机字节数为 $B(P-1)/P \times s$。剩下的 $B/P$ 个 token 命中的是本地专家，走 NVLink 或直接路由，不算「网络」通信。
- **第二步，算全网总量**：$P$ 个 rank 各发 $B(P-1)/P$，全网总搬移量为：

$$V_{\text{alltoall}} = P \times \frac{B(P-1)}{P} \times s = B(P-1)\, s$$

- **第三步，看规模趋势**：若固定每 rank 的 $B$，$V_{\text{alltoall}}$ 随 $P$ **线性增长**——rank 越多，同样的 token 被拆得越碎、搬得越远，总通信越大。对比数据并行的 AllReduce（量约 $2N$ 与 $P$ 无关），**MoE 的专家并行是「通信随规模涨」的并行方式**。这也是为什么 MoE 训练对互连带宽极度敏感：搬的是和计算等量的数据，网络带宽几乎决定了扩展效率。

**代入一组具体数字**。设一个 MoE 层有 $E=8$ 个专家、分布在 $P=8$ 个 rank，每 rank 有 $B=1024$ 个 token，隐藏维 $d=4096$，BF16（$\tau=2$ 字节），则 $s = 4096 \times 2 = 8$ KB。每 rank 跨机 token 数为 $1024 \times 7/8 = 896$，即每 rank 每次分发搬 $896 \times 8$ KB $= 7$ MB；全网共 $8 \times 7 = 56$ MB。**分发 + 收集要跑两遍**，所以单层每步实际搬 $112$ MB。一个 32 层的 MoE 模型，光 AllToAll 每步就搬约 3.6 GB——**通信量已经与梯度同步相当，甚至更高**。

## 4 通信特征与瓶颈

除了量大，MoE 的 AllToAll 还有三个与稠密模型截然不同的特征：

**特征一：突发且不可预测。** 路由由数据决定，每步的 AllToAll 量都在变。梯度同步是「固定周期、固定大小」，可以用预算好的方式重叠；MoE 的通信大小在 kernel 运行时才知道，**通信与计算的依赖也更紧**——必须先知道路由结果才能组包，这让重叠（overlap）比稠密模型难做得多。<span class="marginnote">工程上的对策是「按层重叠」：上一层专家算前向的同时，下一层的路由与分发 AllToAll 已经在另一个流上跑了。把「路由-分发-计算-收集」四条流水错开，是 MoE 推理/训练引擎的核心技巧。</span>

举个具体的突发例子：一批样本里若恰好有很多 token 都被路由到同一个专家（比如一批代码里的关键词高度相似），该专家所在 rank 会在瞬间收到远超平均值的 token，对应的网络链路瞬时打满、其他链路闲置。这种「一会儿挤爆、一会儿空闲」的流量形态对拥塞控制也很不友好——这也解释了为什么 MoE 集群对**无损网络与低延迟**的依赖比稠密模型更强。

**特征二：负载不均。** 现实中路由不会完美均衡，某些专家会「爆单」。为此引入 **capacity factor（容量因子）$c$**：每个专家最多处理 $\lceil c \times \frac{B \cdot k}{E}\rceil$ 个 token，超出的 token 要么被丢弃（Switch Transformer 的做法），要么溢出到其他机制。$c$ 越大越不丢数据，但通信与空转也越多——**$c$ 是「精度 vs 效率」的旋钮**。

**特征三：变长消息。** 每个 rank 发给各方的 token 数不等，必须用 AllToAllv 而非定长 AllToAll。变长消息让网卡难以预取、让通信调度更难，也放大了「小报文」效应——某些方向的报文可能小到填不满一个 MTU，协议开销占比飙升。

**特征三的工程应对**：先对 token 做**按目标 rank 的排序/重排**，让发往同一 rank 的 token 在显存里连续，一次 AllToAllv 就能用尽量大的连续块发送；再配合「预留最大容量 + 按需填充」的缓冲策略，避免每次通信都动态分配显存。<span class="marginnote">这和 AllReduce 的「分桶合并」同理：把碎报文聚成大块，网络协议与 RDMA 的效率才能上去。MoE 引擎里「重排（permutation）kernel」的耗时常常被低估，值得单独剖析。</span>

## 5 优化手段：把搬移量压下去、把搬移藏起来

**手段一：辅助负载均衡损失（aux load-balancing loss）。** 与其被动承受偏斜，不如在路由上加一个惩罚项，逼门控把 token 摊均匀。经典形式（Switch Transformer）：

$$L_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot g_i$$

其中 $f_i$ 是分给专家 $i$ 的 token 比例，$g_i$ 是门控平均分数。两者乘积在「分得多且分数高」时最小——**鼓励每个专家被均匀、稳定地选中**，从源头上减少偏斜与「爆单」。

**手段二：分组 / 层次 AllToAll（grouped / hierarchical all-to-all）。** 先在本机内用 NVLink 做一次「局部交换」，把**本机内就能满足**的 token 消化掉，只把真正要出机的 token 送上跨机网络。这样跨机 AllToAll 的 $P$ 从「全 rank 数」缩小到「机架/节点数」，总搬移量与跨机跳数都大幅下降。<span class="marginnote">这就是「两层拓扑匹配两层通信」：NVLink 管机内、RDMA 管网间，token 只在其目标专家真正在别处时才跨机。配合《训练集群的网络拓扑》一节的 rail-optimized 拓扑，跨机带宽也能喂满。</span>

层次 AllToAll 的实现思路是把一次全局 AllToAll 拆成「先机内、再机间、再机内」三段：先按「目标 rank 是否在本机」分两组，本机的走 NVLink 直接送达，要出机的先汇总到本机的「出口 buffer」，再做一次**只发生在机与机之间的** AllToAll，最后由目标机内部再分发一次。这样跨机通信的报文从「碎小的单 token 束」变成「按节点聚合的大块」，协议效率更高，也更容易喂满网卡。

**手段三：通信计算重叠。** 利用上一节的机制：分发的 AllToAll 与**上一层**专家计算重叠，收集的 AllToAll 与**本层**专家计算重叠。因为 MoE 层内「先路由后计算」依赖紧密，重叠只能做到「层与层之间」与「分组内错开」，比稠密模型更精细，也需要更多缓冲。

在实践上，MoE 的 overlap 通常配合 **token 分片（chunked dispatch）**：把本层的 token 先切一半，让路由对前一半先执行、先分发，专家算前一半时再对后一半路由分发——把「通信 vs 通信」与「通信 vs 计算」的并行粒度都做细。代价是需要双份缓冲与更复杂的流同步，是成熟的 MoE 框架（如 Megatron、DeepSpeed-MoE）才做的深度优化。

**手段四：选型与工程参数。** 用 AllToAllv（变长接口）减少定长填充浪费；给通信留独立流与足够缓冲；在部署上**优先把常被同时选中的专家放在同机/同机架**（专家共置，expert co-location），让跨机流量最小化。

专家共置的做法不改变路由本身，而是改变**专家的物理排布**：把「经常一起被选中的专家」放进同一张卡或同一个 NVLink 域，让它们的 token 交换走 NVLink 而非跨机网络。因为路由模式在相似数据上往往有规律（同类样本偏好同类专家），共置能把相当一部分 AllToAll 的流量「降级」成机内交换，是部署层面最省事的优化。下面是一段 PyTorch 层的分发骨架：

```python
import torch
import torch.distributed as dist


def moe_dispatch(x, gate_probs, expert_to_rank, top_k=2):
    """MoE 分发骨架：token 按路由结果搬到目标专家所在的 rank。

    x              : [B, d] 本 rank 的 token 隐藏向量
    gate_probs     : [B, E] 门控概率（已 softmax）
    expert_to_rank : [E]   专家 -> 所在 rank 的映射
    """
    # 1. 路由：每个 token 选出 top-k 个专家
    _, top_experts = gate_probs.topk(top_k, dim=-1)          # [B, k]

    # 2. 带序号展开：一个 token 要去 k 个专家，复制 k 份并记录原位置
    tokens = x.repeat_interleave(top_k, dim=0)               # [B*k, d]
    token_id = torch.arange(x.size(0), device=x.device).repeat_interleave(top_k)

    # 3. 分桶：按目标 rank 把 token 组装成发送列表
    target_ranks = expert_to_rank[top_experts.flatten()]     # [B*k]
    send_tokens = [tokens[target_ranks == r] for r in range(dist.get_world_size())]
    send_ids    = [token_id[target_ranks == r] for r in range(dist.get_world_size())]

    # 4. 变长 AllToAllv：把 token 与序号一并送到目标 rank
    recv_tokens = [torch.empty_like(t) for t in send_tokens]
    recv_ids    = [torch.empty_like(t) for t in send_ids]
    dist.all_to_all(recv_tokens, send_tokens)
    dist.all_to_all(recv_ids, send_ids)

    # 5. 返回给接收方的专家 FFN；收集（combine）时按 token_id 插回原序列
    return recv_tokens, recv_ids
```

## 6 小结

- **MoE 把通信从 AllReduce 换成 AllToAll**：token 按路由结果搬到目标专家所在的 rank，通信是**纯搬运、无归约**。
- 路由由门控决定：$g(x)=\operatorname{softmax}(xW_g)$ 取 top-k；通信模式**动态、随数据变化**。
- 一次 AllToAll 全网搬移 $V = B(P-1)s$，**随 rank 数线性增长**——与 AllReduce 的「量不随 $P$ 变」形成鲜明对比。
- 特征与瓶颈：**突发、负载不均（capacity factor）、变长消息**。
- 优化四手段：**辅助负载均衡损失、分组/层次 AllToAll、通信计算重叠、专家共置与工程参数**。
- 实战顺序：**先治偏斜（aux loss）→ 再减跨机量（分组/共置）→ 最后藏通信（overlap）**——从根因到症状逐层处理。

在下一节，我们将进入并行策略的正式第一课：**数据并行（DP）原理：梯度同步的实现与开销分析**——从单卡训练到多卡训练的第一次跃迁。那是 AllReduce 回归主场的地方，而它的每一步，都用得上前面四节打下的 RDMA、无损网络与重叠功底。
