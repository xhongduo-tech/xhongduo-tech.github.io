---
title: 专家并行（EP）与 MoE 模型推理
date: 2026-08-07
---

# 专家并行（EP）与 MoE 模型推理

<div class="epigraph">
<p>每一张卡只跑被点名的专家，这是 MoE 的天然并行。</p>
<footer>—— MoE 部署实践（源自 Switch Transformer 与 Mixtral 社区）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ Switch Transformer / Mixtral 论文 ｜ 2026-08-07</p>
</div>

## 为什么从专家并行开始

MoE（Mixture of Experts，混合专家）模型把 FFN 换成一组「专家」网络，每次只激活少数几个。MoE 模型的参数量很大（Mixtral 8×7B 有 47B 参数），但**每次推理只用到一小部分**。这给部署带来一个独特的机会：把不同的专家放到不同的 GPU 上，每个 token 的路由把请求「分发」给被点名的专家——这就是**专家并行（Expert Parallelism, EP）**。<span class="marginnote">EP 与 TP/PP 都不同：<strong>TP 切同一层的矩阵，PP 切层，EP 切「同一层里的不同专家」</strong>。MoE 层天然是「专家之间互不相干」的并行结构，EP 是它的最优切法。</span>

本篇讲 MoE 推理的结构、EP 的「专家分布 + token 路由」机制、以及它与 TP 的组合（EP 还是 TP，取决于 token 数）。

## 1 MoE 层的推理结构

一个 MoE transformer 层 = 共享的 attention + 一个 MoE FFN。MoE FFN 的结构：

**路由器（router / gate）**：一个线性层，对每个 token 输出「每个专家的权重」，取 top-$k$（通常 $k=1$ 或 $2$）个专家；
**被选中的专家**：每个专家就是一个标准 FFN（升维 + 降维）；
**输出**：被选专家的输出按路由权重加权求和。

推理时每个 token 只经过 $k$ 个专家，所以**计算量只占「全部专家都算」的 $k/E$**（$E$ 是专家总数）——这是 MoE「参数多但算得省」的来源。<span class="marginnote">路由是<strong>逐 token</strong>的：一个 batch 里不同 token 可能被路由到不同的专家。这意味着「专家间的负载天然不均衡」——有的专家忙、有的专家闲，这是 EP 调度的核心难点。</span>

对显存来说，MoE 的**全部专家权重都要驻留显存**（虽然每次只算几个），所以 MoE 模型比同计算量的 dense 模型「更占显存」——EP 把专家摊到多卡，正好解决这个驻留问题。

## 2 EP 的机制：专家分布 + token 路由

EP 的部署策略：

1. **专家分布**：$E$ 个专家均匀分布到 $P$ 张卡上（每卡 $E/P$ 个专家），共享层（attention）通常每卡都有一份（或用 TP 切）；
2. **token 路由**：路由层先算出每个 token 要去哪些专家，然后把 token **按目标专家分组，通过 all-to-all 通信发到对应卡**；
3. **专家计算**：每卡在自己负责的专家上计算收到的 token；
4. **结果回收**：算完的 token 通过另一次 all-to-all 发回原卡，加权求和。

**通信形态是 all-to-all**（每卡都向每卡发数据），而不是 TP 的 all-reduce。通信量取决于「被路由的 token 数 × 隐藏状态大小」，与 TP 的「每层全量同步」不同——**EP 的通信量与路由稀疏性相关**。<span class="marginnote">All-to-all 是 EP 的关键开销：<strong>token 要在卡间「搬家」</strong>。设计路由时让「同一批 token 尽量去同一专家」能显著降低通信量，这也是 batch 越大 EP 越划算的原因。</span>

## 3 EP vs TP：何时用哪个

MoE 层的两种并行切法直接竞争：

**TP 切专家内部**：每个专家被切成多卡算（把 8 专家的 FFN 矩阵行切）。好处是不用 all-to-all（保持 TP 的通信模式），但**所有卡的注意力都在每个专家上**——token 少时很浪费。
**EP 切专家之间**：每个专家独占一张（或几张）卡，token 靠 all-to-all 分发。好处是**每个 token 只在一张卡上算**，专家内部是全量计算，利用率高；坏处是 all-to-all 通信。

经验法则：**token 数多（大 batch / 长序列 prefill）时 EP 赢**——每卡收到的 token 多，all-to-all 摊销得薄，专家计算饱满；**token 数少（小 batch / decode）时 TP 赢**——all-to-all 的开销占比过大。<span class="marginnote">这也是为什么主流引擎对 MoE 常采用「<strong>decode 用 TP、prefill 用 EP</strong>」的混合策略（如 DeepSeek、vLLM 的 MoE 支持）。</span>

**辨析｜易错点：EP 不解决「单 token 延迟」。** EP 解决的是「专家驻留显存」与「大批量吞吐」；对单个 token，它只是把计算换了个位置，延迟改善有限。**别把 EP 当成「让 MoE 变快」的万能解，它是「让 MoE 放得下 + 大并发吞吐」的工具**。

## 4 公式解析：EP 的通信量与利用率

设 $T$ 个 token、$E$ 个专家、$P$ 张卡、$k$ 个路由专家，隐藏状态大小 $d$。EP 下 all-to-all 的通信量：

- **第一步，算每卡收发**：每个 token 要发到 $k$ 个目标卡（其路由专家所在的卡），总「token-专家」配对 $T \cdot k$ 个。理想均衡时，每卡收发约 $T \cdot k / P$ 个 token 的隐藏状态：

$$B_{\text{comm}} \approx 2 \cdot \frac{T \cdot k}{P} \cdot d \cdot \text{bytes}$$

- **第二步，算专家计算量**：每个专家处理约 $T/P$ 个 token（均衡时），FFN 计算量 $\propto T \cdot k / P \cdot (4 d^2)$。
- **第三步，看拐点**：当 token 数 $T$ 大时，通信 $B_{\text{comm}}$ 与计算都随 $T$ 线性增长，但**通信与「每卡 token 数」相关、计算与「每卡 token 数」也相关**——比值固定；真正决定 EP vs TP 的是「all-to-all 单次开销」与「TP 的每层全量同步」谁更大。实践中，$T$ 超过几百 token 后 EP 开始占优。

负载不均衡时（某些专家被路由得多），上面的「均衡」假设破功，慢专家成为瓶颈——**EP 的调度器需要做负载均衡（重路由、复制热门专家）**，这是 MoE 部署的进阶话题。

## 5 小结

- **MoE 每 token 只激活 $k$ 个专家**：参数多、算得省、但全部专家要驻留显存。
- **EP 把专家分布到多卡**：token 经 all-to-all 分发到目标专家，算完再回收加权。
- **EP vs TP**：token 多时 EP 优（专家内部全量计算），token 少时 TP 优（all-to-all 占比过大）。
- **混合策略常见**：prefill 用 EP、decode 用 TP，平衡吞吐与延迟。
- **负载不均衡是难点**：热门专家成为瓶颈，需要重路由与复制。

在下一节，我们进入推理架构层面的解耦——**PD 分离：Prefill 与 Decode 的解耦部署**。
