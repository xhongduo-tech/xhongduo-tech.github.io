## 核心结论

Expert Parallelism，简称 EP，可以直译为“专家并行”：在 Mixture of Experts（MoE，专家混合模型）里，不再让每张 GPU 都保存全部专家，而是把不同专家完整地分配到不同 GPU 上。这样做的直接收益是单卡显存压力显著下降，因为每张卡只存自己负责的专家权重；代价是 token 必须在卡之间来回传递。

MoE 可以先用一句白话解释：它不是让所有前馈网络都参与计算，而是先由一个 router（路由器，负责给 token 选专家的模块）挑出少数几个 expert（专家，本质上是若干独立的 FFN），只让这些专家处理当前 token。EP 则是为这种“少量专家被激活”的结构专门设计的并行方式。

如果一句话概括 EP 的价值，就是：**用更多跨卡通信，换更低的单卡参数存储成本，以及更适合大规模 MoE 的计算组织方式。**

下面这张表先把它和更常见的 Tensor Parallelism（张量并行，简称 TP，把同一层权重切成多份分布到多卡）对比清楚：

| 对比项 | Expert Parallelism | Tensor Parallelism |
| --- | --- | --- |
| 参数分布 | 每卡只存自身专家 | 每卡存所有专家的切片 |
| 通信模式 | All-to-All 路由 + 结果归并 | All-Reduce / Reduce-Scatter / All-Gather |
| 激活稀疏性 | 仅激活 top-K experts | 通常整层都参与 |
| 单卡显存压力 | 更低，适合专家很多的 MoE | 专家很多时仍可能很高 |
| 主要瓶颈 | 动态 token 路由通信 | 线性层切片同步 |
| 适合场景 | 大规模稀疏 MoE | 密集模型或较小 MoE |

一个最直观的玩具例子是：有 2 张 GPU、8 个专家。GPU-A 只放 expert0 到 expert3，GPU-B 只放 expert4 到 expert7。每个 token 先在本地经过 router，选出 top-1 或 top-2 专家；如果选中的专家不在本卡，就通过 all-to-all（全互连交换，意思是每张卡都可能同时向其他卡发送和接收数据）把 token 发出去。专家处理完，再把结果送回 token 原来的位置，继续后面的残差连接和下一层计算。

所以 EP 的本质不是“把模型平均切开”，而是“把专家作为独立单元放到不同设备上，然后围绕 token 的动态派送来组织计算”。这也是它成为大规模 MoE 训练关键技术的原因。

---

## 问题定义与边界

要理解 EP，先要明确它解决的是什么问题。

在普通 dense 模型里，一个 token 通常会经过整层网络；在 MoE 里，一个 token 只会经过少数几个专家。问题在于，**这些被选中的专家未必在 token 当前所在的 GPU 上**。因此，系统必须把 token 送到正确的专家所在设备，再把结果送回来。

这就是 EP 的核心问题定义：**当专家分布在不同 GPU 上时，如何高效完成 token 到专家的跨卡路由，并控制通信开销、显存占用和负载不均衡。**

它的边界也很清楚：

1. 它主要针对 MoE，不是通用并行策略。
2. 它默认专家可以被“整块”放到不同 GPU 上，而不是像 TP 那样把每个专家内部再切碎。
3. 它是否划算，强依赖互联带宽。如果 GPU 之间通信太慢，EP 的收益会被 all-to-all 抵消。
4. 它不自动解决路由倾斜。router 如果总把 token 发给少数专家，就会让少数 GPU 过载。

可以用集合来表示一次路由。设第 $i$ 张 GPU 上，本地 token 中被送往专家 $e$ 的集合为：

$$
T_i^e = \{t \mid t \text{ 位于 GPU } i,\ \text{且 router 选择 expert } e\}
$$

那么专家 $e$ 实际收到的 token 数量是：

$$
T^e = \sum_{i=1}^{N} |T_i^e|
$$

这里的 $|T_i^e|$ 就是“来自第 $i$ 张卡、发给专家 $e$ 的 token 个数”。这个式子非常关键，因为它直接决定两件事：

1. 专家 $e$ 的计算负载。
2. 发往持有专家 $e$ 的那张 GPU 的通信流量。

看一个 2 卡 4 token 的玩具例子。假设：

- GPU0 持有 expert0、expert1
- GPU1 持有 expert2、expert3

现在 batch 中 4 个 token 分布在两张卡上，各卡本地 router 的选择如下：

| Token 所在卡 | Token | 选中的 expert |
| --- | --- | --- |
| GPU0 | t0 | expert2 |
| GPU0 | t1 | expert3 |
| GPU1 | t2 | expert0 |
| GPU1 | t3 | expert1 |

那么 dispatch 阶段会发生：

- GPU0 把 t0、t1 发给 GPU1
- GPU1 把 t2、t3 发给 GPU0

各卡只计算自己拥有的专家，然后 combine 阶段再把结果发回 token 原位置。这个例子说明了 EP 的边界条件：**它不是让数据始终留在本地，而是接受跨卡交换，前提是交换带来的成本小于节省的显存与提升的专家计算效率。**

---

## 核心机制与推导

EP 的执行流程可以拆成三个阶段：dispatch、compute、combine。

- dispatch：把 token 发到目标专家所在 GPU
- compute：每张 GPU 上的本地专家执行 FFN
- combine：把专家输出送回原 token 所在 GPU，并按 gating score 加权合并

这里的 gating score（门控分数）可以白话理解为“router 对每个被选专家打的权重”。如果使用 top-2 routing，一个 token 会被送到两个专家，最后按两个分数加权相加。

设共有 $N$ 张 GPU，专家集合为 $E$。如果每个 token 只去 top-$K$ 个专家，那么总通信量近似与下面这个量成正比：

$$
\text{Comm} \propto \sum_{i=1}^{N}\sum_{e \in E} |T_i^e|
$$

如果 top-$K=1$，每个 token 只会被派送一次；如果 top-$K=2$，通信量大致翻倍。这里先忽略 padding、capacity 和元数据开销，讨论主量级。

进一步看一次前向的时间，可以粗略写成：

$$
T_{\text{moe}} \approx T_{\text{router}} + T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}}
$$

但真实系统不会傻等四步串行完成，而是尽量做重叠。理想情况是：

$$
T_{\text{dispatch}} + T_{\text{combine}} \le T_{\text{other compute}}
$$

这里的 $T_{\text{other compute}}$ 指其他能并行执行的计算，比如注意力、共享层、反向的部分算子，或者同一层流水线中的其他阶段。如果能把通信埋进这些计算空隙里，EP 的通信惩罚就会显著下降。

为什么 EP 对大规模 MoE 特别重要？因为它改变了显存增长方式。

假设总共有 $E$ 个专家，每个专家参数量为 $P$。如果不做 EP，而让每张卡都完整保存所有专家，那么单卡需要存储约：

$$
M_{\text{dense-like}} \propto E \cdot P
$$

如果采用 EP，把专家平均分配到 $N$ 张卡，单卡存储大致变成：

$$
M_{\text{EP}} \propto \frac{E}{N} \cdot P
$$

也就是说，单卡专家权重压力大约按 $N$ 倍下降。这就是 EP 的第一性收益。

但这里立刻出现第二个问题：router 可能不均匀。假设某个专家特别受欢迎，那么对应 GPU 就会成为热点。于是训练吞吐不再由平均卡决定，而由最慢那张卡决定，这叫 straggler（拖后腿的慢节点）。

可以把一次批次里专家的负载不均衡写成：

$$
L_{\text{imbalance}} = \max_{e \in E} T^e - \frac{1}{|E|}\sum_{e \in E} T^e
$$

这个式子不是标准训练损失，而是一个直观指标：最大专家负载和平均负载差得越多，系统越不均衡。工程里常会配合 auxiliary load balancing loss（辅助负载均衡损失，意思是额外加一个正则项，逼 router 不要只偏爱少数专家）去压这个问题。

再看一个稍大一点的玩具例子。4 张卡，8 个专家，每张卡持有 2 个专家。某次前向里有 8 个 token，本地 router 给出的目标如下：

| GPU | 本地 token | 目标专家 |
| --- | --- | --- |
| GPU0 | t0, t1 | e5, e2 |
| GPU1 | t2, t3 | e2, e7 |
| GPU2 | t4, t5 | e1, e5 |
| GPU3 | t6, t7 | e0, e2 |

如果 e2 很热门，那么持有 e2 的那张卡会收到 3 个 token，而其他专家可能只有 0 或 1 个。这时即便总 token 数不大，也会出现局部拥塞。EP 的关键不只是“能路由”，而是“路由后仍然均衡且可重叠”。

真实工程例子里，这种问题会被放大。比如一个 256 expert 的大模型部署在多节点集群上，专家跨 8 张甚至更多 GPU 分布。此时 all-to-all 不再只是单机 NVLink 内交换，还可能跨节点走 InfiniBand。带宽分层后，调度通常会改成“先机内聚合，再机间交换，再机内分发”的分级通信，否则全局 all-to-all 的尾延迟会很差。这也是 Hybrid EP 出现的背景。

---

## 代码实现

下面先给一个最小可运行的 Python 示例，用来模拟 EP 的 dispatch 和 combine 逻辑。它不依赖 GPU，也不做真正的神经网络计算，只展示“token 如何被派送到对应专家，再合并回原位置”。

```python
from collections import defaultdict

def dispatch_tokens(tokens, token_gpus, expert_for_token, expert_owner):
    """
    tokens: token 值
    token_gpus: 每个 token 原本所在的 GPU
    expert_for_token: 每个 token 被 router 选中的 expert
    expert_owner: expert -> 持有该 expert 的 GPU
    """
    send_buffers = defaultdict(list)
    metadata = []

    for idx, token in enumerate(tokens):
        src_gpu = token_gpus[idx]
        expert = expert_for_token[idx]
        dst_gpu = expert_owner[expert]
        send_buffers[dst_gpu].append((idx, token, expert, src_gpu))
        metadata.append((idx, src_gpu, dst_gpu, expert))

    return send_buffers, metadata

def local_expert_compute(send_buffers):
    """
    用一个玩具规则替代真实 FFN:
    expert e 对 token x 的输出为 x * 10 + e
    """
    outputs = defaultdict(list)
    for gpu, items in send_buffers.items():
        for idx, token, expert, src_gpu in items:
            out = token * 10 + expert
            outputs[gpu].append((idx, out, src_gpu))
    return outputs

def combine_outputs(outputs, num_tokens):
    restored = [None] * num_tokens
    for gpu, items in outputs.items():
        for idx, out, src_gpu in items:
            restored[idx] = out
    return restored

def run_demo():
    tokens = [1, 2, 3, 4]
    token_gpus = [0, 0, 1, 1]
    expert_for_token = [2, 3, 0, 1]
    expert_owner = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }

    send_buffers, metadata = dispatch_tokens(
        tokens, token_gpus, expert_for_token, expert_owner
    )

    # GPU0 上的 token 都被发往 GPU1；GPU1 上的 token 都被发往 GPU0
    assert len(send_buffers[0]) == 2
    assert len(send_buffers[1]) == 2

    outputs = local_expert_compute(send_buffers)
    combined = combine_outputs(outputs, len(tokens))

    # 按玩具规则验证输出
    assert combined == [12, 23, 30, 41]
    return metadata, combined

meta, result = run_demo()
print(meta)
print(result)
```

这个例子对应的流程是：

1. 每个 token 先决定去哪个 expert。
2. 根据 expert 所属 GPU，把 token 放进不同发送缓冲区。
3. 每张 GPU 在本地批量执行专家计算。
4. 结果根据原 token 索引恢复回去。

真实训练框架里，当然不会像上面这样用 Python 字典；它们通常会做几件更底层的事：

- 先对 token 按目标 expert 排序或分桶，形成连续内存块，减少碎片。
- 用 `all_to_all` 或者分级 `all_to_allv` 完成跨卡交换。
- 在本地把属于同一 expert 的 token 拼成一个大矩阵，执行 GroupGEMM。GroupGEMM 可以白话理解为“把多个相似的小矩阵乘法打包成一批，提高 GPU 利用率”。
- combine 时按 token 原位置和 gating score 做 scatter 或加权归并。

新手可以先记住下面这个顺序化伪代码：

```python
for layer in moe_layers:
    scores, expert_idx = router(x)                 # 选 top-K expert
    x_dispatch = all_to_all_dispatch(x, expert_idx)
    x_local = local_expert_ffn(x_dispatch)         # 本卡专家计算
    x_combine = all_to_all_combine(x_local)
    x = residual_and_weighted_sum(x, x_combine, scores)
```

如果再把 backward 考虑进去，工程实现通常还会把通信句柄、发送索引、专家分桶结果缓存下来，避免反向重新构建全部元数据。对于多机环境，还会把 dispatch 拆成两层：

1. 机内交换
2. 机间交换

这样可以优先利用 NVLink，再尽量压缩跨节点流量。

一个真实工程例子是大规模稀疏模型训练。假设 64 张 GPU，256 个专家，每张卡放 4 个专家。每一层前向都要完成：

- 本地 router 打分
- token 按目标专家打包
- 机内和机间 all-to-all
- 本地专家 FFN
- 结果 combine 回原 token 顺序

这时代码真正关注的不是“能不能跑通”，而是“如何避免小包太多、如何让专家 batch 足够大、如何隐藏通信、如何监控每个专家 token 数”。EP 的实现重点始终是数据搬运和调度，而不是单个专家 FFN 的数学本身。

---

## 工程权衡与常见坑

EP 很少是“打开开关就自动更快”的策略。它通常是在模型足够大、专家足够多、互联足够强时，才明显优于简单方案。

最常见的坑有三个：通信饱和、负载失衡、专家批次太小。

| 坑 | 影响 | 规避方式 |
| --- | --- | --- |
| All-to-All 饱和 | GPU 大量时间耗在等通信 | 做分级通信、通信计算重叠、压缩精度 |
| Router 负载不平衡 | 少数专家成为 straggler | 加负载均衡损失，监控 `tokens_per_expert` |
| 专家批次太小 | GEMM 效率差，吞吐下降 | 调整 batch、capacity、并行粒度 |
| 单卡专家过多 | 显存重新变紧张 | 提高 EP 宽度，或和 TP 组合 |
| 跨节点比例过高 | 延迟长、尾部更差 | 优先机内放置热点专家，减少远程路由 |

先看通信饱和。EP 的收益建立在“通信成本可控”这个前提上。如果 all-to-all 把 NVLink 或 InfiniBand 打满，计算单元就会空转。常见手段是：

- 使用 BF16、FP8 之类更低精度传输格式，减小消息体积。
- 让 dispatch 和 combine 与其他层计算重叠。
- 做 hierarchical routing，也就是先局部聚合再跨节点交换。
- 尽量让 token 在本地命中更多专家，减少远程流量。

再看负载失衡。假设一次批次里 80% token 都被分给 GPU3 上的 expert5，那么会出现：

- GPU3 专家计算很慢
- 其他卡很快做完本地计算，但 combine 阶段必须等 GPU3
- 最终整层时间被 GPU3 决定

这是 EP 最典型的“平均看着不错，尾部却很差”的问题。实际工程里通常会持续记录：

- 每个 expert 的 token 数直方图
- 每张 GPU 的 dispatch 输入量和输出量
- 每层的 dispatch / compute / combine 时间
- top-K 路由命中熵或负载均匀度指标

对初学者来说，一个简单判断标准是：**如果专家负载分布长期明显偏斜，先别急着优化 kernel，先修 router。** 因为很多吞吐问题不是算子慢，而是调度不均。

第三个坑是专家批次太小。MoE 的“稀疏”不等于自然高效。如果 batch 太小、sequence 太短、top-K 太低、专家太多，那么每个专家实际收到的 token 数可能很少。这样虽然看起来把计算分散了，但每个 expert 的 GEMM 都变成了小矩阵乘法，GPU 利用率会很差。也就是说，EP 节省了显存，却不一定自动提升算力利用率。

一个常见误区是把“更多专家”直接等同于“更高效率”。实际上，专家数增加会带来三种相反趋势：

1. 单卡存储更轻松。
2. 单专家收到的 token 可能更少。
3. 路由与通信复杂度上升。

所以工程上的核心不是盲目把专家数堆高，而是找到一个平衡点，让每个专家既不会太大装不下，也不会太小吃不满 GPU。

---

## 替代方案与适用边界

EP 不是唯一答案。它适合的是“专家很多、单卡放不下、且高速互联可用”的场景。如果这些条件不成立，别的并行方式可能更直接。

最常见的替代方案是 TP。TP 把同一层权重切片到多卡上，通信更规则，不需要动态路由。对于较小模型、较小 MoE 或者网络带宽一般的环境，TP 往往更稳定，调试也更容易。

下面用表格做一个整体比较：

| 策略 | 何时选 | 通信模式 | 主要优势 | 主要限制 |
| --- | --- | --- | --- | --- |
| Tensor Parallel | 模型较小，或通信更希望规则化 | All-Reduce / All-Gather | 实现成熟，行为稳定 | 专家很多时单卡存储仍重 |
| Expert Parallel | 专家多到单卡难以承载 | All-to-All dispatch/combine | 显存压力低，适合稀疏 MoE | 动态通信复杂，负载易失衡 |
| Data Parallel | 单卡能放下完整模型 | 梯度同步 | 最简单 | 无法解决模型过大问题 |
| Hybrid EP/TP | 模型特别大，拓扑分层明显 | 分级 all-to-all + 线性同步 | 兼顾显存和通信 | 系统复杂度最高 |

什么时候不该优先用 EP？

1. 专家总量还不大，单卡完全能装下。
2. GPU 间带宽一般，比如只有普通以太网，all-to-all 成本过高。
3. 业务更关注延迟稳定性，而不是极致吞吐。
4. 团队还没有能力监控和调试复杂通信链路。

什么时候 EP 明显更合适？

1. MoE 模型的专家参数已经成为单卡显存主瓶颈。
2. 集群内有强互联，如 NVLink、NVSwitch、InfiniBand。
3. 训练或推理批次较大，能让每个专家吃到足够多 token。
4. 已经准备好处理 router 负载均衡和分级通信。

真实工程里，最常见的不是纯 EP，而是混合方案。比如共享层或 shared experts 用 TP，稀疏专家层用 EP；或者机内做更宽的 EP，机间再控制 TP/DP 比例。这类 Hybrid EP/TP 的本质是承认一个事实：**不同层、不同拓扑、不同瓶颈，不该强行用同一种并行方式。**

因此，EP 的适用边界可以概括成一句话：**它不是通用最优，而是大规模稀疏 MoE 在强互联集群上的高性价比方案。**

---

## 参考资料

| 来源 | 类型 | 核心信息 |
| --- | --- | --- |
| NVIDIA TensorRT-LLM Expert Parallelism 文档 | 官方文档 | 给出 EP 的基本定义、放置方式与实现接口 |
| NVIDIA 关于 Wide Expert Parallelism 的技术博客 | 技术博客 | 展示大规模机架上更宽 EP 带来的吞吐收益 |
| NVIDIA 关于 Hybrid Expert Parallel 的技术博客 | 技术博客 | 讨论分级通信、通信计算重叠与低精度传输 |
| ApX / ApXML 的 MoE 分布式训练教程 | 教程 | 对 all-to-all、dispatch/combine 和负载均衡做了系统说明 |
| MoE 相关工程论文与系统实现 | 论文/实现 | 关注多机通信、路由稳定性、专家计算组织方式 |

阅读这些资料时，建议按这个顺序：

1. 先理解“token 为什么要跨卡路由”。
2. 再理解“为什么显存收益会换来 all-to-all 通信”。
3. 最后再看“真实系统如何把通信隐藏到计算后面”。

如果你的目标是判断自己的集群是否适合 EP，最应该核对的是三件事：

1. GPU 间的有效带宽是否足够高。
2. 每层每个专家平均能分到多少 token。
3. router 是否能维持长期稳定的负载均衡。
