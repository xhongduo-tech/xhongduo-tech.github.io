## 核心结论

Context Parallel，简称 CP，是把一条很长的序列按长度方向平均切到多张卡上执行的并行策略。白话说，不是把模型参数切开，而是把一篇超长文章拆成几段，让不同 GPU 各自负责其中一段 token。

它解决的核心问题是：当上下文长度从 32k、128k 增长到 1M token 时，单卡上最先爆掉的通常不是参数，而是注意力计算和 KV cache。CP 的做法是把总序列长度 $s$ 切成 $cp$ 份，每张卡只保留本地长度 $B=s/cp$ 的子序列；查询向量 $Q$ 留在本地，键值向量 $KV$ 在环形拓扑里轮流传递。这样每个查询最终仍然能“看到”全局上下文，但单卡显存压力下降到原来的约 $1/cp$。

对零基础读者，最重要的判断是：

| 问题 | 结论 |
| --- | --- |
| CP 在切什么 | 切序列，不切模型结构 |
| 哪一层改动最大 | 主要改注意力层 |
| 为什么能支持百万上下文 | 因为每卡只算本地 query，并通过 Ring Attention 分步看完整序列 |
| 适合训练还是推理 | 两者都能用，但工程上更常用于超长 prefill |
| decode 阶段是否总是收益 | 不是，decode 常因对齐和通信导致 TTIT 上升 |

玩具例子：把 1M token 输入切到 8 张卡，每张卡只处理 125k token 的 $Q$。第 1 轮用本地 $KV$ 算注意力，第 2 到第 8 轮依次接收其他卡传来的 $KV$ 块。8 轮结束后，每张卡上的查询都已经和整条 1M token 序列交互过一次。

真实工程例子：在大模型超长上下文推理里，CP2+TP8 处理 128k token 时，prefill 的首 token 延迟可以显著下降，因为每张卡只承担一半序列长度；但 decode 阶段单 token 延迟可能变差，所以常见做法是“prefill 开 CP，decode 退回普通 TP”。

---

## 问题定义与边界

问题定义可以写成一句话：在不改模型数学定义的前提下，把长度为 $s$ 的超长序列分布到 $cp$ 张卡，让每张卡只保存 $s/cp$ 长度的数据，同时保证注意力结果与原始全序列计算一致。

这里有几个边界必须先讲清楚。

第一，CP 不是“近似注意力”。它仍然在算完整注意力，只是把完整的 $QK^T$ 计算拆成多轮局部计算再累积。

第二，CP 主要针对长序列瓶颈。若上下文只有 4k 或 8k，纯 Tensor Parallel，简称 TP，也就是把矩阵乘法按张量维度切分，通常更简单，额外通信也更少。

第三，很多实现要求序列长度满足：
$$
seq\_len \bmod (cp\_size \cdot 2) = 0
$$
原因不是数学本身，而是工程实现里的双向分块、环形调度和 kernel 对齐。不能整除时，一般通过 padding 补齐。

下面这个表能快速看出 padding 的影响。假设原始长度是 130k：

| `cp_size` | 对齐单位 `cp_size*2` | 是否需 padding | 补齐后长度示意 | 额外开销 |
| --- | --- | --- | --- | --- |
| 2 | 4 | 可能很小 | 130000 或 130004 | 很低 |
| 4 | 8 | 可能很小 | 向 8 的倍数补 | 很低 |
| 8 | 16 | 更常见 | 向 16 的倍数补 | 仍低，但 decode 更敏感 |

如果用更直观的新手例子：8 卡处理 128k token，每卡负责 16k token。若实现要求按 `8*2=16` 对齐，而实际长度不是 16 的倍数，就需要补一些空 token。prefill 阶段这点 padding 往往还能接受，但 decode 阶段每一步都很短，小额外开销也会变得明显。

判断 CP 是否值得启用，通常看两个量：

1. 上下文是否足够长，长到单卡注意力和 KV cache 成为瓶颈。
2. prefill 的收益是否大于 ring 通信成本。

直观上，TTFT，也就是 Time To First Token，首 token 延迟，会随着每卡本地序列缩短而下降；但 TTIT，也就是 Time To Incremental Token，增量单 token 延迟，未必同步下降。

---

## 核心机制与推导

先定义符号。设批大小为 $b$，总序列长度为 $s$，CP 组大小为 $cp$，每卡本地序列长度为
$$
B = s/cp
$$
隐藏维度为 $h$。那么每张卡的输入从长度 $s$ 变成长度 $B$。

注意力原式是：
$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

在 CP 里，每张卡只保留自己那一段的 $Q$。白话说，问题不是“谁来算 query”，而是“如何让本地 query 看到远端的 key 和 value”。

答案是 Ring Attention。它是一个环形传输策略：把各卡的 $KV$ 块接成一个环，每一轮把当前 $KV$ 发给下一张卡，同时从上一张卡接收新的 $KV$。每张卡在每轮都用“本地 $Q$ + 当前收到的 $KV$”计算一部分注意力贡献，循环 $cp$ 轮后完成全局注意力。

玩具例子：4 张卡，序列分成 A、B、C、D 四段。
- 第 1 轮，卡 0 用 $Q_A$ 对 $KV_A$ 算；
- 第 2 轮，卡 0 接收 $KV_D$，算 $Q_A$ 对 $KV_D$；
- 第 3 轮，算 $Q_A$ 对 $KV_C$；
- 第 4 轮，算 $Q_A$ 对 $KV_B$。
最终 $Q_A$ 实际已经和 A、B、C、D 全部发生过注意力。

通信量的量级可以直观写成：
- 单轮每卡传输一个本地 chunk，量级为 $O(b \cdot B \cdot h)$；
- 共执行 $cp$ 轮，总量级为 $O(b \cdot s \cdot h)$；
- 从整个 CP 组看，总通信量常写为 $O(b \cdot s \cdot h \cdot cp)$。

这不是说每卡都要保存全量数据，而是整个并行组在完成一次全局注意力时，总共发生了与组规模成正比的环形传输。

在 GQA，Grouped Query Attention，中文可理解为“多个查询头共享较少的 KV 头”的设置下，pass-KV 和 pass-Q 两种策略的通信量不同：
- pass-KV：固定本地 $Q$，轮流传 $KV$
- pass-Q：固定本地 $KV$，轮流传 $Q$

常见比较式是：
$$
\frac{T}{T+P} \ge \frac{2N_{KV}}{N_H}
$$
其中 $T$ 可理解为真实 token 数，$P$ 是 padding 数，$N_H$ 是 query 头数，$N_{KV}$ 是 KV 头数。

白话解释：如果有效 token 占比足够高，且 KV 头相对更少，那么传 KV 更划算；如果 padding 很多，或者缓存命中很高导致传 KV 不再便宜，那么传 Q 可能更优。

下面这个表适合记忆：

| 策略 | 保留本地什么 | 环上传什么 | 适用情况 |
| --- | --- | --- | --- |
| pass-KV | 本地 `Q` | `K,V` | 长序列 prefill、GQA 下常见 |
| pass-Q | 本地 `K,V` | `Q` | padding 多、KV miss 低时可能更优 |

真实工程例子：在大模型 128k 到 1M 上下文推理中，通常会在 prefill 采用 pass-KV，因为此时整段序列都要参与计算，本地 query 块固定、远端 KV 块循环经过，最容易把通信隐藏在计算后面；但当 miss rate 很低，比如 2.5% 左右，继续传 KV 反而可能不划算，需要切到 pass-Q。

---

## 代码实现

工程上，CP 的改动主要集中在注意力模块。线性层、MLP、LayerNorm 仍然在本地执行，不需要改模型数学定义。

下面给一个可运行的 Python 玩具实现。它不依赖分布式库，而是用列表模拟 4 张卡上的 ring 传输。目的是说明“本地保留 Q，轮流接收 KV，再把分块结果拼回去”这个核心过程。

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def attention(q_chunk, k_all, v_all):
    out = []
    scale = math.sqrt(len(q_chunk[0]))
    for q in q_chunk:
        scores = [dot(q, k) / scale for k in k_all]
        probs = softmax(scores)
        y = [0.0 for _ in v_all[0]]
        for p, v in zip(probs, v_all):
            for i in range(len(v)):
                y[i] += p * v[i]
        out.append(y)
    return out

def ring_attention(q_chunks, k_chunks, v_chunks):
    cp = len(q_chunks)
    cur_k = list(k_chunks)
    cur_v = list(v_chunks)
    partial_k = [[] for _ in range(cp)]
    partial_v = [[] for _ in range(cp)]

    # 每张卡只保留本地 Q；K/V 在 ring 上轮流传递
    for _ in range(cp):
        for rank in range(cp):
            partial_k[rank].extend(cur_k[rank])
            partial_v[rank].extend(cur_v[rank])

        # send to next rank, recv from prev rank
        cur_k = [cur_k[(rank - 1) % cp] for rank in range(cp)]
        cur_v = [cur_v[(rank - 1) % cp] for rank in range(cp)]

    outputs = []
    for rank in range(cp):
        outputs.append(attention(q_chunks[rank], partial_k[rank], partial_v[rank]))
    return outputs

# 4 个 token，切到 2 张“卡”
Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
]
K = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
]
V = [
    [10.0, 0.0],
    [0.0, 20.0],
    [30.0, 30.0],
    [5.0, 5.0],
]

q_chunks = [Q[:2], Q[2:]]
k_chunks = [K[:2], K[2:]]
v_chunks = [V[:2], V[2:]]

ring_out = ring_attention(q_chunks, k_chunks, v_chunks)
full_out = attention(Q, K, V)

merged = ring_out[0] + ring_out[1]

for a, b in zip(merged, full_out):
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-9

print("ring attention matches full attention")
```

如果换成真实分布式伪代码，核心结构通常是这样：

```python
def cp_attention_forward(local_q, local_k, local_v, cp_group):
    kv_k = local_k
    kv_v = local_v
    acc = init_attention_accumulator(local_q)

    for step in range(cp_group.size):
        acc = update_attention(acc, local_q, kv_k, kv_v)
        send_to_next_rank((kv_k, kv_v), cp_group)
        kv_k, kv_v = recv_from_prev_rank(cp_group)

    return finalize_attention(acc)
```

这里有两个容易忽略的点。

第一，只保留本地 `Q` 是为了把每张卡的 query 长度稳定在 $s/cp$，这样计算量和显存占用都随 CP 规模下降。

第二，不能简单把每轮 softmax 结果直接相加。真实实现里需要做分块 softmax 的数值稳定合并，也就是在线维护局部最大值和归一化因子，否则长序列下会出现数值误差。

---

## 工程权衡与常见坑

CP 最大的价值，是把超长上下文从“单卡根本放不下”变成“多卡可控地跑起来”。但它不是免费收益，代价主要在通信、对齐和实现复杂度。

先看常见坑：

| 坑点 | 现象 | 原因 | 处理方式 |
| --- | --- | --- | --- |
| 序列不能整除 | 自动 padding 后速度变差 | ring 调度需要对齐 | prefill 容忍 padding，decode 尽量不用 CP |
| 低 miss 仍传 KV | 通信盖不住 | pass-KV 不再最优 | 按阈值切 pass-Q |
| 只关注 TTFT | decode 反而更慢 | TTIT 更受通信影响 | prefill/decode 分开配置 |
| 忽略数值稳定 | 长序列输出漂移 | 分块 softmax 合并错误 | 用 online softmax 累积 |

对于 miss rate，可以用一个经验表来理解：

| miss rate | 推荐策略 | 原因 |
| --- | --- | --- |
| 2.5% | pass-Q 倾向更强 | KV 传输收益不明显 |
| 10% | pass-KV 常可接受 | 通信仍可能被计算覆盖 |
| 20% | pass-KV 更常见 | 远端 KV 访问价值更高 |

更工程化的判断，是把论文里的比较条件当作静态开关。除了前面的
$$
\frac{T}{T+P} \ge \frac{2N_{KV}}{N_H}
$$
还可以结合带宽条件估算：
$$
T \ge \frac{N \cdot C \cdot N_{KV} \cdot e}{2 \cdot N_H \cdot BW}
$$
这里 $BW$ 是网络带宽，$e$ 是元素字节数，$N,C$ 可理解为并行规模和 chunk 数。白话说，序列越长、网络越快、GQA 越明显，pass-KV 越容易划算。

真实工程例子：一个在线推理系统同时服务 128k prefill 和短 decode。若把 CP8 全程打开，prefill 可能收益明显，但 decode 因为每步 token 太短，通信无法被覆盖，TTIT 反而升高。更稳妥的做法是：
- prefill：`CP + TP`
- decode：`TP + DP`，关闭 CP

这也是为什么很多实现把 CP 视为“超长 prefill 加速器”，而不是“所有阶段通吃的默认并行策略”。

---

## 替代方案与适用边界

如果不使用 CP，常见替代方案有三类。

| 方案 | 切分对象 | 适合什么场景 | 局限 |
| --- | --- | --- | --- |
| TP | 权重和矩阵乘法 | 中短上下文、算力扩展 | KV cache 压力不随序列降低 |
| DP | 数据批次 | 训练吞吐扩展 | 对单条超长样本帮助有限 |
| PP | 模型层 | 超大模型放不下单机 | pipeline 气泡和调度复杂 |
| CP | 序列长度 | 超长上下文、百万 token | 注意力通信复杂，decode 不一定优 |

TP 更像“把一层算子拆给多卡”，CP 更像“把一条超长输入拆给多卡”。两者不是互斥关系，反而经常叠加。比如 CP4+TP8 的意思是：先把序列切成 4 份，再把每份内部的矩阵计算切成 8 份。

对初学者，一个非常实用的经验边界是：
- 32k 以内，优先考虑 TP/DP，系统简单。
- 128k 以上，开始认真评估 CP。
- 1M 级别，没有 CP 或近似等价方案，工程上通常很难做。

如果业务场景是“prefill 很长，decode 很短”，可以只在 prefill 启用 CP。配置上常见做法类似下面这样：

```python
def build_runtime_config(is_prefill: bool):
    return {
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "use_context_parallel": is_prefill,
        "context_parallel_size": 4 if is_prefill else 1,
    }

prefill_cfg = build_runtime_config(is_prefill=True)
decode_cfg = build_runtime_config(is_prefill=False)

assert prefill_cfg["use_context_parallel"] is True
assert decode_cfg["use_context_parallel"] is False
```

玩具例子：一个 32k decode pipeline，如果用户上传 128k 长文档，系统可以先用 `CP4` 做 prefill，把长文编码进 KV cache；等进入逐 token 生成，再切回 `TP8` 普通解码路径。

这类“阶段切换”比“一套并行策略跑到底”更符合真实工程。

---

## 参考资料

| 资料 | 来源 | 重点内容 |
| --- | --- | --- |
| Context Parallelism for Scalable Million-Token Inference | MLSys 2025 论文 | CP 定义、Eq.1-3、Ring Attention、Table 6/8 实验 |
| Context Parallelism for Scalable Million-Token Inference | Graphcore Research 博客 | 新手可读的机制解释、为什么 CP 与其他并行正交 |
| Distributing Training 中的 Sequence/Context Parallel 说明 | Hugging Face 文档 | 工程接入方式、对齐与 padding 约束、训练配置建议 |

阅读顺序建议如下：
1. 先看 Graphcore 博客，建立“序列切分 + 环上传 KV”的直觉。
2. 再看 MLSys 论文，重点读 Eq.1-3、算法逻辑和 128k/1M 的实验表。
3. 最后看 Hugging Face 文档，理解真实框架里的配置、padding 和组合方式。
