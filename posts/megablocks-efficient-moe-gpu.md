## 核心结论

MegaBlocks 的核心价值，不是“把 MoE 再做快一点”，而是把 **MoE 的计算形状** 改掉。MoE，Mixture of Experts，直译是“专家混合模型”：同一层里有很多个 FFN 专家，但每个 token 只会进入其中很少几个专家，通常是 top-1 或 top-2。传统实现会先把 token 按专家分组，再调用 `batched GEMM`，也就是“同时做很多个形状相同的小矩阵乘法”。问题在于，真实训练里每个专家接收到的 token 数量波动很大，而 GPU 更擅长大而规则的矩阵，不擅长一堆动态变化的小矩阵。于是工程上常见做法是给每个专家设置容量上限 `capacity factor`，超出的部分要么 padding，要么直接 drop token。

MegaBlocks 的做法是：不再把每个专家当成一个独立小批次来处理，而是把整个路由结果重写成一个 **block-sparse matrix**，即“按固定小块组织的稀疏矩阵”。之后前向和反向的主要计算都落到 block-sparse GEMM 上。这样做的直接结果有两个：

| 方案 | 核心算子 | 是否需要大量 per-expert padding | 是否可能丢 token | GPU 利用率特点 |
|---|---|---:|---:|---|
| 传统 capacity-based MoE | token permutation + batched GEMM | 是 | 是，超出 capacity 时会 drop | 专家负载不均时下降明显 |
| Tutel 一类优化实现 | 更高效的 batched GEMM | 仍需要 | 仍可能 | 比朴素实现好，但仍受 capacity 约束 |
| MegaBlocks / dMoE | block-sparse GEMM | 只在块粒度补齐 | 否 | 更接近大矩阵乘法的吞吐 |

第一，**dropless**。dMoE，dropless MoE，意思是“训练时一个 token 都不丢”。MegaBlocks 不靠限制每个专家的容量来维持效率，而是靠块稀疏矩阵把不规则路由重排成 GPU 更容易执行的形状。

第二，**高利用率来自块级规则化，而不是样本级规则化**。传统方案要求“每个专家接收到的 token 数尽量接近”，否则 batched GEMM 的 shape 很差。MegaBlocks 只要求“块大小固定，比如 128×128”，于是动态路由虽然仍不均匀，但最终仍能映射到稳定的 kernel。

如果把 MoE 想成“token 被路由到不同工厂加工”，传统方法是先把每个工厂前的队伍排整齐，再分别开工；MegaBlocks 是直接把所有工厂的加工关系编码进一张分块稀疏调度表，然后让 GPU 按块执行。前者依赖人工分桶，后者依赖统一的块调度。

---

## 问题定义与边界

先定义问题。设输入 hidden states 为 $X \in \mathbb{R}^{T \times d}$，其中 $T$ 是 token 数，$d$ 是隐藏维度。设共有 $E$ 个专家，第 $e$ 个专家是一个两层 MLP：

$$
f_e(x)=W_{2,e}\,\phi(W_{1,e}x)
$$

其中 $W_{1,e}\in\mathbb{R}^{h\times d}$，$W_{2,e}\in\mathbb{R}^{d\times h}$，$h$ 是专家内部的中间维度，$\phi$ 通常是 GELU、SiLU 或 SwiGLU 之类的激活函数。

路由器会为每个 token 选择若干专家，通常是 top-1 或 top-2。若记路由概率为 $p_{t,e}$，则一个标准 top-$k$ MoE 层可以写成：

$$
Y_t=\sum_{e\in \mathrm{TopK}(t)} p_{t,e}\,f_e(X_t)
$$

这里最重要的信息不是“公式能不能写出来”，而是 **哪些 token 会被送到哪些专家**。把这种连接关系记成稀疏拓扑 $S$ 更方便理解。所谓拓扑，直白说就是“token 到 expert 的连边图”。

传统 MoE 的主要问题不在数学定义，而在 GPU 执行形状：

1. 每个专家接收到的 token 数天然不均匀。
2. GPU 更擅长大而规则的矩阵乘法，不擅长大量 shape 各不相同的小矩阵。
3. 如果直接按专家分组，就得给每个专家预留容量并补齐。
4. 如果容量不够，就只能 drop token，训练语义会被改变。

因此，`capacity factor` 本质上不是模型定义的一部分，而是为了适配硬件形状做出的工程妥协。

在 dMoE 里，MegaBlocks 把前向过程抽象成围绕稀疏拓扑 $S$ 的两类运算。为了便于说明，可以把所有专家的第一层权重和第二层权重分别看成“按专家分块拼接”的大矩阵，再把 MoE 主要计算写成：

$$
H_1 = \mathrm{SDD}(S, X, W_1), \qquad
Y = \mathrm{DSD}(S^\top, \phi(H_1), W_2)
$$

这里：

- `SDD` 可以理解成 sparse-dense-dense：先依据稀疏路由 $S$ 决定哪些 token block 和哪些 expert block 需要参与计算，再与稠密权重块相乘。
- `DSD` 可以理解成 dense-sparse-dense：专家内部算完之后，再按转置路由 $S^\top$ 把结果聚回 token 空间。
- $S^\top$ 不是为了写得“像线代论文”，而是因为回写方向和前向分发方向相反。

如果只看“模型做了什么”，MegaBlocks 和普通 MoE 并没有变成另一个模型；它变的是 **数据布局和算子形式**。这也决定了它的边界。MegaBlocks 主要解决的是 **训练阶段 MoE 层在 GPU 上的高效实现问题**，尤其适合：

- token 路由高度动态
- 专家数较多
- batch 较大
- 不希望丢 token
- 可以接受 Triton/CUDA 自定义 kernel

它不直接解决：

- 路由器如何更快收敛
- 负载均衡损失怎么设计更好
- 多机 all-to-all 通信瓶颈
- 推理阶段极低 batch 下的尾延迟问题

也就是说，MegaBlocks 主要是“算子与数据布局层”的答案，不是 MoE 全部问题的统一解。

---

## 核心机制与推导

核心思想可以压缩成一句话：**把“按专家分组再算”改成“按块组织后统一算”**。

### 1. 从 token 路由到块稀疏拓扑

先看一个最小例子。假设有 4 个 token、2 个专家，每个 token 只选 1 个专家：

| token | 路由到的 expert |
|---|---|
| t0 | e0 |
| t1 | e1 |
| t2 | e1 |
| t3 | e0 |

传统做法会先重排成：

- expert e0: `[t0, t3]`
- expert e1: `[t1, t2]`

如果真实训练里 e0 收到 350 个 token，而 kernel 需要按 128 对齐，那么就得补到 384。这里多出来的 34 个位置不携带真实数据，只是为了让矩阵 shape 合法。若容量设成 256，那么多出来的 94 个 token 甚至可能被直接丢掉。

MegaBlocks 改成块视角。设块大小为 $B=128$。这时不再记录“第 37 个 token 去 expert 5”，而是记录“第几个 token block 与第几个 expert block 之间存在非零块”。从单元素视角看，路由很不规则；从块视角看，它变成了“固定大小的小方块是否存在”。

块对齐关系可以写成：

$$
n^{(e)}_{\text{pad}} = \left\lceil \frac{n^{(e)}}{B} \right\rceil B
$$

其中 $n^{(e)}$ 是 expert $e$ 实际接收到的 token 数，$n^{(e)}_{\text{pad}}$ 是向上补齐到块大小后的长度。补齐只发生在尾块，不再需要像 capacity-based MoE 那样给每个专家预留一个大的静态上限。

这个区别很关键：

| 补齐方式 | 补齐对象 | 浪费来源 |
|---|---|---|
| capacity padding | 预设专家容量 | 负载波动越大，浪费越大 |
| block padding | 固定块尾部 | 只浪费最后一个不完整块 |

因此 MegaBlocks 并不是“完全没有 padding”，而是把 padding 从“容量级浪费”降成了“块尾浪费”。

### 2. 为什么 SDD/DSD 能串起整个 MoE

设输入 $X \in \mathbb{R}^{T \times d}$。如果每个 token 只路由到一个专家，那么前向可以理解成三步：

1. 按路由把 token 收集到对应专家。
2. 在各个专家内部做两层 MLP。
3. 再把结果写回原 token 顺序。

普通实现会把第 1 步和第 2 步拆开：先 permutation，再 batched GEMM。MegaBlocks 的关键是把第 1 步和第 2 步合并进块稀疏计算里：

$$
H_1 = \mathrm{SDD}(S, X, W_1)
$$

这一步的直觉是：只对那些“路由存在”的块做乘法，不存在的块根本不进入计算。于是专家侧得到中间激活 $H_1$。

然后做非线性，再通过反向映射写回 token 空间：

$$
Y = \mathrm{DSD}(S^\top, \phi(H_1), W_2)
$$

其中 $S^\top$ 只是说明：前面是“token 发给 expert”，后面是“expert 结果回到 token”。连接方向反过来了。

更完整一点，若是 top-$k$ 路由并带门控权重，可以写成：

$$
Y = \sum_{r=1}^{k} G^{(r)} \odot
\mathrm{DSD}\!\Bigl((S^{(r)})^\top,\ \phi(\mathrm{SDD}(S^{(r)}, X, W_1)),\ W_2\Bigr)
$$

其中：

- $r$ 表示第几个被选中的专家分支
- $S^{(r)}$ 表示第 $r$ 个路由分支的稀疏拓扑
- $G^{(r)}$ 表示对应的门控权重
- $\odot$ 表示按 token 逐元素缩放

这说明 MegaBlocks 不是只能处理最简单的 top-1；top-2 甚至更一般的 top-$k$，本质上都是“多份稀疏拓扑 + 同样的块稀疏算子”。

### 3. 为什么它比 permutation + batched GEMM 更稳

传统方案的问题在于，它把每个专家都当成一个独立小 batch。专家一多，就会同时出现几件事：

- 有的专家很忙，有的专家很闲
- 每个专家对应的矩阵形状不同
- scatter / gather / permutation 很重
- batched GEMM 虽然“批量”了，但批里的每个问题仍然很小

而 GPU 的优势来自两点：规则访存和高 tile 复用。MegaBlocks 的做法是，把“不规则性”限制在“哪些块非零”这个元数据层，而不让 kernel 直接面对一堆大小各异的小矩阵。

可以把它理解成两层分离：

1. 模型层保留稀疏性：token 仍然只去少数专家。
2. 系统层恢复规则性：实际执行统一落在固定块大小上。

这就是为什么 MegaBlocks 既能做到 dropless，又能比传统 permutation + batched GEMM 更稳定地接近大矩阵乘法吞吐。

还有一个容易忽略的点：**稳定** 比 **峰值快** 更重要。普通实现可能在负载均衡很好时也不慢，但一旦路由分布波动，吞吐会明显下滑；MegaBlocks 的块机制把波动收敛到了元数据层，因此更容易在不同 batch、不同阶段保持相对稳定的性能。

### 4. 真实工程例子

在大规模 LLM 训练里，比如 dMoE 集成到 Megatron-LM 或 LLM Foundry 的场景，真正麻烦的不是“写一个 MoE 层”，而是“当 batch、序列长度、top-k 路由分布都在变化时，GPU 仍然能保持高利用率”。

传统 capacity-based 实现通常要反复调 `capacity_factor`：

- 设小了，会 drop token，训练语义被改写
- 设大了，padding 很重，显存和算力浪费增加
- 同一组超参数在不同训练阶段可能还不一样好用

MegaBlocks 的意义在于把这个问题改写成另一类问题：不再围绕“每个专家最多收多少 token”做人肉折中，而是围绕以下几件更可控的事情做优化：

- 块大小怎么选
- 元数据怎么编码
- 正向与转置索引怎么复用
- 本地算子与跨卡通信怎么重叠

换句话说，MegaBlocks 不是让 MoE 从此没有工程难题，而是把难题从“容量调参”转移到了“块稀疏执行栈”。对大规模训练来说，这通常是更值得的方向。

---

## 代码实现

下面先给一个 **可直接运行** 的 Python 玩具实现。它不是 Triton kernel，也不追求速度，目标只有两个：

1. 把“按专家分桶、按块补齐、保存元数据、执行专家前向、再回写”的流程讲清楚。
2. 让新手可以真的跑起来，看见中间结果，而不是只看伪代码。

这个版本用纯 Python 标准库实现，不依赖第三方包。

```python
from math import ceil
from pprint import pprint

BLOCK = 4  # 为了演示方便，故意设小一点；真实实现常见是 128

def pad_to_block(n, block=BLOCK):
    return ceil(n / block) * block if n > 0 else 0

def group_tokens_by_expert(assignments, num_experts):
    expert_to_tokens = {e: [] for e in range(num_experts)}
    for token_id, expert_id in enumerate(assignments):
        if not (0 <= expert_id < num_experts):
            raise ValueError(f"invalid expert id {expert_id} for token {token_id}")
        expert_to_tokens[expert_id].append(token_id)
    return expert_to_tokens

def build_block_metadata(assignments, num_experts, block=BLOCK):
    """
    assignments[i] = expert_id，表示 token i 路由到哪个 expert
    返回:
      - expert_to_tokens: 每个专家实际接收的 token 索引
      - blocks: 每个专家的块描述
    """
    expert_to_tokens = group_tokens_by_expert(assignments, num_experts)
    metadata = {
        "block_size": block,
        "experts": {}
    }

    for expert_id, token_ids in expert_to_tokens.items():
        padded_len = pad_to_block(len(token_ids), block)
        padded_token_ids = token_ids + [-1] * (padded_len - len(token_ids))

        blocks = []
        for block_start in range(0, padded_len, block):
            block_token_ids = padded_token_ids[block_start:block_start + block]
            blocks.append({
                "expert_id": expert_id,
                "block_index": block_start // block,
                "token_ids": block_token_ids,
                "valid_count": sum(tid != -1 for tid in block_token_ids),
            })

        metadata["experts"][expert_id] = {
            "real_tokens": len(token_ids),
            "padded_tokens": padded_len,
            "blocks": blocks,
        }

    return metadata

def expert_mlp_scalar(values, expert_id):
    """
    用一个确定性的两层“标量 MLP”代替真实矩阵乘法。
    真实专家是向量 -> 向量；这里为了易读，只处理标量。
    """
    w1 = expert_id + 2
    w2 = expert_id + 3

    hidden = []
    for v in values:
        if v is None:
            hidden.append(None)
        else:
            h = v * w1
            h = max(h, 0)  # 代替激活函数 ReLU
            hidden.append(h)

    out = []
    for h in hidden:
        out.append(None if h is None else h * w2)
    return out

def forward_dropless_toy(x, assignments, num_experts, block=BLOCK):
    """
    x: token 标量值列表，例如 [1, 2, 3, 4, 5]
    assignments: 每个 token 去哪个 expert，例如 [0, 1, 1, 0, 1]
    """
    if len(x) != len(assignments):
        raise ValueError("x and assignments must have the same length")

    metadata = build_block_metadata(assignments, num_experts, block=block)
    output = [None] * len(x)

    for expert_id in range(num_experts):
        expert_meta = metadata["experts"][expert_id]
        for block_info in expert_meta["blocks"]:
            token_ids = block_info["token_ids"]
            token_values = [None if tid == -1 else x[tid] for tid in token_ids]

            block_output = expert_mlp_scalar(token_values, expert_id)

            for tid, val in zip(token_ids, block_output):
                if tid != -1:
                    output[tid] = val

    return output, metadata

def dense_reference(x, assignments, num_experts):
    """
    不做块补齐，直接按 token 逐个走专家。
    用它验证 dropless block 版本的数值结果是否一致。
    """
    output = [None] * len(x)
    for token_id, (value, expert_id) in enumerate(zip(x, assignments)):
        result = expert_mlp_scalar([value], expert_id)[0]
        output[token_id] = result
    return output

if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7]
    assignments = [0, 1, 1, 0, 1, 2, 0]
    num_experts = 3

    dropless_out, meta = forward_dropless_toy(
        x=x,
        assignments=assignments,
        num_experts=num_experts,
        block=BLOCK,
    )
    ref_out = dense_reference(x, assignments, num_experts)

    print("dropless block output:", dropless_out)
    print("dense reference output:", ref_out)
    print("\nmetadata:")
    pprint(meta)

    assert dropless_out == ref_out
    assert meta["experts"][0]["real_tokens"] == 3
    assert meta["experts"][0]["padded_tokens"] == 4
    assert meta["experts"][1]["real_tokens"] == 3
    assert meta["experts"][1]["padded_tokens"] == 4
    assert meta["experts"][2]["real_tokens"] == 1
    assert meta["experts"][2]["padded_tokens"] == 4

    print("\nAll checks passed.")
```

这段代码有几个值得对照理解的点。

第一，`assignments` 是“路由结果”的最简版本。它只表示 top-1，即每个 token 只去一个专家。真实训练里常见 top-2，这时一个 token 会在逻辑上被复制到两个专家分支中，再在最后按门控权重加权求和。

第二，`-1` 是 padding 槽位，不是真实 token。真实 kernel 不会对它们做有意义的计算，通常会通过 mask 或边界检查跳过。

第三，`metadata` 就是块稀疏执行真正依赖的那部分信息。它至少要回答这些问题：

| 元数据 | 作用 |
|---|---|
| 哪些块存在 | 决定哪些块需要参与乘法 |
| 每个块对应哪些 token | 决定从哪里读入数据 |
| 每个块属于哪个 expert | 决定乘哪个专家权重 |
| 转置后索引如何映射 | 决定结果如何高效写回 |

如果换成更接近真实训练的写法，流程一般是下面这样：

```python
# 伪代码：展示真实 dMoE 前向的数据流，而不是可执行实现

BLOCK_M = 128
BLOCK_N = 128

def moe_forward(x, router_logits, w1, w2, metadata_cache):
    # 1. 路由：每个 token 选 top-k 专家，并得到门控权重
    topk_experts, topk_probs = router_topk(router_logits, k=2)

    # 2. 基于路由结果构建块元数据
    signature = route_signature(topk_experts)
    meta = metadata_cache.get(signature)
    if meta is None:
        meta = build_block_sparse_metadata(
            topk_experts=topk_experts,
            block_size=128,
            formats=["BCSR", "BCOO", "TRANSPOSE"],
        )
        metadata_cache[signature] = meta

    # 3. SDD: token -> expert hidden
    h = block_sparse_sdd(
        sparse_topology=meta.forward_topology,
        dense_input=x,
        expert_weight=w1,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
    )

    # 4. 专家内部非线性
    h = activation(h)

    # 5. DSD: expert hidden -> token
    y = block_sparse_dsd(
        dense_input=h,
        sparse_topology=meta.transpose_topology,
        expert_weight=w2,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
    )

    # 6. 若 top-k > 1，再按门控权重聚合多个专家分支
    y = combine_with_router_probs(y, topk_probs)

    return y
```

这段伪代码要表达的不是 API 细节，而是工程实现中的四个关键点：

1. **metadata 要缓存**  
   路由拓扑决定块结构。每次都重建元数据，预处理时间会很高。实际系统通常会缓存可复用的块布局和转置索引。

2. **正向索引和转置索引都要准备**  
   前向分发和回写的访存方向不同，只维护单向索引，反向和回写阶段很容易退化成低效 scatter/gather。

3. **padding 最好和 permutation 融合**  
   如果先重排 token，再单独做块补齐，就会多一次内存搬运。实际高性能实现倾向于一次完成“重排 + 对齐 + 记录索引”。

4. **块大小不是随便写个常数**  
   128×128 常见，是因为它和 GPU tile、共享内存、寄存器压力、向量化访存之间比较容易平衡。块太小，索引和调度开销重；块太大，尾块浪费上升。

---

## 工程权衡与常见坑

MegaBlocks 不是“免费午餐”。它省掉的是 capacity-based padding 的大量浪费，但换来的是更复杂的稀疏 kernel、元数据组织和调试成本。

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| expert token 数不是块大小倍数 | 尾块效率下降 | 块大小固定 | 在 permutation 阶段补齐到块倍数 |
| 只维护一种稀疏格式 | 前向快，反向或回写慢 | 访问方向不同 | 同时维护前向与转置索引，必要时用 BCSR/BCOO 组合 |
| 每轮都重建 metadata | CPU/GPU 预处理开销高 | 路由拓扑构建成本被放大 | 对重复出现的拓扑做缓存 |
| 块太大 | 尾部 padding 浪费增多 | 粒度过粗 | 在吞吐与尾块浪费之间折中 |
| 块太小 | 稀疏索引、调度和 launch 开销上升 | metadata 占比提高 | 选择与硬件 tile 更匹配的块尺寸 |
| 只盯前向吞吐 | 端到端速度不升反降 | 反向与转置写回变慢 | 前向、反向、转置一起测 |
| 忽略负载均衡 | 个别专家长期过热 | 路由偏斜 | 仍需 load balancing loss 或辅助路由正则 |
| 忽略跨卡通信 | 单卡 kernel 很快，整机不快 | all-to-all 成为瓶颈 | 将本地算子优化与专家并行通信重叠起来看 |

最容易误解的一点是：**MegaBlocks 消除了“大量无意义的 per-expert padding”，但没有消灭所有 padding**。它只是把 padding 从“每个专家都预留大容量”变成了“只补齐最后一个不完整块”。这两者的浪费规模完全不同。

另一个常见误区是只关注前向吞吐，忽略反向与转置访问。MoE 训练不是“前向跑通就结束”。如果没有为回写和梯度传播准备好合适的转置索引，系统很容易在 `DSD` 或其反向阶段退化成低效 scatter-gather，最后表现为“局部 kernel benchmark 很漂亮，但整层速度并不理想”。

还有通信边界。MegaBlocks 优化的是 **本地专家计算的执行形状**。如果专家按 expert parallelism 分布在多张卡上，那么 token dispatch 和返回时的 all-to-all 依然可能占据很大时间。也就是说，它把“本地算子”做快了，但不会自动把“跨设备移动数据”一起做没。

---

## 替代方案与适用边界

MegaBlocks 不是唯一方案。是否值得上它，取决于模型规模、训练环境和团队的底层工程能力。

| 方案 | 适用条件 | 优势 | 劣势 |
|---|---|---|---|
| 朴素 MoE + capacity factor | 小规模实验、验证想法 | 实现简单，容易调试 | padding 浪费大，容易 drop token |
| Tutel 类优化 | 需要更快的 batched GEMM，专家数不算太多 | 工程成熟度较高，接入成本低 | 仍依赖 capacity，仍可能丢 token |
| MegaBlocks / dMoE | 大模型、大 batch、路由波动大、不想丢 token | dropless，高吞吐，减少容量调参负担 | 需要 block-sparse kernel 与更复杂的元数据管理 |
| 回退到稠密 FFN | 模型较小，部署或平台环境受限 | 稳定、可移植、维护成本低 | 参数效率和计算效率都不如 MoE |

可以用几个非常实际的问题做判断。

第一，你的主要瓶颈是不是已经变成 `capacity_factor`？如果你发现：

- 经常要在“不 drop token”和“不浪费太多算力”之间反复调参
- 不同 batch 或不同训练阶段需要不同 capacity
- padding 和 token drop 已经明显影响吞吐或训练稳定性

那说明传统 capacity-based MoE 已经碰到边界，MegaBlocks 才会真正体现价值。

第二，你的系统栈能不能承受更复杂的内核实现？MegaBlocks 的收益来自块稀疏 kernel、稀疏格式编码和元数据复用。如果团队无法稳定维护 Triton/CUDA 自定义算子，或者平台环境对驱动、编译链、运行时兼容性限制很多，那么它未必是当前阶段最优解。

第三，你是更在乎“尽快验证模型想法”，还是更在乎“把大规模训练吞吐压榨出来”？前者通常更适合 Tutel 或普通 capacity-based MoE，后者更适合 MegaBlocks。

可以把选择原则压缩成一句话：

- 小规模实验阶段：先用更简单的 MoE 实现，接受少量 padding。
- 大规模训练阶段：当 capacity-based 方案的浪费已经成为主要瓶颈时，再上 MegaBlocks。

所以，MegaBlocks 的适用边界不是“只要做 MoE 就应该用”，而是“当传统 MoE 的容量机制已经明显妨碍训练效率和稳定性时，它才是更好的系统答案”。

---

## 参考资料

1. [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)，Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia，MLSys 2023。原始论文，核心价值在于给出 dMoE 的 block-sparse 重写方式，以及 block-sparse GPU kernel、blocked-CSR-COO encoding、transpose indices 等关键实现。  
2. [论文 PDF](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf)。如果要看公式、图示和实验表，直接看 PDF 更高效。  
3. [Training MoEs at Scale with PyTorch](https://pytorch.org/blog/training-moes/)，PyTorch Blog，2024-06-23。工程视角最有价值，重点是 MegaBlocks 在 PyTorch、LLM Foundry、专家并行、HSDP/FSDP 下的大规模训练实践。  
4. [stanford-futuredata/megablocks](https://github.com/stanford-futuredata/megablocks)。官方代码仓库，适合对照具体 API、稀疏格式和集成方式理解论文。  
5. 如果你想补足背景知识，建议先回看 MoE 基础论文 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)。它不是 MegaBlocks 的实现论文，但能帮助理解“为什么 MoE 会天然带来动态路由和负载不均”。
