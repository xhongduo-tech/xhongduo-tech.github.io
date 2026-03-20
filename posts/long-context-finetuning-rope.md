## 核心结论

长上下文微调的核心不是“让模型看到更长的 token 序列”，而是“让位置编码在更长范围内仍然可解释”。RoPE 是 Rotary Position Embedding，白话讲就是把 token 的位置信息编码成一组按不同频率旋转的角度。问题在于，这组角度只在预训练最大长度 $L$ 内被充分学习过，直接把上下文从 4K 外推到 32K 或 100K，模型会进入没有学过的角度区间，注意力分数容易失真甚至数值不稳定。

Position Interpolation, PI 的思路很直接：不让位置 $p$ 真的走到 32K 或 100K 对应的“新角度”，而是先压回训练期熟悉的范围。公式是

$$
p' = p \cdot \frac{L}{L'}
$$

其中 $L$ 是原始训练长度，$L'$ 是目标长度。这样模型虽然读的是更长序列，但用于 RoPE 的位置索引仍落在原训练分布附近。

YaRN 在 PI 之上更进一步。它发现不同频率维度承担的信息不同，低频更适合表示长距离结构，高频更敏感于局部顺序，所以不能“一刀切”缩放。LongLoRA 则解决另一个工程问题：长上下文训练很贵，自注意力复杂度接近 $O(n^2)$。它用 S²-Attention，也就是 Shifted Sparse Attention，白话讲就是把长序列切块，只算局部稀疏注意力，再通过错位分组让信息跨块流动，从而把训练成本压下来。

一句话概括：RoPE 长上下文扩展的第一步通常不是直接训练更长，而是先做位置缩放；PI 解决“能不能稳定扩”，YaRN 解决“扩了以后长短兼顾”，LongLoRA 解决“扩的时候算不算得起”。

---

## 问题定义与边界

先把问题说清楚。我们讨论的是“已经使用 RoPE 的预训练模型”，例如很多 LLaMA 系列变体。目标是在不从头预训练的前提下，把上下文长度从原始训练窗口扩到更长，比如 4K 扩到 32K、64K 或 100K。

RoPE 的基础角频率可写成：

$$
\theta_j = \text{base}^{-2j/d}
$$

其中 $d$ 是 head 维度，$j$ 是频率维度索引。白话讲，不同维度以不同转速旋转，低维通常转得快，高维转得慢。位置越大，旋转角度越大。模型在训练时只见过 $[0, L]$ 这个区间内的角度组合，所以“位置扩展”的本质是“角度分布扩展”。

如果直接把 4K 训练的模型用于 32K，风险不只是“性能下降”，而是“位置相对关系开始失真”。可以把它想成一个颜色环：模型只认得 4K 内那一圈颜色，你把 32K 直接送进去，相当于颜色环绕了很多圈，局部颜色会重复但全局语义不再对应，模型就会误判距离与顺序。

下面是边界表：

| 原训练长度 $L$ | 目标长度 $L'$ | 直接外推风险 | 是否建议直接用 |
|---|---:|---|---|
| 4096 | 8192 | 中等，部分任务可勉强工作 | 不建议作为正式方案 |
| 4096 | 32768 | 高，角度漂移明显，长距注意力失真 | 不建议 |
| 4096 | 100000 | 很高，局部与全局关系都可能错乱 | 不可取 |
| 8192 | 32768 | 中高，需要缩放与再训练 | 需要 PI/YaRN |

一个玩具例子最容易说明问题。假设原训练长度 $L=4096$，目标长度 $L'=32768$。原本位置 20000 会产生“20K 级别”的 RoPE 角度，但 PI 先把它映射为：

$$
p' = 20000 \cdot \frac{4096}{32768} = 2500
$$

也就是说，模型虽然在读 20000 这个远位置的 token，但实际喂给 RoPE 的是 2500 对应的角度区间。这样旋转仍在训练期熟悉范围内。

可以用一个 ASCII 示意：

```text
原训练可见角度区间:
position: 0 ---------------- 4096
angle:    熟悉--------------熟悉

直接外推到 32768:
position: 0 ---- 4096 ---- 8192 ---- ... ---- 32768
angle:    熟悉   越界      更越界            严重越界

PI 压缩后:
原始位置: 0 ------------------------------ 32768
映射位置: 0 ---------------- 4096
angle:    仍然落在熟悉区间
```

边界还包括“不能只看长任务”。如果你只在 32K 或 100K 样本上继续训练，模型可能适应了缩放后的长文位置分布，却损伤原先短文本任务。原因不是模型“忘了知识”，而是它重新适应了位置统计。实际工程里，短文和长文混合微调几乎是必需项。

---

## 核心机制与推导

RoPE 的关键性质是：它通过旋转变换把“绝对位置”转成“相对位置可比较”的向量关系。若查询向量和键向量分别位于 $p$ 与 $q$，它们的内积会体现相对位移 $p-q$。这个性质建立在“旋转角度分布仍在模型学过的区域”上。

### 1. Position Interpolation

PI 的核心只有一步：把目标长度下的位置压缩回原长度范围。

$$
p' = p \cdot \frac{L}{L'}
$$

如果原模型训练长度是 4K，要扩到 32K，则缩放因子是 $1/8$。位置 0 到 32768 会被线性映射到 0 到 4096。优点是简单、稳定、改动小。缺点是位置分辨率被压缩，原本相隔 8 个 token 的位置在极端长上下文下可能被投到更接近的角度区间。

这也是钟表秒针类比的准确版本。RoPE 像一组不同转速的秒针，每来一个 token 就转一点。直接从 4K 推到 32K，相当于把秒针推得远超训练时的表盘范围。PI 则是先把 32K 的刻度重新映射回 4K 的表盘，让秒针继续在熟悉区间运动。

### 2. YaRN 的非均匀缩放

YaRN 认为所有频率统一压缩不够细。低频分量负责更长程的结构，过度保留原频率会限制长距离表示；高频分量负责局部顺序，过度缩放会伤害邻近 token 的辨别能力。于是它为不同频率引入缩放系数 $\lambda_j$。

可以用分段函数表达：

$$
\lambda_j =
\begin{cases}
\alpha, & j \in \text{低频区} \\
g(j), & j \in \text{中频过渡区} \\
1, & j \in \text{高频区}
\end{cases}
$$

这里 $\alpha$ 表示线性插值强度，$g(j)$ 表示过渡区的平滑函数，可结合 NTK-aware 的伸缩策略。白话讲，低频“拉长”一点，保留长期依赖；高频“不动”，守住局部顺序；中间部分平滑过渡，避免频率断层。

| 频率段 | 主要语义 | 处理方式 | 目的 |
|---|---|---|---|
| 低频 | 长距离结构 | 线性插值缩放 | 支持更长依赖 |
| 中频 | 过渡层 | NTK 风格平滑伸缩 | 减少断裂 |
| 高频 | 局部顺序 | 保持 $\lambda_j=1$ 或接近 1 | 保住短程精度 |

### 3. LongLoRA 的 S²-Attention

即便位置编码问题解决了，长上下文训练还是贵。标准自注意力需要计算所有 token 对之间的关系，序列长度从 4K 到 32K，成本不是 8 倍，而接近 64 倍。

LongLoRA 的 S²-Attention 采用分块加错位的办法。先把序列切成固定长度的 block，每个 block 内做局部注意力；然后让一部分 attention head 的块边界向前或向后平移半个 block，这样相邻块的信息也能被看到。

ASCII 示意：

```text
普通局部块:
[0 1 2 3] [4 5 6 7] [8 9 10 11]

shift 后的另一组 head:
  [2 3 4 5] [6 7 8 9] [10 11 ...]

结果:
一部分 head 看原块内关系
一部分 head 看跨块邻接关系
```

这不是精确全局注意力，而是“可训练、可扩展、近似全局”的稀疏注意力。对于长上下文微调，这种近似通常够用，因为任务重点是让模型学会在更长距离上传播关键信息，不一定要在所有位置对之间都做全连接计算。

### 4. 真实工程例子

真实工程里，一个典型流程是：把 4K 预训练的 LLaMA 类模型扩展到 32K。第一步修改 RoPE 位置映射，用 PI 缩放；第二步准备混合长度指令数据，例如 4K、8K、32K 混合；第三步用 LoRA 或 QLoRA 做若干百到一千多步 SFT；第四步在长文检索、passkey retrieval、多文档问答和原始 4K 基准上同时验证。

如果资源有限，比如只有单机 8 张 A100，那么 32K 以上继续拉长就要认真考虑 LongLoRA 一类稀疏训练策略，否则 batch size 很快降到不合理水平，训练会变得既慢又不稳定。

---

## 代码实现

下面先给一个最小可运行的 PI 实现。它不依赖深度学习框架，只演示“位置压缩”与 RoPE 角度计算的关系。

```python
import math

def rope_thetas(dim: int, base: float = 10000.0):
    assert dim % 2 == 0
    return [base ** (-2 * j / dim) for j in range(dim // 2)]

def interpolate_position(pos: float, train_max_len: int, target_max_len: int) -> float:
    assert train_max_len > 0 and target_max_len >= train_max_len
    return pos * (train_max_len / target_max_len)

def rope_angles(pos: float, dim: int, base: float = 10000.0):
    return [pos * theta for theta in rope_thetas(dim, base)]

# 玩具例子：4K 扩到 32K，位置 20000 被压到 2500
train_L = 4096
target_L = 32768
pos = 20000

scaled = interpolate_position(pos, train_L, target_L)
assert scaled == 2500.0

angles_direct = rope_angles(pos, dim=8)
angles_scaled = rope_angles(scaled, dim=8)

# 最高频分量在 PI 后角度显著减小
assert angles_scaled[0] < angles_direct[0]

# 目标长度末端会被映射回训练长度末端
assert interpolate_position(target_L, train_L, target_L) == train_L

print("scaled_pos =", scaled)
print("direct_first_angle =", angles_direct[0])
print("scaled_first_angle =", angles_scaled[0])
```

如果放到实际模型代码里，伪代码通常是这样的：

```python
def build_rope_positions(pos_ids, train_max_len, target_max_len):
    scale = train_max_len / target_max_len
    scaled_pos = pos_ids * scale
    return scaled_pos

def apply_pi_rope(q, k, pos_ids, train_max_len, target_max_len, base=10000.0):
    scaled_pos = build_rope_positions(pos_ids, train_max_len, target_max_len)
    cos, sin = precompute_rope_cos_sin(scaled_pos, q.shape[-1], base=base)
    q = rotate_with_rope(q, cos, sin)
    k = rotate_with_rope(k, cos, sin)
    return q, k
```

关键点有两个。第一，`base` 和维度定义要与原模型完全一致，否则你不是在“扩展 RoPE”，而是在“换一种 RoPE”。第二，PI 只改位置，不改模型主体结构，所以很适合做低成本 SFT。

YaRN 的实现本质上是在频率维度加一个 `lambda` 张量：

```python
def apply_yarn_scaling(thetas, lambdas):
    assert len(thetas) == len(lambdas)
    return [t * l for t, l in zip(thetas, lambdas)]

def make_piecewise_lambdas(n_freq, low=0.5, mid=0.75):
    lambdas = []
    for j in range(n_freq):
        ratio = j / max(1, n_freq - 1)
        if ratio < 0.33:
            lambdas.append(low)      # 低频缩放更多
        elif ratio < 0.66:
            lambdas.append(mid)      # 中频过渡
        else:
            lambdas.append(1.0)      # 高频保持
    return lambdas

thetas = rope_thetas(dim=8)
lambdas = make_piecewise_lambdas(len(thetas))
scaled_thetas = apply_yarn_scaling(thetas, lambdas)

assert scaled_thetas[-1] == thetas[-1]
assert scaled_thetas[0] < thetas[0]
```

在真实框架里，`lambdas` 一般会做成 shape 可广播的 tensor，例如 `[1, 1, 1, head_dim/2]`，这样可以直接乘到按频率组织的 `inv_freq` 或角度张量上。

S²-Attention 的核心不是复杂数学，而是 block mask 设计。下面是一个简化示意：

```python
def make_local_blocks(seq_len, block_size):
    blocks = []
    for start in range(0, seq_len, block_size):
        blocks.append((start, min(start + block_size, seq_len)))
    return blocks

def shift_blocks(blocks, shift):
    shifted = []
    for s, e in blocks:
        shifted.append((max(0, s - shift), max(0, e - shift)))
    return shifted

blocks = make_local_blocks(seq_len=16, block_size=4)
shifted = shift_blocks(blocks, shift=2)

assert blocks[0] == (0, 4)
assert shifted[1] == (2, 6)
```

它表达的是：一组 head 看 `(0,4)(4,8)(8,12)` 这样的原始块；另一组 head 看错位后的 `(0,2)(2,6)(6,10)` 之类的窗口。真正实现时需要处理 causal mask、padding 和跨块边界，但思路就是“局部块 + 错位覆盖”。

---

## 工程权衡与常见坑

长上下文扩展最常见的误区，是把它理解成“只要把 `max_position_embeddings` 改大就行”。这在 RoPE 模型上通常不成立，因为真正决定可扩展性的不是配置文件里的一个数字，而是旋转角度是否仍处在训练分布内。

下面是常见坑与规避方法：

| 常见坑 | 现象 | 根因 | 规避方法 |
|---|---|---|---|
| 直接外推 RoPE | 长文检索差、注意力混乱 | 角度越界 | 先用 PI 或 YaRN |
| 所有频率统一缩放 | 长短任务顾此失彼 | 高频和低频职责不同 | 用 YaRN 式分段缩放 |
| 只训长文不训短文 | 4K 任务退步 | 位置统计偏移 | 混合短文与长文 SFT |
| 只测 passkey 不测通用任务 | 误判模型已稳定 | 指标过窄 | 同时测短文、长文、多任务 |
| 长度拉太快 | 训练不稳定 | 分布跳变过大 | 分阶段扩展，例如 4K→16K→32K |
| 忽略显存预算 | batch 太小、训练慢 | 注意力成本过高 | 用 LongLoRA/S²-Attn 或更短阶段训练 |

一个实际可执行的训练流程通常是：

| 阶段 | 长度设置 | 数据比例建议 | 目标 |
|---|---|---|---|
| 适配阶段 | 4K + 8K | 短:长 = 3:1 | 先稳定位置缩放 |
| 扩展阶段 | 4K + 8K + 32K | 2:1:1 | 保住短文并学习长依赖 |
| 强化阶段 | 4K + 32K | 1:1 | 面向目标场景调优 |

这个比例没有唯一标准，但方向很明确：短文不能被完全挤掉。原因是模型部署后，大多数请求仍然是短输入。如果为了追求 32K 或 100K 的基准分数，把常见 1K 到 4K 任务性能换掉，工程收益往往是负的。

再给一个真实工程例子。假设你在做企业知识库问答，文档长度经常达到 20K 到 50K token。此时你希望模型能在单次上下文里读完多份规范文档，而不是分段检索后拼接摘要。PI 或 YaRN 可以解决“模型读得下”的问题，但如果训练资源有限，LongLoRA 才能解决“你是否训得起”的问题。相反，如果你的应用主要是客服单轮问答，平均输入不到 2K，那长上下文微调优先级可能并不高，做 retrieval 或 rerank 往往更划算。

---

## 替代方案与适用边界

PI、YaRN、LongLoRA 都是在“继续沿用 RoPE”这个前提下成立的方案。如果模型已经是 RoPE 架构，且你希望以最小改动获得更长上下文，它们通常是第一选择。

但这不是唯一路线。下面给出常见替代方案：

| 方案 | 是否依赖 RoPE | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| PI | 是 | 实现简单、改动小、稳定 | 分辨率被线性压缩 | 已有 RoPE 模型，先从 4K 扩到 16K/32K |
| YaRN | 是 | 长短语义兼顾更好 | 实现更复杂，超参更多 | 需要兼顾短文精度和长文能力 |
| LongLoRA | 是 | 显著降低长文训练成本 | 训练逻辑更复杂 | 单机多卡、资源有限的长文微调 |
| ALiBi | 否 | 天然线性扩展，不依赖旋转外推 | 需模型原生支持或重训适配 | 新模型设计或结构可重构场景 |
| T5 相对位置偏置 | 否 | 相对距离建模直接 | 与 decoder-only RoPE 路线不同 | encoder-decoder 或相对位置架构 |
| BigBird/稀疏注意力 | 否/可组合 | 长序列计算更省 | 稀疏模式设计复杂 | 极长文档预训练或专门长文模型 |
| RAG/检索切块 | 无关 | 不必真的扩大上下文 | 召回错误会传播 | 知识库问答、成本敏感部署 |

适用边界可以概括为三条：

第一，若你手里已经有一个 RoPE 预训练模型，并且想在最短时间内从 4K 扩到 32K，优先 PI；如果你发现短文本能力掉得明显，再考虑 YaRN 式分频缩放。

第二，若瓶颈不是“位置编码不稳”，而是“训练算力不够”，优先 LongLoRA 一类稀疏训练方法。它解决的是成本问题，不直接替代 PI/YaRN，通常是配套关系。

第三，若你的业务并不要求模型在一次前向中真正理解 50K 到 100K 连续上下文，而只是需要访问大量外部知识，那么检索增强生成往往比硬扩上下文更经济。长上下文不是默认更好，它只是在“信息必须同时保留在同一上下文窗口里”时才真正有价值。

---

## 参考资料

| 文献/资源 | 核心贡献 | 对应章节 |
|---|---|---|
| Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023) | 提出 PI，用位置压缩替代直接外推，使 RoPE 模型能稳定扩展上下文 | 核心机制与推导、代码实现 |
| YaRN / LongRoPE 相关技术解读与实现资料 | 强调按频率分段缩放，兼顾长距依赖与局部顺序 | 核心机制与推导、工程权衡与常见坑 |
| LongLoRA, ICLR 2024 | 提出 LoRA 配合 S²-Attention，在有限资源下完成超长上下文微调 | 核心机制与推导、代码实现、替代方案与适用边界 |
| Zheng et al., *When Long Helps Short* (EMNLP 2025) | 说明长短上下文混合训练的重要性，长文训练不应以短文退化为代价 | 工程权衡与常见坑 |

1. Chen et al., arXiv 2023, *Extending Context Window of Large Language Models via Positional Interpolation*。核心结论是：RoPE 直接外推风险高，位置插值是更稳定的扩展路径。  
2. YaRN / LongRoPE 相关公开资料。核心价值在于解释为什么不同频率维度不能统一缩放，以及如何通过分段策略兼顾长短语义。  
3. LongLoRA, ICLR 2024。核心贡献是 S²-Attention，让长上下文微调从“理论可行”变成“资源上可做”。  
4. Zheng et al., EMNLP 2025, *When Long Helps Short*。核心提醒是：长上下文训练要看短任务回归，不要只看长文基准。
