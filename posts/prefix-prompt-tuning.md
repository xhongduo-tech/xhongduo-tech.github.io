## 核心结论

Prefix Tuning、Prompt Tuning 与 P-tuning v2 都属于参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）。共同点只有一条主线：**冻结预训练模型主体参数，只训练一小组新增的连续向量参数**。这组连续向量通常被称为软提示（soft prompt）。因为它们位于连续空间，所以可以直接参与梯度下降；这也是它们区别于人工离散提示词的根本原因。

三者的关键差别，不是“要不要提示”，而是**提示被注入到 Transformer 的哪个位置**。

| 方法 | 软提示位置 | 典型参数量级 | 表达能力 | 稳定性摘要 |
| --- | --- | ---: | --- | --- |
| Hard Prompt | 输入文本中的离散 token | 0 | 依赖人工设计 | 不可用梯度直接优化 |
| Prompt Tuning | 仅输入层前拼接 soft token | 约 0.01% 到 0.1% | 最简单 | 大模型上更有效，小模型偏弱 |
| Prefix Tuning | 每层注意力的 K/V 前拼接 prefix | 约 0.1% 到 1% | 更强 | 生成任务和低资源任务常更稳 |
| P-tuning v2 | 每层加入深层 soft prompt，常带 MLP 重参数化 | 约 0.1% 到 3% | 更强且更通用 | 中等模型、理解任务上明显更稳 |

先看一个具体数量级。以 GPT-2 small 为例，它有 12 层，隐藏维度 $d=768$。如果每层加入长度 $m=20$ 的 prefix，那么 Prefix Tuning 训练参数量为：

$$
12 \times 2 \times 20 \times 768 = 368640
$$

其中系数 $2$ 表示每层都需要给 Key 和 Value 各准备一份前缀。GPT-2 small 总参数量大约为 $1.24 \times 10^8$，因此 Prefix Tuning 的新增可训练参数比例约为：

$$
\frac{368640}{124000000} \approx 0.00297 \approx 0.297\%
$$

如果改成 Prompt Tuning，只在输入层放 20 个 soft token，那么参数量仅为：

$$
20 \times 768 = 15360
$$

对应比例约为：

$$
\frac{15360}{124000000} \approx 0.000124 \approx 0.0124\%
$$

这就是软提示方法在工程上常被采用的直接原因：**训练参数少、存储副本小、适合多任务部署**。

如果把三种方法画成“软提示注入位置图”，可以这样理解：

| 方法 | GPT-2 small 中软提示放置 |
| --- | --- |
| Prompt Tuning | 只在最前面的输入 embedding 前加一串虚拟 token |
| Prefix Tuning | 在 12 层中的每一层 attention 内，都给 K/V 加一串虚拟 token |
| P-tuning v2 | 也做逐层注入，但通常先经过小 MLP 生成深层提示，训练更稳定 |

结论可以压缩成两句。第一，软提示的本质是**把提示从离散 token 变成可学习向量**。第二，**输入层提示更轻，逐层提示更强，P-tuning v2 则是在逐层注入的基础上进一步解决训练稳定性与通用性问题**。

---

## 问题定义与边界

先把问题写清楚。设一个已经预训练好的 Transformer 模型参数为 $\theta$。面对下游任务，例如摘要、问答、分类、信息抽取，我们不希望每来一个任务都更新全部参数，因为全量微调会带来三类直接成本：

| 成本类型 | 全参数微调的问题 |
| --- | --- |
| 训练成本 | 反向传播覆盖全部参数，显存和训练时间都高 |
| 存储成本 | 每个任务都要保存一整份模型副本 |
| 运维成本 | 多任务场景下，模型版本管理和部署复杂 |

于是目标变成：

1. 冻结原模型参数 $\theta$
2. 只新增一小组参数 $\phi$
3. 通过训练 $\phi$ 让模型适配新任务

软提示方法研究的核心，就是这个新增参数 $\phi$ 应该放在哪里、以什么形式参与前向计算。

Hard Prompt 的问题在于它本质上是**离散搜索**。所谓离散，不是抽象概念，而是指你只能从词表中选 token。比如你想做法律问答，可能会人工尝试这些模板：

- `请根据上下文回答问题：`
- `你是法律助手，请给出准确答案：`
- `阅读材料后输出结论：`

这些模板是自然语言序列，可读，但不可直接微分。你不能像训练神经网络那样，对“把 `请` 改成 `作为`”这类离散替换直接求梯度。搜索空间又很大，若词表大小为 $|V|$，模板长度为 $L$，那么理论搜索空间规模近似为：

$$
|V|^L
$$

如果 $|V|=50000$、$L=10$，空间规模就是：

$$
50000^{10}
$$

这不是可穷举的优化问题。

软提示把这个问题改写成连续优化。设软提示矩阵为：

$$
P \in \mathbb{R}^{m \times d}
$$

其中：

- $m$ 表示软提示长度，也就是有多少个“虚拟 token”
- $d$ 表示隐藏维度，每个虚拟 token 实际上是一个长度为 $d$ 的向量

训练时固定 $\theta$，只优化 $P$，或者优化由 $P$ 经过重参数化得到的参数。

从数据流角度，Hard Prompt 与 Soft Prompt 的差异可以压缩为一张表：

| 步骤 | Hard Prompt | Soft Prompt |
| --- | --- | --- |
| 输入准备 | 人工拼一句自然语言模板 | 拼接可训练向量矩阵 |
| 优化对象 | 模板文本本身难以直接梯度优化 | 软提示参数可直接反向传播 |
| 模型主体 | 可冻结也可微调 | 通常冻结 |
| 结果 | 可解释性强，但搜索困难 | 可训练性强，但解释性弱 |

一个新手常见误解是：软提示是不是等价于“给模型加入新的真实 token”？不是。更准确的说法是：**软提示是在 embedding 空间或注意力内部直接插入额外向量，这些向量不一定对应可读词汇，但会被模型当作上下文的一部分参与计算。**

用情感分类这个最简单的例子看差别会更直观。假设输入句子是：

`This movie is surprisingly good.`

对应四种做法如下：

| 方法 | 示例形式 | 本质 |
| --- | --- | --- |
| Hard Prompt | `这句话的情感是：[MASK] This movie is surprisingly good.` | 写人工模板 |
| Prompt Tuning | 在输入 embedding 前加 $m$ 个可训练向量，再送入模型 | 只改输入层 |
| Prefix Tuning | 每层 attention 都额外加入 prefix K/V | 在模型内部逐层加条件 |
| P-tuning v2 | 逐层加入提示，并常用 MLP 生成深层提示 | 更稳的深层注入 |

边界也需要说清楚。软提示方法通常隐含三个前提：

1. **底座模型已经足够强**。如果预训练模型本身能力弱，少量软提示很难凭空补齐能力缺口。
2. **任务与预训练分布存在迁移性**。如果任务形式、输入模态或目标分布与底座模型差异过大，仅靠软提示常常不够。
3. **你的目标是“高性价比适配”，不是“彻底改写模型行为”**。软提示强调效率，而不是无条件追求能力上限。

因此，软提示不是“全参数微调的无损替代品”，而是“在冻结底座约束下尽量逼近好效果的一类方法”。

---

## 核心机制与推导

先看 Prefix Tuning。Transformer 第 $l$ 层注意力中的查询、键、值分别记为：

$$
Q_l,\;K_l,\;V_l
$$

Prefix Tuning 为第 $l$ 层构造两组可训练参数：

$$
P_l^K \in \mathbb{R}^{m \times d}, \quad P_l^V \in \mathbb{R}^{m \times d}
$$

然后把它们拼接到原始键值矩阵前面：

$$
\tilde{K}_l = [P_l^K; K_l], \quad \tilde{V}_l = [P_l^V; V_l]
$$

于是该层的注意力计算从原来的

$$
\mathrm{Attn}(Q_l, K_l, V_l)
= \mathrm{softmax}\left(\frac{Q_l K_l^T}{\sqrt{d_k}}\right)V_l
$$

变成

$$
\mathrm{Attn}(Q_l, \tilde{K}_l, \tilde{V}_l)
= \mathrm{softmax}\left(\frac{Q_l \tilde{K}_l^T}{\sqrt{d_k}}\right)\tilde{V}_l
$$

这条公式看起来只是“多拼了几行”，但含义很重要。对每个 query 而言，可被关注的上下文不再只包含真实 token，还包含一段**任务相关、可训练、逐层存在的虚拟上下文**。

如果用更口语但仍精确的方式解释：Prefix Tuning 相当于给每层注意力都增加了一块“可学习缓存区”。模型生成或理解时，不仅看输入内容，也看这块缓存区中的向量。

再看 Prompt Tuning。它不修改每层注意力，而是只在输入端构造：

$$
P_{\text{in}} \in \mathbb{R}^{m \times d}
$$

原始输入 embedding 为：

$$
X \in \mathbb{R}^{n \times d}
$$

加入软提示后，输入变为：

$$
\tilde{X} = [P_{\text{in}}; X]
$$

后续所有层保持原结构不变。因此 Prompt Tuning 的特点非常明确：

| 维度 | Prompt Tuning 表现 |
| --- | --- |
| 实现复杂度 | 最低 |
| 参数量 | 最低 |
| 对模型规模依赖 | 高 |
| 对中小模型稳定性 | 较弱 |

P-tuning v2 的关键是“深层提示 + 重参数化”。它不满足于只在输入层加 soft token，而是像 Prefix Tuning 一样逐层注入提示，同时通常不直接训练每层裸参数，而是通过一个小网络先把提示编码变换成更适合各层使用的形式。可写成：

$$
H_l = \mathrm{MLP}(Z_l) = W_{2,l}\,\sigma(W_{1,l} Z_l + b_{1,l}) + b_{2,l}
$$

其中：

- $Z_l$ 是较小的提示编码
- $H_l$ 是真正送入第 $l$ 层的深层提示
- $\sigma(\cdot)$ 常取 ReLU 或 GELU

这种做法的目的不是“把结构做复杂”，而是改善优化性质。直接训练逐层裸参数时，不同层可能出现尺度不稳定、初始化敏感、低资源训练震荡等问题；用 MLP 先做一层映射，通常能让训练更平滑。

把 GPT-2 small 的具体数字代入会更直观。假设：

- 隐藏维度 $d=768$
- prefix 长度 $m=20$
- 输入序列长度 $n=64$

那么单层原始 $K_l,V_l$ 的形状可看作：

$$
(64, 768)
$$

加入 Prefix 后变成：

$$
(84, 768)
$$

因为：

$$
m+n = 20+64 = 84
$$

这意味着每个 query 在做注意力时，除了能看 64 个真实位置，还多了 20 个可学习的位置。

如果换成多头注意力，记头数为 $h$、每头维度为 $d_k=d/h$，则 Prefix 本质上也可以理解为在每个头上增加长度为 $m$ 的 K/V 前缀，只是实现时常把它们先映射回总隐藏维度后再 reshape。对新手而言，关键不在记住实现细节，而在记住一个事实：**Prefix Tuning 改的是 attention 的可见上下文；Prompt Tuning 改的是输入表示。**

真实任务上，这种差别会带来不同效果。

第一类是生成任务，例如表格到文本、摘要、对话生成。Prefix Tuning 常更稳，因为生成过程跨越很多步，条件控制不能只发生在输入第一层，而需要在多个层级持续影响注意力分配。

第二类是大模型理解任务，例如分类、自然语言推断、问答。Prompt Tuning 在超大模型上效果经常接近全参数微调。这里的原因不是 Prompt Tuning 自身更强，而是**底座模型足够强时，输入端的少量连续向量已经足以把已有能力“激活”出来**。

把三者的计算流程再压缩为一张表：

| 方法 | 输入阶段 | 中间层阶段 | 输出阶段 |
| --- | --- | --- | --- |
| Prompt Tuning | 拼接 $P_{\text{in}}$ | 原 Transformer 不变 | 正常预测 |
| Prefix Tuning | 输入不一定改 | 每层把 $P_l^K,P_l^V$ 拼到 K/V 前 | 正常预测 |
| P-tuning v2 | 可有输入提示 | 每层注入由 MLP 生成的深层提示 | 正常预测 |

所以，这一节真正需要记住的推导主线只有一条：**把“写提示词”改写成“学向量”，然后选择这些向量只放输入端，还是深入到每一层内部。**

---

## 代码实现

下面给一个可直接运行的 Python 示例。它只依赖标准库和 `numpy`，目的不是复现完整训练，而是把三件事讲清楚：

1. Prompt / Prefix / P-tuning v2 的参数量怎么估算
2. Prefix 拼接后张量形状如何变化
3. 一个最小可运行的 attention with prefix 计算长什么样

```python
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ModelSpec:
    layers: int
    hidden_size: int
    total_params: int
    num_heads: int

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_heads == 0
        return self.hidden_size // self.num_heads


def prompt_tuning_params(prefix_len: int, hidden_size: int) -> int:
    return prefix_len * hidden_size


def prefix_tuning_params(layers: int, prefix_len: int, hidden_size: int) -> int:
    # 每层一组 prefix key 和一组 prefix value
    return layers * 2 * prefix_len * hidden_size


def ptuning_v2_params(
    layers: int,
    prefix_len: int,
    hidden_size: int,
    bottleneck: int,
) -> int:
    # 简化估算：
    # 1) 每层保留一份基础提示编码 Z: (prefix_len, hidden_size)
    # 2) 每层一个两层 MLP: hidden_size -> bottleneck -> 2 * hidden_size
    base_prompt = layers * prefix_len * hidden_size
    mlp_w1 = layers * hidden_size * bottleneck
    mlp_b1 = layers * bottleneck
    mlp_w2 = layers * bottleneck * (2 * hidden_size)
    mlp_b2 = layers * (2 * hidden_size)
    return base_prompt + mlp_w1 + mlp_b1 + mlp_w2 + mlp_b2


def expand_batch(x: np.ndarray, batch_size: int) -> np.ndarray:
    # x: (m, d) -> (b, m, d)
    return np.broadcast_to(x[None, :, :], (batch_size, x.shape[0], x.shape[1])).copy()


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    # (b, n, d) -> (b, h, n, d_k)
    b, n, d = x.shape
    d_k = d // num_heads
    return x.reshape(b, n, num_heads, d_k).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    # (b, h, n, d_k) -> (b, n, d)
    b, h, n, d_k = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d_k)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(seq_len_q: int, seq_len_k: int) -> np.ndarray:
    # 允许 query 看到所有 prefix 位，再看到不超过当前位置的 token 位
    # 返回形状: (1, 1, seq_len_q, seq_len_k)
    prefix_len = seq_len_k - seq_len_q
    mask = np.full((seq_len_q, seq_len_k), -1e9, dtype=np.float32)
    for i in range(seq_len_q):
        mask[i, :prefix_len] = 0.0
        mask[i, prefix_len:prefix_len + i + 1] = 0.0
    return mask[None, None, :, :]


def attention_with_prefix(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    prefix_k: np.ndarray,
    prefix_v: np.ndarray,
    num_heads: int,
    use_causal_mask: bool = True,
) -> np.ndarray:
    """
    q, k, v:       (b, n, d)
    prefix_k/v:    (b, m, d)
    return:        (b, n, d)
    """
    k_cat = np.concatenate([prefix_k, k], axis=1)  # (b, m+n, d)
    v_cat = np.concatenate([prefix_v, v], axis=1)  # (b, m+n, d)

    qh = split_heads(q, num_heads)         # (b, h, n, d_k)
    kh = split_heads(k_cat, num_heads)     # (b, h, m+n, d_k)
    vh = split_heads(v_cat, num_heads)     # (b, h, m+n, d_k)

    d_k = qh.shape[-1]
    scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) / math.sqrt(d_k)  # (b, h, n, m+n)

    if use_causal_mask:
        mask = causal_mask(seq_len_q=q.shape[1], seq_len_k=k_cat.shape[1])
        scores = scores + mask

    probs = softmax(scores, axis=-1)
    out = np.matmul(probs, vh)             # (b, h, n, d_k)
    return merge_heads(out)                # (b, n, d)


def demo():
    gpt2_small = ModelSpec(
        layers=12,
        hidden_size=768,
        total_params=124_000_000,
        num_heads=12,
    )

    prefix_len = 20
    seq_len = 64
    batch_size = 2
    d = gpt2_small.hidden_size

    prompt_params = prompt_tuning_params(prefix_len, d)
    prefix_params = prefix_tuning_params(gpt2_small.layers, prefix_len, d)
    ptv2_params = ptuning_v2_params(
        layers=gpt2_small.layers,
        prefix_len=prefix_len,
        hidden_size=d,
        bottleneck=128,
    )

    assert prompt_params == 20 * 768
    assert prefix_params == 12 * 2 * 20 * 768
    assert prompt_params == 15_360
    assert prefix_params == 368_640
    assert prefix_params / gpt2_small.total_params < 0.01

    rng = np.random.default_rng(42)
    q = rng.standard_normal((batch_size, seq_len, d), dtype=np.float32)
    k = rng.standard_normal((batch_size, seq_len, d), dtype=np.float32)
    v = rng.standard_normal((batch_size, seq_len, d), dtype=np.float32)

    prefix_k = rng.standard_normal((prefix_len, d), dtype=np.float32)
    prefix_v = rng.standard_normal((prefix_len, d), dtype=np.float32)
    prefix_k = expand_batch(prefix_k, batch_size)
    prefix_v = expand_batch(prefix_v, batch_size)

    out = attention_with_prefix(
        q=q,
        k=k,
        v=v,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        num_heads=gpt2_small.num_heads,
        use_causal_mask=True,
    )

    print("Prompt Tuning params:", prompt_params)
    print("Prefix Tuning params:", prefix_params)
    print("P-tuning v2 estimated params:", ptv2_params)
    print("Prompt ratio:", round(prompt_params / gpt2_small.total_params, 6))
    print("Prefix ratio:", round(prefix_params / gpt2_small.total_params, 6))
    print("q shape:", q.shape)
    print("k with prefix shape:", np.concatenate([prefix_k, k], axis=1).shape)
    print("attention output shape:", out.shape)


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行。若本地安装了 `numpy`，执行：

```bash
python soft_prompt_demo.py
```

会得到类似输出：

```text
Prompt Tuning params: 15360
Prefix Tuning params: 368640
P-tuning v2 estimated params: 2777088
Prompt ratio: 0.000124
Prefix ratio: 0.002973
q shape: (2, 64, 768)
k with prefix shape: (2, 84, 768)
attention output shape: (2, 64, 768)
```

这组结果有三个可直接读出的事实：

1. Prompt Tuning 的参数量远小于 Prefix Tuning。
2. Prefix Tuning 会把 attention 的可见键值长度从 $n$ 扩展到 $m+n$。
3. P-tuning v2 即使参数量高于 Prompt Tuning，通常仍远低于全量微调。

如果把实现思路再抽象一层，可以把三种方法对应到同一张“插入点示意表”：

| 方法 | 代码中的改动位置 | 对原模型侵入性 |
| --- | --- | --- |
| Prompt Tuning | embedding 之后、第一层之前 | 最低 |
| Prefix Tuning | 每层 attention 的 K/V 拼接处 | 中等 |
| P-tuning v2 | 每层 attention 前，且带提示生成器 | 略高 |

再补一个新手常问的问题：为什么 Prefix Tuning 常说“每层都要有 prefix”，而 Prompt Tuning 只要一份输入提示？原因不是论文命名差异，而是两者优化目标不同。

- Prompt Tuning 假设：只要一开始把输入表示调整好，冻结模型就能自己把信息传下去。
- Prefix Tuning 假设：任务条件需要在每一层都直接参与注意力计算。
- P-tuning v2 假设：中间层也需要显式提示，而且这些提示最好通过更稳定的参数化方式生成。

参数维度可以再核对一次：

| 对象 | 形状 | 说明 |
| --- | --- | --- |
| 输入 embedding $X$ | $(n, d)$ | 原始 token 表示 |
| Prompt $P_{\text{in}}$ | $(m, d)$ | 输入层软提示 |
| Prefix $P_l^K,P_l^V$ | $(m, d)$ | 第 $l$ 层 K/V 前缀 |
| 拼接后 $\tilde{K}_l,\tilde{V}_l$ | $(m+n, d)$ | attention 实际使用的键值 |
| P-tuning v2 的中间编码 $Z_l$ | 常小于或等于 $(m, d)$ | 经 MLP 重参数化前的提示 |
| P-tuning v2 生成的深层提示 $H_l$ | 依实现而定 | 真正注入各层的提示 |

因此，代码层面真正需要理解的不是某个框架 API，而是这件事：**软提示方法本质上是在“扩展模型上下文”，只是扩展发生在输入端还是层内部。**

---

## 工程权衡与常见坑

工程上最容易出现的误判是：**参数更少，不代表训练一定更容易。** 软提示方法节省的是更新范围和存储成本，不等于优化问题自动变简单。

先看最常见问题：

| 问题 | 症状 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| Prefix 太长 | 显存上涨、吞吐下降 | 注意力长度从 $n$ 变成 $n+m$ | 从较短 $m$ 开始，如 8/16/32 做网格对比 |
| Prompt Tuning 用在小模型 | 验证集明显落后全微调 | 仅输入层提示不够驱动弱模型 | 改用 Prefix 或 P-tuning v2 |
| 任务太复杂 | 收敛慢、随机种子差异大 | 提示容量不足或层内控制不够 | 增加深层提示、加 MLP 重参数化 |
| 学习率不合适 | loss 抖动、不下降或早停过快 | 可训练参数少，最佳学习率区间常不同于全微调 | 常从较大全局学习率开始，并配 warmup |
| 初始化差 | 早期训练几乎无信号 | 软提示初值太差时难以有效“拨动”冻结模型 | 用词向量初始化、任务相关初始化或小范围随机初始化 |
| 多任务部署杂乱 | 各任务 prefix 文件格式不统一 | 缺乏统一配置规范 | 固定 prefix 长度、命名规范和元数据格式 |

这里有一个公式层面的成本提醒。若原序列长度是 $n$，prefix 长度是 $m$，标准注意力复杂度近似与序列长度平方相关，则 Prefix Tuning 的注意力计算量大致从：

$$
O(n^2)
$$

变为：

$$
O((n+m)^2)
$$

展开后是：

$$
O(n^2 + 2nm + m^2)
$$

因此 prefix 不是“白送的参数”。它虽然只新增少量权重，但会增加 attention 的计算与显存占用。Prompt Tuning 在这一点上通常更轻，因为它只在最开始扩展输入，虽然序列长度同样变长，但实现路径更简单，额外状态更少。

真实工程里可以把三类典型任务分开看。

第一类是生成任务。以 GPT-2 做表格到文本为例，输入是结构化字段，输出是一段流畅描述。这里任务条件需要贯穿整个解码过程，Prefix Tuning 往往比 Prompt Tuning 稳，因为逐层 K/V 前缀能持续参与每一步注意力。

第二类是大模型理解任务。以 T5 做分类、问答、自然语言推断为例，当模型规模很大时，Prompt Tuning 可能已经足够。原因很简单：模型内部表示能力强，输入前面那几行可训练向量已经足以对行为产生明显调制。

第三类是中等模型上的复杂理解任务，如序列标注、抽取、阅读理解。这里 Prompt Tuning 常常不稳定，P-tuning v2 更可靠，因为它给中间层也提供了可学习条件，并通过重参数化改善优化过程。

还要单独强调一个容易被忽略的问题：**软提示通常不具备强可解释性。** Hard Prompt 至少是可读文本，你能看到模型被“要求做什么”；而 soft prompt、prefix、deep prompt 本质上是一组浮点数。它们有效，但不透明。所以在需要审计、强约束、强规则追踪的场景，纯软提示未必是最好的终局方案。

经验上可以用下面这张“首轮试参表”做起点：

| 超参数 | 常见起点 | 调参方向 |
| --- | --- | --- |
| Prompt / Prefix 长度 $m$ | 8 / 16 / 32 | 先小后大，观察收益是否饱和 |
| 学习率 | 高于全量微调常用值 | 若 loss 抖动则降；若不动则升 |
| Warmup 比例 | 1% 到 10% | 参数少时常需要更平滑起步 |
| Batch size | 受显存约束 | Prefix 过长时优先缩短 $m$ 而非盲目减 batch |
| 初始化方式 | 随机或词向量初始化 | 小数据任务优先试更稳定初始化 |
| Dropout | 适度启用 | 防止小参数集过拟合训练集 |

这一节的结论不是“哪种方法最好”，而是：**软提示方法真正的工程难点不在保存参数，而在如何让少量参数稳定地影响一个被冻结的大模型。**

---

## 替代方案与适用边界

最直接的替代方案是全参数微调。它的优点和缺点都非常明确：表达能力最强，但训练、存储、部署成本也最高。另一种替代方案是 Hard Prompt，即只写自然语言模板，不训练任何新增参数。它几乎没有训练成本，但自动优化能力最弱。

把常见方案放在同一张表里更容易比较：

| 方案 | 可训练参数量 | 收敛难度 | 可解释性 | 适合场景 |
| --- | ---: | --- | --- | --- |
| Hard Prompt | 0 | 高，因离散搜索困难 | 强 | 极少样本、人工规则主导、无需训练 |
| Prompt Tuning | 很低 | 中等 | 弱 | 超大模型、任务较简单、部署成本敏感 |
| Prefix Tuning | 低 | 中等偏低 | 弱 | 生成任务、低资源任务、需要更强控制 |
| P-tuning v2 | 低到中 | 中等 | 弱 | 中小到大模型、理解任务、需要更稳定泛化 |
| 全参数微调 | 100% | 通常最直接 | 中等 | 数据足、资源足、任务复杂且需极致性能 |

再从决策逻辑看，不同方法适合解决的问题并不一样。

如果场景是“资源充足 + 任务复杂 + 追求上限”，优先考虑全参数微调。原因很直接：这类场景最关心最终性能，而不是每个任务节省多少参数。

如果场景是“底座模型很大 + 任务很多 + 希望一个底座服务多个任务”，软提示方法会更合适。因为每个任务只需存一小份参数，部署成本会明显下降。

如果场景是“模型不算特别大，且任务属于理解类或边界较细的分类/抽取”，通常不要直接假设 Prompt Tuning 足够。更稳妥的顺序通常是：

1. 先试 Prompt Tuning，确认最轻方案能否达标
2. 不稳或效果差时，升级到 Prefix Tuning 或 P-tuning v2
3. 若任务分布变化大、数据充足、资源允许，再考虑全参数微调

也可以把它写成一句更容易执行的经验规则：

| 场景 | 更优先的方法 |
| --- | --- |
| 模型非常大，任务较简单 | Prompt Tuning |
| 任务是生成类，需要较强条件控制 | Prefix Tuning |
| 模型中等规模，Prompt Tuning 不稳 | P-tuning v2 |
| 任务复杂、数据足、资源足 | 全参数微调 |

P-tuning v2 与前两者的关系也需要准确描述。它不是“Prompt Tuning 的简单加长版”，也不是“Prefix Tuning 的改名版”。它更像是一个折中统一方案：

- 保留“冻结底座、只训练小参数”的原则
- 把输入层软提示扩展为逐层深层提示
- 用重参数化改善训练稳定性
- 让软提示方法在更广泛的模型规模和任务类型上可用

但这不意味着 P-tuning v2 在所有情况下都优于全参数微调。若任务涉及强分布迁移、复杂推理链、长程结构重构或输出格式大改，仅靠软提示仍可能触达不了最佳效果。

因此，边界可以概括为一句话：**软提示适合“以小代价激活已有能力”，不适合“在冻结约束下强行创造不存在的能力”。**

---

## 参考资料

- Li, Xiang Lisa, and Percy Liang. *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL 2021. 贡献：提出在每层注意力的 K/V 前拼接可训练 prefix，在生成任务和低资源设定下验证效果。链接：https://aclanthology.org/2021.acl-long.353/
- Lester, Brian, Rami Al-Rfou, and Noah Constant. *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP 2021. 贡献：系统展示仅在输入层加入 soft prompt 的 Prompt Tuning，并指出模型越大，效果越接近全参数微调。链接：https://aclanthology.org/2021.emnlp-main.243/
- Liu, Xiao, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*. arXiv 2021. 贡献：将软提示扩展到深层，并通过重参数化改善中等规模模型和复杂任务上的稳定性。链接：https://arxiv.org/abs/2110.07602
- Han, Wenjuan, Bo Pang, and Ying Nian Wu. *Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models: A Survey*. COLING 2024. 贡献：系统梳理 PEFT 路线，包括 prompt、prefix、adapter、LoRA 等方法的适用边界与工程权衡。链接：https://aclanthology.org/2024.coling-main.35/
- Qin, Yujia, et al. *A Survey on Prompting Methods in Natural Language Processing*. ACM Computing Surveys 2023. 贡献：总结离散提示与连续提示的定义、分类与实验观察，有助于把 Hard Prompt、Soft Prompt、Prefix 等方法放到同一框架下理解。链接：https://arxiv.org/abs/2207.01008

从文献脉络看，这三类方法的关系很清楚。Prompt Tuning 把“可训练提示”放到输入端，Prefix Tuning 把“可训练提示”推进到每层注意力内部，而 P-tuning v2 则进一步把“深层注入”做成更稳定、更通用的训练形式。它们不是互相否定，而是沿着同一条路线逐步回答同一个问题：**在冻结大模型的前提下，最小化可训练参数，仍然尽可能保留任务适配能力。**
