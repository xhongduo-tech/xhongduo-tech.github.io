## 核心结论

QK-Norm 的核心不是重写 attention，而是在进入 attention 之前先把 `Q` 和 `K` 的尺度压住。`Q` 是 Query，表示“当前 token 想找什么”；`K` 是 Key，表示“每个候选 token 提供什么线索”。标准 attention 同时受方向和长度影响，长度一旦失控，$QK^T$ 会变得很大，softmax 很快饱和，模型过早把注意力压成接近 one-hot，训练稳定性变差。

标准形式是：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

QK-Norm 的通用写法是：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\alpha \cdot \frac{\mathrm{Norm}(Q)\mathrm{Norm}(K)^T}{\sqrt{d_k}}\right)V
$$

其中 `Norm` 可以是 `L2`、`LayerNorm` 或 `RMSNorm`，$\alpha$ 是可学习缩放，作用是把被归一化压平的 logits 再拉回到合适范围。

一句话理解：QK-Norm 先把双方拉到同一量级，再比较它们是否相似。

| 对比项 | 标准 attention | QK-Norm attention |
| --- | --- | --- |
| logits 来源 | 方向 + 范数共同决定 | 更接近方向相似度 |
| softmax 状态 | 更容易饱和 | 更可控 |
| 主要收益 | 表达力直接 | 稳定性更强 |
| 典型用途 | 通用默认结构 | 大模型训练稳定化 |
| 是否是“性能魔法” | 否 | 否，主要是稳定性改造 |

结论先给清楚：QK-Norm 主要是稳定性改造，不是炫技技巧。它最有价值的场景，是 attention logits 过大导致训练不稳的大模型预训练。

---

## 问题定义与边界

设输入隐状态为 $X$，线性投影得到：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

这里 `V` 是 Value，表示真正要被加权汇总的内容。问题出在 $QK^T$。点积不是只看方向，还看向量长度。两个方向差不多的向量，如果有一个长度特别大，点积也会被放大。attention 看到的是“大”，不一定是“像”。

玩具例子先看最小情况。设：

$$
q=(3,4),\quad k_1=(6,8),\quad k_2=(-6,8)
$$

原始点积为：

$$
q\cdot k_1=50,\quad q\cdot k_2=14
$$

如果直接做 softmax，即使再除以 $\sqrt{2}$，两个 logits 仍然偏大，模型会非常偏向 $k_1$。但这里真正重要的，不只是“谁绝对更大”，而是“谁方向更接近”。

做归一化后：

$$
\tilde Q=\mathrm{Norm}(Q),\quad \tilde K=\mathrm{Norm}(K)
$$

如果用 `L2` 归一化，向量会被缩放到单位长度，点积就更接近余弦相似度。余弦相似度可以理解为“两个方向有多对齐”。

QK-Norm 的边界也要说清楚。

| 版本 | 归一化对象 | 归一化维度 | 是否常配 $\alpha$ | 说明 |
| --- | --- | --- | --- | --- |
| 原始 `L2` QKNorm | `Q` 和 `K` | 通常按每个 token 的 head 内特征维 | 是 | 点积更接近余弦相似度 |
| 工程版 `LayerNorm` | `Q` 和 `K` | head 内特征维 | 常见 | 同时处理均值和方差 |
| 工程版 `RMSNorm` | `Q` 和 `K` | head 内特征维 | 常见 | 只按均方根缩放，计算更直接 |
| 普通 attention | 不做 | 无 | 仅固定 $1/\sqrt{d_k}$ | 范数漂移直接进入 logits |

常见误区是把“只对 `Q` 做归一化”也叫 QK-Norm。这不完整。因为 `K` 的范数仍然会把 logits 拉爆，训练不稳定的问题并没有被根治。

另一个边界是：QK-Norm 不是所有模型都必须加。它解决的是 attention logits 数值不稳，不是数据质量、模型容量、优化器配置等所有问题。

---

## 核心机制与推导

把机制压缩成一句：QK-Norm 把 $QK^T$ 从“长度 + 方向混合信号”，改造成更接近“方向相似度信号”。

先看原始 attention：

$$
\mathrm{score}_{ij}=\frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

其中：

$$
q_i \cdot k_j = \|q_i\|\,\|k_j\|\cos\theta_{ij}
$$

这说明分数由三部分共同决定：$\|q_i\|$、$\|k_j\|$ 和夹角 $\theta_{ij}$。如果 $\|q_i\|$ 或 $\|k_j\|$ 很大，那么即使方向信息一般，分数也可能被放大。

若先做单位归一化，则：

$$
\hat q_i=\frac{q_i}{\|q_i\|},\quad \hat k_j=\frac{k_j}{\|k_j\|}
$$

于是：

$$
\hat q_i \cdot \hat k_j = \cos\theta_{ij}
$$

这时 logits 范围更可控，softmax 不容易被异常大值打穿。继续用前面的玩具例子：

$$
\hat q=(0.6,0.8),\quad \hat k_1=(0.6,0.8),\quad \hat k_2=(-0.6,0.8)
$$

归一化后点积变成：

$$
\hat q\cdot \hat k_1=1,\quad \hat q\cdot \hat k_2=0.28
$$

原来是 `50 vs 14`，现在变成 `1 vs 0.28`。排序没变，但尺度被压住了。softmax 不再过早极化。

但只压尺度会带来另一个问题：logits 可能过平，注意力区分能力下降。所以工程上通常加可学习缩放 $\alpha$：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\alpha \cdot \frac{\tilde Q\tilde K^T}{\sqrt{d_k}}\right)V
$$

$\alpha$ 可以理解为一个可学的温度。温度的白话解释是“控制分布有多尖锐”。$\alpha$ 大，分布更尖；$\alpha$ 小，分布更平。于是 QK-Norm 的逻辑变成两步：

1. 先用归一化消掉不受控的范数漂移。
2. 再用 $\alpha$ 把表达范围学回来。

三种常见归一化的差异如下。

| 方法 | 保留均值信息 | 保留方向信息 | 数值范围控制 | 典型特点 |
| --- | --- | --- | --- | --- |
| `L2` | 不强调 | 最直接，最接近余弦相似度 | 强 | 更贴近原始 QKNorm 定义 |
| `LayerNorm` | 会中心化 | 会改变原始向量几何形状 | 强 | 工程实现常见 |
| `RMSNorm` | 不减均值 | 相对保留尺度结构 | 强 | 计算更轻，常见于大模型 |

真实工程例子可以看超大模型预训练。训练规模上去后，某些层的 `Q/K` 范数会逐渐漂移，attention logits 方差随之变大，softmax 熵下降。熵是“分布有多分散”的量，熵过低通常说明注意力过尖。此时 loss 会震荡，极端时直接 divergence。QK-Norm 的收益不一定表现在单步速度，而是让训练更稳、更少炸。

如果画图，最值得画的不是 fancy 架构图，而是两张分布图：
1. 归一化前后的 logits 直方图。
2. softmax 熵随训练步数的变化曲线。

这两张图最能解释 QK-Norm 在做什么。

---

## 代码实现

实现原则只有三个：

1. 归一化发生在 attention 点积之前。
2. 归一化通常按每个 head 的特征维做，不跨 head。
3. 保留可学习缩放 $\alpha$。

先给一个最小可运行的 `python` 例子，验证玩具数据的行为。

```python
import math

def l2_norm(x, eps=1e-12):
    s = sum(v * v for v in x)
    scale = math.sqrt(max(s, eps))
    return [v / scale for v in x]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

q = [3.0, 4.0]
k1 = [6.0, 8.0]
k2 = [-6.0, 8.0]

raw_scores = [dot(q, k1), dot(q, k2)]
raw_probs = softmax(raw_scores)

q_hat = l2_norm(q)
k1_hat = l2_norm(k1)
k2_hat = l2_norm(k2)

norm_scores = [dot(q_hat, k1_hat), dot(q_hat, k2_hat)]
norm_probs = softmax(norm_scores)

assert raw_scores == [50.0, 14.0]
assert round(norm_scores[0], 2) == 1.00
assert round(norm_scores[1], 2) == 0.28
assert raw_probs[0] > 0.999
assert 0.60 < norm_probs[0] < 0.70
```

上面这个例子说明两件事：原始 attention 很容易变成“几乎全选一个”，归一化后分布明显更可控。

下面是接近工程实现的 PyTorch 风格版本：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKNormAttention(nn.Module):
    def __init__(self, d_model, num_heads, eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def _reshape(self, x):
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, T, D]

    def forward(self, x, mask=None):
        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))

        q = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2, dim=-1, eps=self.eps)

        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = logits * self.alpha / (self.head_dim ** 0.5)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out_proj(out)
```

实现项对照如下。

| 项目 | 建议 |
| --- | --- |
| 位置 | attention 前 |
| 对象 | `Q` 和 `K` 同时处理 |
| 维度 | head 内特征维 `dim=-1` |
| 参数 | 保留可学习 `alpha` |
| 精度 | 混合精度下重点观察 logits 范围 |

调试时不要只盯 loss。更有效的检查清单是：

| 检查项 | 你要看什么 |
| --- | --- |
| logits 均值/方差 | 是否明显失控 |
| softmax 熵 | 是否过早塌缩 |
| `Q/K` 范数分布 | 是否层间漂移 |
| 梯度范数 | 是否突然尖峰 |
| loss 曲线 | 是否震荡或发散 |

---

## 工程权衡与常见坑

QK-Norm 的主要收益是稳定性，但代价也真实存在。它改变了 attention 的分数生成方式，等于主动限制范数对 logits 的影响。如果模型原本确实需要通过范数编码某些强信号，过强的归一化可能损失部分表达力。

最常见的坑如下。

| 常见坑 | 后果 | 规避方法 |
| --- | --- | --- |
| 只归一化 `Q` 或只归一化 `K` | 另一侧仍可放大 logits | 必须双侧归一化 |
| 去掉 `alpha` | attention 过平，区分度下降 | 保留可学习缩放 |
| 归一化维度写错 | 跨 head 污染表示 | 只在 head 内特征维处理 |
| 把 `L2` 与 `RMSNorm` 视为等价 | 调参经验失真 | 先确认参考实现 |
| 忽略混合精度 | 低精度下数值异常 | 监控 logits 和梯度 |
| 与 `RoPE` 组合时位置不一致 | 行为和参考论文不一致 | 固定顺序并对齐实现 |

这里有一个实际排查顺序。若训练初期 loss 震荡，同时 attention logits 方差很大，不要第一反应就改学习率。先查三件事：`Q/K` 的范数分布、是否双侧归一化、是否保留 `alpha`。因为如果根因是 attention 分数本身数值过大，单纯改学习率往往只能缓解，不能消掉源头。

和 `RoPE` 的耦合也要注意。`RoPE` 是旋转位置编码，白话说是把位置信息通过旋转写进向量方向里。如果你在错误的位置做归一化，可能把本来依赖方向的结构改弱，或者让行为与参考实现不一致。工程里最稳妥的做法不是“凭感觉调”，而是严格对照同类模型的顺序。

建议监控三张图：
1. logits 分布直方图。
2. attention 熵曲线。
3. 梯度范数曲线。

这比只看最终验证集指标更能快速定位问题。

---

## 替代方案与适用边界

QK-Norm 不是唯一的稳定化方法。它只是针对 attention logits 这个局部问题的一种强力修正。

| 方法 | 是否改 attention 本身 | 稳定性收益 | 表达力影响 | 适用场景 |
| --- | --- | --- | --- | --- |
| QK-Norm | 是 | 高 | 可能略受约束 | logits 过大、训练易炸 |
| 调小初始化 | 否 | 中 | 小 | 训练初期不稳 |
| `LayerNorm/RMSNorm` 结构调整 | 间接 | 中到高 | 依实现而定 | 全局稳定性问题 |
| 调整 temperature | 是 | 中 | 明显影响分布尖锐度 | 想快速试验 attention 形状 |
| logit clipping | 是 | 高但粗暴 | 可能损伤表达 | 明显溢出或极端值 |
| 优化器/学习率修正 | 否 | 中 | 小 | 全局优化不稳 |
| 数据清洗 | 否 | 取决于数据 | 无直接结构副作用 | 训练目标本身脏乱 |

适用边界可以用两句话概括：

1. 当训练不稳定主要来自 attention logits 过大时，优先考虑 QK-Norm。
2. 当主要矛盾不在 attention 时，先查数据、优化器和初始化。

新手最容易犯的错误，是把所有训练问题都归因到 attention。比如数据噪声很大、标签错误很多、tokenizer 切分异常、优化器超参不合理，这些问题不会因为加了 QK-Norm 就自动解决。

因此，QK-Norm 更像一个“定点修复器”，不是“万能稳定器”。如果日志里已经显示某几层 attention logits 方差异常、softmax 熵急剧下降、梯度出现尖峰，那它很值得优先试。反过来，如果 attention 各项统计都正常，而模型还是学不动，就不该继续在 QK-Norm 上耗时间。

---

## 参考资料

| 来源 | 类型 | 贡献点 | 适合引用的位置 |
| --- | --- | --- | --- |
| Henry et al., *Query-Key Normalization for Transformers* | 原始论文 | 给出 QKNorm 的核心定义与动机 | 定义、理论机制 |
| Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters* | 工程论文 | 展示大规模训练中 QK 归一化类技术的稳定化价值 | 工程动机、扩展理解 |
| Team OLMo, *2 OLMo 2 Furious* | 工程报告 | 说明大模型训练里 QK-norm / RMSNorm 的实际使用 | 真实工程例子 |
| `CyndxAI/QKNorm` | 源码仓库 | 提供可运行参考实现 | 代码实现、细节对齐 |

需要区分两个概念：原始论文里的 QKNorm 更接近严格定义，通常指 `L2` 归一化；大模型工程里常说的 “QK-Norm” 往往是广义叫法，可能具体落成 `LayerNorm` 或 `RMSNorm` 版本。写文章或做实验时，必须明确自己指的是哪一种，否则结论会混淆。

- Henry et al., *Query-Key Normalization for Transformers*: https://huggingface.co/papers/2010.04245
- Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*: https://proceedings.mlr.press/v202/dehghani23a.html
- Team OLMo, *2 OLMo 2 Furious*: https://huggingface.co/papers/2501.00656
- CyndxAI/QKNorm: https://github.com/CyndxAI/QKNorm
