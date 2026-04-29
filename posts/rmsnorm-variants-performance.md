## 核心结论

RMSNorm 的定义很直接：它对输入向量按均方根做缩放，也就是只控制“整体幅度”，不做“均值居中”。白话说，它负责把信号调到合适大小，但不强行把信号中心挪到 0。

它在大模型里的价值，不是“数学上更高级”，而是更省计算、更少数值操作、更适合现代 Transformer 的 pre-norm 结构。pre-norm 的意思是“先归一化，再过子层”，这样残差路径更稳定。

最重要的判断有三条：

| 方法 | 是否减均值 | 是否按 RMS/方差缩放 | 常见可学习参数 | 工程特征 |
|---|---:|---:|---:|---|
| LayerNorm | 是 | 是 | `γ, β` | 生态最成熟，开销更高 |
| RMSNorm | 否 | 是 | 通常只有 `γ` | 更轻，常见于 LLM |
| pRMSNorm | 否 | 近似是 | 通常只有 `γ` | 追求更低开销，但收益依赖实现 |

如果只看结论，RMSNorm 可以理解为“保留方向，控制长度”的归一化。这里的方向是指向量各维相对比例，长度是指数值整体大小。

---

## 问题定义与边界

Transformer 里为什么需要归一化？因为层数加深后，激活值会漂移。激活值就是神经网络中间层输出的数值。它如果忽大忽小，训练会更难收敛，混合精度下也更容易不稳定。

归一化要解决的是“尺度稳定”，不是“把数据变成高斯分布”，也不是“让每一层都长得一样”。RMSNorm 只做一件事：控制最后一维特征的幅度。

设输入为 $x \in \mathbb{R}^d$，RMSNorm 的作用范围是最后一维，也就是单个 token 的隐藏向量。它不跨 batch，不跨序列长度，不统计整批样本的均值和方差。

一个玩具例子最容易看出边界：

- 输入向量是 `[1, 2, 3, 4]`
- RMSNorm 会把这四个数整体缩放到较稳定的幅度
- 但输出仍然整体偏正，不会围绕 0 对称
- LayerNorm 会先减去均值 `2.5`，所以输出会分布在 0 的两边

这说明 RMSNorm 不是“LayerNorm 去掉 `β`”。`β` 是可学习偏置项，白话说是“最后再整体平移一点”。RMSNorm 去掉的不只是 `β`，更关键的是它连“减均值”这一步也没有。

可以用一个边界表来记：

| 问题 | RMSNorm 解决吗 | 说明 |
|---|---:|---|
| 控制隐藏状态尺度波动 | 是 | 核心目标 |
| 让输出零均值 | 否 | 不做均值中心化 |
| 替代 batch 统计 | 否 | 不看 batch 维 |
| 保证任何任务都优于 LayerNorm | 否 | 依赖模型结构与实现 |

真实工程里，RMSNorm 常见于 decoder-only 大模型，比如 Llama 一类结构。原因不是“别的都不能用”，而是这类模型通常更关心长深网络里的吞吐、稳定性和内核融合效率。

---

## 核心机制与推导

RMSNorm 的公式很短：

$$
RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
y_i = \gamma_i \cdot \frac{x_i}{RMS(x)}
$$

这里的 $\epsilon$ 是一个很小的正数，用来防止分母过小；$\gamma_i$ 是可学习缩放参数，白话说是“每个维度最后再调一下比例”。

LayerNorm 则多了两步：先算均值，再减均值。

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i,\quad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
y_i = \gamma_i \cdot \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i
$$

两者的核心差别可以压缩成一句话：LayerNorm 同时改“中心”和“尺度”，RMSNorm 只改“尺度”。

为什么只改尺度也能工作？因为很多 Transformer 层真正怕的是数值幅度失控，而不是均值必须为 0。尤其在残差网络里，保留一定的偏移信息并不一定是坏事，反而减少了额外数值操作。

继续看那个玩具例子，设：

- $x = [1,2,3,4]$
- $\epsilon = 0$
- $\gamma = [1,1,1,1]$

先算 RMS：

$$
RMS(x)=\sqrt{\frac{1^2+2^2+3^2+4^2}{4}}=\sqrt{7.5}\approx 2.7386
$$

所以 RMSNorm 输出约为：

| 维度 | 原值 | RMSNorm 输出 |
|---|---:|---:|
| 1 | 1 | 0.3651 |
| 2 | 2 | 0.7303 |
| 3 | 3 | 1.0954 |
| 4 | 4 | 1.4606 |

如果换成 LayerNorm：

- 均值 $\mu = 2.5$
- 方差 $\sigma^2 = 1.25$
- 标准差 $\sqrt{1.25} \approx 1.1180$

输出约为：

`[-1.3416, -0.4472, 0.4472, 1.3416]`

差异非常直观。RMSNorm 保留了原向量“全为正”的趋势；LayerNorm 则把它拉成以 0 为中心的分布。

这也能从几何上理解。设向量长度近似由 $\sqrt{\sum x_i^2}$ 决定，那么 RMSNorm 实际上是在按长度归一化，只是多了一个 $\sqrt{d}$ 的尺度因子。换句话说，它更接近“只修正模长”。

### pRMSNorm 是什么

pRMSNorm 中的 `p` 指只用部分维度估计 RMS。它的思想是：如果完整求 $\sum x_i^2$ 太贵，那就抽样一部分维度，得到一个近似值。

形式上可以写成：

$$
\widehat{RMS}(x)=\sqrt{\frac{1}{|S|}\sum_{i\in S}x_i^2+\epsilon}
$$

其中 $S$ 是被采样的维度集合。

它的本质不是改归一化目标，而是把“精确估计”换成“近似估计”。因此它带来的收益主要是性能，不是训练性质上的根本改变。

但要注意，近似少算几次平方和，不等于端到端就一定更快。现代 GPU 上很多操作的瓶颈不是算术次数，而是访存模式、kernel 启动、融合能力。抽样如果打乱内存访问，理论更省，实测反而可能没优势。

---

## 代码实现

工程实现里最容易出错的地方，不是公式，而是精度和广播细节。广播的意思是参数形状较小，但会自动扩展到整张量上参与运算。

下面先给一个纯 Python 可运行版本，方便理解数值行为：

```python
import math

def rms_norm(x, weight=None, eps=1e-6):
    assert len(x) > 0
    if weight is None:
        weight = [1.0] * len(x)
    assert len(weight) == len(x)

    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    y = [w * v / rms for v, w in zip(x, weight)]
    return y

def layer_norm(x, weight=None, bias=None, eps=1e-6):
    assert len(x) > 0
    n = len(x)
    if weight is None:
        weight = [1.0] * n
    if bias is None:
        bias = [0.0] * n
    assert len(weight) == n
    assert len(bias) == n

    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    y = [w * (v - mean) * inv_std + b for v, w, b in zip(x, weight, bias)]
    return y

x = [1.0, 2.0, 3.0, 4.0]
r = rms_norm(x, eps=0.0)
l = layer_norm(x, eps=0.0)

assert len(r) == 4
assert len(l) == 4
assert all(v > 0 for v in r)
assert abs(sum(l)) < 1e-6
assert round(r[0], 4) == 0.3651
assert round(l[0], 4) == -1.3416
```

如果放到 PyTorch 里，推荐实现会多两点：

1. 先转 `fp32` 计算 RMS  
2. 最后再转回原始 dtype

`fp32` 是 32 位浮点数，白话说就是比 `fp16/bf16` 更稳，但更占显存与带宽。归一化这一步通常值得用更高精度算。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_fp32 * inv_rms * self.weight
        return y.to(orig_dtype)
```

这里有几个实现点值得单独说明：

| 实现点 | 推荐做法 | 原因 |
|---|---|---|
| 归一化维度 | `dim=-1` | 对齐 Transformer 隐藏维 |
| 精度 | 先转 `fp32` | 低精度平方和更容易抖 |
| 分母计算 | 用 `rsqrt` | 通常更适合底层优化 |
| 参数 | 只保留 `weight` | 符合常见 RMSNorm 设计 |

`pRMSNorm` 的代码思路则应该明确和标准版分开：

```python
# x: [..., dim]
# sampled_idx: 预先选好的部分维度索引
sample = x[..., sampled_idx].float()
inv_rms = torch.rsqrt(sample.pow(2).mean(dim=-1, keepdim=True) + eps)
y = x * inv_rms * weight
```

真实工程例子是 Llama 风格 decoder block。一个 block 通常会在 attention 前和 MLP 前各做一次 pre-norm。如果这里用 RMSNorm，就意味着每经过一个子层前，都先把当前 token 表示的幅度拉回稳定区间，但不强行改写其均值结构。

---

## 工程权衡与常见坑

RMSNorm 常被说成“更快”，但这个说法必须拆开看。

第一层是理论复杂度。它确实比 LayerNorm 少了均值计算、减均值和偏置处理，操作更少。

第二层是实际 wall time，也就是端到端运行时间。这里就不一定了。现代训练里，很多归一化已经被 fused kernel 融合。fused kernel 的意思是多个小操作被合成一个底层内核，减少中间读写。如果 LayerNorm 的融合更成熟，而 RMSNorm 版本没优化好，理论更轻也可能跑不赢。

常见坑主要有五类：

| 常见坑 | 典型表现 | 规避方式 |
|---|---|---|
| 把 RMSNorm 当成 LayerNorm 小改版 | 理解输出分布出错 | 明确它不减均值 |
| 直接在 `fp16/bf16` 上算平方均值 | loss 抖动、偶发 NaN | 先转 `fp32` |
| `eps` 不一致 | 同权重不同实现结果对不上 | 训练与推理统一默认值 |
| 参数命名或形状不同 | checkpoint 加载失败 | 对齐框架命名与 shape |
| 盲目使用 pRMSNorm | 理论节省但实测无收益 | 先做 profile |

这里特别强调一个认知误区：RMSNorm 更轻，不代表它一定更稳定。它少做了一些事，也意味着它少约束了一些分布特征。如果你的模型、初始化、优化器设置更依赖“零均值”这一性质，LayerNorm 可能更稳。

另一个常见问题是兼容性。比如你从一个使用 `eps=1e-5` 的实现切到 `eps=1e-6`，即使只差一个数量级，长链路推理时也可能累积出可见差异。再比如有的框架把参数叫 `weight`，有的实现沿用 `gamma` 命名，加载 checkpoint 时就需要映射。

真实工程里，最稳妥的做法是：

1. 保持训练和推理的 RMSNorm 实现一致  
2. 固定 `eps` 和参数命名  
3. 混合精度下显式检查是否转 `fp32`  
4. 上 `pRMSNorm` 之前先 profile，而不是先相信论文里的 FLOPs

---

## 替代方案与适用边界

RMSNorm 不是唯一选择。更准确的说法是：它是在“性能、稳定性、生态兼容性”三者之间做了一个偏工程化的折中。

先看选择空间：

| 方案 | 优点 | 缺点 | 更适合的场景 |
|---|---|---|---|
| LayerNorm | 传统稳定、文献多、兼容性强 | 更重 | 通用 Transformer、老模型迁移 |
| RMSNorm | 更轻、实现简洁、LLM 常见 | 不做均值中心化 | pre-norm 大模型，尤其 decoder-only |
| pRMSNorm | 理论更省 | 近似误差与实现复杂度更高 | 特定高性能内核场景 |
| 无归一化 | 最少算子 | 训练风险大 | 研究对照或特殊初始化方案 |

可以把选择逻辑压缩成三条：

1. 如果你要一个稳定、保守、和老 checkpoint 兼容的基线，用 LayerNorm。  
2. 如果你做的是现代 LLM，尤其是 pre-norm decoder，RMSNorm 往往是更自然的默认项。  
3. 如果你在极限优化吞吐，才考虑 pRMSNorm，但前提是有 profile 证明它真的带来收益。

还有一个边界必须说清：RMSNorm 更适合“已经知道自己在做什么”的系统性替换，不适合把任意旧模型中的 LayerNorm 全部机械替掉。因为归一化不是孤立部件，它会和残差比例、初始化、学习率、混合精度策略一起作用。

对初级工程师来说，一个实用判断标准是：

- 小模型实验、教学代码、兼容旧仓库：优先 LayerNorm
- 大模型训练、追求更轻实现、参考 Llama 风格：优先 RMSNorm
- 追求极致性能数字：不要凭感觉选 pRMSNorm，先测

---

## 参考资料

1. [Root Mean Square Layer Normalization](https://www.research.ed.ac.uk/en/publications/root-mean-square-layer-normalization/)
2. [PyTorch `torch.nn.RMSNorm` Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
3. [PyTorch `torch.nn.LayerNorm` Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
4. [torchtune RMSNorm Source](https://docs.pytorch.org/torchtune/0.1/_modules/torchtune/modules/rms_norm.html)
5. [Hugging Face Transformers: Llama](https://huggingface.co/docs/transformers/main/model_doc/llama)
