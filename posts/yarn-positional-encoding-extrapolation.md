## 核心结论

YaRN 的核心不是“把所有位置统一缩放”，而是“按频段分别处理 RoPE 频率，再补一个 attention 温度修正”。RoPE 是旋转位置编码，白话说就是把 token 位置变成一组不同转速的二维旋转，低维转得快，高维转得慢。YaRN 发现，不同转速对应的信息职责不同，不能一刀切。

具体做法是：

| 频段 | 直观职责 | 处理方式 |
| --- | --- | --- |
| 低频 | 管长距离结构 | 线性插值，频率缩到 $\theta/s$ |
| 中频 | 兼顾局部与全局 | 用斜坡函数在 $\theta/s$ 与 $\theta$ 之间平滑过渡 |
| 高频 | 管局部细节 | 保持原频率，不做插值 |

再加上 softmax 前的温度修正，YaRN 能在不改主干参数的前提下，把 LLaMA 这类 RoPE 模型的上下文扩展到 32K、64K、128K，并且所需微调 token 和步数显著少于早期方法。论文报告它比此前方法少约 10 倍 token、少约 2.5 倍训练步数。

一个容易混淆但必须说清的点是：论文经验式写的是

$$\frac{1}{\sqrt{t}}=0.1\ln(s)+1$$

不是 $t=0.1\ln(s)+1$。很多工程实现直接把右边这个量记成 `mscale`，然后乘到 RoPE 或 q/k 上，所以代码里常见的是 `0.1 * log(s) + 1`，但它对应的是 $1/\sqrt{t}$，不是 $t$ 本身。

---

## 问题定义与边界

问题定义很直接：模型训练时只见过长度 $L$ 的序列，例如 2048 或 4096，推理时却希望它稳定处理 $L' = sL$ 的更长序列。RoPE 在训练窗口内表现很好，但直接外推会出两个方向的问题。

第一类问题是高频维度过早“绕圈”。高频维度就是波长短、转得快的维度，白话说它负责很细的局部位置差异。长度一拉长，这些维度会反复旋转，导致远距离位置出现别名，模型把不同位置看得太像。

第二类问题是低频维度被拉得过松。低频维度就是波长长、转得慢的维度，白话说它负责全局布局。若完全不压缩，模型在更长区间里学不到稳定的全局距离感；若把所有维度一起压缩，又会把高频细节一并抹掉。

因此 YaRN 的边界条件是：它适用于“底座模型已经使用 RoPE”这一前提，目标是尽量少改动地扩展上下文。它不是通用位置编码理论，更不是替代长序列训练本身。若目标是百万级乃至更长上下文，YaRN 通常要和 Dynamic Scaling、分阶段扩展或更激进的方法联合使用，而不是单独解决全部问题。

用一个玩具例子看最直观。假设原始窗口是 2048，要扩到 32768，即 $s=16$。如果你把所有维度都统一改成 $\theta/16$，高频维度本来负责“相邻几十个 token 的细粒度顺序”，现在也被压成慢速旋转，局部辨别力会下降。YaRN 的观点是：低频该压，高频别碰，中间再平滑过渡。

---

## 核心机制与推导

YaRN 先定义每个 RoPE 维度的波长 $\lambda_d$ 与旋转频率 $\theta_d$：

$$\lambda_d=\frac{2\pi}{\theta_d}$$

再定义这个维度在原训练窗口 $L$ 内会转多少圈：

$$r(d)=\frac{L}{\lambda_d}$$

这个 $r(d)$ 很关键。它的白话含义是：这个维度在“训练时看过的最长长度”里到底活跃到什么程度。

- 若 $r(d)$ 很小，说明一个训练窗口里它连一圈都转不完，它更像低频全局信号。
- 若 $r(d)$ 很大，说明它在训练窗口里已经转了很多圈，它更像高频局部信号。

YaRN 用两个阈值 $\alpha,\beta$ 把频谱切成三段。论文在 LLaMA 系列上给出的经验值是 $\alpha=1,\beta=32$。斜坡函数定义为：

$$
\gamma(r)=
\begin{cases}
0,& r<\alpha \\
\frac{r-\alpha}{\beta-\alpha},& \alpha\le r\le\beta \\
1,& r>\beta
\end{cases}
$$

然后把频率改写成：

$$
\theta_d'=(1-\gamma(r(d)))\frac{\theta_d}{s}+\gamma(r(d))\theta_d
$$

这条式子的意义非常直接：

- $\gamma=0$ 时，完全走插值，$\theta_d'=\theta_d/s$
- $\gamma=1$ 时，完全保留原频率，$\theta_d'=\theta_d$
- 中间值时，两者线性混合

这就是 “NTK-by-parts”。NTK 是神经切线核，白话说它常被用来分析网络对不同频率信号的学习能力。这里不需要完整掌握 NTK 理论，只要理解一点：统一插值会让模型丢失高频表达，而分段插值能更接近模型训练时对不同频率的敏感性。

接着是温度修正。YaRN 在 attention logits 上引入：

$$
\text{Attention}(m,n)=\text{softmax}_n\left(\frac{q_m^\top k_n}{t\sqrt{D}}\right)
$$

其中 $t$ 是温度。温度的白话作用是控制 softmax 的尖锐程度。$t$ 变大，分布更平；$t$ 变小，分布更尖。长上下文扩展后，注意力熵会变化，如果不修正，就可能出现“只盯少数 token”或“过度平均化”两种退化。

论文经验拟合得到：

$$\frac{1}{\sqrt{t}}=0.1\ln(s)+1$$

例如 $s=16$ 时，

$$\frac{1}{\sqrt{t}}=0.1\ln(16)+1\approx1.277$$

因此

$$t\approx \frac{1}{1.277^2}\approx0.613$$

很多工程代码不会显式改 softmax，而是把 `1/sqrt(t)` 直接乘到 q、k 或旋转后的 RoPE embedding 上。这和改 logits 分母是等价的，只是更容易兼容 FlashAttention 之类的高性能实现。

---

## 代码实现

下面给一个最小可运行的 Python 版本，演示如何按频段修改 RoPE 频率，并计算温度缩放因子。这里用纯 `math`，不依赖第三方库。

```python
import math

def gamma_ramp(r: float, alpha: float = 1.0, beta: float = 32.0) -> float:
    if r < alpha:
        return 0.0
    if r > beta:
        return 1.0
    return (r - alpha) / (beta - alpha)

def yarn_theta(theta: float, r: float, s: float, alpha: float = 1.0, beta: float = 32.0) -> float:
    g = gamma_ramp(r, alpha, beta)
    return (1.0 - g) * (theta / s) + g * theta

def rope_wavelength(theta: float) -> float:
    return 2.0 * math.pi / theta

def rope_ratio(L: int, theta: float) -> float:
    return L / rope_wavelength(theta)

def yarn_mscale(s: float) -> float:
    # paper Eq.15: 1/sqrt(t) = 0.1 * ln(s) + 1
    if s <= 1:
        return 1.0
    return 0.1 * math.log(s) + 1.0

def yarn_temperature(s: float) -> float:
    mscale = yarn_mscale(s)
    return 1.0 / (mscale * mscale)

# 玩具例子：三个频段
L = 2048
s = 16

theta_low = 2 * math.pi / 8192    # 低频，波长长
theta_mid = 2 * math.pi / 256     # 中频
theta_high = 2 * math.pi / 32     # 高频，波长短

r_low = rope_ratio(L, theta_low)
r_mid = rope_ratio(L, theta_mid)
r_high = rope_ratio(L, theta_high)

new_low = yarn_theta(theta_low, r_low, s)
new_mid = yarn_theta(theta_mid, r_mid, s)
new_high = yarn_theta(theta_high, r_high, s)

assert r_low < 1.0
assert 1.0 <= r_mid <= 32.0
assert r_high > 32.0

assert abs(new_low - theta_low / s) < 1e-12
assert theta_mid / s < new_mid < theta_mid
assert abs(new_high - theta_high) < 1e-12

mscale = yarn_mscale(s)
temp = yarn_temperature(s)

assert 1.27 < mscale < 1.28
assert 0.61 < temp < 0.62

print("r_low, r_mid, r_high =", r_low, r_mid, r_high)
print("mscale =", mscale, "temperature =", temp)
```

这个实现对应真实工程里的最少侵入改法：

1. 在生成 `inv_freq` 或 `theta_d` 的地方，根据 $r(d)$ 改成 $\theta_d'$
2. 在 RoPE 应用前后，把旋转向量整体乘上 `mscale = 1/sqrt(t)`
3. attention 主干、KV cache、FlashAttention 路径尽量不动

玩具例子里，波长 8192 的维度在训练窗口 2048 内只转四分之一圈，所以被视为低频，直接压缩；波长 32 的维度会转很多圈，所以被保留。中间频段只做部分压缩。

真实工程例子来自论文设置：LLaMA 7B 从 2048 扩到 32K，等价于 $s=16$，训练使用 PG19 切成长段，global batch size 64，微调 400 steps。结论不是“完全不训练就白嫖 32K”，而是“只用非常少的长文本微调，就能把位置外推从脆弱状态拉回可用状态”。

---

## 工程权衡与常见坑

第一，最常见的错误是把 YaRN 当成“一个全局 rope scale 参数”。如果你只是把所有频率统一改成 $\theta/s$，你做的是 PI，不是 YaRN。这样高频会一起被压扁，短距离顺序感知下降，模型常见表现是代码缩进、局部引用、近邻 token 对齐变差。

第二，温度公式经常被抄错。很多文章把 $t$ 和 $1/\sqrt{t}$ 混成一个量。工程上应先明确你改的是哪一层：

- 若你改 softmax 分母，用的是 $t$
- 若你改 q/k 或旋转向量幅度，用的是 $1/\sqrt{t}$

这两种写法等价，但数值不是一回事。把它们混掉，效果会直接反过来。

第三，长上下文可用不等于所有任务都变强。位置编码只解决“模型能否在更长距离上保持相对位置感知”，不自动补足检索能力、长期记忆策略和数据分布。若训练长文本 mostly 是小说，部署却是仓库级代码问答，效果仍可能偏弱。

第四，Dynamic Scaling 和固定 scale 的目标不同。固定 scale 适合“模型明确要跑到某个上限，例如 32K”。Dynamic Scaling 则在推理时按当前长度动态调节 $s$，可以减轻“短输入反而退化”的问题，也能让超出训练扩展上限时的退化更平滑。

第五，评估要分短文和长文两类。只看 64K passkey retrieval 容易高估方法；只看短文 benchmark 又看不出扩窗价值。论文同时报告了长文 perplexity、Proof-pile、passkey retrieval 和标准 Open LLM benchmark，这才更接近真实结论。

---

## 替代方案与适用边界

常见替代方案可以简化成下表：

| 方法 | 核心思路 | 是否分频段 | 是否显式温度修正 | 适用边界 |
| --- | --- | --- | --- | --- |
| PI | 所有维度统一插值 | 否 | 否 | 简单，但大倍率扩展容易丢高频 |
| NTK-aware | 调整 base 或频率分布 | 部分 | 否 | 比 PI 好，但仍偏统一处理 |
| NTK-by-parts | 分段插值/外推 | 是 | 否 | 已经接近 YaRN 的频率部分 |
| YaRN | NTK-by-parts + 温度缩放 | 是 | 是 | 中大倍率扩展下更稳 |
| Dynamic Scaling | 推理时动态更新 scale | 可组合 | 可组合 | 适合无微调或柔性退化场景 |
| ALiBi | 在注意力分数加距离偏置 | 不适用 | 不适用 | 不是 RoPE 修补路线，改动点不同 |

YaRN 相对 ALiBi 的优势是：它不需要重写 attention 公式主体，主要改动集中在 RoPE 频率生成和缩放，兼容现有高性能 attention 内核更容易。相对 PI/NTK-aware，它的优势是把“不同频率承担不同职责”这件事显式写进了公式里。

但 YaRN 也有边界。它最适合“已有 RoPE 底座，想低成本扩几倍到几十倍上下文”的场景。若你从一开始就要为超长上下文设计新模型，或者目标是百万级上下文，YaRN 往往只是阶段性方案，需要配合更长训练、检索增强、分层记忆或更激进的位置外推方法。

---

## 参考资料

- [YaRN: Efficient Context Window Extension of Large Language Models, ICLR 2024 Abstract](https://proceedings.iclr.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html)
- [YaRN: Efficient Context Window Extension of Large Language Models, ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)
- [ICLR 2024 Poster / Slides for YaRN](https://iclr.cc/virtual/2024/poster/17499)
- [YaRN 官方代码仓库](https://github.com/jquesnelle/yarn)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409)
