## 核心结论

FFN 的第一层扩展比，指的是 $r = d_{ff} / d_{\text{model}}$，也就是“中间层宽度相对输入宽度放大了多少倍”。白话说，模型会先把每个 token 的表示临时拉宽，在更大的空间里做一次非线性变换，再压回原始宽度。

主流 Transformer 把这个比率设成 4×，不是因为 4 这个数字有理论最优证明，而是因为它长期在“表达能力、参数量、算力成本”三者之间形成了稳定折中。对标准两层 FFN 而言，参数量近似满足：

$$
\text{Params}_{\text{FFN}} \approx 2 d_{\text{model}} d_{ff}
= 2 r d_{\text{model}}^2
$$

这说明扩展比每增加一倍，FFN 参数和 FLOPs 也几乎跟着增加一倍。相比改层数或改注意力头数，扩展比是一个非常直接的“容量旋钮”。

一个最简单的玩具例子是：若 $d_{\text{model}}=1024$，那么 2×、4×、8× 分别对应 $d_{ff}=2048,4096,8192$。主权重参数量分别约为 419 万、839 万、1678 万。4× 往往能明显强于 2×，而 8× 相比 4× 常常只带来有限收益，却把 FFN 成本再翻一倍。

真实工程里，LLaMA 一类模型经常不用标准 ReLU/GELU FFN，而改用 SwiGLU。SwiGLU 是一种门控前馈结构，白话说就是“让一条分支学内容，另一条分支学开关”。这时常见设计不是 4×，而是约 $\frac{8}{3}\times$，目的是在引入额外门控分支后，尽量把总参数预算维持在接近标准 4× FFN 的水平。

---

## 问题定义与边界

这里讨论的对象，是 Transformer block 中 attention 后面的前馈网络 FFN。标准形式可以写成：

$$
\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

其中：

- $W_1: d_{\text{model}} \rightarrow d_{ff}$
- $W_2: d_{ff} \rightarrow d_{\text{model}}$

$d_{\text{model}}$ 是模型主干宽度，也就是 token 表示的基础维度；$d_{ff}$ 是 FFN 中间层宽度；扩展比就是两者之比。

本文只讨论“FFN 第一层扩展比如何设计”，不讨论以下问题：

| 范围内 | 不在范围内 |
|---|---|
| 标准 FFN 与门控 FFN 的宽度设计 | 注意力头数如何选择 |
| 扩展比对参数、算力、效果的影响 | 训练数据质量对最优扩展比的影响 |
| 2×/4×/8× 与 $8/3\times$ 的工程折中 | MoE 路由策略的细节 |

边界很重要，因为很多初学者会把“模型更强”直接归因于扩展比更大。实际不是这样。扩展比只是 FFN 的一个局部宽度参数，它只决定“在非线性子层里临时展开到多宽”。如果模型整体太浅、训练 token 不够、优化器设置不对，仅靠把 4× 改成 8× 往往得不到成比例提升。

再看完整参数公式：

$$
\text{Params}_{\text{FFN}} = d_{\text{model}}d_{ff} + d_{ff}d_{\text{model}} + d_{ff} + d_{\text{model}}
= 2d_{\text{model}}d_{ff} + d_{ff} + d_{\text{model}}
$$

当 $d_{ff} \gg d_{\text{model}}$ 时，偏置项几乎可以忽略，于是有：

$$
\text{Params}_{\text{FFN}} \approx 2d_{\text{model}}d_{ff}
$$

这就是为什么工程上讨论扩展比时，大家首先想到的是“它几乎线性决定 FFN 成本”。

---

## 核心机制与推导

FFN 的作用不是做 token 间信息交换，那是 attention 的任务；FFN 做的是“对每个 token 单独进行特征重组”。白话说，attention 负责“看别人”，FFN 负责“改自己”。

它的核心机制可以拆成三步：

1. 先投影到更高维：$x \rightarrow W_1 x$
2. 在高维空间施加非线性：$\sigma(\cdot)$
3. 再压回主干维度：$W_2$

可以把它理解成一个“先展开，再筛选，再压缩”的过程。高维展开提供更多中间通道，让不同特征组合有机会被分离和激活。

一个玩具例子：

假设输入向量只有 2 维，表示一个 token 的两个粗糙特征：

$$
x = [\text{语法强度}, \text{语义强度}]
$$

如果直接在 2 维里做很少的线性变换，模型能表达的组合非常有限。若先扩展到 8 维，中间层就可以出现诸如“名词但带数字”“动词且位于句首”“和上一层激活共同出现”这类组合方向。虽然这些维度不是人工定义的，但从函数表达能力看，宽一些的中间层更容易容纳这类组合特征。

为什么 4× 常见，而不是固定 2× 或 8×？因为 FFN 不是越宽越好。若设扩展比为 $r$，则：

$$
d_{ff} = r d_{\text{model}}
$$

代回参数近似式：

$$
\text{Params}_{\text{FFN}} \approx 2r d_{\text{model}}^2
$$

所以：

- 2× 到 4×：参数翻倍，常常能换来明显收益
- 4× 到 8×：参数再翻倍，但收益通常不再同样明显

以 $d_{\text{model}}=1024$ 为例：

| 扩展比 | $d_{ff}$ | 主权重参数量 $2d_{\text{model}}d_{ff}$ | 相对 4× |
|---|---:|---:|---:|
| 2× | 2048 | 4,194,304 | 0.5x |
| 4× | 4096 | 8,388,608 | 1.0x |
| 8× | 8192 | 16,777,216 | 2.0x |

这里能看出一个工程事实：扩展比是线性放大，但效果不是线性增长。也就是说，容量增加是确定的，收益增加是不确定且递减的。

对门控变体也可以做类似推导。以 SwiGLU 为例，其常见写法是：

$$
\text{SwiGLU}(x) = W_2(\text{SiLU}(W_g x) \odot W_v x)
$$

这里 $\odot$ 表示逐元素乘法。因为前面有两条投影分支 $W_g, W_v$，参数开销比标准 FFN 更大。如果仍用 4× 宽度，总参数会明显上涨。于是很多实现会把中间维度改成：

$$
d_{ff} \approx \frac{8}{3} d_{\text{model}}
$$

原因很直接：标准 FFN 约有 $2d_{\text{model}}(4d_{\text{model}})=8d_{\text{model}}^2$ 个主权重参数；SwiGLU 有三块主权重，约是 $3d_{\text{model}}d_{ff}$。令两者近似相等：

$$
3d_{\text{model}}d_{ff} \approx 8d_{\text{model}}^2
\Rightarrow d_{ff} \approx \frac{8}{3}d_{\text{model}}
$$

这不是神秘常数，而是“参数预算守恒”推出来的。

---

## 代码实现

实现上最重要的不是某个具体库，而是把扩展比显式暴露成超参数。下面是一个可运行的 Python 示例，演示标准 FFN 和参数量计算：

```python
import math

def ffn_param_count(d_model: int, expansion: float, bias: bool = True) -> int:
    d_ff = int(d_model * expansion)
    params = d_model * d_ff + d_ff * d_model
    if bias:
        params += d_ff + d_model
    return params

def swiglu_param_count(d_model: int, expansion: float, bias: bool = False) -> int:
    d_ff = int(d_model * expansion)
    # gate proj + value proj + output proj
    params = d_model * d_ff + d_model * d_ff + d_ff * d_model
    if bias:
        params += d_ff + d_ff + d_model
    return params

assert ffn_param_count(1024, 4, bias=False) == 2 * 1024 * 4096
assert ffn_param_count(1024, 2, bias=False) * 2 == ffn_param_count(1024, 4, bias=False)
assert swiglu_param_count(1024, 8/3, bias=False) < swiglu_param_count(1024, 4, bias=False)
assert abs(swiglu_param_count(1024, 8/3, bias=False) - ffn_param_count(1024, 4, bias=False)) < 5000

print("standard 4x FFN:", ffn_param_count(1024, 4, bias=False))
print("SwiGLU 8/3x:", swiglu_param_count(1024, 8/3, bias=False))
```

如果写成接近框架代码的伪实现，结构通常是这样：

```python
def ffn(x, d_model, expansion=4):
    d_ff = int(d_model * expansion)
    h = linear(x, d_model, d_ff)
    h = activation(h)
    y = linear(h, d_ff, d_model)
    return y
```

这里的 `expansion` 就是扩展比。它本质上控制“中间层有多宽”。

若换成 SwiGLU，常见形式类似：

```python
def swiglu_ffn(x, d_model, expansion=8/3):
    d_ff = int(d_model * expansion)
    gate = silu(linear(x, d_model, d_ff))
    value = linear(x, d_model, d_ff)
    h = gate * value
    y = linear(h, d_ff, d_model)
    return y
```

真实工程例子是 LLaMA 系列：它并不是简单继承“标准 FFN 用 4×”这一规则，而是因为用了门控结构，转而选取接近 $\frac{8}{3}\times$ 的中间宽度。比如当 $d_{\text{model}}=8192$ 时，常见配置会把 $d_{ff}$ 设到 28672，接近 $8192 \times \frac{8}{3}$。这反映的不是“更宽一定更强”，而是“结构改了，宽度也要按参数预算重算”。

再对比不同结构的预算关系：

| 结构 | 主权重近似 | 若要接近标准 4× 参数预算，推荐中间宽度 |
|---|---|---|
| 标准 FFN | $2d_{\text{model}}d_{ff}$ | $4d_{\text{model}}$ |
| GLU / SwiGLU | $3d_{\text{model}}d_{ff}$ | $\frac{8}{3}d_{\text{model}}$ |

---

## 工程权衡与常见坑

工程上最常见的误区，是把扩展比当成“免费涨点按钮”。它不是。它的收益和代价都很直接。

先看经验对比：

| 扩展比 | 参数成本 | 常见效果趋势 | 主要问题 |
|---|---|---|---|
| 2× | 低 | 往往明显弱于 4× 基线 | 容量不足，FFN 表达偏紧 |
| 4× | 中 | 通常是稳定默认值 | 无明显结构性问题 |
| 6× | 较高 | 可能略有提升，但不稳定 | 成本上涨快，收益开始递减 |
| 8× | 高 | 常见为小幅提升或接近持平 | 显存、吞吐、并行效率恶化 |

第一个坑是只看参数，不看吞吐。FFN 不只是参数占用，它还直接增加矩阵乘的计算量与激活缓存。训练时会影响显存；推理时会影响延迟和吞吐。

第二个坑是忽视边际递减。把 4× 改成 8×，FFN 成本几乎翻倍，但验证集收益通常远小于“从 2× 升到 4×”那次跃迁。这说明在很多规模区间里，4× 已经覆盖了主要收益区。

第三个坑是错误比较不同结构。标准 FFN 的 4×，和 SwiGLU 的 $\frac{8}{3}\times$，不能只看倍数大小。因为两者前面的投影条数不同，参数公式不同。直接说“$\frac{8}{3}$ 比 4 更小，所以更省”是片面的，必须在相同结构预算下比较。

第四个坑是脱离整体架构单独调 FFN。若模型已经是“宽而浅”，继续增大扩展比可能不如增加层数有效；若模型已经很深但单层表达不足，适度提高扩展比才更有价值。

一个真实工程判断方式是：假设某层 $d_{\text{model}}=1024$，标准 4× FFN 主权重约 839 万；8× 会到 1678 万。对几十层堆叠模型来说，这不是“小改动”，而是会在总参数、激活、训练吞吐上产生连锁影响。若实验只带来极小验证收益，这个改动通常不值。

---

## 替代方案与适用边界

如果 4× 不够，替代方案不只有“继续拉宽”。

第一种替代方案是增加层数。层数的白话意思是“多做几次逐层变换”。在很多场景下，增加深度比增加单层 FFN 宽度更能提升层级抽象能力。比如把扩展比保持 4×，但从 24 层加到 32 层，常常比把每层从 4× 拉到 8× 更均衡。

第二种替代方案是增加 $d_{\text{model}}$。这会同时增大注意力子层和 FFN 子层的基础表示能力。代价也高，但它提升的是主干宽度，而不是只给 FFN 加料。

第三种替代方案是改结构而不是硬拉宽。SwiGLU、GEGLU 等门控结构的核心价值，就是在相近参数预算内提高有效容量。所谓“有效容量”，白话说就是“不是单纯多参数，而是更会用参数”。

对比可以概括为：

| 方案 | 提升方式 | 适合场景 | 风险 |
|---|---|---|---|
| 扩展比从 4× 提到 6×/8× | 增大单层 FFN 宽度 | 单层表达明显不足，且算力充足 | 成本线性上升，收益递减 |
| 增加层数 | 增强层级组合能力 | 任务依赖更深的表示变换 | 训练更慢，优化更难 |
| 增大 $d_{\text{model}}$ | 提升整体主干宽度 | 需要全面提升模型容量 | 参数和注意力成本同步上升 |
| 改为 SwiGLU 等 | 提升参数利用效率 | 追求更强 FFN 表达但需控预算 | 实现复杂度更高 |

适用边界也要明确：

- 小模型或资源紧张场景，2× 到 4× 的差距往往大于 4× 到 8× 的差距，因此优先保证至少 4× 是常见做法。
- 中大模型里，若已采用门控 FFN，则优先按预算推导 $\frac{8}{3}\times$ 或类似设置，而不是机械继承标准 4×。
- 超大模型和 MoE 场景下，FFN 宽度问题会和路由、专家数、激活 sparsity 一起耦合，这时“固定 4×”就不再是完整答案。

简化地说：标准 dense Transformer 里，4× 是一个稳健默认值；门控结构里，$\frac{8}{3}\times$ 是预算对齐后的常见默认值；当你想突破默认值时，必须把它放回“总参数、总 FLOPs、吞吐、显存、训练曲线”一起看。

---

## 参考资料

| 主题 | 来源 | 可复核内容 |
|---|---|---|
| FFN 结构与参数推导 | [Feed-Forward Networks in Transformers](https://mbrenndoerfer.com/writing/transformer-feed-forward-networks?utm_source=openai) | 标准 FFN 两层结构、参数近似为 $2d_{\text{model}}d_{ff}$ |
| FFN 扩展比直观解释 | [Feed Forward Networks in Transformers](https://emberverse.ai/piece/feed_forward_networks_in_transformers?utm_source=openai) | 扩展比作为“先放大再压缩”的容量控制 |
| 2×/4×/8× 的经验比较 | [Advanced FFN Ablation / Emberverse 相关文章](https://emberverse.ai/piece/feed_forward_networks_in_transformers?utm_source=openai) | 宽度增大带来收益递减的经验结论 |
| LLaMA 与现代 Transformer 设计 | [LLMs from Scratch: Modern Transformer Architectures](https://datahacker.rs/llms-from-scratch-003-modern-transformer-architectures-a-deep-dive-into-design-principles-and-training/?utm_source=openai) | SwiGLU、$8/3\times$ 宽度与现代结构取舍 |
| 工程经验补充 | [CSDN: FFN 扩展比经验讨论](https://blog.csdn.net/weixin_30700095/article/details/152473050?utm_source=openai) | 2×/4×/6×/8× 的工程权衡与常见坑 |
