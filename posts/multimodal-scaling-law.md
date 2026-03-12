## 核心结论

Scaling Law，直译是“扩展定律”，指模型损失如何随参数量和数据量变化的经验规律。对纯文本模型，常见写法是 $L(N, D)$：$N$ 是参数规模，$D$ 是训练 token 数。但在图文联合训练里，损失不能再写成“文本损失 + 图像损失”的简单相加，因为图像和文本会通过交叉注意力发生相互作用。

Aghajanyan 等人的核心结论可以压缩成一句话：多模态训练的有效计算量不是各模态 token 的算术和，而是由 $L(N,D_t,D_i)$ 这样的三变量幂律决定，其中既有协同增益，也有资源竞争。

| 项目 | 单模态 scaling law | 多模态 scaling law |
|---|---|---|
| 输入变量 | $N, D$ | $N, D_t, D_i$ |
| 默认假设 | 数据彼此独立 | 模态之间会相互影响 |
| 损失来源 | 模型容量不足、数据不足 | 单模态项 + 图文交叉项 |
| 是否存在协同收益 | 通常不单独建模 | 明确存在 |
| 是否存在竞争区 | 不突出 | 明确存在 competition barrier |

这里的交叉注意力，白话讲，就是模型在看文本时也能“看见”图像特征，反过来也一样。正因为有这个桥梁，图像 token 并不只是“另一种输入”，而是可能替代一部分文本 token 的语义负担。研究中常见的经验说法是：1 个图像 token 的信息价值，大约可折合为 0.3 个文本 token 的有效补充，换句话说，图像 token 与文本 token 之间存在非 1:1 的兑换关系。

玩具例子：如果一个任务需要理解“红色三角形在蓝色圆形左边”，纯文本需要把位置、颜色、形状都写出来；一张图像里这些关系是同时给出的。所以图像不是简单多加一份数据，而是把关系信息压缩进更少的上下文结构里。

---

## 问题定义与边界

先把变量说清楚：

- $N$：模型参数量。白话讲，就是模型能记住和拟合多少规律。
- $D_t$：文本 token 数。白话讲，就是喂给模型的文本训练量。
- $D_i$：图像 token 数。白话讲，就是图像经过编码器切分后得到的离散视觉表示数量。
- $L(N,D_t,D_i)$：联合训练的损失，通常可理解为模型“还犯多少错”。

如果把多模态损失粗暴写成
$$
L_{naive} = L(N,D_t) + L(N,D_i),
$$
会遗漏一个关键事实：文本和图像不是两支互不干扰的队伍，而是在共享同一个模型容量。模型小、数据少时，图像分支和文本分支会争抢参数表达能力，这就是 competition barrier，直译是“竞争屏障”，意思是系统还没大到足以释放协同收益之前，先出现了互相拖累。

可以把这个过程理解成下面的简化流程：

$$
(N, D_t, D_i) \rightarrow \text{联合训练结构} \rightarrow \text{单模态项 + 交叉项} \rightarrow L(N,D_t,D_i)
$$

其中交叉项决定你是在协同区，还是困在竞争区。

适用边界也要提前说清楚：

| 条件 | 是否适合使用多模态 scaling law |
|---|---|
| 任务只有文本输入输出 | 通常不需要 |
| 图文共用一个主干模型 | 需要 |
| 模态之间存在 cross-attention | 强烈建议使用 |
| 图像只是离线特征，不参与联合优化 | 作用有限 |
| 模型很小、数据很少 | 仍可用，但更可能落在竞争区 |

真实工程例子：做商品检索时，如果只把商品标题和商品图分别训练，再把分数相加，常会出现“标题拟合变好了，图片分支却拖累整体召回”的现象。问题不一定出在数据脏，而是因为你默认它们互不影响，实际上它们在抢同一套参数容量。

---

## 核心机制与推导

Aghajanyan 等人给出的双模态形式可以写成：

$$
L(N,D_t,D_i)=\frac{L(N,D_t)+L(N,D_i)}{2}-C_{t,i}+\frac{A_{t,i}}{N^{\alpha_{t,i}}}+\frac{B_{t,i}}{(|D_t|+|D_i|)^{\beta_{t,i}}}
$$

这条式子可以拆成四部分理解。

| 项 | 含义 | 白话解释 |
|---|---|---|
| $\frac{L(N,D_t)+L(N,D_i)}{2}$ | 单模态基线 | 先把两边各自能做到的水平取平均 |
| $-C_{t,i}$ | 协同上限 | 图文互补最多能白送你多少性能 |
| $\frac{A_{t,i}}{N^{\alpha_{t,i}}}$ | 参数驱动误差 | 模型太小，吃不下跨模态关系 |
| $\frac{B_{t,i}}{(|D_t|+|D_i|)^{\beta_{t,i}}}$ | 数据驱动误差 | 总数据不够，协同规律学不稳 |

这里的 $C_{t,i}$ 最重要。它表示当模型和数据都足够时，图文交互最多能把损失再压下去多少。可以把它理解成“synergy ceiling”，也就是协同收益上限。

最小数值例子如下。设：

- $L(N,D_t)=1.0$
- $L(N,D_i)=1.6$
- $C_{t,i}=0.12$
- $A_{t,i}=0.02$
- $B_{t,i}=0.01$
- $\alpha_{t,i}=0.5$
- $\beta_{t,i}=0.4$
- $N=10^9$
- $|D_t|+|D_i|=5\times 10^{10}$

代入后：

$$
\frac{1.0+1.6}{2}=1.3
$$

$$
\frac{A_{t,i}}{N^{\alpha}}=\frac{0.02}{(10^9)^{0.5}} \approx \frac{0.02}{31622.8}\approx 6.3\times 10^{-7}
$$

$$
\frac{B_{t,i}}{(|D_t|+|D_i|)^\beta}=\frac{0.01}{(5\times10^{10})^{0.4}} \approx 5.3\times10^{-7}
$$

所以总损失约为：

$$
L \approx 1.3 - 0.12 + 6.3\times10^{-7}+5.3\times10^{-7}\approx 1.18000116
$$

这说明什么？说明哪怕模型项和数据项几乎已经很小，仅仅一个 $C_{t,i}=0.12$ 的协同项，就能显著把联合损失从 1.3 拉到 1.18 左右。这个 0.12 可以理解成“免费 token”带来的收益，不是真的送你 token，而是模态之间的信息复用降低了本来需要更多数据才能学到的误差。

看趋势时，可以用一个简化图示理解：

| 区域 | 现象 |
|---|---|
| 小 $N$、小数据 | 竞争主导，模态互相拖累 |
| 中等 $N$、中等数据 | 开始跨过 barrier，协同项可见 |
| 大 $N$、大数据 | 更接近 $-C_{t,i}$ 所代表的上限收益 |

因此，多模态 scaling law 的关键不是“多一个模态”，而是“是否能让交叉项从负担变成收益”。

---

## 代码实现

工程里不需要直接复现论文全部拟合流程，但需要把单模态项和交叉项拆开监控。下面给一个可运行的 Python 玩具实现：

```python
from math import pow

def compute_multimodal_loss(
    N,
    loss_text,
    loss_image,
    Dt,
    Di,
    C,
    A,
    B,
    alpha,
    beta,
):
    base = 0.5 * (loss_text + loss_image)
    model_term = A / pow(N, alpha)
    data_term = B / pow(Dt + Di, beta)
    total = base - C + model_term + data_term
    components = {
        "base": base,
        "synergy": -C,
        "model_term": model_term,
        "data_term": data_term,
        "total": total,
    }
    return total, components

loss, comp = compute_multimodal_loss(
    N=1e9,
    loss_text=1.0,
    loss_image=1.6,
    Dt=3e10,
    Di=2e10,
    C=0.12,
    A=0.02,
    B=0.01,
    alpha=0.5,
    beta=0.4,
)

assert comp["base"] == 1.3
assert 1.17 < loss < 1.19
assert comp["synergy"] == -0.12

def suggest_image_ratio(image_tokens, text_tokens):
    ratio = image_tokens / text_tokens
    if ratio > 0.6:
        return "reduce_img_tokens"
    if ratio < 0.4:
        return "increase_img_tokens"
    return "keep_ratio"

assert suggest_image_ratio(500, 1000) == "keep_ratio"
assert suggest_image_ratio(800, 1000) == "reduce_img_tokens"
```

如果把它放进真实训练循环，最关键的是日志拆分，而不是只盯总 loss：

```python
def log_components(step, components):
    print(
        f"step={step} "
        f"base={components['base']:.4f} "
        f"synergy={components['synergy']:.4f} "
        f"model={components['model_term']:.6f} "
        f"data={components['data_term']:.6f} "
        f"total={components['total']:.4f}"
    )
```

参数含义和调节方向如下：

| 参数 | 含义 | 过小时的现象 | 调节思路 |
|---|---|---|---|
| $C$ | 协同收益上限 | 图文互补收益看不见 | 提升跨模态对齐质量 |
| $A$ | 模型不足惩罚 | 小模型误差高 | 扩大主干或提升共享层表达 |
| $B$ | 数据不足惩罚 | 数据不够时 loss 降不动 | 增加高质量图文对 |
| $\alpha$ | 参数扩展效率 | 扩模型收益慢 | 检查架构是否限制容量利用 |
| $\beta$ | 数据扩展效率 | 加数据收益慢 | 检查数据重复和噪声 |

真实工程例子：做 OCR 文档问答时，可以分别记录文字识别误差、图像布局误差、图文联合问答误差。如果总 loss 下降但联合问答指标不升，很可能不是学习率问题，而是交叉项没学起来，模型仍在 competition barrier 里。

---

## 工程权衡与常见坑

多模态训练最大的误区，是把“多一种输入”误认为“多一份信息”。实际上它同时也多了一份资源竞争。模型参数、KV cache、训练吞吐、显存预算，都会因为第二模态进入而重排。

常见坑可以直接列出来：

| 坑 | 典型表现 | 后果 | 规避策略 |
|---|---|---|---|
| 忽略 cross-term | 只看总 loss，拆不出交叉收益 | 误判训练有效，实际卡在竞争区 | 单独记录跨模态项 |
| 固定模态比 | 全程图像 token 比例不变 | 某一阶段过量视觉输入拖慢优化 | 做 ratio scheduler |
| 只追求更多图像 token | 图像越多越贵 | 账单涨、收益不稳定 | 利用压缩和缓存 |
| 把图片当免费上下文 | 忽略编码成本 | 推理成本失控 | 评估端到端成本 |
| 数据配对质量差 | 图文语义错位 | $C_{t,i}$ 被噪声吃掉 | 优先清洗对齐数据 |

一个常见经验是把图像 token 比例控制在文本 token 的 40% 到 60% 区间附近，而不是无上限增加。原因并不神秘：过低时协同不够，过高时视觉分支开始占用过多容量，文本建模反而退化。

可以用一个极简调度器表达这个思路：

```python
def adjust_ratio(loss_cross, image_tokens, text_tokens, threshold=0.05):
    ratio = image_tokens / text_tokens
    if loss_cross > threshold or ratio > 0.6:
        return "reduce_img_tokens()"
    if ratio < 0.4:
        return "increase_img_tokens()"
    return "keep_current_ratio()"
```

真实工程例子：在法律合同抽取里，把 5000 字合同渲染成高分辨率页面图片，再交给视觉入口处理，账单可能明显低于直接走长文本入口。这件事说明“令牌经济”不能只看 token 数，还要看不同模态编码后的价格结构。但这个策略只在任务目标仍能保留关键信息时成立；如果任务高度依赖逐字精确编辑，图像路径未必合适。

---

## 替代方案与适用边界

不是所有场景都必须上多模态 scaling law。判断标准很简单：模态之间到底有没有强交互。

| 方案 | 核心思想 | 适用场景 | 局限 |
|---|---|---|---|
| 真正的 multi-modal scaling | 显式建模 $L(N,D_t,D_i)$ 与交叉项 | 图文问答、视觉指令、文档理解 | 建模和监控更复杂 |
| pseudo multi-modal | 把图像先转文字，再按纯文本训练 | OCR 后摘要、结构化抽取 | 丢失空间关系和视觉细节 |
| token compression | 压缩视觉 token 数 | 图像分辨率高、成本敏感 | 可能损失细粒度信息 |
| 专用 cross-attention mask | 控制哪些层跨模态交互 | 交互稀疏、显存紧张 | 设计不当会压掉协同 |
| 单模态 scaling law | 只建模一个模态 | 纯文本、纯图像任务 | 无法解释协同与竞争 |

一个简单判断流程可以写成：

$$
\text{若 } \Delta = |L_t - L_{t+i}| < \varepsilon,\ \text{且视觉信息只起弱辅助作用，则继续用单模态近似}
$$

$$
\text{若图文交互显著改变任务结果，则必须使用多模态逻辑}
$$

玩具例子：纯文本摘要任务，输入输出都只有文字，哪怕数据来源原本包含图片，也可以继续沿用单模态 scaling law。因为图片没有进入最终推理链路。

真实工程例子：电商商品理解里，标题“夏季连衣裙”无法告诉你袖长、材质纹理、图案布局，这些信息来自图片，而且会直接影响分类和检索结果。这时如果继续用单模态近似，就会系统性低估跨模态收益。

所以边界很清楚：当另一模态只是在外围辅助预处理时，可以继续用老公式；当另一模态进入主模型、参与联合优化并改变最终输出时，就必须把交叉项写进系统理解中。

---

## 参考资料

| 来源 | 主要内容 | 章节关联 |
|---|---|---|
| Aghajanyan et al., 2023, *Scaling Laws for Generative Mixed-Modal Language Models* | 给出多模态 loss 的核心拟合形式与图文协同/竞争分析 | 核心结论、机制推导、工程坑 |
| Scaling Law Survey 中对 Equation 27 的整理 | 用更统一的视角概括三变量幂律写法 | 问题定义、机制推导 |
| Visual Context Scaling 相关综述与工程案例 | 讨论视觉 token 的信息密度、压缩与成本 | 核心结论、工程权衡、替代方案 |

1. Aghajanyan, A. et al. *Scaling Laws for Generative Mixed-Modal Language Models*. 2023.
核心贡献：提出图文联合训练不应视为单模态 loss 的线性和，并给出包含协同项与竞争项的拟合形式。

2. *How to Upscale Neural Networks with Scaling Law: A Survey and Practical Guidelines*.
核心贡献：从更高层次总结多变量 scaling law，把双模态关系放回统一的扩展定律框架中。

3. Visual Context Scaling 与跨模态 token 经济相关资料。
核心贡献：把“图像 token 是否值得”这个问题落到上下文压缩、吞吐和成本上，补上论文之外的部署视角。
