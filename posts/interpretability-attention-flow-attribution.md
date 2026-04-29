## 核心结论

`注意力流`和 `attribution` 都在回答“模型到底依赖了哪些输入”，但它们不是同一种解释。

`注意力流`可以先理解成“信息路由图”，也就是看 Transformer 内部把哪些 token 的信息沿哪些层、哪些路径传到了目标位置。它更像在回答：信息是怎么走过去的。

`Attribution` 可以先理解成“贡献分数”，也就是看某个输入一旦变化，输出分数会变化多少。它更像在回答：这个输入到底把结果推高了多少，或者拉低了多少。

这两个问题本身就不同，所以结果不一致不是异常，而是正常现象。一个常见误解是：某个 token 的 attention 权重很高，就说明它对最终输出贡献最大。这个结论并不成立。权重高，表示模型在某一层更关注它；贡献大，表示它真的改变了输出值。两者之间隔着残差连接、MLP、层间混合、多头聚合，以及输出头本身的非线性映射。

先看一个总览表：

| 维度 | 注意力流 / rollout | Attribution / IG |
|---|---|---|
| 定义 | 把各层 attention 当作传播矩阵，累计信息传递强度 | 计算输入变化对输出变化的贡献 |
| 回答的问题 | 信息如何在结构里传播 | 哪些输入真正改变了输出 |
| 输出对象 | token 到 token 或 token 到 `[CLS]` 的路径强度 | 每个输入 token 的贡献值 |
| 优点 | 直观看路由，适合分析层间传递 | 更接近“贡献”，能比较正负影响 |
| 局限 | 不直接等于因果贡献 | 依赖 baseline，可能受梯度问题影响 |

玩具例子可以直接说明“权重高”和“贡献大”不是一回事。假设模型输出是

$$
F(x_1, x_2) = 0.9x_1 + 0.1x_2
$$

如果输入是 $x=(1, 10)$，从“结构权重”看，$x_1$ 的系数更大；但从贡献看：

$$
0.9 \times 1 = 0.9,\quad 0.1 \times 10 = 1.0
$$

最终反而是 $x_2$ 对输出贡献更大。这个例子很小，但已经说明：传播权重和输出贡献不是同一个量。

---

## 问题定义与边界

本文比较的不是“所有可解释性方法”，而是两类非常常见、但经常被混用的方法：

1. `attention flow / attention rollout`
2. `gradient-based attribution / Integrated Gradients`

边界先说清楚。`attention flow` 研究的是 Transformer 结构内部的信息传递近似；`Integrated Gradients` 研究的是从某个 baseline 到真实输入这条路径上，输入维度对输出的累计贡献。前者偏结构，后者偏敏感度。

用文本分类举例。句子是 `this movie is not good`，模型预测为负面。

- 如果你用 `注意力流`，你更关心的是：`not` 的信息有没有通过若干层 attention 路由传到 `[CLS]`，再进入分类头。
- 如果你用 `IG`，你更关心的是：`not` 这个 token 是否真的把“负面”分数推高了，或者把“正面”分数压低了。

这两个问题看起来接近，但并不等价。模型可能在结构上大量“看”某个 token，却不一定让它显著改变最终分数；也可能某个 token 在最后输出上很关键，但它在单层 attention 图里并不显眼。

边界表如下：

| 维度 | 注意力流能回答什么 | 不能回答什么 | 适合模型 | 不适合直接使用的场景 |
|---|---|---|---|---|
| 结构传播 | 哪些 token 的表示更容易传到目标位置 | 某 token 是否真正导致输出变化 | Transformer 类模型 | 需要严格因果结论时 |
| 输出贡献 | 不擅长 | 哪些 token 拉高或拉低目标分数 | 可求梯度模型 | 离散输入且 baseline 难定义时 |
| 多层整合 | 能看层间累计路径 | 不能覆盖 MLP 的全部作用 | 有显式 attention 的架构 | 非 attention 架构 |

结论是：结果不一致，先不要急着判断哪一个错。很多时候只是“问题没对齐”。

---

## 核心机制与推导

### 1. 注意力流为什么是“传播视角”

先把 attention 矩阵理解成“这一层里谁把信息分给谁”。第 $l$ 层的 attention 记作 $A^{(l)} \in \mathbb{R}^{n \times n}$。为了考虑残差连接，常见近似是把单位矩阵 $I$ 加进去，再做行归一化：

$$
\hat{A}^{(l)} = \text{RowNorm}(A^{(l)} + I)
$$

这里的白话解释是：每个 token 不只是看别人，也保留自己原来的信息。

如果有 $L$ 层，那么 rollout 的累计传播可以写成：

$$
R^{(L)} = \hat{A}^{(L)} \hat{A}^{(L-1)} \cdots \hat{A}^{(1)}
$$

其中 $R^{(L)}_{i,j}$ 表示：第 $j$ 个输入 token 的信息，经过多层传播后，对第 $i$ 个高层表示的累计传递强度。

一个 2 层、3 token 的玩具例子：

- token 顺序：`[CLS]`, `not`, `good`
- 第一层里 `[CLS]` 主要看 `not`
- 第二层里 `[CLS]` 又看了上一层中更依赖 `good` 的表示

这时如果只看第二层 raw attention，可能觉得 `[CLS]` 主要依赖 `good`；但 rollout 会把两层路径连起来，得到“`not -> good -> [CLS]`”这样的累计传播关系。这就是它的价值：它不是单层快照，而是层间连乘后的近似路径强度。

### 2. Attribution 为什么是“敏感度视角”

Integrated Gradients 的核心思想是：别只看某一点的梯度，而是从一个参考输入 $x'$ 走到真实输入 $x$，沿这条路径把梯度积分起来。公式是：

$$
IG_i(x; x') = (x_i - x'_i)\int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha
$$

白话解释是：第 $i$ 个输入维度从“参考状态”移动到“真实状态”时，一路上对输出造成了多少累计影响。

它有一个很重要的性质：

$$
\sum_i IG_i = F(x) - F(x')
$$

这叫完整性。意思是：所有输入贡献加起来，应当等于真实输入和 baseline 的输出差值。这是很多简单梯度方法不具备的。

### 3. 同一个输入，为何两种方法可能冲突

还是看前面的线性例子。attention 侧可能给出“token1 更重要”，因为它在传播结构里的权重是 0.9；但 IG 会得到“token2 贡献更大”，因为它的输入值本身更大。

机制对比表如下：

| 维度 | 注意力流 | IG |
|---|---|---|
| 传播对象 | 层间表示与路径强度 | 输出对输入的累计敏感度 |
| 是否依赖 baseline | 否 | 是 |
| 是否受梯度饱和影响 | 不直接受影响 | 会受影响，但比单点梯度稳 |
| 是否保留路径信息 | 保留层间传播路径 | 不强调结构路径 |
| 是否直接回答贡献 | 不直接 | 更直接 |

因此可以把两者理解为：

- `注意力流`：谁把信息递给了谁。
- `IG`：谁真正改变了结果。

---

## 代码实现

下面给最小可运行实现。为了让原理清晰，代码不用真实大模型，而用简化张量演示。

### 1. Attention Rollout 最小实现

```python
import numpy as np

def row_norm(x):
    s = x.sum(axis=-1, keepdims=True)
    return x / s

def attention_rollout(attn_layers):
    """
    attn_layers: List[np.ndarray], each shape [n, n]
    return: rollout matrix R, shape [n, n]
    """
    n = attn_layers[0].shape[0]
    R = np.eye(n)
    for A in attn_layers:
        A_hat = row_norm(A + np.eye(n))  # residual + normalization
        R = A_hat @ R
    return R

# toy example: [CLS], not, good
A1 = np.array([
    [0.1, 0.8, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
], dtype=float)

A2 = np.array([
    [0.1, 0.2, 0.7],
    [0.3, 0.5, 0.2],
    [0.2, 0.2, 0.6],
], dtype=float)

R = attention_rollout([A1, A2])

# 看最终 [CLS]（索引0）累计依赖哪些输入 token
cls_flow = R[0]
assert np.isclose(cls_flow.sum(), 1.0)
assert cls_flow.shape == (3,)
assert cls_flow[2] > cls_flow[1]  # 在这个构造里，good 的累计传播更强
print(cls_flow)
```

这个函数的输入就是每层 attention 矩阵，输出是累计传播矩阵。实际工程里通常要做三件事：

1. 从每层多头 attention 里先做 head 聚合，比如均值或加权均值。
2. 把残差加进去，否则会低估 token 保留自身信息的能力。
3. 最后取目标位置，例如 `[CLS]` 对所有输入 token 的累计分数，并排序可视化。

### 2. Integrated Gradients 最小实现

下面用一个可微的线性函数模拟输出，重点展示 baseline、插值采样和累积梯度。

```python
import numpy as np

def f(x):
    # mock model output
    return 0.9 * x[0] + 0.1 * x[1]

def grad_f(x):
    # gradient of f wrt x
    return np.array([0.9, 0.1], dtype=float)

def integrated_gradients(x, x_baseline, steps=100):
    total_grad = np.zeros_like(x, dtype=float)
    for k in range(1, steps + 1):
        alpha = k / steps
        x_interp = x_baseline + alpha * (x - x_baseline)
        total_grad += grad_f(x_interp)
    avg_grad = total_grad / steps
    return (x - x_baseline) * avg_grad

x = np.array([1.0, 10.0])
x0 = np.array([0.0, 0.0])
ig = integrated_gradients(x, x0)

assert np.allclose(ig, np.array([0.9, 1.0]), atol=1e-6)
assert np.isclose(ig.sum(), f(x) - f(x0), atol=1e-6)
print(ig)
```

真实工程里，文本输入不是直接对 token id 求梯度，而是对 embedding 求梯度，再把 embedding 维度上的 attribution 聚合回 token 级别。常见做法是对每个 token 的 embedding attribution 取范数或求和，然后画成 token 高亮图。

### 3. 结果可视化怎么读

一个合理的可视化应该同时给出三类信息：

- `rollout 热力图`：看 `[CLS]` 最终接收了哪些 token 的传播。
- `IG token 高亮`：看哪些 token 真正推高或压低目标分数。
- `ablation 对照`：删掉某个 token 后，输出分数掉了多少。

真实工程例子：客服工单分类里，句子是 `customer cannot login after password reset`。

- rollout 可能显示 `cannot` 和 `login` 被稳定传播到分类头。
- IG 可能显示真正把“高优先级故障”分数推高的是 `cannot`、`password reset`。
- 如果删掉 `login` 分数变化很小，说明它被模型“看到”了，但不是主要贡献者。

---

## 工程权衡与常见坑

工程里最危险的不是“方法不够高级”，而是把一种解释当成另一种解释。

坑位表如下：

| 问题 | 现象 | 后果 | 规避方式 |
|---|---|---|---|
| 只看单层 attention | 某层某 token 权重很高 | 误判为最终依赖 | 至少做 rollout 或 flow |
| baseline 选得不合理 | IG 波动很大 | 结论不稳定 | 选零向量、PAD、均值 embedding 并做对照 |
| 梯度饱和 | 单点梯度接近 0 | 低估真实贡献 | 用 IG 替代原始梯度 |
| 子词切分碎片化 | 一个词被拆成多个子词 | 解释难读 | 按词级聚合 attribution |
| 忽略 residual / MLP | attention 图很漂亮 | 误把局部路由当完整机制 | 结合 ablation 与输出验证 |

这里再给一个判断原则：如果 rollout、IG、ablation 三者一致，解释可信度会明显更高；如果不一致，不要先相信图更好看的那个，而是先检查假设是否成立。

例如在情感分类里：

- rollout 说模型主要“看到了” `good`
- IG 说真正拉低正面分数的是 `not`
- ablation 删除 `not` 后模型从负面变正面

这时更合理的结论是：`not` 才是输出层面的关键因素，而 `good` 只是结构传播中的高连接节点。attention 在这里更像路由，不像证据强度。

---

## 替代方案与适用边界

`attention rollout` 适合回答“信息怎么传”，`IG` 适合回答“哪些输入真的推高了输出”，但它们都不是严格因果解释。只要你没有真正做干预，解释就仍然是近似。

在客服工单分类中，如果目标是排查模型是否“看到了关键词”，rollout 很快，因为它直接沿结构检查关键词是否被传到分类位置。如果目标是定位“哪些词真正导致高风险分类”，IG 或 token ablation 更合适，因为它们更接近输出贡献。

常见替代方法如下：

| 方法 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| attention rollout | 分析 Transformer 内部信息路由 | 结构直观、实现快 | 不等于贡献 |
| IG | 需要 token 贡献排序 | 满足完整性、比原始梯度稳 | 依赖 baseline |
| ablation / occlusion | 需要更接近干预的证据 | 解释直接、容易验证 | 计算慢，组合爆炸 |
| gradient × input | 快速粗排重要特征 | 成本低 | 不如 IG 稳定 |
| SHAP | 需要统一特征贡献框架 | 理论表达强 | 计算很贵，文本上常需近似 |

适用边界可以压缩成一句话：

- 想看“模型有没有把信息传过去”，优先 `rollout`。
- 想看“哪些输入真正影响结果”，优先 `IG`。
- 想要更接近因果判断，用 `ablation / occlusion` 做验证。
- 想在生产环境里稳定落地，不要只保留一种解释图，最好做交叉检查。

---

## 参考资料

1. [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/)：提出 attention flow 与 rollout，用来量化 Transformer 中的信息传播路径。  
2. [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)：提出 Integrated Gradients，解决原始梯度不稳定且缺少公理保证的问题。  
3. [Captum: Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients)：给出 IG 的工程实现方法，适合从论文过渡到代码。  
4. [Attention is not Explanation](https://aclanthology.org/N19-1357/)：说明 attention 权重不能直接当作充分解释，帮助理解“高注意力不等于高贡献”。  
5. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：适合新手建立 Transformer 中 attention、残差和层间传播的直观认识。
