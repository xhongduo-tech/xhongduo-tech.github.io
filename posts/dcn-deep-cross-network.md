## 核心结论

DCN（Deep & Cross Network，深度交叉网络）是一种把显式特征交叉和深层非线性表示联合训练的推荐排序模型。显式特征交叉，指模型直接构造并学习多个输入特征之间的组合关系；深层非线性表示，指 DNN 通过多层神经网络学习更自由、更抽象的模式。

DCN 的核心价值不是替代 DNN，而是在 CTR 预估、推荐排序、广告排序这类场景中，更高效地学习明确的特征组合。例如 `user_country × device_type × item_category` 这种组合，纯 DNN 也可能学到，但它需要在参数空间里自己摸索；DCN 的 Cross Network 会反复把原始输入 $x_0$ 拉回来，与当前层表示 $x_l$ 做交叉，因此更直接。

| 维度 | 普通 DNN | DCN |
|---|---|---|
| 特征交叉方式 | 隐式学习 | 显式交叉 + 隐式学习 |
| 是否保留原始输入参与每层计算 | 不一定 | 是，Cross 层每层都使用 $x_0$ |
| 适合场景 | 复杂非线性模式 | 推荐、广告、排序中的明确组合规则 |
| 主要风险 | 需要更多数据学交叉 | Cross 层过深容易过拟合 |
| 典型结构 | Embedding + MLP | Embedding + Cross Network + Deep Network |

可以把 DCN 写成：

$$
\text{DCN}(x)=\text{Combine}(\text{Cross}(x),\text{Deep}(x))
$$

Cross Network 负责有界阶数的显式交叉，Deep Network 负责更自由的隐式表达，二者联合训练，共同服务最终排序目标。

---

## 问题定义与边界

DCN 解决的问题是：在稀疏离散特征很多的推荐任务里，如何更高效地学习特征交互。稀疏离散特征，指取值空间很大但每条样本只命中少量取值的类别特征，例如用户 ID、广告 ID、城市、设备类型、商品类目。数值特征，指可以直接表示为连续数值的特征，例如价格、历史点击率、停留时长。

在 CTR 预估中，`user_country`、`device_type`、`hour`、`ad_category` 单独看信息有限，但组合后经常很关键。例如“美国用户 + iOS 设备 + 晚上 9 点 + 游戏广告”可能对应一种点击模式。DCN 的目标就是让模型自动学习这类组合，而不是靠工程师手工把所有组合特征枚举出来。

| 场景 | 是否适合 DCN | 原因 |
|---|---:|---|
| 广告 CTR 预估 | 适合 | 类别特征多，组合关系明显 |
| 推荐系统排序 | 适合 | 用户、物品、上下文之间存在交互 |
| 搜索排序粗排/精排 | 视情况适合 | 若结构化特征丰富，DCN 有价值 |
| 图像分类 | 通常不适合 | 卷积或视觉 Transformer 更自然 |
| 大语言模型建模 | 不适合 | 问题主体是序列建模，不是结构化特征交叉 |
| 直接输入超高维 one-hot | 不适合 | 维度过大，参数和计算都会失控 |

DCN 的输入通常不是原始高维 one-hot，而是“类别特征经过 embedding 后的向量 + 少量数值特征”。Embedding 是把离散 ID 映射成稠密向量的查表层，例如把 `device_type=ios` 映射成一个 16 维向量。多个 embedding 拼接后得到 $x_0$，再送入 Cross Network 和 Deep Network。

边界也要明确：DCN 适合推荐、排序、CTR 预估等结构化特征任务，但不是所有深度模型的通用替代品。如果任务的主要信息来自文本、图像、语音或长序列行为，DCN 通常只能作为一部分特征建模模块，而不是主模型。

---

## 核心机制与推导

Cross Layer 是 DCN 的核心。原始 vector 版 Cross Layer 常写成：

$$
x_{l+1}=x_0\odot(x_l^\top w_l)+b_l+x_l
$$

其中 $x_0$ 是原始输入向量，$x_l$ 是第 $l$ 层输出，$w_l$ 是可训练权重，$b_l$ 是偏置，$\odot$ 表示按元素乘。$x_l^\top w_l$ 会得到一个标量，这个标量像门控一样控制原始输入 $x_0$ 如何参与本层交叉。门控，白话说就是一个控制强弱的系数。

也有实现会写成矩阵版或 DCN-V2 形式：

$$
x_{l+1}=x_0\odot(W_lx_l+b_l)+x_l
$$

为了降低参数量，DCN-V2 还可以使用低秩分解：

$$
W_l=U_lV_l
$$

低秩分解，白话说就是用两个小矩阵近似一个大矩阵，从而减少参数和计算量。

玩具例子：取 $x_0=[1,2]$，第一层参数 $w_0=[0.5,1.0]$，$b_0=[0,0]$。先计算标量门控：

$$
x_0^\top w_0=1\times0.5+2\times1.0=2.5
$$

再计算输出：

$$
x_1=x_0\odot2.5+x_0=[1\times2.5+1,\ 2\times2.5+2]=[3.5,\ 7.0]
$$

这个过程可以拆成三步：

| 步骤 | 计算 | 结果 |
|---|---|---|
| 输入 | $x_0$ | $[1,2]$ |
| 标量门控 | $x_0^\top w_0$ | $2.5$ |
| 输出 | $x_0\odot2.5+x_0$ | $[3.5,7.0]$ |

为什么说层数会提高交互阶数？因为每一层都把原始输入 $x_0$ 与上一层结果 $x_l$ 再组合一次。上一层已经包含低阶交互，本层再乘一次原始输入，就能形成更高阶的多项式项。通常可以理解为：堆叠 $l$ 层 Cross Layer，最高可表达到 $l+1$ 阶交互。

| Cross 层数 | 可理解的最高交互阶数 | 例子 |
|---:|---:|---|
| 0 | 1 阶 | 单特征作用 |
| 1 | 2 阶 | `user_country × device_type` |
| 2 | 3 阶 | `country × device × category` |
| 3 | 4 阶 | `country × device × category × hour` |

真实工程例子：广告 CTR 排序中，模型输入可能包括用户国家、设备类型、广告类目、小时、历史点击率、广告出价。Cross Network 可以学习“某国家用户在某设备上对某类广告更敏感”这类交叉关系；Deep Network 同时学习更复杂的非线性模式，例如用户历史行为与广告内容 embedding 的隐式匹配。

---

## 代码实现

实现 DCN 时先分清版本：原始 vector 版 Cross Layer 使用 $w_l$ 产生标量门控；DCN-V2 或矩阵版 Cross Layer 使用 $W_l$ 产生向量门控。两者形状不同，参数量也不同。

基本结构如下：

```text
类别特征 -> embedding
数值特征 -> 归一化
拼接得到 x0
        |-----------------> Cross Network
        |-----------------> Deep Network
Cross 输出 + Deep 输出 -> 拼接 -> 输出层 -> 预测分数
```

核心伪代码：

```python
x = x0
for layer in cross_layers:
    gate = matmul(x, w) + b
    x = x0 * gate + x
```

形状说明：

| 符号 | vector 版形状 | 含义 |
|---|---|---|
| `x0` | `[batch, d]` | 原始输入向量 |
| `x_l` | `[batch, d]` | 第 `l` 层 Cross 输出 |
| `w_l` | `[d, 1]` | 当前层权重 |
| `x_l @ w_l` | `[batch, 1]` | 标量门控 |
| `b_l` | `[d]` 或 `[1, d]` | 偏置 |
| `x_{l+1}` | `[batch, d]` | 下一层输出 |

下面是一个可运行的最小 Python 实现，只依赖 `numpy`：

```python
import numpy as np

class CrossLayer:
    def __init__(self, dim):
        self.w = np.zeros((dim, 1), dtype=float)
        self.b = np.zeros((dim,), dtype=float)

    def set_params(self, w, b=None):
        self.w = np.asarray(w, dtype=float).reshape(-1, 1)
        if b is not None:
            self.b = np.asarray(b, dtype=float)

    def __call__(self, x0, xl):
        gate = xl @ self.w          # [batch, 1]
        return x0 * gate + self.b + xl

x0 = np.array([[1.0, 2.0]])
layer = CrossLayer(dim=2)
layer.set_params(w=[0.5, 1.0], b=[0.0, 0.0])

x1 = layer(x0, x0)

assert x1.shape == x0.shape
assert np.allclose(x1, np.array([[3.5, 7.0]]))

# 堆叠第二层时，输入形状仍然保持 [batch, d]
x2 = layer(x0, x1)
assert x2.shape == x0.shape
```

注意这里的 `gate` 是 `[batch, 1]`，会广播到 `[batch, d]`。广播，白话说就是小形状数组在计算时自动扩展成兼容的大形状。这个行为很方便，但也容易隐藏 bug，所以工程实现里要主动检查张量形状。

---

## 工程权衡与常见坑

Cross 层不是越深越好。层数增加会提高可表达的交互阶数，但真实数据中的有效交互通常有限，过深会带来过拟合、训练不稳定和收益递减。实际调参通常从 1 到 3 层开始，再根据验证集指标决定是否加深。

| 常见坑 | 问题 | 规避建议 |
|---|---|---|
| 把 Cross 写成 `x0 * xl` | 这只是逐元素乘，不是原始 DCN 机制 | 先用 `xl` 计算门控，再和 `x0` 交叉 |
| 直接喂高维稀疏 ID | 维度爆炸，训练慢且不稳定 | 类别特征先 embedding |
| Cross 层堆太深 | 高阶交互过多，容易过拟合 | 从 1-3 层开始，用验证集选择 |
| 混淆 DCN 与 DCN-V2 | vector 版和 matrix/low-rank 版参数形状不同 | 明确实现公式和张量维度 |
| 忽略数值特征归一化 | 数值尺度过大影响训练 | 对连续特征做标准化或分桶 |
| 拼接方式随意 | Cross 与 Deep 输出尺度不一致 | 使用规范初始化、归一化和验证集监控 |

调参优先级通常是：

1. 先检查 embedding 维度是否合理。维度太小表达不足，太大容易过拟合。
2. 再调 Cross 层数。多数任务先试 1、2、3 层。
3. 再调 Deep 分支宽度和层数。Deep 太大可能掩盖 Cross 的收益。
4. 最后调正则化，例如 dropout、L2、early stopping。

工程上还要注意线上一致性。训练时的特征处理、embedding 字典、缺失值策略、数值归一化参数，必须和线上推理一致。否则模型离线 AUC 很高，上线 CTR 却可能下降。

---

## 替代方案与适用边界

DCN 适合“既需要显式交叉，又需要深层表达”的结构化特征任务，但它不是唯一选择。模型选择应取决于数据规模、特征类型、交互模式和线上延迟预算。

| 模型 | 是否显式交叉 | 可解释性 | 参数复杂度 | 对稀疏特征适配性 | 适用边界 |
|---|---|---|---|---|---|
| DCN | 是 | 中等 | 中等 | 好 | 明确组合规则较多的排序任务 |
| DNN | 否，主要隐式学习 | 较弱 | 可大可小 | 好 | 数据量大、模式复杂但不要求显式交叉 |
| Wide & Deep | Wide 部分显式 | 较好 | 取决于人工特征 | 好 | 有成熟人工交叉特征的系统 |
| FM / DeepFM | 是，常见为二阶交叉 | 中等 | 较低到中等 | 很好 | 二阶交互强、稀疏特征多 |
| xDeepFM | 是，压缩交互网络 | 中等 | 较高 | 好 | 希望建模更复杂显式交互 |

如果任务里最重要的是少量明确组合规则，例如广告排序中的国家、设备、时间、类目组合，DCN 往往值得优先尝试。如果交互主要集中在二阶，FM 或 DeepFM 可能更简单。如果已有大量人工交叉特征且业务规则稳定，Wide & Deep 仍然有效。如果交互模式极复杂、数据量很大、显式交叉收益不明显，纯 DNN 或更复杂的序列模型可能更合适。

结论是：当输入是结构化推荐特征，且你怀疑“特征组合”比单特征更关键时，DCN 是一个高性价比基线；当任务主体不是结构化特征交叉，或者特征工程和数据规模不足以支撑高阶交互时，不应盲目使用 DCN。

---

## 参考资料

1. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)：原始论文，适合理解 DCN 的设计动机、Cross Layer 公式和广告点击预估场景。
2. [TensorFlow Recommenders: Deep & Cross Network](https://www.tensorflow.org/recommenders/examples/dcn)：官方教程，适合从工程角度理解如何把 DCN 用在推荐任务中。
3. [TensorFlow Recommenders API: tfrs.layers.dcn](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/dcn)：API 文档，适合查实现参数、输入输出形状和可选配置。
4. [TensorFlow Recommenders dcn.py Source Code](https://github.com/tensorflow/recommenders/blob/v0.7.7/tensorflow_recommenders/layers/feature_interaction/dcn.py)：源码实现，适合确认 vector、matrix、low-rank 等工程细节。
5. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)：扩展论文，适合理解 DCN-V2、低秩交叉和大规模排序系统中的实践经验。
