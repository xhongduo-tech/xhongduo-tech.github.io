## 核心结论

DeepFM 是一个用于点击率预估和推荐排序的端到端模型：FM Component 显式建模二阶特征交叉，Deep Component 隐式学习高阶特征交互，二者共享同一套 embedding 输入。

核心公式是：

$$
p = \sigma(z_{FM} + z_{DNN})
$$

其中，$p$ 是点击概率，$\sigma$ 是 sigmoid 函数，用来把任意实数 logit 映射到 $0$ 到 $1$ 之间。

DeepFM 的关键价值不在于“用 DNN 替代 FM”，而在于把两类能力放在同一个模型里共同训练：

| 模块 | 负责内容 | 交互类型 | 特点 |
|---|---|---|---|
| FM Component | 显式计算任意两个特征之间的关系 | 二阶交叉 | 结构清晰，适合稀疏特征 |
| Deep Component | 从 embedding 拼接结果中学习复杂组合 | 高阶交互 | 表达能力强，但可解释性弱 |
| Shared Embedding | 为两个分支提供同一套特征向量 | 表示共享 | 避免两套表示割裂 |

新手版例子：一个广告点击任务里，输入有 `user_id=U1`、`item_id=I7`、`device=iPhone`、`hour=22`。这些离散 ID 特征本身不能直接表达相似性，所以先映射成向量，也就是 embedding。FM 分支先判断 `U1` 和 `I7` 是否天然相关，DNN 分支再学习“某类用户在晚上用手机浏览某类商品更容易点击”这种更复杂组合。

结构可以简化为：

```text
稀疏特征 / 数值特征
        |
        v
  Shared Embedding
    /           \
   v             v
FM Component   Deep Component
二阶显式交叉     高阶隐式交互
    \           /
     v         v
   z_FM + z_DNN
        |
        v
   sigmoid 输出点击概率
```

DeepFM 适合的核心问题是：特征高维、稀疏、交叉关系多，但人工构造交叉特征成本很高。

---

## 问题定义与边界

DeepFM 解决的是“高维稀疏特征下的点击率预估或推荐排序”问题。点击率预估，简称 CTR Prediction，指预测一次曝光被用户点击的概率。推荐排序，指在候选物品已经召回之后，对这些物品按用户可能感兴趣的程度重新排序。

一条真实曝光日志可能长这样：

| 字段 | 示例值 | 含义 |
|---|---:|---|
| `user_id` | `U1024` | 用户 ID |
| `item_id` | `I7788` | 商品、广告或内容 ID |
| `query` | `running shoes` | 搜索词 |
| `category` | `sports` | 类目 |
| `device` | `ios` | 设备类型 |
| `hour_bucket` | `night` | 时间桶 |
| `region` | `shanghai` | 地域 |
| `clicked` | `1` | 是否点击，训练标签 |

模型输入是曝光日志里的特征，输出是点击概率：

| 项目 | 内容 |
|---|---|
| 输入特征类型 | 稀疏 ID 特征、数值特征、上下文特征 |
| 稀疏 ID 特征 | `user_id`、`item_id`、`category_id`、`query_id` |
| 数值特征 | 历史点击率、价格、用户活跃天数、曝光次数 |
| 上下文特征 | 时间桶、设备、地域、入口页面 |
| 输出目标 | 点击概率 $p(y=1 \mid x)$ |
| 适合场景 | 广告 CTR、商品排序、内容推荐、信息流排序 |
| 不适合场景 | 纯图像端到端建模、纯文本端到端建模、特征极少的简单任务 |

分类边界要明确：DeepFM 是稀疏交互建模模型，不是所有分类问题的通用替代品。它依赖结构化特征，尤其擅长处理大量 one-hot 或 multi-hot 特征。one-hot 是一种把类别变成稀疏向量的编码方式，例如 `device=ios` 在所有设备类别里只激活一个位置。multi-hot 是一次可以激活多个位置的编码方式，例如一个用户最近浏览过多个类目。

如果任务只有十几个稳定数值特征，逻辑回归、GBDT 或小型 MLP 可能已经足够。如果任务主要输入是图片像素或长文本 token，应该优先考虑 CNN、Transformer 或多模态模型，而不是直接套 DeepFM。

---

## 核心机制与推导

DeepFM 的输入通常由多个 field 组成。field 是特征域的意思，例如 `user_id` 是一个 field，`item_id` 是一个 field，`device` 也是一个 field。每个 field 经过 embedding lookup 得到一个低维向量。embedding lookup 是按 ID 查表，把高维稀疏 ID 转成稠密向量。

FM 部分的公式是：

$$
z_{FM} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle v_i, v_j \rangle x_i x_j
$$

其中：

| 符号 | 含义 |
|---|---|
| $w_0$ | 全局偏置 |
| $x_i$ | 第 $i$ 个特征的输入值 |
| $w_i$ | 一阶权重，表示单个特征本身对点击的影响 |
| $v_i$ | 第 $i$ 个特征的 embedding，表示该特征的低维向量 |
| $\langle v_i, v_j \rangle$ | 两个 embedding 的内积，用来表示两个特征之间的二阶交互强度 |

二阶交互是指任意两个特征之间的组合关系。例如 `user_id=U1` 和 `item_id=I7` 经常一起出现点击，FM 可以通过 $\langle v_{U1}, v_{I7} \rangle$ 学到它们之间的关系。

DNN 部分先把多个 field 的 embedding 拼接起来：

$$
h^{(0)} = [v_1; v_2; ...; v_m]
$$

这里 $[;]$ 表示向量拼接，$m$ 是 field 数量。然后送入多层神经网络：

$$
h^{(l+1)} = \phi(W^{(l)} h^{(l)} + b^{(l)})
$$

其中，$\phi$ 是激活函数，例如 ReLU；$W^{(l)}$ 是第 $l$ 层的权重矩阵；$b^{(l)}$ 是偏置。

最后 DNN 输出一个 logit：

$$
z_{DNN} = W^{(L+1)} h^{(L)} + b^{(L+1)}
$$

最终预测为：

$$
p = \sigma(z_{FM} + z_{DNN})
$$

logit 是 sigmoid 之前的原始分数，可以是任意实数。logit 越大，点击概率越高。

玩具例子：只看两个激活特征，$x_1=x_2=1$。设：

$$
w_0=0.1,\quad w_1=0.2,\quad w_2=-0.1,\quad v_1=0.6,\quad v_2=0.4
$$

则：

$$
z_{FM}=0.1+0.2-0.1+0.6 \times 0.4=0.44
$$

假设 DNN 分支输出：

$$
z_{DNN}=0.35
$$

总 logit 为：

$$
z=0.44+0.35=0.79
$$

点击概率为：

$$
p=\sigma(0.79)\approx 0.688
$$

这说明 DeepFM 的最终预测来自两个分支的加和，不是 FM 和 DNN 二选一。

共享 embedding 是 DeepFM 的关键设计。如果 FM 和 DNN 各自维护一套 embedding，FM 学到的是一套“适合二阶交叉”的表示，DNN 学到的是另一套“适合高阶组合”的表示，两者之间没有直接约束。DeepFM 让两个分支共用 $v_i$，训练时 FM 的二阶损失信号和 DNN 的高阶损失信号都会更新同一个向量。这样做有三个直接效果：

| 设计点 | 作用 |
|---|---|
| 避免两套 embedding | 参数更少，表示不割裂 |
| 二阶与高阶互补 | FM 给稳定 pairwise 信号，DNN 学复杂非线性 |
| 梯度共同更新 | embedding 同时服务两个目标，训练更稳定 |

真实工程例子：广告 CTR 预估中，`user_id`、`item_id`、`query`、`类目`、`设备`、`时间桶`、`地域` 同时进入模型。FM 负责学习 `user_id × item_id`、`类目 × 时间桶`、`设备 × 广告样式` 这类稳定二阶关系。DNN 负责学习“用户历史偏好 + 当前搜索词 + 晚间时段 + 移动设备”共同影响点击的复杂模式。

---

## 代码实现

代码层面，DeepFM 通常由三块组成：输入层、FM 层、DNN 层。实现重点不是把公式写得复杂，而是统一管理 embedding，并在 FM 分支和 DNN 分支复用。

最小流程可以写成：

```python
emb = embedding_lookup(features)
z_fm = linear_term + second_order_term(emb)
z_dnn = mlp(concat(emb))
p = sigmoid(z_fm + z_dnn)
```

结构对应关系如下：

| 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 输入特征 | 原始 field | ID / 数值张量 | 表示一条曝光日志 |
| embedding 层 | 稀疏 ID | 稠密向量 | 把离散特征转成可学习表示 |
| FM 分支 | embedding + 一阶权重 | `z_fm` | 计算一阶项和二阶交叉 |
| DNN 分支 | embedding 拼接 | `z_dnn` | 学习高阶非线性交互 |
| logit 相加 | `z_fm + z_dnn` | 总 logit | 融合两个分支 |
| sigmoid | 总 logit | 点击概率 | 输出 $0$ 到 $1$ 的概率 |

下面是一个可运行的 Python 玩具实现，只依赖标准库。为了让逻辑清楚，代码只演示两个 field 的 DeepFM 前向计算：

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def relu(xs):
    return [max(0.0, x) for x in xs]

def linear_layer(weights, bias, xs):
    return [
        sum(w_i * x_i for w_i, x_i in zip(row, xs)) + b
        for row, b in zip(weights, bias)
    ]

# 两个激活特征：user_id=U1, item_id=I7
x = [1.0, 1.0]

w0 = 0.1
linear_w = [0.2, -0.1]

# shared embedding: 同一套向量同时给 FM 和 DNN 使用
embeddings = [
    [0.6, 0.2],  # v_user
    [0.4, 0.5],  # v_item
]

# FM: 一阶项 + 二阶内积
linear_term = w0 + sum(w * xi for w, xi in zip(linear_w, x))
second_order = dot(embeddings[0], embeddings[1]) * x[0] * x[1]
z_fm = linear_term + second_order

# DNN: 拼接 embedding 后过一层 MLP
h0 = embeddings[0] + embeddings[1]
hidden_w = [
    [0.3, -0.2, 0.4, 0.1],
    [-0.1, 0.5, 0.2, 0.2],
]
hidden_b = [0.0, 0.1]
h1 = relu(linear_layer(hidden_w, hidden_b, h0))

out_w = [0.7, -0.3]
out_b = 0.05
z_dnn = sum(w * h for w, h in zip(out_w, h1)) + out_b

p = sigmoid(z_fm + z_dnn)

assert round(z_fm, 4) == 0.44
assert 0.0 < p < 1.0
assert round(p, 4) == round(sigmoid(z_fm + z_dnn), 4)

print(round(p, 4))
```

在 deepctr 或 deepctr-torch 这类工程实现中，对应关系通常是：

| DeepFM 概念 | 工程实现里的常见模块 |
|---|---|
| 特征列定义 | SparseFeat、DenseFeat、VarLenSparseFeat |
| embedding 层 | embedding dict / embedding lookup |
| FM Component | FM layer |
| Deep Component | DNN stack / MLP |
| 输出层 | add logits + PredictionLayer |

实际训练时还会加入 batch normalization、dropout、正则化、优化器配置和样本权重。batch normalization 是一种稳定中间层分布的技术，dropout 是训练时随机丢弃部分神经元来降低过拟合的方法。

---

## 工程权衡与常见坑

DeepFM 的效果高度依赖特征边界、数据切分和训练目标。很多线上问题不是因为 FM 公式写错，而是数据泄漏、过拟合或评估指标不合适。

最常见的坑如下：

| 坑位 | 结果 | 规避方式 |
|---|---|---|
| 低频 ID 直接建大 embedding 表 | 内存高，长尾 ID 泛化差 | 做频次截断、hashing、低频归桶 |
| 随机切分训练集和验证集 | 未来信息泄漏，线下 AUC 虚高 | CTR 任务优先按时间切分 |
| 只看 accuracy | 类别不平衡下指标失真 | 重点看 AUC、logloss、校准情况 |
| embedding 维度过大 | 参数膨胀，训练不稳定 | 从小维度开始，例如 8、16、32 |
| DNN 太深 | 过拟合或收益很小 | 先用 2 到 3 层浅 MLP |
| 只依赖 ID 特征 | 冷启动能力差 | 加入类目、文本侧标签、上下文特征 |
| 负样本采样不一致 | 训练分布和线上分布偏移 | 明确采样策略并做概率校正 |

新手版例子：如果把所有曝光日志随机打散后划分训练集和验证集，那么 4 月 10 日的用户行为可能进入训练集，4 月 8 日的曝光进入验证集。模型间接看到了未来行为，线下 AUC 会很好，但线上遇到真实未来流量时效果变差。CTR 任务通常应该按时间切分，例如用前 7 天训练，第 8 天验证，第 9 天测试。

一个更稳的最小工程顺序是：

| 步骤 | 建议 |
|---|---|
| 1 | 先用小 embedding，控制参数量 |
| 2 | 先用浅 DNN，避免一开始就堆深层 |
| 3 | 先按时间切分数据，避免泄漏 |
| 4 | 先看 AUC / logloss，不只看 accuracy |
| 5 | 再逐步调 embedding 维度、DNN 宽度、正则化和学习率 |

AUC 是排序指标，衡量正样本排在负样本前面的能力。logloss 是概率预测指标，惩罚“预测很自信但预测错”的情况。CTR 任务里点击样本通常远少于未点击样本，因此 accuracy 很容易误导：如果点击率只有 2%，模型永远预测“不点击”也能有 98% accuracy，但这个模型没有排序价值。

DeepFM 还有一个常见误区：认为它“无需特征工程”就等于“无需特征理解”。更准确的说法是，DeepFM 减少了人工构造交叉特征的工作，但仍然需要清楚哪些原始特征能在上线时稳定获得，哪些统计特征可能泄漏未来信息，哪些 ID 特征会带来严重长尾问题。

---

## 替代方案与适用边界

DeepFM 不是所有推荐模型的默认最优解。它适合稀疏特征交互场景，但当数据形态、算力预算或线上目标变化时，其他模型可能更合适。

| 模型 | 核心能力 | 适合场景 | 局限 |
|---|---|---|---|
| 逻辑回归 | 一阶线性建模 | 特征少、关系简单、需要强可解释性 | 不能自动学习交叉 |
| FM | 二阶特征交互 | 稀疏 ID 特征、pairwise 关系明显 | 高阶表达能力有限 |
| Wide & Deep | 宽特征记忆 + 深层泛化 | 有稳定人工交叉特征的工业系统 | 仍依赖 wide 侧特征设计 |
| DeepFM | FM + DNN 共享 embedding | 二阶和高阶交互都重要的 CTR 排序 | 序列兴趣建模能力有限 |
| xDeepFM / DCN | 更强显式交叉能力 | 需要更强特征组合建模 | 结构更复杂，调参成本更高 |
| DIN / DIEN | 用户行为序列兴趣建模 | 点击序列、购买序列强相关 | 对行为序列质量要求高 |
| Transformer 推荐模型 | 长序列和复杂上下文建模 | 大规模序列推荐、多行为建模 | 训练和 serving 成本更高 |

新手版判断方式：

如果只需要很强的二阶交互，FM 可能已经够用。例如 `用户 × 商品`、`类目 × 时间` 的关系是主要信号，数据量也不大，此时 DeepFM 的 DNN 分支未必带来明显收益。

如果已有大量人工交叉特征，并且这些特征在业务上很稳定，Wide & Deep 仍然是常见选择。wide 分支负责记忆确定性组合，deep 分支负责泛化。

如果用户点击强烈依赖历史行为序列，例如“刚看过手机壳，所以接下来更可能点充电器”，DeepFM 只看聚合后的结构化特征可能不够。此时通常会考虑 DIN、DIEN 或 Transformer 类推荐模型，因为它们更擅长建模用户行为序列。

DeepFM 的适用边界可以总结为：

| 条件 | 是否适合 DeepFM |
|---|---|
| 特征高度稀疏 | 适合 |
| 交叉关系多 | 适合 |
| 日志数据量大 | 适合 |
| 人工交叉维护成本高 | 适合 |
| 纯图片输入 | 不适合 |
| 纯长文本输入 | 不适合直接替代文本模型 |
| 特征数量很少 | 未必需要 |
| 强依赖用户行为序列 | 可能需要序列模型增强 |

真实工程中，DeepFM 常作为排序阶段的基线或主模型。它的优势是结构清楚、实现成熟、训练成本可控。缺点是对序列兴趣、跨域迁移、多模态内容理解的表达能力有限。选型时应先看数据形态，而不是只看模型名字。

---

## 参考资料

1. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf). DeepFM 原始论文，本文中 FM Component、Deep Component、共享 embedding 和总公式来自该论文。

2. Steffen Rendle. [Factorization Machines](https://ieeexplore.ieee.org/document/5694074). FM 基础论文，本文中二阶交互公式 $z_{FM}$ 的基础思想来自该论文。

3. DeepCTR. [DeepFM `deepfm.py` 源码](https://deepctr-doc.readthedocs.io/en/latest/_modules/deepctr/models/deepfm.html). 工程实现参考，用于理解特征列、embedding 层、FM layer、DNN stack 的组合方式。

4. DeepCTR-Torch. [DeepFM `deepfm.py` 源码](https://deepctr-torch.readthedocs.io/en/latest/_modules/deepctr_torch/models/deepfm.html). PyTorch 版本工程实现参考，用于理解 DeepFM 在实际代码中的模块拆分。
