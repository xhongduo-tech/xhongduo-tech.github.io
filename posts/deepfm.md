## 核心结论

DeepFM 是一种面向点击率预估和推荐排序的模型。它把 FM 和 Deep 两个分支放在同一个网络里联合训练，并且让两条分支共享同一套 embedding。embedding 可以理解为“把离散类别映射成可学习稠密向量”的方法。这样做的结果是：

1. FM 分支负责学习低阶交互，主要是一阶权重和二阶特征交叉。
2. Deep 分支负责学习高阶交互，也就是多个字段共同作用形成的复杂模式。
3. 两个分支共享输入表示，所以不需要像 Wide & Deep 那样先手工构造大量交叉特征。

对推荐系统而言，这个组合很实用。FM 擅长记住稳定、稀疏、经典的交互信号，比如“某类用户更容易点某类广告”；Deep 擅长从 embedding 中继续组合出更复杂的模式，比如“新用户在晚间更偏好短视频广告，但只在低价商品场景下成立”。最终输出通常写成：

$$
\hat{y} = \sigma\left(y_{\text{FM}} + y_{\text{Deep}}\right)
$$

其中 $\sigma$ 是 sigmoid，用来把实数映射成 0 到 1 的概率。

下表先给出 DeepFM 的职责拆分：

| 分支 | 输入 | 主要学习内容 | 输出作用 | 是否共享 embedding |
|---|---|---|---|---|
| FM | 稀疏特征 + embedding | 一阶权重、二阶交叉 | 提供稳定低阶信号 | 是 |
| Deep | 同一套 embedding 拼接后送入 MLP | 高阶非线性交互 | 提供复杂泛化能力 | 是 |
| 最终融合 | 两分支 logit 相加 | 联合建模 | 输出 CTR / 转化概率 | 不适用 |

一个入门玩具例子：广告系统里有两个字段，`user_gender` 和 `ad_category`。FM 分支可以直接学到“男性用户 + 数码广告”这个二阶组合的稳定效果；Deep 分支则能继续把 `user_gender`、`time_slot`、`ad_category` 三者组合成高阶模式。两者共享 embedding，所以模型不会分别学两套不一致的类别表示。

---

## 问题定义与边界

DeepFM 解决的问题不是“所有机器学习表格任务”，而是更具体的排序和预估问题，典型目标包括：

- CTR 预估：预测用户是否点击
- CVR 预估：预测用户是否转化
- 推荐排序：给候选物品打分并排序

这类任务通常有三个共同特点。

第一，输入非常稀疏。稀疏的意思是“维度很大，但每次只有很少位置非零”。例如用户 ID、商品 ID、城市、设备、广告位、时间段等字段常常先做 one-hot 或 ID 编码。

第二，类别基数很高。基数高的意思是“一个字段可能有几十万甚至几千万个取值”，比如商品 ID、搜索词、作者 ID。

第三，真正有用的信号往往来自特征交叉。单看“用户是北京”“商品是图书”都不够，关键在于“北京用户 + 图书 + 晚间 + 首次曝光”这种组合。

问题边界可以整理成表格：

| 项目 | 内容 |
|---|---|
| Problem | CTR、CVR、推荐排序 |
| Inputs | one-hot 类别特征、ID 特征、少量 dense 数值特征 |
| Constraints | 稀疏、高基数、交互复杂、样本极不均衡 |
| Success Metric | AUC、LogLoss、GAUC、NDCG |
| 不擅长场景 | 样本很少、图像/文本是主要信息源、强时序依赖任务 |

DeepFM 的关键边界在于：它适合“结构化稀疏特征主导”的场景。如果主要信息来自图像、长文本或复杂会话序列，DeepFM 往往只适合作为最后排序层的一部分，而不是完整解法。

一个新手视角的真实定义是：你把用户、物品、上下文都编码成稀疏字段，模型一边显式计算两两交互，一边让多层神经网络从同一套 embedding 里自动找更复杂的组合，最后输出点击概率。这正是 DeepFM 的工作边界。

真实工程例子：电商首页推荐里，一个样本可能包含 `user_id`、`age_bucket`、`item_id`、`brand_id`、`category_id`、`price_bucket`、`hour`、`device_type`。这些字段单独看都有限，真正影响点击的通常是它们的组合。DeepFM 就是在不手写海量交叉特征的前提下学习这些组合。

---

## 核心机制与推导

先看 FM。FM 的目标是高效学习二阶特征交叉。二阶交叉就是“任意两个特征同时出现时的联合作用”。

设输入为 $x \in \mathbb{R}^n$，线性权重为 $w_i$，第 $i$ 个特征的 embedding 为 $v_i \in \mathbb{R}^k$，则 FM 输出可写成：

$$
y_{\text{FM}} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i<j}\langle v_i, v_j \rangle x_i x_j
$$

其中 $\langle v_i, v_j \rangle$ 是向量点积，意思是“两个特征在隐空间里的匹配程度”。

直接枚举所有 $i<j$ 的计算成本较高，所以常用等价形式：

$$
\sum_{i<j}\langle v_i, v_j \rangle x_i x_j
=
\frac{1}{2}\sum_{f=1}^{k}
\left[
\left(\sum_{i=1}^{n} v_{i,f}x_i\right)^2
-
\sum_{i=1}^{n} v_{i,f}^2 x_i^2
\right]
$$

这个形式把原本 $O(n^2)$ 的枚举，变成和 embedding 维度相关的更高效计算。

再看 Deep 分支。Deep 分支把各字段 embedding 拼接起来，送入多层感知机 MLP。MLP 可以理解为“通过多层线性变换和非线性激活学习复杂组合”。

如果第 0 层输入是拼接后的 embedding：

$$
a^{(0)} = [e_1; e_2; \dots; e_m]
$$

那么第 $l$ 层是：

$$
a^{(l+1)} = \phi\left(W^{(l)} a^{(l)} + b^{(l)}\right)
$$

其中 $\phi$ 常用 ReLU。最终得到：

$$
y_{\text{Deep}} = W^{(L)} a^{(L)} + b^{(L)}
$$

最后两条路径直接相加：

$$
\hat{y} = \sigma\left(y_{\text{FM}} + y_{\text{Deep}}\right)
$$

这里最重要的设计不是“相加”，而是“共享 embedding”。共享的含义是：FM 学到的低阶结构和 Deep 学到的高阶模式，都在更新同一套字段表示。这样可以减少参数冗余，也让稀疏类别的表示更稳定。

玩具例子可以直接算。

设只有两个 one-hot 特征 $x_1=1, x_2=1$，它们的一维 embedding 为：

$$
v_1=[0.5], \quad v_2=[0.2]
$$

那么 FM 的二阶项是：

$$
\langle v_1, v_2 \rangle = 0.5 \times 0.2 = 0.1
$$

如果 Deep 分支把两个 embedding 拼接后做一个极简线性层，权重设成把它们求和，再过 ReLU：

$$
\text{ReLU}(0.5 + 0.2) = 0.7
$$

则最终 logit 可以理解为：

$$
y = 0.1 + 0.7 = 0.8
$$

再做 sigmoid 得到概率。这个例子虽然极简，但已经说明了共享 embedding 的意义：同一组向量同时服务 FM 和 Deep。

真实工程里，用户特征、物品特征、上下文特征往往分成多个 field。每个 field 先查 embedding，再进入两条分支。FM 用 field 间二阶交互兜住稳定模式，Deep 用 MLP 去吸收更高阶组合。这就是 DeepFM 的核心机制。

---

## 代码实现

下面给出一个可运行的 Python 版本，只演示前向计算，不依赖深度学习框架。这个版本足够说明 DeepFM 的数据流和共享 embedding 的实现方式。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

class DeepFM:
    def __init__(self, linear_weight, embeddings, deep_w, deep_b, out_w, out_b):
        self.linear_weight = linear_weight      # dict: feature -> scalar
        self.embeddings = embeddings            # dict: feature -> list[float]
        self.deep_w = deep_w                    # list[list[float]]
        self.deep_b = deep_b                    # list[float]
        self.out_w = out_w                      # list[float]
        self.out_b = out_b                      # scalar

    def fm_part(self, active_features):
        linear = sum(self.linear_weight.get(f, 0.0) for f in active_features)
        pairwise = 0.0
        for i in range(len(active_features)):
            for j in range(i + 1, len(active_features)):
                vi = self.embeddings[active_features[i]]
                vj = self.embeddings[active_features[j]]
                pairwise += dot(vi, vj)
        return linear + pairwise

    def deep_part(self, active_features):
        x = []
        for f in active_features:
            x.extend(self.embeddings[f])  # 共享 embedding：与 FM 使用同一套向量
        hidden = []
        for row, b in zip(self.deep_w, self.deep_b):
            z = sum(w * xi for w, xi in zip(row, x)) + b
            hidden.append(max(0.0, z))    # ReLU
        return sum(w * h for w, h in zip(self.out_w, hidden)) + self.out_b

    def predict_proba(self, active_features):
        logit = self.fm_part(active_features) + self.deep_part(active_features)
        return sigmoid(logit)

linear_weight = {
    "user_male": 0.1,
    "ad_digital": 0.05,
    "night": 0.02,
}

embeddings = {
    "user_male": [0.5, 0.1],
    "ad_digital": [0.2, 0.4],
    "night": [0.3, -0.2],
}

# 输入长度 = 3 个特征 * 2 维 embedding = 6
deep_w = [
    [0.4, 0.1, 0.2, 0.3, 0.5, -0.1],
    [-0.2, 0.3, 0.6, 0.1, 0.2, 0.4],
]
deep_b = [0.0, 0.1]
out_w = [0.7, -0.2]
out_b = 0.05

model = DeepFM(linear_weight, embeddings, deep_w, deep_b, out_w, out_b)
p = model.predict_proba(["user_male", "ad_digital", "night"])

assert 0.0 < p < 1.0
assert round(model.fm_part(["user_male", "ad_digital"]), 2) == 0.29
print("pred =", round(p, 4))
```

这段代码对应的结构是：

1. `embeddings` 是共享输入层。
2. `fm_part` 计算线性项和两两点积。
3. `deep_part` 把同一套 embedding 拼接后送入 MLP。
4. 两个分支的 logit 相加后过 sigmoid。

如果换成 PyTorch，工程代码通常会多出以下部分：

| 模块 | 作用 |
|---|---|
| `Embedding` | 为每个 field 或特征 ID 建立共享向量 |
| `Linear` | 学习一阶权重 |
| `FM interaction` | 计算二阶交叉 |
| `MLP` | 学习高阶交叉 |
| `BCEWithLogitsLoss` | 二分类损失 |
| `AUC / LogLoss` | 训练和验证指标 |

一个真实工程实现里，常见可调参数有：

- `embedding_dim`：embedding 维度
- `deep_hidden_units`：MLP 每层宽度
- `dropout`：随机失活，减少过拟合
- `learning_rate`：学习率
- `l2_reg`：正则化强度
- `batch_size`：批大小

经验上，DeepFM 的 baseline 往往先从较小 embedding 开始，例如 8、16、32，而不是一开始堆很大。因为高基数字段一多，embedding 表的内存成本会迅速上升。

---

## 工程权衡与常见坑

DeepFM 的理论并不复杂，真正难的是工程落地。最常见的问题不是公式写错，而是特征系统失控。

下面先给出常见坑表：

| Pitfall | Why it happens | Mitigation |
|---|---|---|
| 线上线下编码不一致 | 训练和服务使用了不同字典或不同清洗规则 | 固定 feature registry，版本化词典，服务前做一致性校验 |
| DNN 过拟合 | Deep 分支容量大，样本稀疏且噪声多 | dropout、L2、early stopping、减小 hidden size |
| FM 信号被 Deep 淹没 | Deep 分支过强，训练时更容易吃掉梯度 | 监控分支输出分布，控制网络深度，必要时做分支权重调整 |
| embedding 内存爆炸 | 高基数字段太多，维度设置过大 | 热门/长尾分桶，hash trick，分字段维度配置 |
| 冷启动效果差 | 新 ID 没有足够曝光 | 加入统计特征、内容特征、类目级回退特征 |
| 样本偏差 | 曝光机制导致训练数据分布偏 | 做曝光建模、重加权或更严格的样本构造 |

先说最容易忽略的一点：共享 embedding 既是优点，也是风险点。因为 FM 和 Deep 都依赖它，一旦特征编码出错，两条分支会同时受损。

新手最典型的错误例子是：训练时字典里有 `country=US`，线上日志却写成 `country=USA`。结果不是“略有误差”，而是直接查不到原 embedding，只能落到默认值或 OOV 桶。对 DeepFM 来说，这等于 FM 的二阶交叉错了，Deep 的输入也错了，损失是双重的。

再说过拟合。DeepFM 并不天然稳健。FM 分支偏“记忆”，Deep 分支偏“泛化”，但如果 Deep 网络过深、过宽，而训练数据又不够，模型会学到很多短期模式和噪声模式。表现通常是训练集 AUC 很高，验证集 LogLoss 恶化。

还有一个常见误区是把 DeepFM 当成“自动特征工程机器”。它确实减少了人工交叉工作，但并不意味着原始特征可以随便喂。无意义、高泄漏、强噪声字段照样会把模型带偏。比如把未来信息混入特征，或者把口径不稳定的统计量直接输入，模型会在离线指标上很好看，线上却崩。

真实工程例子：广告排序系统里，团队常会同时维护 `user_id`、`ad_id`、`campaign_id`、`creative_id`、`slot_id`、`hour`、`weekday`、`device_os`。如果 `creative_id` 更新频繁、长尾极长，而 embedding 维度又给得很大，显存和训练时间会迅速上涨。此时通常要做分字段裁剪，而不是统一给所有字段同样大的 embedding。

---

## 替代方案与适用边界

DeepFM 不是唯一选择。判断是否使用它，关键看你的业务特征、工程投入和可解释性要求。

先看对比表：

| 模型 | 是否需要人工交叉 | 是否共享 embedding | 主打交互层 | 典型适用场景 |
|---|---|---|---|---|
| FM | 否 | 不适用 | 二阶点积 | 需要快速 baseline，关注稀疏二阶交互 |
| Wide & Deep | 是，Wide 侧常需要 | 通常不完全共享 | 线性 + MLP | 已有成熟人工交叉体系，强调可解释性 |
| DeepFM | 否 | 是 | FM 点积 + MLP | 稀疏推荐/CTR，想减少特征工程成本 |
| xDeepFM | 否 | 通常是 | FM + MLP + CIN | 需要更强显式高阶交互建模 |
| DLRM | 否 | 常用 embedding + interaction | embedding 交互 + MLP | 大规模工业推荐，工程体系成熟 |

Wide & Deep 和 DeepFM 的核心差别是：Wide & Deep 往往要求你手工维护一批 cross features；DeepFM 把这部分显式工程尽量交给 FM 和共享 embedding 来完成。因此，如果你的团队已经积累了大量效果稳定的人工交叉特征，并且需要保留线性分支的可解释性，Wide & Deep 仍然有价值。

如果你发现 DeepFM 的 MLP 对高阶交叉学习不够稳定，或者你希望更明确地控制高阶组合结构，可以考虑 xDeepFM。它加入了 CIN，CIN 可以理解为“显式建模高阶交叉的网络层”，不是完全交给普通 MLP 去隐式学习。

DeepFM 的适用边界可以直接总结成一句话：当你的主要输入是稀疏离散特征，希望同时兼顾低阶记忆和高阶泛化，又不想持续手工造交叉特征时，DeepFM 是一个非常自然的默认选项。

但如果任务明显依赖以下因素，DeepFM 就不一定是首选：

- 长序列用户行为是主信息源
- 文本、图像、多模态特征主导效果
- 线上延迟极其敏感，需要更轻量推理
- 对分支可解释性有极高要求

这时可能需要序列模型、双塔、DLRM、DIN、DCN、xDeepFM，或者把 DeepFM 作为多路召回/排序系统中的一个子模型，而不是唯一主模型。

---

## 参考资料

- Guo, Huifeng, et al. “DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.” IJCAI 2017. https://www.ijcai.org/Proceedings/2017/239
- Shaped Docs, “DeepFM.” https://docs.shaped.ai/docs/v2/model_library/deepfm
- Shaped Docs, “Wide & Deep.” https://docs.shaped.ai/docs/v2/model_library/wide_deep/
- MDPI, 关于 FM 公式与 DeepFM 架构综述文章。https://www.mdpi.com/2073-8994/14/10/2123
- ShadeCoder, “DeepFM: A Comprehensive Guide for 2025.” https://www.shadecoder.com/topics/deepfm-a-comprehensive-guide-for-2025
- ShadeCoder, “xDeepFM: A Comprehensive Guide for 2025.” https://www.shadecoder.com/topics/xdeepfm-a-comprehensive-guide-for-2025
