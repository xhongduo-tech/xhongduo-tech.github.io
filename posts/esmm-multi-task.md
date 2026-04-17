## 核心结论

ESMM（Entire Space Multi-Task Model，完整空间多任务模型）适合解决推荐排序里 CTR 和 CVR 的联合建模问题。CTR 是点击率，白话说就是“曝光后会不会点”；CVR 是转化率，白话说就是“点了以后会不会买/注册/下单”。

它的关键不是“多加一个任务”，而是把用户行为链路里的顺序关系写进模型：

$$
\text{impression} \rightarrow \text{click} \rightarrow \text{conversion}
$$

在这个链路下，ESMM不直接只用点击样本训练 CVR，而是同时在全曝光空间训练两个任务：

- $pCTR = P(y=1 \mid x)$
- $pCTCVR = P(y=1,z=1 \mid x)$

再利用：

$$
pCTCVR = pCTR \times pCVR
$$

把 CVR 约束回全曝光空间。这样可以同时缓解两个核心问题：

1. 样本选择偏差：CVR 只在点击样本上有标签，但上线推理面对的是所有曝光样本。
2. 数据稀疏：转化远少于点击，直接学 CVR 很容易学不稳。

工程上，ESMM最值得记住的是三件事：

- 底层 embedding 共享
- 上层 CTR 塔、CVR 塔独立
- 用 $pCTR \times pCVR$ 去拟合 $pCTCVR$

如果你的数据里有完整曝光日志，而且点击到转化是明确顺序事件，ESMM 是一个可直接落地、可直接复现的基线方案。

---

## 问题定义与边界

先把任务边界说清楚。

在推荐或广告系统里，一条样本通常从“曝光”开始。曝光就是“这个商品、广告、内容被展示给用户”。之后可能发生点击，再之后才可能发生转化。

因此同一个曝光样本，会有两个常见标签：

- 点击标签 $y \in \{0,1\}$
- 转化标签 $z \in \{0,1\}$，但通常只有在点击后才有意义

这带来一个直接问题：CTR 和 CVR 的训练空间不一样。

| 任务 | 预测目标 | 训练样本空间 | 正样本定义 | 稀疏程度 |
|---|---|---|---|---|
| CTR | 曝光后是否点击 | 全部曝光 | 点击=1 | 相对不稀疏 |
| CVR | 点击后是否转化 | 仅点击样本 | 转化=1 | 很稀疏 |
| CTCVR | 曝光后是否点击且转化 | 全部曝光 | 点击=1 且转化=1 | 最稀疏 |

“样本选择偏差”可以直白理解成：你训练时只看“已经点过的人”，但预测时却要面对“所有会被曝光的人”。这两群人的分布不是一回事。

一个玩具例子：

- 曝光 10000 次
- 点击 500 次，CTR = 5%
- 转化 25 次，CTCVR = 0.25%
- 在点击样本里看，CVR = 25 / 500 = 5%

如果你只拿 500 个点击样本训练 CVR 模型，模型学到的是“已经愿意点的人里，谁更会转化”。但线上排序时，模型面对的是 10000 个曝光候选。未点击样本的特征分布完全被忽略了，这就是偏差来源。

这里还要补一个术语：MNAR（Missing Not At Random，非随机缺失）。白话说，数据不是“随机没采到”，而是“因为用户没点，所以你天然看不到后续转化”。这不是简单缺失，而是由行为机制决定的缺失。

因此，ESMM 的适用边界也很明确：

- 适合：曝光→点击→转化链路清晰，且曝光日志完整
- 不适合：没有全曝光数据，或者“点击后转化”这层定义本身不稳定

---

## 核心机制与推导

ESMM 的核心推导其实很短，但非常关键。

设：

- $x$：样本特征
- $y$：点击事件
- $z$：转化事件

定义三种概率：

$$
pCTR = P(y=1 \mid x)
$$

$$
pCVR = P(z=1 \mid y=1, x)
$$

$$
pCTCVR = P(y=1,z=1 \mid x)
$$

由于转化一定发生在点击之后，可以写出：

$$
P(y=1,z=1 \mid x)=P(y=1 \mid x)\cdot P(z=1 \mid y=1,x)
$$

因此：

$$
pCTCVR = pCTR \times pCVR
$$

进一步有：

$$
pCVR = \frac{pCTCVR}{pCTR}
$$

这一步的意义不是“拿个公式算一算”，而是把原本只在点击子空间定义的 CVR，嵌进了全曝光空间的联合训练里。

### 玩具例子

假设某个曝光样本的模型输出是：

- $pCTR=0.12$
- $pCVR=0.25$

那么：

$$
pCTCVR = 0.12 \times 0.25 = 0.03
$$

这表示：该样本从曝光走到“点击且转化”的总体概率是 3%。

如果真实标签是：

- 点击标签 $y=1$
- 转化标签 $z=1$

那么：

- CTR 任务会把 0.12 往 1 拉
- CTCVR 任务会把 0.03 往 1 拉
- CVR 塔则通过乘法关系间接受到约束

如果真实标签是：

- 点击标签 $y=0$
- 转化标签 $z=0$

那么：

- CTR 任务会把 0.12 往 0 拉
- CTCVR 任务也会把 0.03 往 0 拉

这样，CVR 相关表示不再只从点击子集学习，而是通过共享 embedding 接触到了全体曝光样本。

### 结构怎么搭

ESMM 不是复杂图模型，它的主干很直接：

1. 稀疏离散特征进入共享 embedding
2. embedding 拼接后形成共享底座表示
3. 分叉到 CTR 塔，输出 $pCTR$
4. 分叉到 CVR 塔，输出 $pCVR$
5. 通过乘法得到 $pCTCVR=pCTR \times pCVR$

可以写成一个层级列表：

- Embedding 层：用户 ID、商品 ID、类目、上下文等共享表示
- CTR Tower：预测是否点击
- CVR Tower：预测点击后是否转化
- CTCVR 辅助目标：用乘法约束全链路概率

这里“塔”就是 task-specific tower，白话说就是“不同任务各自的上层小网络”。

为什么共享 embedding 有用？因为 embedding 是大部分参数所在的位置，而 CVR 标签太稀疏，单独训练很难把这些参数学稳。CTR 任务样本更多，能帮底层表示先学出相对可用的结构。

但要注意一个工程事实：虽然公式上有 $pCVR=pCTCVR/pCTR$，训练里通常不建议真的直接做除法求监督，因为当 $pCTR$ 很小时会数值不稳定。比如：

- 若 $pCTR \approx 0.001$
- 而 $pCTCVR \approx 0.0002$

那么比值会被小分母放大，梯度容易抖动。实际工程里更常见的做法是保留独立的 CVR 塔输出，再用乘法关系做约束，而不是把“除法”作为主路径。

### 真实工程例子

以广告排序为例，一条曝光日志可能包含：

- 用户画像：地域、设备、历史点击序列
- 物料特征：广告主、商品类目、价格段
- 上下文特征：时间、频道位、流量来源

标签构造通常是：

- `label_ctr = 1(click)`
- `label_ctcvr = 1(click and conversion)`

训练时对全曝光样本都算 CTR loss 和 CTCVR loss。这样模型上线后可以直接给每个曝光候选输出：

- 预估点击概率
- 预估点击且转化概率
- 或中间的 CVR 估计

排序阶段再按业务目标组合，例如出价、GMV、ROI 等。

---

## 代码实现

下面先给一个最小可运行的 Python 版本，用来说明 ESMM 的标签构造和损失逻辑。它不是完整深度学习训练代码，但机制是对的。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def bce(prob: float, label: int, eps: float = 1e-8) -> float:
    prob = min(max(prob, eps), 1.0 - eps)
    return -(label * math.log(prob) + (1 - label) * math.log(1 - prob))

# 玩具样本：一次曝光，发生了点击和转化
label_ctr = 1
label_ctcvr = 1  # click=1 and conversion=1

# 两个塔的logit输出
ctr_logit = -2.0
cvr_logit = -1.1

pctr = sigmoid(ctr_logit)
pcvr = sigmoid(cvr_logit)
pctcvr = pctr * pcvr

loss = bce(pctr, label_ctr) + bce(pctcvr, label_ctcvr)

assert 0.0 < pctr < 1.0
assert 0.0 < pcvr < 1.0
assert abs(pctcvr - pctr * pcvr) < 1e-12
assert loss > 0.0

# 数值例子：pCTR=0.12, pCVR=0.25 -> pCTCVR=0.03
assert abs(0.12 * 0.25 - 0.03) < 1e-12
print(round(pctr, 6), round(pcvr, 6), round(pctcvr, 6), round(loss, 6))
```

如果写成常见的伪代码，结构会更直观：

```python
embedding = SharedEmbedding(features)

pctr = sigmoid(CtrTower(embedding))
pcvr = sigmoid(CvrTower(embedding))
pctcvr = pctr * pcvr

loss = BCE(pctr, label_ctr) + BCE(pctcvr, label_ctcvr)
```

标签构造通常如下：

| 输入样本 | `label_ctr` | `label_ctcvr` | 是否进入训练 |
|---|---:|---:|---|
| 曝光未点击 | 0 | 0 | 是 |
| 曝光点击未转化 | 1 | 0 | 是 |
| 曝光点击且转化 | 1 | 1 | 是 |

这张表很重要。ESMM 的本质就是：所有曝光样本都进入训练，而不是只拿点击样本训练后续任务。

如果你用 PaddleRec 或类似实现，核心模块通常包括：

1. 数据预处理  
把曝光日志整理成统一样本，每条样本都带 `ctr_label` 和 `ctcvr_label`。

2. 共享 embedding  
用户、物品、上下文等离散特征共享同一套底层表示。

3. 双塔输出  
CTR 塔输出 `pctr`，CVR 塔输出 `pcvr`。

4. 乘法约束  
`pctcvr = pctr * pcvr`，并对 `pctcvr` 计算 BCE。

更接近工程写法的伪代码如下：

```python
pctr = sigmoid(CtrTower(embedding))
pcvr = sigmoid(CvrTower(embedding))
pctcvr = pctr * pcvr

loss_ctr = BCE(pctr, label_ctr)
loss_ctcvr = BCE(pctcvr, label_ctcvr)
loss = loss_ctr + loss_ctcvr
```

有些实现还会尝试加入 CVR 监督，但一般建议谨慎处理。若直接写：

```python
pcvr_from_ratio = pctcvr / pctr
loss_cvr = BCE(pcvr_from_ratio, label_cvr)
```

当 `pctr` 很小时会不稳定。更稳妥的方式是：

- 保留 `pcvr = sigmoid(CvrTower(...))`
- 用 `pctcvr = pctr * pcvr`
- 主要监督 `loss_ctr + loss_ctcvr`
- 如要增强约束，可在点击样本上额外做 mask 后的 `loss_cvr`

也就是只在 `label_ctr=1` 的样本上给 CVR 辅助损失，而不把除法作为主要训练路径。

---

## 工程权衡与常见坑

ESMM 好用，但它不是“上了就赢”的模型。常见问题主要集中在样本、数值和任务冲突上。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 直接用点击样本训练 CVR | 线上泛化差 | 训练分布和推理分布不一致 | 使用全曝光训练 CTR + CTCVR |
| `pctcvr / pctr` 不稳定 | loss 抖动、梯度爆炸 | 小分母放大噪声 | 保留独立 CVR 塔，用乘法约束 |
| 曝光日志不全 | 训练效果虚高但上线失真 | 全空间假设被破坏 | 先检查日志采集完整性 |
| 任务冲突 | CTR 提升但 CVR 下降 | 共享表示被强任务主导 | 调整 loss 权重、分塔更深 |
| 转化回传延迟 | 标签污染 | 转化在窗口外才到达 | 固定回传窗口，延迟对齐 |

新手最容易踩的一个坑就是数值问题。

例如某批样本里：

- $pCTR \approx 0.001$
- $pCTCVR \approx 0.00001$

如果你强行算：

$$
pCVR = \frac{pCTCVR}{pCTR}
$$

得到的值在数学上没问题，但梯度会很容易被小分母扰乱。工程里更稳的做法通常是让 CVR 塔自己输出 `sigmoid(cvrtower)`，再让 `pctr * pcvr` 去贴近 `label_ctcvr`。这样梯度路径更平滑。

另一个常见误解是“共享越多越好”。并不是。ESMM 通常共享 embedding，但塔头分开，就是因为点击和转化虽然相关，但目标并不相同。点击偏“吸引力”，转化偏“成交意愿”，把上层也完全共享，往往会让任务互相干扰。

从资源角度看，ESMM 的代价主要在两点：

- 必须拿到全曝光样本，数据量会明显变大
- 双塔结构比单任务 CTR 稍重，但通常仍在可控范围内

如果你的系统连全曝光日志都没有，只保留了点击和订单日志，那 ESMM 的前提就不成立了。

---

## 替代方案与适用边界

ESMM 不是唯一方案。是否使用它，要看你的数据条件和偏差假设。

可以把它和几类常见方案放在一起看：

| 方法 | 训练样本 | 解决偏差方式 | 优点 | 局限 |
|---|---|---|---|---|
| 直接 CVR 模型 | 点击样本 | 不显式处理 | 简单 | 样本选择偏差明显 |
| CVR + IPS 校准 | 点击样本为主 | 用逆倾向加权修正 | 理论上更直接 | 倾向估计不准时方差大 |
| ESMM | 全曝光样本 | 用 CTR 与 CTCVR 联合建模 | 工程稳定、易复现 | 依赖完整曝光链路 |
| 序列式多任务模型 | 全链路样本 | 更强的行为顺序建模 | 表达能力更强 | 训练更复杂、调参更重 |

这里的 IPS（Inverse Propensity Score，逆倾向评分）可以白话理解成“给容易被选中的样本降权，给不容易被选中的样本加权”，用来矫正偏差。但它通常依赖额外的倾向估计，方差也可能很大。

一个新手可理解的对比是：

- 传统 CVR：只看“已经点了的人”，判断谁会买
- ESMM：同时看“所有曝光的人”，并把“点”和“点后买”一起建模

所以 ESMM 更像是“把 CVR 问题放回完整漏斗里重写”。

它更适合这些场景：

- 推荐、广告、电商排序
- 曝光→点击→转化链路明确
- 特征在 CTR 和 CVR 间可以共享
- 转化极稀疏，单独建模学不稳

它不太适合这些场景：

- 没有可靠曝光日志
- 点击不是转化前置条件
- 转化目标很多且路径分叉严重
- 任务间共享表示很弱，甚至相互冲突

如果业务已经进入更复杂阶段，比如多转化目标、长链路漏斗、强时序依赖，那么 OMoE、MMoE、AITM、序列因果建模等方法可能更合适。但作为“CTR/CVR 联合建模”的第一层基线，ESMM 仍然是非常实用的起点。

---

## 参考资料

| 来源 | 链接 | 重点 |
|---|---|---|
| ESMM 原论文 | https://arxiv.org/abs/1804.07931 | 原始问题定义、公式推导、Ali-CCP 数据背景 |
| CSDN 论文精读 | https://blog.csdn.net/Dby_freedom/article/details/112464380 | 用中文解释样本选择偏差、$pCTR/pCTCVR/pCVR$ 关系 |
| PaddleRec ESMM 文档 | https://www.aidoczh.com/paddlerec/en/models/multitask/esmm.html | 数据准备、模型复现、工程运行入口 |
| 博客园多任务学习笔记 | https://www.cnblogs.com/makefile/p/multi-task-learn.html | 共享 embedding、乘法约束、数值稳定性经验 |
| SIGIR 2018 论文索引 | https://colab.ws/articles/10.1145/3209978.3210104 | 论文发表信息与摘要入口 |
