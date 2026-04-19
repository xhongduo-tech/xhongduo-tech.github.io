## 核心结论

ESMM（Entire Space Multi-Task Model，全空间多任务模型）不是“再训练一个 CVR 模型”，而是把点击率预估和“点击且转化”预估放到同一个全曝光样本空间里联合学习，再通过概率关系间接得到 CVR。

推荐和广告系统里的现实分布通常是：曝光很多，点击很少，转化更少。如果传统 CVR 模型只看点击样本，它看到的数据已经不是完整曝光分布，而是“被点击行为筛过的一小部分样本”。线上排序却要给所有曝光候选打分，这就形成了训练空间和推理空间不一致。

ESMM 的核心公式是：

$$
pCTR(x)=P(y=1|x)
$$

$$
pCVR(x)=P(z=1|y=1,x)
$$

$$
pCTCVR(x)=P(y=1,z=1|x)=pCTR(x)\cdot pCVR(x)
$$

其中，`pCTR` 是点击概率，`pCVR` 是点击后的转化概率，`pCTCVR` 是曝光后同时发生点击和转化的联合概率。ESMM 直接训练 `pCTR` 和 `pCTCVR`，让两个任务共享底层表示，从而缓解 CVR 训练中的样本选择偏差和数据稀疏问题。

---

## 问题定义与边界

先定义三个变量：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| `x` | 曝光样本特征 | 用户、商品、上下文、位置、时间等输入信息 |
| `y` | 是否点击 | `y=1` 表示用户点击了这次曝光 |
| `z` | 是否转化 | `z=1` 表示用户在点击后完成购买、注册等目标行为 |

这里有一个前提：`z=1` 必须依赖 `y=1`。也就是说，在 ESMM 的经典设定里，转化是 post-click conversion，即点击后的转化。如果业务里存在“未点击也能转化”的归因逻辑，就不再严格满足这个建模前提。

用一个玩具例子说明。假设有：

- `100000` 次曝光
- `5000` 次点击
- `100` 次转化

那么：

$$
pCTR = \frac{5000}{100000}=5\%
$$

$$
pCVR = \frac{100}{5000}=2\%
$$

$$
pCTCVR = \frac{100}{100000}=0.1\%
$$

不能把 `100 / 100000` 直接叫作 `pCVR`。它是从曝光到“点击且转化”的联合概率，即 `pCTCVR`。真正的 CVR 是点击后的条件概率，即 `P(z=1|y=1,x)`。

| 事件 | 标签 | 概率关系 | 传统训练空间 | 线上推理空间 |
|---|---:|---|---|---|
| 曝光 | `x` | 样本入口 | 全部曝光 | 全部候选曝光 |
| 点击 | `y=1` | `P(y=1|x)` | 全部曝光可训练 | 全部候选曝光预测 |
| 转化 | `z=1` | `P(z=1|y=1,x)` | 通常只在点击样本训练 | 全部候选曝光需要打分 |
| 点击且转化 | `y∧z=1` | `P(y=1,z=1|x)` | 全部曝光可构造标签 | 全部候选曝光预测 |

传统 CVR 模型的问题在于：训练集只来自点击样本，推理时却面对全部曝光候选。这叫 sample selection bias，样本选择偏差，意思是训练数据分布被某个选择过程改变了。ESMM 的目标是缓解这种偏差，不是消除所有偏差。曝光日志本身仍可能有展示策略偏差、位置偏差、延迟反馈偏差。

---

## 核心机制与推导

ESMM 的结构是一个共享底层网络加两个任务头。任务头是神经网络最后用于输出某个目标的分支。

```text
输入特征 x
   |
共享底层表示 Shared Bottom
   |
   +--> CTR Head    --> pCTR
   |
   +--> CVR Head    --> pCVR
             |
pCTR × pCVR --> pCTCVR
```

注意，ESMM 训练时通常监督的是 `pCTR` 和 `pCTCVR`，不是在点击样本子集上单独监督 `pCVR`。`pCVR` 通过乘法关系进入 `pCTCVR` 的训练目标。

根据条件概率公式：

$$
P(y=1,z=1|x)=P(y=1|x)\cdot P(z=1|y=1,x)
$$

也就是：

$$
pCTCVR(x)=pCTR(x)\cdot pCVR(x)
$$

训练目标只需要两项二分类交叉熵。交叉熵是分类任务中衡量“预测概率”和“真实标签”差距的常用损失函数。

$$
L = CE(y,pCTR) + CE(y \land z,pCTCVR)
$$

其中：

$$
CE(t,p)=-t\log(p)-(1-t)\log(1-p)
$$

最小数值推导如下：若一个样本群体的点击率是 `5%`，点击后的转化率是 `2%`，那么曝光后同时点击且转化的概率是：

$$
5\% \times 2\% = 0.1\%
$$

这不是数学技巧，而是把“点击后的条件概率”转成“全曝光空间上的联合概率”。这样模型可以用所有曝光样本构造 `y∧z` 标签：没点击的样本，`y=0`，自然 `y∧z=0`；点击但没转化，`y=1,z=0`，所以 `y∧z=0`；点击且转化，`y=1,z=1`，所以 `y∧z=1`。

| 方案 | CVR 训练样本 | 监督信号 | 主要问题 |
|---|---|---|---|
| 传统 CVR | 只用点击样本 | `z` | 训练和推理空间不一致，样本少 |
| ESMM | 全部曝光样本 | `y` 与 `y∧z` | 仍需处理校准、延迟转化、归因噪声 |
| 线上排序 | 全部候选曝光 | 需要 CTR、CVR、CTCVR 等分数 | 依赖训练目标和业务目标一致 |

---

## 代码实现

最小实现需要六个部分：输入特征、共享底层网络、CTR 输出层、CVR 输出层、`pCTCVR = pCTR * pCVR`、联合损失。

伪代码流程是：

```text
读取一批曝光样本 x, y, z
前向计算 shared = shared_network(x)
计算 pCTR = ctr_head(shared)
计算 pCVR = cvr_head(shared)
计算 pCTCVR = pCTR * pCVR
构造 ctcvr_label = y * z
计算 loss = BCE(y, pCTR) + BCE(ctcvr_label, pCTCVR)
反向传播并更新参数
推理时使用 pCVR = pCTCVR / pCTR，并做数值保护
```

下面是一个可运行的 Python 玩具实现，只演示标签构造、概率关系和损失计算，不依赖深度学习框架：

```python
import math

def binary_cross_entropy(label, prob, eps=1e-12):
    prob = min(max(prob, eps), 1 - eps)
    return -label * math.log(prob) - (1 - label) * math.log(1 - prob)

def esmm_forward(p_ctr, p_cvr):
    p_ctcvr = p_ctr * p_cvr
    return p_ctr, p_cvr, p_ctcvr

def esmm_loss(y, z, p_ctr, p_cvr):
    p_ctr, p_cvr, p_ctcvr = esmm_forward(p_ctr, p_cvr)
    ctcvr_label = y * z
    return (
        binary_cross_entropy(y, p_ctr)
        + binary_cross_entropy(ctcvr_label, p_ctcvr)
    )

# 玩具例子：100000 曝光、5000 点击、100 转化
ctr = 5000 / 100000
cvr = 100 / 5000
p_ctr, p_cvr, p_ctcvr = esmm_forward(ctr, cvr)

assert abs(p_ctr - 0.05) < 1e-12
assert abs(p_cvr - 0.02) < 1e-12
assert abs(p_ctcvr - 0.001) < 1e-12

# 标签构造：未点击样本不可能是点击且转化
assert 0 * 0 == 0
assert 1 * 0 == 0
assert 1 * 1 == 1

# 推理阶段：从 pCTCVR 和 pCTR 恢复 pCVR，需要避免除以 0
def recover_cvr(p_ctcvr, p_ctr, eps=1e-8):
    return p_ctcvr / max(p_ctr, eps)

assert abs(recover_cvr(0.001, 0.05) - 0.02) < 1e-12

loss = esmm_loss(y=1, z=1, p_ctr=0.05, p_cvr=0.02)
assert loss > 0
```

真实工程例子是电商广告排序。系统会先从海量商品里召回候选，再用排序模型估计点击和转化。假设某商品在某用户面前的 `pCTR=0.08`，`pCVR=0.03`，那么：

$$
pCTCVR = 0.08 \times 0.03 = 0.0024
$$

这个 `0.24%` 表示“曝光后点击且转化”的概率。排序系统可以继续结合出价、利润、库存、用户体验约束生成最终排序分数。

---

## 工程权衡与常见坑

ESMM 能缓解稀疏和样本选择偏差，但它不是完整的因果修正方案，也不会自动修好所有数据问题。

| 问题表现 | 原因 | 规避方式 |
|---|---|---|
| 线上 CVR 分数整体偏高或偏低 | 概率校准不足 | 在验证集做 calibration，如 Platt scaling 或 isotonic regression |
| `pCTCVR` 被当成 CVR 使用 | 混淆联合概率和条件概率 | 明确定义 `pCVR=P(z=1|y=1,x)` |
| 训练效果离线好、线上差 | 训练和线上特征分布不一致 | 固定特征口径，监控训练-线上特征漂移 |
| 转化标签前后不一致 | 归因窗口变化 | 统一“点击后 N 天内转化”的定义 |
| 正样本极少，模型不稳定 | 转化事件稀疏 | 调整采样、加权、时间窗和校准策略 |
| 负样本被错误截断 | 延迟转化尚未回传 | 留出等待窗口，避免把未来会转化的样本标成负例 |

转化窗口尤其重要。若业务定义是“点击后 7 天内购买算转化”，一个用户第 10 天购买就会被标成负例；若定义是“点击后 30 天内购买算转化”，同一条行为会被标成正例。标签一变，`pCVR` 和 `pCTCVR` 的含义就变了，模型结果不能直接对比。

还要注意归因规则。一个用户可能点击多个广告后购买，转化归给最后一次点击、第一次点击，还是多次点击分摊，会产生不同训练标签。ESMM 只规定概率分解方式，不替业务决定归因口径。

---

## 替代方案与适用边界

ESMM 适合“曝光极多、点击稀疏、转化更稀疏”的推荐和广告场景。例如电商广告排序中，候选商品数量很大，点击率通常只有几个百分点，购买转化率更低。此时只用点击样本训练 CVR，样本量少且分布偏；ESMM 利用全曝光日志，通常是更稳的基线。

但如果转化定义模糊、回传强延迟、归因极不稳定、标签噪声极高，ESMM 可能不够。例如跨端广告中，用户在手机点击广告，三周后在电脑购买，且中间接触过多个渠道。此时问题已经超出“点击后转化率建模”，需要更完整的归因、延迟反馈、反事实或因果校正。

| 方案 | 传统 CVR | ESMM | ESCM^2 |
|---|---|---|---|
| 训练空间 | 点击样本 | 全部曝光样本 | 全部曝光样本 |
| 核心目标 | 直接拟合点击后的转化 | 联合学习 CTR 与 CTCVR | 在 ESMM 基础上处理更复杂偏差 |
| 偏差处理 | 基本不处理样本选择偏差 | 缓解样本选择偏差和稀疏性 | 进一步考虑反事实偏差或估计偏差 |
| 实现复杂度 | 低 | 中 | 较高 |
| 适用场景 | 点击样本充足、分布稳定 | 推荐、广告、电商排序 | 对偏差校正要求更高的研究或大规模系统 |

ESMM 的工程定位是重要基线，不是终点。实际系统中常见做法是先用 ESMM 建立可解释、可上线、可监控的多任务框架，再根据业务问题加入校准、延迟反馈建模、样本重加权、因果修正或更复杂的多目标排序。

---

## 参考资料

阅读顺序建议：如果只想快速上手，先看 DeepCTR 的接口文档，再读 ESMM 原始论文的模型结构部分；如果要做研究，先读原始论文，再读 ESCM^2，重点比较它们对偏差来源的假设差异。

1. [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://doi.org/10.1145/3209978.3210104)
2. [Entire Space Multi-Task Model arXiv Version](https://arxiv.org/abs/1804.07931)
3. [ESCM^2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation](https://doi.org/10.1145/3477495.3531972)
4. [DeepCTR ESMM Documentation](https://deepctr-doc.readthedocs.io/en/v0.9.2/deepctr.models.multitask.esmm.html)
