## 核心结论

Representation Surgery 可以译为“表征手术”。白话说，它不是去改训练数据，也不是去改模型参数，而是在模型前向计算时，直接改一层或几层的隐状态，也就是模型内部用来表达语义的向量。

它解决的问题很具体：如果某种敏感属性，比如性别、种族、毒性、辱骂倾向，在隐状态里形成了明显的方向或子空间，那么可以直接对这部分几何结构动手术，削弱模型沿这条方向继续放大的能力。这样做的目标不是“让模型没有任何价值判断”，而是降低某些偏见或危险输出被激活的概率，同时尽量保住原来的主任务能力。

最常见的两类闭式变换是：

$$
f(H)=H+(\mu_1-\mu_0)
$$

以及

$$
f(H)=\mu_1+\Sigma_1^{1/2}\Sigma_0^{-1/2}(H-\mu_0)
$$

第一种只改均值，第二种同时改均值和协方差。均值就是一堆向量的平均位置；协方差可以理解为这些向量在各个方向上“散开”的形状。

玩具例子很直观。假设“毒性文本”的平均激活是 $(2,0)$，“普通文本”的平均激活是 $(0,0)$。如果当前生成时的隐状态靠近 $(2,0)$，我们就把它往 $(0,0)$ 平移。这样模型更不容易沿着“毒性方向”继续生成。这个思路本质上是在表示空间里修正轨迹，而不是重训整个模型。

真实工程里，它特别适合推理阶段快速迭代。比如你已经有一个线上大模型，不想为了“降低辱骂输出”重新微调一轮，就可以先离线统计若干层的激活均值和协方差，再在解码时按条件触发变换。它轻量、可回滚、可分层测试，但前提是你要承认：它主要处理的是“在表示空间里近似线性可分”的偏差，而不是所有复杂社会问题。

---

## 问题定义与边界

这里的“偏见”不能泛化成所有不喜欢的输出。工程上更可操作的定义是：某个敏感属性或风险属性，能在模型隐状态中被线性探针稳定识别，且这种可识别性会传导到最终输出行为。线性探针就是一个很简单的线性分类器，白话说，它像一把直尺，试图用一条超平面把两类激活分开。如果它很容易分开，说明该属性在表示空间里已经比较“显眼”。

因此，Representation Surgery 的问题边界是：

| 对象 | source 群体 | target 群体 | 统计量 | 变换类型 |
|---|---|---|---|---|
| 性别偏差 | 男性相关表示 | 女性相关表示 | $\mu_0,\mu_1$ | 均值对齐 |
| 毒性控制 | 毒性提示激活 | 中性提示激活 | $\mu_0,\mu_1,\Sigma_0,\Sigma_1$ | OLC |
| 身份属性匿名化 | 含敏感身份表示 | 去身份化表示 | 子空间均值/协方差 | 局部仿射变换 |

这里有两个关键限制。

第一，它作用在“激活几何”上，不直接改世界知识。如果模型训练语料本身已经把很多错误因果关系学进参数里，单纯靠一层平移不可能彻底修复。

第二，它更擅长处理“可分方向”，不擅长处理强非线性偏差。比如某种偏见必须结合多跳上下文、叙事语气、外部知识才能触发，这时简单的线性平移往往不够。

玩具例子可以把它想成在纸上移动一个点。男性样本和女性样本的平均激活是纸上的两个中心点，当前 token 的隐状态也是一个点。Representation Surgery 做的不是擦掉整张纸重画，而是把这个点往另一个中心推一小步或做一次仿射变换。

真实工程例子是内容安全。在自回归语言模型中，下一 token 的分布由当前隐状态决定。如果某些危险上下文使隐状态持续靠近“毒性簇”，那么在 logits 之前插入一个校正函数，可以降低后续 token 向辱骂词、威胁词、仇恨表达词扩散的概率。这个边界很重要：它是“降低风险概率”，不是“形式化保证绝对安全”。

---

## 核心机制与推导

先看最简单的均值对齐。设源分布的激活是 $H\sim(\mu_0,\Sigma_0)$，目标分布是 $(\mu_1,\Sigma_1)$。如果我们只想找一个尽量小的改动，把源类的平均位置推到目标类的平均位置，同时不引入额外旋转和缩放，那么最自然的选择就是固定线性部分为恒等映射 $W=I$，只做平移：

$$
f(H)=H+b
$$

要求变换后的均值满足：

$$
\mathbb{E}[f(H)] = \mathbb{E}[H+b] = \mu_0+b = \mu_1
$$

于是直接得到：

$$
b=\mu_1-\mu_0
$$

所以

$$
f(H)=H+(\mu_1-\mu_0)
$$

这一步的含义很直接：如果你只允许“整体搬家”，那最小改动就是把源均值搬到目标均值。

再看同时对齐均值和协方差的情况。我们考虑仿射变换：

$$
f(H)=A(H-\mu_0)+\mu_1
$$

这样设计是为了先把源分布中心化，再经线性映射 $A$ 改形状，最后平移到目标中心。为了让变换后协方差等于目标协方差，需要：

$$
A\Sigma_0A^\top=\Sigma_1
$$

在高斯近似和二次代价下，一个标准选择是 Wasserstein 意义下的最优线性映射。若简化到可交换或经白化处理后的形式，可以写成：

$$
A=\Sigma_1^{1/2}\Sigma_0^{-1/2}
$$

于是得到：

$$
f(H)=\mu_1+\Sigma_1^{1/2}\Sigma_0^{-1/2}(H-\mu_0)
$$

这就是常见的 OLC 形式。白化这个词的白话解释是：先把原分布拉成“各方向方差都差不多”的标准球，再按目标分布重新拉伸。

玩具例子：

- 源均值 $\mu_0=(0,0)$，目标均值 $\mu_1=(1,1)$
- 若只做均值对齐，则 $f(H)=H+(1,1)$
- 若再设 $\Sigma_0=I,\Sigma_1=4I$，则 $\Sigma_1^{1/2}=2I,\Sigma_0^{-1/2}=I$
- 所以
  $$
  f(H)=(1,1)+2(H-0)
  $$

这意味着不只平移到右上角，还会把分布整体拉伸 2 倍，让“散布形状”也更像目标。

为什么这能降低偏见暴露？一个常见解释是：很多高层语义在隐空间里近似线性可分。既然线性探针能识别“毒性”或“性别相关特征”，那沿这些方向做反向几何修正，就能降低这类特征在后续层被继续读出的概率。它不是严格删除概念，而是降低概念的可读性和支配力。

真实工程例子可以看对话安全。假设你从模型某一层收集了两批激活：

- 一批来自高风险辱骂提示
- 一批来自普通讨论提示

你分别估计出 $\mu_{\text{toxic}},\Sigma_{\text{toxic}}$ 和 $\mu_{\text{normal}},\Sigma_{\text{normal}}$。解码时，如果当前隐状态更像毒性分布，就应用从 toxic 到 normal 的仿射映射。这时模型参数没动，但下一 token 的 logits 已经基于“更接近正常分布”的表示计算，输出风险就会下降。

---

## 代码实现

下面用一个最小可运行的 Python 例子说明均值对齐、协方差对齐和门控触发。这里不用深度学习框架，只演示核心数学。

```python
import numpy as np

def mean_shift(h, mu_source, mu_target):
    h = np.asarray(h, dtype=float)
    mu_source = np.asarray(mu_source, dtype=float)
    mu_target = np.asarray(mu_target, dtype=float)
    return h + (mu_target - mu_source)

def inv_sqrtm_psd(mat, eps=1e-8):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def sqrtm_psd(mat, eps=1e-8):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

def olc_transform(h, mu_source, sigma_source, mu_target, sigma_target):
    h = np.asarray(h, dtype=float)
    mu_source = np.asarray(mu_source, dtype=float)
    mu_target = np.asarray(mu_target, dtype=float)
    sigma_source = np.asarray(sigma_source, dtype=float)
    sigma_target = np.asarray(sigma_target, dtype=float)

    A = sqrtm_psd(sigma_target) @ inv_sqrtm_psd(sigma_source)
    return mu_target + A @ (h - mu_source)

def l2_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x - y)

def gated_surgery(h, mu_risk, mu_safe, threshold):
    # 只有当当前表示足够接近风险均值时才触发，减少对正常样本的扰动
    if l2_distance(h, mu_risk) < threshold:
        return mean_shift(h, mu_risk, mu_safe)
    return np.asarray(h, dtype=float)

# 玩具例子：均值平移
h = np.array([2.2, 0.1])
mu_toxic = np.array([2.0, 0.0])
mu_normal = np.array([0.0, 0.0])

h_new = mean_shift(h, mu_toxic, mu_normal)
assert np.allclose(h_new, np.array([0.2, 0.1]))

# 门控例子：接近风险均值才处理
h_gate = gated_surgery(h, mu_toxic, mu_normal, threshold=1.0)
assert np.allclose(h_gate, np.array([0.2, 0.1]))

h_safe = np.array([5.0, 5.0])
h_safe_after = gated_surgery(h_safe, mu_toxic, mu_normal, threshold=1.0)
assert np.allclose(h_safe_after, h_safe)

# OLC 例子：mu0=(0,0), mu1=(1,1), Sigma0=I, Sigma1=4I
mu0 = np.array([0.0, 0.0])
mu1 = np.array([1.0, 1.0])
sigma0 = np.eye(2)
sigma1 = 4.0 * np.eye(2)

h2 = np.array([1.0, 2.0])
h2_new = olc_transform(h2, mu0, sigma0, mu1, sigma1)
assert np.allclose(h2_new, np.array([3.0, 5.0]))

print("all asserts passed")
```

如果把它映射到真实模型推理流程，伪代码通常是这样：

```python
for each decoding step:
    h = hidden_state[layer_id][last_token]

    risk_score = distance(h, mu_risk)
    if risk_score < threshold:
        h = mu_safe + sqrt_sigma_safe @ inv_sqrt_sigma_risk @ (h - mu_risk)

    logits = lm_head(rest_of_model(h))
    next_token = sample(logits)
```

这里有三个实现点最关键。

第一，统计怎么收集。通常是离线准备两组数据集，例如“高毒性提示”和“中性提示”，在固定层上抽取最后一个 token 或若干关键 token 的隐藏向量，估计 $\mu,\Sigma$。如果样本少，协方差会不稳定，需要加对角正则，也就是给矩阵加一个很小的 $\lambda I$。

第二，在哪一层动手。越浅的层更接近词法和局部句法，越深的层更接近语义和决策。通常中后层更适合做安全 steering，但也更容易影响风格和可读性，所以要做层级敏感性实验。

第三，什么时候触发。不是所有 token 都要改。一个常见办法是距离门控，只在 $\|h-\mu_{\text{risk}}\|_2$ 足够小时触发。还有一种做法是用一个轻量探针先打分，只有高风险时才干预。这样能减少对普通生成的副作用。

真实工程例子里，一个在线聊天系统可以在“回复生成前的最后几层”插入这个钩子。若用户输入包含侮辱、仇恨、威胁上下文，风险探针升高，就用从 toxic 到 safe 的映射修正隐状态；若只是普通争论，则不触发或只做弱平移。这样可以兼顾风险控制和正常表达能力。

---

## 工程权衡与常见坑

Representation Surgery 的优点是轻量，但它的坑也很集中，主要来自统计假设和部署方式。

| 风险 | 原因 | 典型表现 | 缓解措施 |
|---|---|---|---|
| 语义漂移 | 真实隐状态不满足高斯近似 | 回复变得生硬、答非所问 | 分层测试，优先局部子空间映射 |
| 过矫正 | 固定偏移过大 | 温和批评也被压成空话 | 设置门控阈值和最大改变量 |
| 覆盖不足 | 偏差是非线性或跨层联动 | 某些绕写毒性仍通过 | 联合探针、规则、监控 |
| 规范化偏见 | 默认把某类表示推向“标准类” | 掩盖少数群体正常表达 | 明确 target 定义，避免单一规范中心 |
| 数值不稳 | 协方差矩阵病态 | 变换爆炸，生成异常 | 加 $\lambda I$，截断小特征值 |

最常见的误区是把它当成“去偏见万能键”。实际上它只是对表示空间做局部几何校正。只要风险来源超出这个局部线性结构，比如世界知识错误、长链推理错误、工具调用流程错误，手术就无能为力。

另一个常见坑是全局统一偏移。比如你把所有看起来像“毒性”的表示都推向同一个安全中心，结果可能把正常但尖锐的批评、司法语境中的引述、学术讨论中的敏感词分析，一并压扁。这会让模型更“安全”，但也更“迟钝”。工程上更稳妥的方法是分上下文触发：只在高风险提示、开放式续写、低置信度约束场景中启用较强变换。

玩具例子可以这样理解：如果你规定“凡是接近红色区域的点，都统一推回蓝色中心”，那么有些只是“粉红色”的正常点也会被误伤。表示空间里的误伤，最终会表现为语气过度保守、信息缺失或者语义失真。

真实工程里，还要考虑评估指标。只看毒性分数是不够的，至少要同时看：

- 主任务准确率是否下降
- 困惑度是否恶化
- 输出多样性是否被压缩
- 某些群体相关话题是否被过度拒答
- 跨语言、跨提示风格是否稳定

如果这些副作用没有一起评估，你可能只是把问题从“明显毒性”转移成“隐性能力损失”。

---

## 替代方案与适用边界

它不是唯一方案。工程上通常有三类主流手段：表示干预、参数微调、规则化过滤。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Representation Steering / Surgery | 无需回传参数，部署快，可按层按条件触发 | 主要依赖线性可分结构，作用范围有限 | 推理期快速试验、低成本安全加固 |
| 微调 / LoRA | 影响更深，可学复杂非线性行为 | 训练成本高，回滚慢，可能遗忘原能力 | 有标注数据、可接受训练流程时 |
| 规则化过滤 | 简单、透明、易审计 | 易被绕过，误杀高，语义理解弱 | 作为最后一道兜底防线 |

适用边界可以概括成一句话：如果问题主要表现为“某类风险语义在隐空间里有稳定方向”，Representation Surgery 很合适；如果问题是复杂推理错误、工具链误用、长期对话策略偏差，就不能只靠它。

玩具例子是分层防护。先用 steering 把“明显毒性方向”压下去，再用关键词黑名单兜底极端词，再用输出审核模型拦最后一层。三者组合比只靠一个更稳，因为它们覆盖的是不同失效模式。

真实工程例子是客服机器人。对线上大模型直接做全量微调，周期长、风险高；只做规则拦截，又容易把正常投诉一起挡掉。这时可以先在推理期加一层 representation steering，把侮辱性延续概率降下来；若触发高风险，再让规则引擎和审计模块接管。这样成本最低，回滚也最快。

还有一个边界经常被忽略：公平性不等于把所有群体映射到同一个“标准中心”。如果 target 选择不当，系统可能把多数群体表达方式默认成“正常”，反而在工程上放大规范化偏见。因此，真实项目里更合理的做法往往不是单中心对齐，而是：

- 针对明确风险概念做局部抑制
- 只在高风险上下文触发
- 用多中心或子空间方法替代单一中心
- 配合上线监控持续校准

---

## 参考资料

1. Singh 等，`Representation Surgery: Theory and Practice of Affine Steering`，ICML 2024。涵盖内容：理论框架、仿射 steering 目标、实验验证。  
2. Emergent Mind，`Representation Steering Methods`。涵盖内容：均值对齐、OLC、Wasserstein 映射、部署方式与工程案例。  
3. Emergent Mind，`Representation Steering in Neural Models`。涵盖内容：表示干预的直观机制、方向控制、与其他 steering 方法的关系。
