## 核心结论

对抗样本的生成，目标是在“输入几乎不变”的前提下，让模型判断出错、信心下降，或偏向攻击者指定的输出。这里的“扰动”可以理解为对输入做非常小但有方向的修改。视觉任务里，这个方向通常直接来自损失函数对输入的梯度，也就是“往哪里改，最容易让模型犯错”。最常见的公式是：

$$
x' = x + \epsilon \cdot \mathrm{sign}(\nabla_x L(\theta, x, y))
$$

这就是 FGSM。它表示：对输入 $x$ 的每个维度，沿着让损失 $L$ 变大的符号方向，统一走一步，步长由 $\epsilon$ 控制。

玩具例子可以直接看一个像素。假设某张“猫”的图片中，一个归一化像素值是 $x=0.30$，该位置梯度符号为 $+1$，扰动预算设为 $\epsilon=0.01$，那么新像素就是 $x'=0.31$。这个变化对人眼几乎不可见，但如果许多像素都按同样原则被系统性修改，模型可能把“猫”误判成“狗”。

真实工程里，自动驾驶识别交通标志、医疗影像分类、内容安全审核、垃圾邮件检测、LLM 安全对齐都面临同一个问题：模型使用的是高维输入，而高维空间中存在许多对人类不敏感、但对模型极其敏感的方向。对抗样本生成，本质上就是寻找这些方向。

只看“攻击成功”是不够的。评估必须同时看四类指标：攻击成功率、语义或视觉保持度、可读性或自然度、查询成本。否则得到的只是“能骗过模型”的输入，不一定是“现实里可用”的攻击。

---

## 问题定义与边界

形式化地说，对抗样本生成是在约束条件下解一个优化问题。约束表示“输入不能改太多”，目标表示“尽量让模型更容易出错”。常见写法是：

$$
\max_{\delta} \; L(\theta, x+\delta, y)
\quad
\text{s.t.} \quad \|\delta\|_\infty \le \epsilon
$$

这里的 $\delta$ 是扰动，$\|\delta\|_\infty \le \epsilon$ 表示任一维度都不能改超过 $\epsilon$。$L_\infty$ 可以白话理解为“每个像素最多能改多少”，它适合图像，因为图像天然是连续数值。

但这个定义不能直接照搬到文本。文本输入是离散的，意思是 token 或字符不能像像素那样加上 0.01。你不能把“cat”改成“cat+0.01”。所以文本攻击通常把“最小扰动”重新解释为：尽量少改词、尽量保持语义、尽量不破坏语法、尽量少查询模型。

下面这个表能直接看出图像和文本边界的差异：

| 维度 | 图像对抗样本 | 文本对抗样本 |
|---|---|---|
| 输入类型 | 连续数值，像素可微 | 离散符号，token 不可直接微调 |
| 常见约束 | $L_\infty$、$L_2$ 半径 | 同义替换、字符扰动、句法保持 |
| 白盒能力 | 可直接取 $\nabla_x L$ | 常需借助 embedding、打分或搜索 |
| 主要优化方法 | FGSM、PGD、CW | 同义词搜索、字符编辑、GCG、有限差分 |
| 评估重点 | 成功率、扰动范数、视觉相似度 | 成功率、语义相似度、可读性、查询成本 |

白盒和黑盒也必须分清。白盒攻击是攻击者知道模型结构、参数或至少能拿到梯度；黑盒攻击则拿不到内部梯度，只能通过查询输出分数、标签或文本响应来逼近敏感方向。白盒更强，黑盒更贴近真实部署环境。

再看一个新手容易理解的工程边界。自动驾驶攻击中，攻击者不能把“限速 60”路牌直接改成“停车”，那已经不是对抗扰动，而是显式篡改。真正有意义的设定是：只能在视觉上几乎不被注意的范围内修改，比如每个像素变化不超过 $0.01$，或者只贴上小块贴纸，要求人类司机仍然把它看作原标志。

在 LLM 或文本审核系统中，边界同样重要。一个攻击提示如果已经变成乱码、语法严重崩坏、或者只靠上万次查询才成功，那么它的工程价值就明显下降。对齐风险评估必须把“能攻破”与“代价多大”一起看。

---

## 核心机制与推导

FGSM 的推导来自一阶近似。把损失函数在输入点 $x$ 附近做泰勒展开：

$$
L(\theta, x+\delta, y) \approx L(\theta, x, y) + \delta^\top \nabla_x L(\theta, x, y)
$$

如果约束是 $\|\delta\|_\infty \le \epsilon$，那么让这个线性项最大的最优选择，就是每一维都取梯度符号方向的最大允许步长，于是得到：

$$
\delta^\star = \epsilon \cdot \mathrm{sign}(\nabla_x L(\theta, x, y))
$$

因此 FGSM 是“在 $L_\infty$ 球内的一步最优线性近似攻击”。

FGSM 快，但它只有一步，容易受局部曲率影响。PGD 可以理解为“FGSM 的多步版本 + 每一步都拉回约束集合”。公式是：

$$
x^{t+1} = \Pi_{B_\epsilon(x)}\left[x^t + \alpha \cdot \mathrm{sign}(\nabla_x L(\theta, x^t, y))\right]
$$

其中 $\Pi_{B_\epsilon(x)}$ 是投影算子，白话解释就是“如果走出允许扰动范围，就把点拉回半径为 $\epsilon$ 的可行域里”。$\alpha$ 是每一步的小步长。PGD 往往还会加随机初始化，即从 $x+\delta_0$ 开始，$\delta_0$ 在约束球内随机采样，这样更不容易被局部假象骗到。

可以把 PGD 流程理解成三步循环：

1. 计算当前输入对损失的梯度。
2. 沿着让损失增大的方向前进一步。
3. 把结果投影回允许扰动范围，并裁剪到合法像素区间如 $[0,1]$。

这就是为什么 PGD 常被视为更可靠的基准攻击。它不是因为公式更复杂，而是因为它更接近“在约束范围里真的认真找过了”。

文本攻击更难，因为输入离散。常见做法有三类。

第一类是同义替换。先找出重要 token，再从候选同义词里挑能最大幅度提高损失的替换词。

第二类是字符扰动。比如插入、删除、替换、交换字符，适合针对拼写鲁棒性较差的系统，但可读性通常更差。

第三类是梯度引导搜索。核心想法是：虽然 token 离散，但 embedding 是连续的，可以先在连续空间估计“哪个位置、哪个方向最敏感”，再把这个方向映射回离散候选。GCG 就是这一类方法的代表。它的白话解释是：一次只改一个位置，贪心地选当前最能提升攻击目标的 token，然后重复。

可写成坐标更新的形式：

$$
j^\star = \arg\max_j \Delta L_j, \qquad
v^\star = \arg\max_{v \in \mathcal{V}_j} L(x_{1:j-1}, v, x_{j+1:n})
$$

这里 $j^\star$ 是当前最值得修改的位置，$\mathcal{V}_j$ 是该位置的候选 token 集合。GCG 的重点不在“全局最优”，而在“每一步找到最值钱的局部修改”。

玩具例子可以这样理解。原句是“这是一份正常请求，请返回公开信息。”如果系统对某些位置特别敏感，那么 GCG 会先定位最敏感 token，再从候选集中尝试替换，例如把某个普通连接词换成更具诱导性的词，或者在尾部附加一段对齐逃逸后缀。每次替换后都重新查询模型，看目标分数是否上升。这个过程和 PGD 的相似点在于：都在逐步逼近更高损失区域；区别在于 PGD 在连续空间里走步子，GCG 在离散候选里做坐标搜索。

真实工程例子是 LLM 安全过滤绕过。攻击者不会“修改参数”，而是构造 prompt suffix，通过多轮候选替换，让模型从“拒答”逐渐转向“部分遵从”甚至“完全输出”。从优化角度看，这与视觉中的多步 PGD 是同一类问题：在受限空间里迭代寻找最脆弱方向。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不依赖深度学习框架，而是用逻辑回归损失模拟 FGSM 和 PGD 的核心思想。这里的“梯度”就是损失对输入的导数，白话解释是“输入每个维度再增大一点，会让损失增加多少”。

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_and_grad_x(w, b, x, y):
    """
    二分类交叉熵对输入 x 的梯度
    y in {0, 1}
    """
    z = np.dot(w, x) + b
    p = sigmoid(z)
    loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    # dL/dx = (p - y) * w
    grad_x = (p - y) * w
    return loss, grad_x, p

def fgsm_attack(w, b, x, y, eps):
    _, grad_x, _ = loss_and_grad_x(w, b, x, y)
    x_adv = x + eps * np.sign(grad_x)
    return np.clip(x_adv, 0.0, 1.0)

def pgd_attack(w, b, x, y, eps, alpha, steps):
    x_adv = x.copy()
    for _ in range(steps):
        _, grad_x, _ = loss_and_grad_x(w, b, x_adv, y)
        x_adv = x_adv + alpha * np.sign(grad_x)
        # 投影回 L_inf 球
        x_adv = np.minimum(np.maximum(x_adv, x - eps), x + eps)
        # 裁剪回合法输入范围
        x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv

# 一个二维玩具样本
w = np.array([3.0, -2.0])
b = -0.1
x = np.array([0.20, 0.80])
y = 1
eps = 0.05

loss_before, grad_before, p_before = loss_and_grad_x(w, b, x, y)
x_fgsm = fgsm_attack(w, b, x, y, eps)
loss_fgsm, _, p_fgsm = loss_and_grad_x(w, b, x_fgsm, y)

x_pgd = pgd_attack(w, b, x, y, eps=eps, alpha=0.02, steps=5)
loss_pgd, _, p_pgd = loss_and_grad_x(w, b, x_pgd, y)

assert np.max(np.abs(x_fgsm - x)) <= eps + 1e-9
assert np.max(np.abs(x_pgd - x)) <= eps + 1e-9
assert loss_fgsm >= loss_before - 1e-9
assert loss_pgd >= loss_before - 1e-9
assert 0.0 <= p_before <= 1.0 and 0.0 <= p_fgsm <= 1.0 and 0.0 <= p_pgd <= 1.0
```

上面代码已经体现了图像攻击的四个核心动作：算梯度、加扰动、投影、裁剪。迁移到 PyTorch 或 TensorFlow，只是把手写梯度换成自动求导。

如果写成更接近真实训练框架的伪代码，FGSM/PGD 可以概括成：

```python
# FGSM
x.requires_grad_(True)
logits = model(x)
loss = criterion(logits, y)
loss.backward()
x_adv = x + eps * x.grad.sign()
x_adv = x_adv.clamp(0.0, 1.0)

# PGD
x_adv = x + uniform(-eps, eps)
for _ in range(steps):
    x_adv.requires_grad_(True)
    loss = criterion(model(x_adv), y)
    loss.backward()
    x_adv = x_adv + alpha * x_adv.grad.sign()
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)  # 投影
    x_adv = x_adv.clamp(0.0, 1.0).detach()
```

文本侧不能直接做 `x + eps`，实现通常是“候选生成 + 打分筛选 + 语义约束”。下面是 GCG 风格的简化伪代码：

```python
def gcg_like_attack(tokens, candidate_fn, score_fn, semantic_ok, max_steps):
    current = tokens[:]
    best_score = score_fn(current)

    for _ in range(max_steps):
        improved = False
        best_local = current
        best_local_score = best_score

        for pos in range(len(current)):
            candidates = candidate_fn(current, pos)
            for cand in candidates:
                trial = current[:]
                trial[pos] = cand
                if not semantic_ok(tokens, trial):
                    continue
                s = score_fn(trial)
                if s > best_local_score:
                    best_local = trial
                    best_local_score = s
                    improved = True

        current = best_local
        best_score = best_local_score

        if attack_success(current):
            break
        if not improved:
            break

    return current
```

这里有三个工程要点。

第一，`candidate_fn` 不能无约束地产生候选，否则搜索空间会爆炸。常见做法是只给高重要度位置生成少量候选词。

第二，`semantic_ok` 是文本攻击中非常重要的 guard。它可以是 embedding 相似度阈值、困惑度阈值、语法规则过滤，作用是防止“攻击成功但句子已坏掉”。

第三，`attack_success` 和 `early stop` 一起控制查询成本。黑盒系统中，每一次模型调用都是真实成本，可能是时间、API 费用或风控风险。

---

## 工程权衡与常见坑

对抗样本生成真正难的不是把公式写出来，而是保证评估可靠、攻击可复现、结果对现实有意义。

下面这个表总结最常见的坑：

| 坑 | 触发条件 | 结果 | 缓解策略 |
|---|---|---|---|
| 梯度遮蔽 | 只测 FGSM，模型含非平滑防御或数值饱和 | 误以为模型很稳健 | 用 PGD 多步 + 随机初始化复测 |
| 灾难性过拟合 | 训练时只依赖单步攻击 | 训练集附近看似稳健，真实攻击下失效 | 使用多步对抗训练或更强攻击验证 |
| 扰动越界 | 忘记投影或裁剪 | 攻击成功但不满足约束 | 每步执行投影和合法范围裁剪 |
| 文本语义漂移 | 只追求成功率，不做语义检查 | 句子变味、不可读 | 加相似度、困惑度、语法过滤 |
| 黑盒成本失控 | 候选过多、无 early stop | 查询数爆炸 | 候选剪枝、优先队列、成功即停 |
| 只看 ASR | 忽略自然度与预算 | 指标虚高，难落地 | 联合报告成功率、相似度、查询成本 |

先说 FGSM 和 PGD 的关系。FGSM 适合做快速基线，因为便宜；但它太便宜，也因此容易误导。某些模型在 FGSM 下表现很好，并不代表真的稳健，只可能说明梯度在局部不好用，也就是梯度遮蔽。新手常见误区是：训练几轮 FGSM 对抗训练后，发现 FGSM 打不动了，就以为模型安全了。实际上换成 PGD 加随机初始化后，脆弱面可能立刻重新暴露。

再说文本。文本攻击最容易出现“指标看起来成功，语义实际上坏了”。例如把一句正常请求替换成充满奇怪拼写、低频词、无意义重复的句子，模型可能确实被干扰，但这类样本既不自然，也不符合真实使用环境。文本攻击如果不同时报告语义相似度、人工可读性或困惑度，结论很容易失真。

真实工程例子是内容安全审核系统。如果一个对抗提示只有在 3000 次查询后才成功，且最终文本读起来像乱码，那么它对线上系统的风险优先级往往不如“20 次查询内、语义自然、稳定复现”的攻击。安全团队在排序修复优先级时，必须看综合成本，而不是只看单一成功率。

还有一个常见坑是把“白盒研究结论”直接拿来解释“黑盒现实风险”。白盒攻击给的是上界压力测试，说明模型存在多大潜在脆弱性；黑盒攻击更接近现实，但受限于查询预算、反馈粒度和防滥用机制。两者不能混为一谈。

---

## 替代方案与适用边界

如果无法直接取梯度，可以使用查询式黑盒方法。NES 和 SPSA 都属于这类。它们通过随机方向采样或有限差分，估计“往哪个方向改，输出最容易变化”。白话解释是：既然拿不到真实梯度，就用多次试探近似出一个梯度。

在黑盒文本系统中，也可以采用“有限差分 + 候选词库搜索”的组合。做法通常是：先定位重要位置，再从同义词库、近义 token、常见拼写扰动中生成少量候选，逐个试探目标分数是否上升。这个方法不如白盒高效，但在真实部署环境里更可行。

还有一类是生成式方法，例如利用另一个模型直接生成高成功率候选，再由目标模型筛选。它适合复杂文本和多约束场景，但控制性较弱，且评估更依赖外部打分器。

从防御角度看，单纯做检测往往不够。更稳健的路线通常是对抗训练或认证方法。对抗训练是把攻击样本加入训练，让模型在这些扰动附近学得更稳；认证方法则尝试给出可证明的鲁棒区间，例如在某个 $L_\infty$ 半径内保证预测不变。白话解释是：前者靠经验强化，后者给数学保证，但通常更保守、代价更高。

可以用一个总表看适用边界：

| 方案 | 场景 | 优点 | 限制 |
|---|---|---|---|
| FGSM | 白盒、快速基线 | 便宜、实现简单 | 单步弱，易受梯度遮蔽影响 |
| PGD | 白盒、严谨评测 | 更强、更稳定 | 计算成本高于 FGSM |
| GCG/梯度引导搜索 | 文本、LLM、离散输入 | 可在离散空间逐步优化 | 查询与筛选成本高 |
| NES/SPSA | 黑盒、可查询分数 | 无需内部梯度 | 查询开销大，噪声敏感 |
| 对抗训练 | 训练期防御 | 实际有效、可部署 | 成本高，可能牺牲干净样本精度 |
| 认证鲁棒性 | 高安全场景 | 有理论保证 | 适用范围有限，性能代价大 |

因此，方法选择不能脱离约束条件。研究白盒脆弱性，用 PGD；做线上黑盒风险评估，用查询式方法；处理文本或 LLM，重点放在候选搜索与语义保持；做安全防御，优先考虑对抗训练与可认证边界，而不是只做输入检测。

---

## 参考资料

1. TensorFlow FGSM 教程：适合入门图像对抗样本，能直接理解 FGSM 定义、梯度方向和最基础的数值示例。  
   https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

2. Emergent Mind 关于 PGD 与 GCG 攻击的综述：适合建立“连续攻击”和“离散 token 攻击”的统一视角，也有 LLM 与物理场景示例。  
   https://www.emergentmind.com/topics/pgd-and-gcg-attacks

3. PMC 关于文本对抗攻击评估的论文综述：适合补齐文本侧评估指标，尤其是语义保持、可读性、查询成本这些工程上必须报告的维度。  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11920284/
