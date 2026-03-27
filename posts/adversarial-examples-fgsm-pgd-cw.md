## 核心结论

对抗样本生成的核心不是“随便加噪声”，而是在给定扰动预算内，系统地寻找最容易让模型出错的方向。这里的“扰动预算”就是允许改动的最大幅度，常用 $L_\infty$ 或 $L_2$ 范数来约束。FGSM、PGD、CW 都利用输入梯度，也就是“输入每个维度朝哪个方向改，会让损失变大得最快”的信息，但三者回答的问题不同：

| 攻击名 | 步数 | 是否投影 | 主要约束 | 核心思路 |
| --- | --- | --- | --- | --- |
| FGSM | 1 步 | 否 | 常见为 $L_\infty$ | 一次走到边界，快速制造误判 |
| PGD | 多步 | 是 | 常见为 $L_\infty$ / $L_2$ | 多次试探并回到约束球内，寻找更强攻击 |
| CW | 多步优化 | 通常用变量变换代替直接投影 | 常见为 $L_2$ | 把“误判且扰动小”写成联合优化问题 |

结论可以直接记成一句话：FGSM 是快但粗的单步攻击，PGD 是强基线攻击，CW 是更偏优化视角、往往能找到更小扰动的精细攻击。

玩具例子很简单。设一个灰度像素值 $x=0.5$，梯度为 $\nabla_x J=0.2$，预算 $\epsilon=0.05$。FGSM 会直接得到

$$
x' = x + \epsilon \cdot \mathrm{sign}(\nabla_x J)=0.5+0.05=0.55
$$

这个改动只有 0.05，但如果模型在该维度附近本来就接近决策边界，就可能导致分类翻转。

真实工程例子是图像分类鲁棒性评估。对一个在 VGG16 上原本预测为 “beagle” 的 ImageNet 样本，FGSM 往往能用一次扰动打断正确分类；PGD 因为会多步搜索，通常更稳定地把样本推入错误类别；CW 则进一步尝试以更小的可见扰动实现误判。工程上把三种攻击都跑一遍，是为了分别覆盖“快速冒烟测试”“强鲁棒性基准”“最小扰动分析”三个目的。

---

## 问题定义与边界

对抗样本问题可以严格写成：给定模型 $g(x)$、输入 $x$、真实标签 $y$，寻找扰动 $\delta$，使得模型在 $x+\delta$ 上预测错误，同时扰动足够小。

无目标攻击的常见形式是：

$$
\text{find } \delta \quad \text{s.t. } \arg\max g(x+\delta)\neq y,\quad \|\delta\|_p \le \epsilon,\quad x+\delta\in[0,1]^n
$$

这里几个符号先讲清楚：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $x$ | 原始输入 | 比如一张图片的像素向量 |
| $\delta$ | 扰动 | 加在输入上的小改动 |
| $\epsilon$ | 扰动预算 | 允许改动的最大幅度 |
| $J(\theta,x,y)$ | 损失函数 | 模型“错得有多严重”的量化 |
| $B_\epsilon(x)$ | 约束球 | 围绕原始输入、半径为 $\epsilon$ 的允许区域 |

$L_\infty$ 范数表示“所有维度中最大改动不能超过 $\epsilon$”。对白话来说，就是每个像素最多只能改这么多。$L_2$ 范数表示“整体改动能量有限”，允许有些维度改得稍大，但总量受控。

三类攻击的目标分别可以写成：

$$
\text{FGSM: } x' = x + \epsilon \cdot \mathrm{sign}(\nabla_x J(\theta,x,y))
$$

$$
\text{PGD: } x_{t+1} = \Pi_{B_\epsilon(x)}\left(x_t + \alpha \cdot \mathrm{sign}(\nabla_x J(\theta,x_t,y))\right)
$$

$$
\text{CW: } \min_\delta \|\delta\|_2^2 + c\cdot f(x+\delta)
\quad \text{s.t. } x+\delta\in[0,1]^n
$$

其中 $\Pi$ 是投影算子，意思是“如果走出了允许区域，就拉回去”。$f(\cdot)$ 是攻击目标函数，常见设计是让真实类别 logit 低于其他类别 logit。logit 可以理解为 softmax 之前的原始分类分数。

边界也要说清楚。本文主要讨论连续输入，尤其是图像。因为图像像素是实数，可以直接对输入求梯度。文本模型的 token 是离散符号，不能直接对“词本身”做无穷小改动，所以 FGSM/PGD/CW 的原始形式不能直接照搬到文本上。

---

## 核心机制与推导

FGSM 的推导最简单。它来自一阶线性近似。把损失在输入 $x$ 附近展开：

$$
J(\theta, x+\delta, y)\approx J(\theta,x,y)+\delta^\top \nabla_x J(\theta,x,y)
$$

如果约束是 $\|\delta\|_\infty \le \epsilon$，那要让线性项尽可能大，最优解就是让每一维都取与梯度同号的最大值，于是得到：

$$
\delta^\star = \epsilon \cdot \mathrm{sign}(\nabla_x J)
$$

这就是 FGSM。它的优点是只要一次反向传播，几乎是最便宜的白盒攻击。白盒的意思是“攻击者知道模型参数并能拿到梯度”。

但 FGSM 只有一步，它隐含假设是“局部线性近似已经足够好”。模型越非线性，这个假设越容易失效。于是就有了 PGD。

PGD 可以看成“很多个很小的 FGSM，再加约束修正”。每一步先沿让损失上升最快的方向走一小步，再投影回约束球：

$$
x_{t+\frac12} = x_t + \alpha \cdot \mathrm{sign}(\nabla_x J(\theta,x_t,y))
$$

$$
x_{t+1} = \Pi_{B_\epsilon(x)}(x_{t+\frac12})
$$

为什么要投影？因为多步累积后很容易超出预算。对于 $L_\infty$ 约束，投影通常就是逐维裁剪：

$$
x_{t+1} = \mathrm{clip}(x_{t+\frac12}, x-\epsilon, x+\epsilon)
$$

同时还要保证像素合法：

$$
x_{t+1}=\mathrm{clip}(x_{t+1}, 0, 1)
$$

PGD 之所以常被视为强基线，是因为它更接近“在约束集合内求最大损失”：

$$
\max_{\|\delta\|_p\le\epsilon} J(\theta,x+\delta,y)
$$

它不是严格求全局最优，但比 FGSM 更能逼近局部最坏情况。再加上 random restart，也就是从多个随机初始点开始重复 PGD，通常还能进一步提高成功率，因为非凸损失面里不同起点会落入不同局部极值。

CW 的视角进一步变化。它不再直接“最大化损失”，而是把“最小扰动”和“攻击成功”合成一个优化问题。以常见的无目标 $L_2$ 版本为例，可写成：

$$
\min_\delta \|\delta\|_2^2 + c \cdot f(x+\delta)
$$

其中一个常见的 $f$ 设计是：

$$
f(x') = \max\big(z_y(x') - \max_{i\neq y} z_i(x'), -\kappa\big)
$$

这里 $z_i(x')$ 是第 $i$ 类 logit，$\kappa$ 是置信度边界。直观上，只有当“其他某类分数超过真实类分数足够多”时，攻击才算充分成功。$\kappa$ 越大，通常样本越强，但也更难优化。

CW 的关键不是步子多，而是目标函数更精细。FGSM/PGD 更像“在预算内尽量把损失推高”，CW 更像“用尽可能小的代价完成误判”。所以 CW 常常能找到视觉上更不明显的样本，但计算更贵，还要调参数 $c$，常见做法是二分搜索。

如果用一个力学直觉总结三者：

| 方法 | 可以想成什么 |
| --- | --- |
| FGSM | 朝最陡方向猛推一次 |
| PGD | 小步多次试探，每步都不许越界 |
| CW | 明确优化“越小越好，但必须推过边界” |

---

## 代码实现

先用一个可运行的 Python 玩具代码把三者的差异讲清楚。这里不用深度学习框架，只构造一个二维线性分类器，并直接写出梯度。这样读者可以先看懂机制，再迁移到 PyTorch。

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def predict_label(w, b, x):
    # 二分类：logit >= 0 预测为 1，否则为 0
    return 1 if dot(w, x) + b >= 0 else 0

def logistic_loss_grad_x(w, b, x, y):
    # y in {0,1}; 使用 sigmoid + BCE 的输入梯度
    z = dot(w, x) + b
    p = 1.0 / (1.0 + math.exp(-z))
    coeff = p - y
    return [coeff * wi for wi in w]

def sign(v):
    return [1.0 if x > 0 else -1.0 if x < 0 else 0.0 for x in v]

def clip(v, lo, hi):
    return [max(lo_i, min(hi_i, x)) for x, lo_i, hi_i in zip(v, lo, hi)]

def fgsm(x, grad, eps):
    s = sign(grad)
    return [xi + eps * si for xi, si in zip(x, s)]

def pgd_linf(x, y, w, b, eps, alpha, steps):
    x0 = x[:]
    xt = x[:]
    for _ in range(steps):
        grad = logistic_loss_grad_x(w, b, xt, y)
        s = sign(grad)
        xt = [a + alpha * d for a, d in zip(xt, s)]
        xt = clip(xt, [xi - eps for xi in x0], [xi + eps for xi in x0])
    return xt

# 一个玩具分类器
w = [2.0, -1.0]
b = -0.2
x = [0.30, 0.40]
y = 1

orig = predict_label(w, b, x)
grad = logistic_loss_grad_x(w, b, x, y)
x_fgsm = fgsm(x, grad, eps=0.25)
x_pgd = pgd_linf(x, y, w, b, x, eps=0.25, alpha=0.05, steps=5)

# 原样本应被分类为正类
assert orig == 1

# FGSM/PGD 都必须满足 L_inf 约束
assert max(abs(a - b) for a, b in zip(x_fgsm, x)) <= 0.25 + 1e-9
assert max(abs(a - b) for a, b in zip(x_pgd, x)) <= 0.25 + 1e-9

# PGD 至少不应比 FGSM 更“离谱”地越界
assert all(abs(a - b) <= 0.25 + 1e-9 for a, b in zip(x_pgd, x))

print("orig:", x, "label:", orig)
print("fgsm:", x_fgsm, "label:", predict_label(w, b, x_fgsm))
print("pgd :", x_pgd, "label:", predict_label(w, b, x_pgd))
```

上面这个例子不是为了做强攻击，而是为了验证两个事实：第一，FGSM 就是一行“加符号”；第二，PGD 的本质是“循环 + 投影”。

迁移到真实工程时，FGSM 在 PyTorch 中的核心通常只有四步：

1. 输入张量 `x` 开启梯度。
2. 前向计算 loss。
3. 对 `x` 反向传播，拿到 `x.grad`。
4. 执行 `x_adv = clamp(x + eps * sign(x.grad))`。

简化版伪代码如下：

```python
# x: [B, C, H, W]
x.requires_grad_(True)
logits = model(x)
loss = criterion(logits, y)
loss.backward()

x_adv = x + eps * x.grad.sign()
x_adv = x_adv.clamp(0.0, 1.0)
```

PGD 只是把这个过程包进循环里，并加上投影：

```python
x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
x_adv = x_adv.clamp(0.0, 1.0)

for _ in range(steps):
    x_adv.requires_grad_(True)
    loss = criterion(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]

    x_adv = x_adv.detach() + alpha * grad.sign()
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    x_adv = x_adv.clamp(0.0, 1.0)
```

CW 会明显更长，因为它不是直接对 `x_adv` 做简单迭代，而是要定义扰动变量、构造联合损失、配优化器，还经常要对参数 $c$ 做多轮搜索。

真实工程例子是评估一个图像分类服务。流程通常是：

| 步骤 | 目的 |
| --- | --- |
| 先跑 FGSM | 快速检查模型是否对微小扰动极度脆弱 |
| 再跑 PGD | 给出更接近最坏情况的白盒结果 |
| 最后抽样跑 CW | 分析最小扰动误判样本，辅助定位边界问题 |

如果一个模型在 FGSM 下很稳，但在 PGD 下迅速失守，通常说明它对单步扰动有一定抵抗，但对局部多步搜索仍然脆弱；如果连 CW 都难以找到小扰动误判，才说明边界附近确实更稳健。

---

## 工程权衡与常见坑

先给一个工程复核表，最常见的问题基本都在这里。

| 问题 | 征兆 | 规避方式 |
| --- | --- | --- |
| FGSM 的 $\epsilon$ 过大 | 图像明显失真，评估失去意义 | 先按数据归一化尺度设预算，再做人眼检查 |
| FGSM 的 $\epsilon$ 过小 | 攻击几乎无效，误以为模型鲁棒 | 画出准确率随 $\epsilon$ 变化曲线 |
| PGD 步长 $\alpha$ 太大 | 震荡、效果不稳定 | 常用经验是 $\alpha < \epsilon$，并随步数协调 |
| PGD 步数太少 | 结果和 FGSM 差不多 | 增加迭代次数，看成功率是否继续上升 |
| 忽略 random restart | 高估模型鲁棒性 | 多起点重复攻击，取最坏结果 |
| 忘记投影或裁剪 | 扰动超预算或像素非法 | 每步都做 `project + clip` |
| CW 参数 $c$ 不合适 | 要么攻击失败，要么扰动过大 | 用二分搜索调 $c$ |
| 只报告攻击成功率 | 看不出扰动是否合理 | 同时报告范数、可见性、置信度变化 |

参数没有统一金科玉律，但可以先用下面的起点：

| 方法 | 常调参数 | 常见起点 |
| --- | --- | --- |
| FGSM | $\epsilon$ | 依据输入归一化范围选，例如图像常见小预算 |
| PGD | $\epsilon,\alpha,\text{steps},\text{restarts}$ | 小步长、多迭代、若干重启 |
| CW | $c,\kappa,\text{iterations},\text{lr}$ | 先小 $\kappa$，再逐步加大 |

有几个坑值得单独展开。

第一，数据预处理尺度容易搞错。很多模型输入不是 $[0,1]$，而是先减均值再除方差。此时 $\epsilon$ 不能直接照抄像素空间值，否则你以为自己在做“小扰动”，实际可能已经放大了很多倍。

第二，梯度遮蔽会误导评估。梯度遮蔽是“模型让梯度变得不稳定或无用，看起来不好攻，但并不真的稳健”。如果 FGSM 效果差、PGD 稍强、黑盒迁移攻击却又很强，就要警惕是不是出现了梯度遮蔽，而不是模型真的安全。

第三，对抗训练不能只看训练时用的那一种攻击。用 FGSM 做对抗训练，常常只能提升对 FGSM 的抗性；如果用更强的 PGD 评估，效果可能并不好。工程上通常把 PGD 视为更可信的训练内环或评估基线。

第四，CW 不适合全量高频上线评估。它太慢，更适合离线分析、论文实验、或对重点样本做精查，而不是替代 FGSM/PGD 的日常扫描。

---

## 替代方案与适用边界

如果输入是连续空间，比如图像、语音频谱、部分传感器数据，FGSM/PGD/CW 的原理都可以直接成立，因为“对输入求梯度”这件事本身是合法的。

但文本和大语言模型不是这样。token 是离散符号，不能把一个词“微调 0.03 个单位”。这时常见做法是把梯度先作用在 embedding 空间，或者改成离散坐标搜索。Greedy Coordinate Gradient，简称 GCG，就是这类方法的代表。它的白话解释是：每次只替换少量 token，优先选那些最能推动攻击目标的词。

这说明一个重要边界：PGD 的思想是通用的，PGD 的输入形式不是。通用的是“沿梯度方向逐步优化并受约束”，不通用的是“直接对原始输入做连续加法”。

可以把替代方案按场景分成下面几类：

| 场景 | 支持域 | 主要手段 |
| --- | --- | --- |
| 图像分类 | 连续像素空间 | FGSM、PGD、CW |
| 语音或频谱模型 | 连续信号空间 | PGD、CW 类优化 |
| 文本分类/LLM | 离散 token 空间 | GCG、token 替换、embedding 近似 |
| 黑盒服务 | 无梯度访问 | 迁移攻击、查询攻击、进化搜索 |

攻击选择也可以做成简单决策树：

| 目标 | 更合适的方案 |
| --- | --- |
| 快速发现明显脆弱性 | 先用 FGSM |
| 做强白盒鲁棒性评估 | 用 PGD，多次重启 |
| 找更小扰动样本 | 用 CW |
| 处理文本或 LLM | 用 GCG 或离散搜索 |
| 只能访问 API | 用黑盒攻击或迁移攻击 |

一个实用判断标准是：如果你关心“模型最坏情况下有多脆弱”，优先 PGD；如果你关心“最小要改多少才能骗过模型”，优先 CW；如果你关心“流水线里能不能快速扫出问题”，优先 FGSM。

---

## 参考资料

- True Geometry, “What is the core difference in approach between FGSM and PGD calculation?”
- MDPI, “Evaluating the Robustness...” 关于 ImageNet/VGG16 上 FGSM、PGD、CW 的实验分析
- PyTorch 官方教程, “Adversarial Example Generation (FGSM Tutorial)”
- CodeGenes, “Carlini-Wagner’s L2 Attack in PyTorch”
- Emergent Mind, “PGD and GCG Attacks” 与 “Greedy Coordinate Gradient (GCG) Attack”
- Emergent Mind, “Adversarial Machine Learning Techniques”
