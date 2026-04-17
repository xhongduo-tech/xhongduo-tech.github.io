## 核心结论

对抗训练的定义很直接：在训练阶段，不只让模型看正常样本，还主动为每个样本构造“最难回答的一小圈邻域样本”，再要求模型在这些最坏样本上也保持正确预测。这里的“邻域”可以理解为原始输入附近允许出现的小范围改动，“最坏样本”就是在这个范围内最容易把模型带偏的输入。

它的核心目标通常写成一个 min-max 问题：

$$
\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\max_{\|\delta\|_p \le \epsilon}\mathcal{L}(f_\theta(x+\delta), y)\right]
$$

这行公式的意思是：外层最小化是在更新模型参数 $\theta$，内层最大化是在找扰动 $\delta$，也就是找“最会捣乱”的输入改动。训练完成后，模型不是只会答标准题，而是更擅长在恶意扰动附近保持稳定输出。

一个玩具例子是猫狗分类。假设一张猫图的像素每个位置都允许上下浮动 $0.03$，训练前先让攻击算法去找一组最容易把“猫”改成“狗”的像素扰动，再用这张被故意“调坏”的猫图继续训练模型。久而久之，模型学到的不是“这张图长得像训练集里的猫”，而是“即使附近有恶意微调，它仍然是猫”。

| 阶段 | 执行主体 | 目标 |
|---|---|---|
| 内层 `max` | 攻击器，例如 FGSM、PGD | 在约束范围内制造最坏扰动，使损失尽量大 |
| 外层 `min` | 模型训练器，例如 SGD、Adam | 用这些最坏样本更新参数，使损失重新变小 |
| 训练结果 | 模型参数 | 在输入邻域内更稳定，不容易被小扰动带偏 |

对安全与对齐场景也是同样逻辑。把“像素扰动”换成“越狱提示词、边界样本、诱导性上下文”，对抗训练就变成：训练时持续给模型喂最会越狱的提示，让它学会在这些攻击附近仍输出拒答、降权或安全完成。真正的难点不是“多塞一点攻击样本”，而是防止模型只记住历史攻击模板，遇到没见过的新攻击又失效。

---

## 问题定义与边界

要理解对抗训练，先要明确它解决的不是“所有攻击”，而是“给定威胁模型下的攻击”。威胁模型就是你事先规定攻击者能做什么、不能做什么。没有这个边界，“鲁棒性”就是一句空话。

标准形式仍是：

$$
\min_\theta \mathbb{E}_{(x,y)}\left[\max_{\|\delta\|_p\leq\epsilon} \mathcal{L}(f_\theta(x+\delta),y)\right]
$$

其中：

- $\delta$ 是扰动，意思是攻击者对输入施加的改动。
- $\|\delta\|_p\le\epsilon$ 是约束，意思是改动不能无限大，只能在一个规定半径内活动。
- $\epsilon$ 是攻击预算，可以白话理解成“攻击者最多能改多少”。

如果采用 $L_\infty$ 约束，那么 $\|\delta\|_\infty \le \epsilon$ 的意思是：每个维度都只能上下浮动不超过 $\epsilon$。对图片来说，就是每个像素最多只能调亮或调暗一点点。若 $\epsilon=0.03$，可以把它理解为“给每个像素画一个小框，攻击只能在框里移动”。

这一定义的边界很重要，因为它同时限制了结论的适用范围。

| 维度 | 常见取值 | 含义 |
|---|---|---|
| 输入空间 | 图像、文本嵌入、状态向量 | 攻击在哪个空间里改输入 |
| 范数约束 | $L_\infty$、$L_2$ | 攻击改动的形状和度量方式 |
| 预算 $\epsilon$ | 例如 `0.03`、`8/255` | 攻击最大强度 |
| 攻击算法 | FGSM、PGD、多步搜索 | 攻击有多强 |
| 标签保持假设 | 原标签不变 | 扰动后仍认为语义没变 |

这里有一个初学者最容易忽略的点：对抗训练学到的是“在某种约束下更稳”，不是“对任何变化都稳”。例如你只按 $L_\infty$ 小扰动训练，结论通常不能直接外推到模糊、雨雪、压缩损伤，更不能直接外推到自然语言越狱提示。因为这些攻击不一定能被“小范数扰动”准确描述。

在大模型安全里，这个边界更加明显。文本提示词攻击通常不是“每个 token 改一点点”，而是改写结构、插入上下文、构造角色扮演或跨轮诱导。所以如果把经典对抗训练原样照搬，很可能只提升了“模板内攻击”的防御能力，却没有覆盖“语义级攻击”。

---

## 核心机制与推导

对抗训练可以拆成两步：先造敌人，再学会扛住敌人。

第一步是内层最大化，即找扰动。最常见的方式是用梯度法，因为梯度告诉我们“往哪个方向改输入，损失涨得最快”。FGSM 是一步法，PGD 是多步法。PGD 常写成：

$$
x^{(t+1)}=\Pi_{\|\cdot-x\|_p\le\epsilon}\left(x^{(t)}+\alpha \cdot \mathrm{sign}(\nabla_x \ell(f_\theta(x^{(t)}),y))\right)
$$

这里：

- $\alpha$ 是步长，意思是每次往攻击方向走多远。
- $\mathrm{sign}$ 是取符号，意思是只关心梯度方向，不关心绝对大小。
- $\Pi$ 是投影，意思是如果走出允许攻击范围，就拉回约束集合内。

“投影”这个词第一次看容易抽象，可以把它理解成“走路时不能翻出围栏，翻出去了就把人拽回围栏内”。PGD 比 FGSM 强，就强在它不是只推一下，而是多次试探、每次纠偏，因此更接近“最坏扰动”。

第二步是外层最小化，即更新模型参数。做法反而很普通：把刚才生成的对抗样本当成训练输入，继续做常规反向传播，让模型在这些难样本上的损失下降。

训练循环的逻辑可以写成：

1. 取一个 batch 的干净样本 $(x,y)$。
2. 固定模型参数，用 FGSM 或 PGD 找出对抗样本 $x_{adv}$。
3. 用 $x_{adv}$ 和原始标签 $y$ 计算损失。
4. 对模型参数做梯度下降。

这个过程像一个双人博弈：

- 攻击器负责找最容易出错的点。
- 模型负责把这些点重新学对。

玩具例子可以继续用猫狗分类。假设模型原本靠“耳朵边缘的一圈亮度”识别猫。攻击器会优先微调那一圈像素，让模型把猫误判成狗。模型随后被迫改掉这种脆弱依赖，转而学习更稳定的特征，例如整体脸型、眼睛位置或多区域组合特征。也就是说，对抗训练的价值不只是“多见几张难图”，而是逼模型放弃脆弱捷径。

真实工程里，这套机制经常不是直接作用在原始像素，而是作用在更高层的攻击空间。例如自动驾驶感知系统会在模拟环境中系统生成雾、雪、眩光、压缩伪影等扰动。它们未必严格是 $L_p$ 小球，但思想一致：训练时持续暴露在难条件下，让部署时的错误边界更平滑、更可控。

从优化角度看，普通训练只在点 $x$ 上把损失压低；对抗训练试图把 $x$ 周围一个小区域内的最大损失也压低。换句话说，普通训练优化的是单点表现，对抗训练优化的是邻域内最坏表现。这也是为什么它经常能改善安全性，但也更难、更贵。

---

## 代码实现

下面先给一个最小可运行的玩具实现。它不是深度学习框架版，而是用 NumPy 手写一个二维线性分类器，用 FGSM 风格扰动演示“先攻击，再训练”的流程。代码里的 `assert` 可以直接运行。

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_and_grad_w(X, y, w):
    logits = X @ w
    probs = sigmoid(logits)
    eps = 1e-12
    loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    grad = X.T @ (probs - y) / len(X)
    return loss, grad

def grad_x_single(x, y, w):
    p = sigmoid(x @ w)
    return (p - y) * w

def fgsm_attack(X, y, w, eps=0.2):
    X_adv = X.copy()
    for i in range(len(X)):
        gx = grad_x_single(X[i], y[i], w)
        X_adv[i] = X[i] + eps * np.sign(gx)
    return X_adv

# 玩具数据：二维点分类
X = np.array([
    [2.0, 1.0],
    [1.5, 1.2],
    [-1.0, -1.5],
    [-1.2, -0.8],
], dtype=float)
y = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)

w_clean = np.zeros(2)
w_adv = np.zeros(2)
lr = 0.5

# 普通训练
for _ in range(200):
    _, grad = loss_and_grad_w(X, y, w_clean)
    w_clean -= lr * grad

# 对抗训练
for _ in range(200):
    X_adv = fgsm_attack(X, y, w_adv, eps=0.2)
    _, grad = loss_and_grad_w(X_adv, y, w_adv)
    w_adv -= lr * grad

# 在对抗样本上评估
X_test_adv = fgsm_attack(X, y, w_clean, eps=0.2)

def accuracy(X_eval, y_eval, w):
    preds = (sigmoid(X_eval @ w) >= 0.5).astype(float)
    return np.mean(preds == y_eval)

acc_clean_model = accuracy(X_test_adv, y, w_clean)
acc_adv_model = accuracy(X_test_adv, y, w_adv)

assert acc_adv_model >= acc_clean_model
print("clean model adversarial acc:", acc_clean_model)
print("adv-trained model adversarial acc:", acc_adv_model)
```

这个例子省略了神经网络，但机制完整：

- `fgsm_attack(...)` 是内层攻击，负责生成 `X_adv`。
- `loss_and_grad_w(X_adv, y, w_adv)` 是外层训练，负责让模型在攻击样本上把损失降下来。

如果换成 PyTorch 或 JAX，逻辑也类似。下面是更贴近工程实践的 PGD 训练伪代码：

```python
for x, y in loader:
    x_adv = pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=10)  # inner max
    logits = model(x_adv)
    loss = cross_entropy(logits, y)  # outer min
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

其中 `pgd_attack` 的核心一般是：

```python
def pgd_attack(model, x, y, eps, alpha, steps):
    x_adv = x.clone().detach()
    x_adv = x_adv + uniform_noise(-eps, eps)  # 随机初始化，减少陷入固定攻击模式
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = cross_entropy(model(x_adv), y)
        grad = autograd(loss, x_adv)
        x_adv = x_adv.detach() + alpha * sign(grad)
        x_adv = clip(x_adv, x - eps, x + eps)  # 投影回扰动域
        x_adv = clip(x_adv, 0.0, 1.0)          # 保证像素合法
    return x_adv
```

对大模型安全而言，代码形态会变，但结构不变。例如你可以维护一个“攻击提示池”，每个训练 batch 先生成或检索最容易诱导违规输出的提示，再用安全目标函数训练模型。此时 `pgd_attack` 不再是改像素，而可能是“重写 prompt”“插入角色扮演前缀”“拼接越狱上下文”。本质仍是 inner max 生成最坏样本，outer min 让模型在这些样本上学会安全响应。

---

## 工程权衡与常见坑

对抗训练最常见的误解是：“把攻击样本加进训练集，鲁棒性自然就上去了。”这句话只说对了一半。真正困难的地方在于，你加入的攻击样本是否足够强、足够多样、足够接近部署时的真实攻击。

第一个权衡是算力成本。假设普通训练一次前向加一次反向记为 1 个单位成本，PGD 用 $K$ 步时，训练成本通常近似放大到 $K+1$ 倍。因为每一步攻击都要额外算梯度。若 $K=7$ 到 $10$，训练成本上升 5 到 10 倍并不夸张。

$$
\text{Cost}_{AT} \approx (K+1)\cdot \text{Cost}_{clean}
$$

第二个权衡是干净精度和鲁棒精度的拉扯。干净精度就是正常样本上的表现，鲁棒精度就是攻击样本上的表现。很多情况下，鲁棒性上去以后，干净样本精度会下降，因为模型被迫放弃一部分“在训练集上很有效、但对攻击脆弱”的特征。

第三个坑是鲁棒过拟合。它的白话解释是：模型学会了防住“练过的拳路”，但遇到没练过的拳路还是挨打。比如你只用固定参数的 PGD 训练，模型可能对这组 PGD 很强，但对不同步长、不同初始化、不同失真类型的攻击仍然脆弱。安全与对齐场景中，这对应“模型会拒绝那几种常见越狱模板，但换个措辞、换个多轮上下文又被绕过去”。

下面是常见方案对比：

| 方法 | 时间成本 | 干净精度 | 鲁棒性 | 过拟合风险 |
|---|---|---|---|---|
| 干净训练 | 低 | 通常最高 | 低 | 对攻击极脆弱 |
| 标准对抗训练 | 高 | 常有下降 | 高于干净训练 | 会对特定攻击模板过拟合 |
| TRADES | 中高 | 可调 | 通常较稳 | 仍需关注攻击多样性 |

一个新手能立刻理解的例子是天气扰动。假设你只在“雾天模拟”上做对抗训练，模型可能在雾里变稳，但上线后遇到雪、逆光、污渍时依旧大幅退化。原因不是训练没做，而是威胁模型过窄。大模型越狱也是同一个问题：只针对“忽略之前所有指令”这种模板训练，部署后对“角色扮演”“翻译转述”“多轮诱导”仍可能失守。

真实工程里，常见规避策略有三类：

- 增强攻击多样性：不同初始化、不同步长、不同损失函数、不同攻击器混合。
- 做早停和独立验证：不要只看训练攻击上的损失，要看独立攻击集上的鲁棒指标。
- 结合正则或辅助目标：避免模型在鲁棒训练后出现明显的表示塌缩或干净精度过低。

还有一个很实际的坑是“梯度遮蔽”。它的意思是模型看起来很难攻击，不是因为真的更稳，而是因为梯度变得不可靠，导致你用的攻击器失效了。判断是否出现梯度遮蔽，通常要换更强的白盒攻击、更长步数、更强初始化，甚至换黑盒迁移攻击验证。否则你可能只是“把测量工具搞坏了”，并没有真正提升鲁棒性。

---

## 替代方案与适用边界

对抗训练不是唯一方案，也不是任何场景下的最优方案。它适合“你能明确攻击空间，并且愿意为更高训练成本买单”的任务。如果算力不足、攻击空间不稳定、或者你更关心干净精度和部署效率，替代方案往往更实用。

TRADES 是一个常见替代。它不是只最小化对抗样本上的交叉熵，而是在干净样本的分类损失之外，再加一个“干净输出分布和对抗输出分布要接近”的 KL 项。KL 散度可以白话理解成“两个概率分布差多远”。其典型形式是：

$$
\mathcal{L}_{TRADES}
=
\mathcal{L}_{CE}(f_\theta(x), y)
+
\beta \cdot D_{KL}(f_\theta(x)\,\|\,f_\theta(x_{adv}))
$$

这里 $\beta$ 是权衡系数：

- $\beta$ 大，说明更重视鲁棒性。
- $\beta$ 小，说明更重视干净样本精度。

对应伪代码如下：

```python
for x, y in loader:
    x_adv = pgd_attack_for_kl(model, x)  # 让对抗样本尽量偏离干净输出分布
    logits_clean = model(x)
    logits_adv = model(x_adv)

    loss_ce = cross_entropy(logits_clean, y)
    loss_kl = kl_div(log_softmax(logits_adv), softmax(logits_clean).detach())
    loss = loss_ce + beta * loss_kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

对资源有限的团队，TRADES 往往比“全强度 PGD 对抗训练”更容易调，因为它把“准确率下降多少、鲁棒性提升多少”的旋钮显式暴露出来了。

再往外看，还有几类替代思路：

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| TRADES / MART | 需要精细平衡干净精度与鲁棒性 | 可调权衡项，训练目标更细 | 仍依赖攻击构造 |
| 随机平滑 | 需要概率意义下的认证鲁棒性 | 有数学保证 | 适用攻击类型有限，推理更慢 |
| 形式化验证 | 小模型或高价值模块 | 可给出严格安全边界 | 成本高，难扩展到大模型 |
| 数据增强 | 自然分布扰动明显的任务 | 实现简单 | 不等于最坏情况鲁棒性 |
| 行为级安全训练 | LLM、多轮 agent | 更贴近真实越狱 | 很难穷举攻击空间 |

在多模态和大模型场景中，经典 $L_p$ 扰动假设经常不够用。比如越狱防护更接近“语义对抗”，攻击者会改写任务包装、伪造开发者指令、操纵工具调用上下文。这些攻击并不在一个固定范数球里。因此，直接把图像领域的 PGD 套过来，往往只能解决很窄的一部分问题。更合理的做法通常是把对抗训练升级为“行为级对抗训练”：让模型在训练中反复暴露于不同类型的越狱样本、角色扮演、提示注入、多轮诱导和工具滥用案例，并在这些案例上优化安全目标。

所以适用边界可以总结成一句话：如果攻击空间能被定义得足够清楚，对抗训练是把鲁棒性直接写进目标函数的强方法；如果攻击空间本身不断变化，尤其是语义级攻击，对抗训练必须和多样化评估、策略约束、行为级数据构造一起使用，单独依赖它通常不够。

---

## 参考资料

- EmergentMind, “Adversarial Training”: https://www.emergentmind.com/topics/adversarial-training  
- OpenAI, “Testing robustness against unforeseen adversaries”: https://openai.com/index/testing-robustness  
- Bitdefender InfoZone, “What is Adversarial Training”: https://www.bitdefender.com/en-us/business/infozone/what-is-adversarial-training.html
