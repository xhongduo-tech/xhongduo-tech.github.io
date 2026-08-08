---
title: Adam 及其变体（AdamW、AMSGrad）
date: 2026-08-07
---

# Adam 及其变体（AdamW、AMSGrad）

<div class="epigraph">
<p>站在巨人的肩膀上，才能看得更远。</p>
<footer>—— 艾萨克 · 牛顿（Isaac Newton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§8.5.3、李沐《动手学深度学习》§4.8 ｜ 2026-08-07</p>
</div>

## 为什么从 Adam 及其变体开始

AdaGrad 给了「逐维缩放」，动量给了「惯性」，但它们各管一摊。**Adam（Adaptive Moment Estimation）**（Kingma & Ba, 2015）把两者**合二为一**：用**一阶矩**（梯度均值，动量）决定「往哪走」，用**二阶矩**（梯度平方均值）决定「走多快」——一阶矩管方向、二阶矩管步长。它一经提出就以「好上手、少调参、收敛快」成为深度学习的事实默认优化器。

但 Adam 不是终点，而是起点：**AdamW**（解耦权重衰减，Loshchilov & Hutter, 2019）修复了 Adam 里「权重衰减被逐维缩放搅乱」的问题，如今是大模型预训练的标配；**AMSGrad**（Reddi 等, 2018）修复了 Adam 在某些凸问题上「不收敛」的理论缺陷。本节把「Adam 是什么 → 两个著名变体修什么 → 怎么选」一次讲透。<span class="marginnote">Adam 的论文有 10 万+ 引用，是深度学习史上被引用最多的算法论文之一。它的成功很大程度来自「工程友好」：几乎不用调参（默认 lr=1e-3、β₁=0.9、β₂=0.999 通吃大多数任务）、对尺度不敏感、收敛快。但也正因「太省心」，它把「学习率调度」的负担悄悄藏进了默认值里——理解内部机制才能正确使用。</span>

## 1 Adam 的更新规则：动量 + 自适应步长

Adam 维护两个滑动平均（都初始化为零）：

$$
\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t \qquad (\text{一阶矩，动量})
$$

$$
\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t\odot\boldsymbol{g}_t \qquad (\text{二阶矩，RMSProp})
$$

其中 $\beta_1=0.9$、$\beta_2=0.999$ 是默认的矩衰减系数。由于 $\boldsymbol{m},\boldsymbol{v}$ 从零初始化，早期被「低估」，Adam 做了**偏差校正（bias correction）**：

$$
\hat{\boldsymbol{m}}_t = \frac{\boldsymbol{m}_t}{1-\beta_1^t}, \qquad
\hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1-\beta_2^t}
$$

最终更新：

$$
\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_{t-1} - \eta\,\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}
$$

三步拆解这个「看起来复杂」的更新：

- **第一步，看方向**：$\hat{\boldsymbol{m}}_t$ 是动量的偏差校正版——决定**往哪走**（历史梯度的加权平均）。
- **第二步，看步长**：$\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}$ 是「一阶矩 ÷ 二阶矩的根」——二阶矩缩放步长，让每个维度按「典型梯度大小」归一化。
- **第三步，看为什么默认值通吃**：$\beta_1=0.9$ 让动量窗口约 10 步，$\beta_2=0.999$ 让二阶矩窗口约 1000 步——「短期方向 + 长期尺度」的组合，配合逐维归一化，对大多数任务都稳健。<span class="marginnote">「一阶矩管方向、二阶矩管步长」是理解 Adam 的总纲。它等价于「<strong>对梯度做归一化后再带动量</strong>」：$\frac{\hat{m}}{\sqrt{\hat{v}}}$ 把梯度量级归一化到 $\pm 1$ 左右，再让动量决定「该方向的累积强度」。这也是为什么 Adam 的学习率几乎独立于损失尺度——它内部自带尺度校准。</span>

## 2 偏差校正：为什么必须有

Adam 的 $\boldsymbol{m}, \boldsymbol{v}$ 从零初始化。训练第一步：$\boldsymbol{m}_1 = 0.1\boldsymbol{g}_1$——**只有真实值的一成**。若不做校正，早期步长被严重低估，训练「起步极慢」。偏差校正把早期估计「放大」到无偏：

$$
\mathbb{E}[\hat{\boldsymbol{m}}_t] = \mathbb{E}[\boldsymbol{g}_t], \qquad \mathbb{E}[\hat{\boldsymbol{v}}_t] = \mathbb{E}[\boldsymbol{g}_t^2]
$$

$t=1$ 时，$\frac{1}{1-\beta_1^1} = \frac{1}{0.1} = 10$，恰好把「低估的 0.1」拉回真实量级。<span class="marginnote">偏差校正只在早期显著（$1-\beta_1^t$ 随 $t$ 指数趋近 1）。$t=10$ 时校正系数约 1.1，$t=100$ 时约 1.0。所以「省掉校正」在长训练里影响渐小，但在短训练、小步数任务里是「起步慢 + 训练不足」的隐患——实现 Adam 时不要「偷懒省略」。</span>

**易错点一：** 实现 Adam 若忘记偏差校正（或实现错误），症状是「训练前期收敛异常慢」。这是「自己手写优化器」最常见的 bug 之一——框架的 `torch.optim.Adam` 类已内置校正，但自定义实现时极易遗漏。

## 3 AdamW：解耦权重衰减

Adam 与 L2 正则化（权重衰减）的组合有一个**隐蔽问题**（见《L2 正则化与权重衰减》）：在 Adam 里，$\lambda\boldsymbol{\theta}$（L2 正则项）的惩罚梯度也会被「逐维缩放」。这意味着**权重的衰减量随该维度梯度大小而变**——本该「均匀衰减」的 L2 惩罚被扭曲了。

**AdamW（Adam with decoupled weight decay）**的修复极其简单：把权重衰减**从梯度路径中剥离**，直接在更新时对权重做衰减：

$$
\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_t - \eta\Big(\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}\Big) - \eta\lambda\,\boldsymbol{\theta}_{t-1}
$$

与「把 $\lambda\boldsymbol{\theta}$ 加进梯度」的区别：AdamW 的衰减项**不被** $\sqrt{\hat{\boldsymbol{v}}_t}$ 缩放，它始终以「固定比例」压向零——这才是「权重衰减」的本义。<span class="marginnote">为什么 AdamW 在实践上更好？Loshchilov & Hutter 实验发现：Adam+L2 的等效衰减「随梯度大小波动」，导致正则强度不一致；AdamW 让衰减恒定，训练更稳、泛化更好。如今 <strong>LLaMA、GPT 等主流大模型全部使用 AdamW</strong>——这个「看起来只是实现细节」的改动，实际是预训练工程里最重要的一行。</span>

**易错点二：** 「weight_decay」在 PyTorch 的 `torch.optim.Adam` 与 `torch.optim.AdamW` 里语义不同。`torch.optim.Adam` 是「L2 进梯度」，`torch.optim.AdamW` 是「解耦衰减」。跨框架、跨优化器移植超参数时，weight_decay 的**语义**必须确认。

## 4 AMSGrad：修复「步长单调」的理论缺陷

Reddi 等（2018）指出 Adam 的一个理论问题：在**某些凸问题**上，Adam 会「忘掉」过去的大梯度——当近期梯度变小，二阶矩 $\boldsymbol{v}_t$ 变小，步长 $\frac{\eta}{\sqrt{\hat{v}_t}}$ 反而**变大**，可能「跳过头」，导致**不收敛**。

**AMSGrad** 的修复：二阶矩取「历史最大值」而非滑动平均：

$$
\hat{\boldsymbol{v}}_t = \max(\hat{\boldsymbol{v}}_{t-1}, \boldsymbol{v}_t), \qquad
\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_{t-1} - \eta\,\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}
$$

$\max$ 操作保证 $\hat{\boldsymbol{v}}_t$ **单调不减**，步长单调不增——从理论上恢复了收敛保证。<span class="marginnote">AMSGrad 的实践定位很微妙：<strong>它修的是凸问题的理论缺陷，在深度学习的非凸实践里改善通常微小甚至不稳定</strong>。它的价值更多是「理论完整性」——提醒我们「自适应方法的收敛证明」与「实际训练」之间存在落差。工程上，AdamW 的使用率远高于 AMSGrad。</span>

**易错点三：** 不要把「Adam 在凸问题不收敛」误读为「Adam 训练会发散」。深度学习任务是非凸的，Adam 实践收敛良好；AMSGrad 是为「理论正确」的强迫症准备的，不是「Adam 用不了」时的替代。

## 5 公式解析：Adam 的「信任比」

Adam 的每一步更新，本质是在维护一个「**信任比（signal-to-noise ratio）**」：

$$
\text{SNR}_t = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}}
$$

- **第一步，看分子**：$\hat{\boldsymbol{m}}_t$ 是梯度的估计（信号），度量「该方向平均往哪走、多确定」。
- **第二步，看分母**：$\sqrt{\hat{\boldsymbol{v}}_t}$ 是梯度的大小（噪声尺度），度量「该方向抖动多剧烈」。
- **第三步，看更新语义**：$\Delta\theta_t = -\eta\,\text{SNR}_t$。**信号强、噪声小**的方向（梯度一致）→ 信任比大 → 大步走；**噪声大**的方向（梯度正负乱跳）→ 信任比小 → 小步走。**Adam 自动在「确定的方向快走、不确定的方向慢走」**——这是它「自适应」的最精确含义。<span class="marginnote">「信任比」还解释了 Adam 的一个行为：训练后期梯度变得稀疏时，$\hat{m}$ 与 $\sqrt{\hat{v}}$ 一起变小，信任比保持稳定——Adam 不会像 AdaGrad 那样「步长死锁」，也不会像固定步长 SGD 那样「后期震荡」。这个「信号/噪声比恒定」的性质，是它长期训练稳定性的数学根源。</span>

## 6 选型与实践建议

| 优化器 | 核心特点 | 首选场景 |
| --- | --- | --- |
| SGD + 动量 | 泛化好、可解释 | CNN、成熟 pipeline |
| Adam | 好上手、少调参 | 通用默认 |
| AdamW | 解耦权重衰减、泛化更稳 | 大模型、Transformer |
| AMSGrad | 理论收敛保证 | 理论验证、凸子问题 |

**实践要点**：

1. **默认起步**：AdamW + 预热 + 余弦退火（大模型配方）；小任务 Adam 即可。
2. **学习率**：Adam 家族通常 $3\times10^{-4}$ 到 $1\times10^{-3}$ 起步；比 SGD 的典型值小 1–2 个数量级。
3. **权重衰减**：大模型常用 $0.01$–$0.1$（AdamW 的 $\lambda$），比 L2 时代的 $10^{-4}$–$10^{-3}$ 大得多——因为「解耦」后语义变了。
4. **验证集裁决**：AdaGrad/RMSProp/Adam/SGD 没有绝对胜者，最终以验证集为准。

**易错点四：** Adam 的 $\epsilon$ 也影响步长。默认 $10^{-8}$ 在「梯度极小」时保护除零；过大的 $\epsilon$ 会「抑制」所有步长。大模型训练里偶有把 $\epsilon$ 调大到 $10^{-6}$ 以稳定 AdamW 的做法——但不要随意动默认值。<span class="marginnote">「Adam 忘掉 SGD 的手感」是一个真实现象：Adam 的逐维归一化让「学习率」的语义从「绝对步长」变成「相对信任」——所以 Adam 学习率不能直接与 SGD 学习率对比。「先 SGD 找到好参数、再 Adam 微调」之类的混合策略在强化学习等领域仍有使用，但深度监督学习的主流已全面倒向 AdamW。</span>

## 7 小结

- **Adam** = 动量（一阶矩 $\boldsymbol{m}$）+ 自适应步长（二阶矩 $\boldsymbol{v}$），配**偏差校正**。
- 更新语义：$\Delta\theta = -\eta\,\frac{\hat{\boldsymbol{m}}}{\sqrt{\hat{\boldsymbol{v}}}+\epsilon}$——「确定方向快走、不确定方向慢走」。
- **AdamW**：把权重衰减从梯度路径**解耦**，衰减恒定不被缩放——大模型标配。
- **AMSGrad**：二阶矩取历史最大值，恢复凸问题收敛保证；实践价值有限。
- 「信任比」$\hat{m}/\sqrt{\hat{v}}$ 是理解 Adam 行为的钥匙。
- 选型：通用 Adam、大模型 AdamW、CNN 常 SGD+动量；一切以验证集为准。

在下一节，我们回到「用二阶信息」的路线，看看牛顿法在深度学习中如何「打折」使用——这就是**二阶优化近似：牛顿法与拟牛顿法**。
