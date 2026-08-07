---
title: DPO 推导：从 RLHF 目标到闭式最优解的完整数学链条
date: 2026-08-07
---

# DPO 推导：从 RLHF 目标到闭式最优解的完整数学链条

<div class="epigraph">
<p>最优策略本身，就藏着一个奖励模型。</p>
<footer>—— DPO 论文副标题：Your Language Model Is Secretly a Reward Model</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第六章 ｜ 2026-08-07</p>
</div>

## 为什么从 DPO 推导开始

前三节讲的 PPO、拒绝采样都围绕「用奖励模型优化策略」。2023 年，斯坦福的 **DPO（Direct Preference Optimization，直接偏好优化）** 论文抛出了一个更激进的论断：**奖励模型和强化学习步骤都是多余的——最优策略本身就能直接作为「隐式奖励模型」来用**。

DPO 是直接偏好优化路线的开山之作，也是本专题最重要的方法之一。它的价值在于**数学推导的优雅**：从 RLHF 的目标函数出发，几步代数就能得到「只用偏好数据、不用 RM、不用 RL」的训练损失。本节把这条推导链完整走一遍——四步，环环相扣——并解释每一步的直觉。读完后，「DPO 为什么能省掉两个阶段」对你不再是黑箱。<span class="marginnote">先给这条推导链一张地图：<strong>RLHF 目标 → 闭式最优策略 → 用策略「解出」奖励 → 代入 Bradley-Terry → 得到 DPO 损失</strong>。整条链只有「带 KL 约束的 RL 目标」有公认的闭式解，其余全是代数替换——这正是 DPO 推导「干净」的原因。</span>

## 1 第一步：RLHF 目标的闭式最优解

一切的起点，是 RLHF 的目标函数（RLHF 总览一节已见过）：

$$
\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot\mid x)}\big[ r(x, y) \big] - \beta\, \mathbb{E}_{x \sim \mathcal{D}}\big[ \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\, \pi_{\mathrm{ref}}(\cdot\mid x)\big) \big]
$$

这个「奖励最大化 + KL 约束」的目标，有一个**已知的闭式解**——带 KL 约束的 RL 最优策略是：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\mathrm{ref}}(y \mid x)\, \exp\Big(\frac{r(x, y)}{\beta}\Big)
$$

其中 $Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x) \exp(r(x,y)/\beta)$ 是配分函数（归一化常数）。逐项拆解：

- 最优策略 $\pi^*$：正比于「参考模型的概率」乘以「奖励的指数」——**奖励高的回答被放大，奖励低的被压小**；
- $Z(x)$：保证 $\pi^*$ 是合法概率分布（所有回答概率加起来为 1）；
- $\beta$：温度——$\beta$ 大，$\exp(r/\beta)$ 的放大作用被削弱，最优策略更贴近参考模型（更保守）；$\beta$ 小则更激进。

**这个闭式解是整个推导的支点**：它告诉我们「给定奖励 $r$ 与参考 $\pi_{\mathrm{ref}}$，最优策略长什么样」。DPO 的妙处，是把它**反过来用**。

## 2 第二步：把奖励「解」出来

闭式解是一个等式，那就可以**反解出奖励**。对两边取对数、整理：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)
$$

逐项拆解：

- 左边的 $r(x,y)$：**奖励**；
- 右边的第一项：**策略比的对数**——最优策略相对参考模型的「偏爱程度」；
- 第二项 $\beta\log Z(x)$：只依赖提示 $x$、不依赖回答 $y$ 的常数。

**关键洞察出现了**：奖励函数 $r(x,y)$ 完全可以用「最优策略 $\pi^*$ 与参考模型 $\pi_{\mathrm{ref}}$ 的比值」表达——**策略本身就编码了奖励**。这就是「你的语言模型就是个奖励模型」这句话的数学来源：**只要知道最优策略，就能推出隐式奖励**。

## 3 第三步：代入 Bradley-Terry，消掉配分函数

现在把「用策略表达的奖励」代入 Bradley-Terry 模型（偏好数据收集一节的公式）：

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

代入第 2 步的表达式：

$$
P = \sigma\Big( \big[\beta \log \tfrac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} + \beta\log Z(x)\big] - \big[\beta \log \tfrac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)} + \beta\log Z(x)\big] \Big)
$$

**配分函数 $Z(x)$ 消掉了**——因为它在「胜者项」与「败者项」里相同，相减即归零。最终：

$$
P(y_w \succ y_l \mid x) = \sigma\Big( \beta \log \frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta \log \frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)} \Big)
$$

这个式子极其漂亮：**偏好概率完全由「策略比」决定，不需要任何显式奖励**。配分函数被消掉，是因为「两两比较」天然不依赖归一化常数——这与「比较优于评分」的哲学再次呼应。

## 4 公式解析：DPO 损失，最终形态

最后一步：把 $\pi^*$ 换成我们正在训练的策略 $\pi_\theta$（$\pi^*$ 是理想的、$\pi_\theta$ 是逼近它的），用最大似然估计写出训练损失：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[ \log \sigma\Big( \beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)} \Big) \Big]
$$

逐项拆解：

- $\pi_\theta$：正在训练的策略（对齐模型）；
- $\pi_{\mathrm{ref}}$：冻结的参考模型（通常是 SFT 模型）；
- $\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}$：胜者的**隐式奖励**——模型相对参考更「偏爱」胜者多少；
- 括号里是「胜者的隐式奖励 − 败者的隐式奖励」，取 sigmoid 后就是「偏好概率」；
- 整体取负对数：**最大化偏好概率**——与奖励模型的 BT 损失形式完全一致，但「奖励」换成了「策略比」。

**DPO 损失与 BT 损失的对应**：奖励模型的损失是 $-\log\sigma(r_w - r_l)$，DPO 的损失是 $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)})$——**把「显式奖励」换成了「隐式奖励（策略比）」**。DPO 等于把奖励模型的训练目标「内嵌」进了策略训练本身。<span class="marginnote">有一个值得记住的「梯度直觉」：DPO 的梯度在「胜者隐式奖励 > 败者」时为正、促进提高胜者概率；但<strong>当模型已经开始偏好转捩时，梯度会减弱（sigmoid 饱和）</strong>——这给了 DPO 一种天然的「自我限制」，避免把某个回答的概率推得过高。这是 DPO 比「直接最大化胜者概率」更稳的原因之一。</span>

## 5 推导的意义：为什么「省掉 RM 与 RL」没有付出代价

把四步推导合起来，DPO 的意义浮出水面：

| 维度 | RLHF（PPO） | DPO |
| --- | --- | --- |
| 训练数据 | 提示（RL 阶段） + 偏好（RM 阶段） | 仅偏好对 |
| 需要奖励模型 | ✅ | ❌（隐式） |
| 需要强化学习 | ✅ | ❌ |
| 模型数 | 4（策略+参考+RM+价值） | 2（策略 + 参考） |
| 训练方式 | 在线采样 + 策略梯度 | 离线监督损失 |

**DPO 没有「白省」**——它省掉的 RM 与 RL，对应的「隐式奖励」信息仍然保留在损失里。推导链展示的等价性说明：**在「离线偏好数据」这个前提下，RLHF 的目标可以被闭式求解，不需要数值优化**。代价是：DPO 是**离线**的——它不能像 PPO 那样在训练中采样新数据、探索新回答（这正是《在线与离线偏好优化的取舍》要讨论的）。

## 6 小结

- 推导四步：**RLHF 目标 → 闭式最优策略 → 反解奖励 → 代入 BT 得 DPO 损失**。
- 闭式解 $\pi^* = \frac{1}{Z(x)}\pi_{\mathrm{ref}}\exp(r/\beta)$：最优策略正比于「参考概率 × 奖励指数」。
- 反解奖励 $r = \beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}} + \beta\log Z$：**策略本身就是奖励模型**。
- 代入 BT 后 $Z(x)$ 消掉：偏好概率只由「策略比」决定——两两比较天然不依赖归一化常数。
- **DPO 损失** $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)})$：把 BT 的显式奖励换成隐式策略比。
- 意义：离线偏好数据下 RLHF 目标可闭式求解——省掉 RM 与 RL，代价是失去在线采样能力。

在下一节，我们把 DPO 从推导落到实践：**DPO 实践——参考模型、温度系数 β 与常见训练陷阱**。
