---
title: DPO及后RLHF时代的对齐算法：隐式奖励与直接偏好优化
date: 2026-08-07
---

# DPO及后RLHF时代的对齐算法：隐式奖励与直接偏好优化

<div class="epigraph">
<p>如果奖励函数已经被「最优策略」倒推出来，那还要奖励函数干什么？</p>
<footer>—— 改编自拉斐尔 · 拉法伊洛夫（Rafael Rafailov）等，2023</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 强化学习与大型语言模型 ｜ 原文：Rafailov et al. 2023 ｜ 2026-08-07</p>
</div>

## 为什么可以「绕过奖励模型和 PPO」

RLHF 的三段式（奖励模型 + PPO + KL）虽然有效，却笨重：**要训奖励模型、要跑 RL、要处理 PPO 的一堆稳定性细节**。**DPO（Direct Preference Optimization，直接偏好优化）** 用一个漂亮的数学观察砍掉中间环节：**RLHF 的优化目标有一个闭式最优解，把这个解「倒过来」，奖励函数可以用「策略」直接表达**——于是「偏好损失」可以直接写成策略的函数，**不需要奖励模型、不需要 RL，只是监督学习**。这一课是「后 RLHF」对齐算法的主线——DPO 以及它的众多后继（KTO、IPO 等），共同指向「更轻、更稳、可扩展」的对齐。<span class="marginnote">DPO 的论文标题《你的语言模型其实是隐式的奖励模型》点破核心：<strong>给定参考策略 $\pi_{\text{ref}}$，任何一个策略 $\pi_\theta$ 都「隐含」定义了一个奖励函数</strong>（$r = \beta\log(\pi_\theta/\pi_{\text{ref}}) + \text{const}$）。优化策略 = 优化这个隐式奖励——偏好数据直接训策略，无需显式奖励。</span>

## 1 核心推导：从 RLHF 目标反解出奖励

RLHF 的优化目标（第68课）是：

$$
\max_\pi\; \mathbb{E}\big[R(x,y)\big] - \beta\, \mathbb{E}\big[D_{\text{KL}}(\pi \,\|\, \pi_{\text{ref}})\big]
$$

这是「带 KL 正则的奖励最大化」——它的**闭式最优解**（带 KL 正则的 RL 的经典结果）是：

$$
\pi^*(y \mid x) \;=\; \frac{1}{Z(x)}\, \pi_{\text{ref}}(y\mid x)\, \exp\Big(\frac{1}{\beta} R(x,y)\Big), \qquad Z(x) = \sum_y \pi_{\text{ref}}(y|x)\,e^{R(x,y)/\beta}
$$

**倒过来解出奖励**：

$$
R(x,y) \;=\; \beta \log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$

**这就是「隐式奖励」**：给定参考策略与最优策略，奖励被「翻译」成「两个策略的对数比」——**奖励不是独立学出来的，而是策略之间差异的另一种说法**。<span class="marginnote">这一步是 DPO 的全部魔法：RLHF 需要「先学奖励、再优化策略」，DPO 发现「奖励本来就藏在『策略与参考策略的比值』里」——于是<strong>学策略就直接在优化偏好</strong>，奖励模型成了多余的中介。配分函数 $Z(x)$ 只依赖 $x$（不依赖 $y$），在成对比较里会被消掉——这就是 DPO 可行的关键。</span>

## 2 DPO 损失：把隐式奖励塞进偏好似然

把隐式奖励代回 **Bradley–Terry 偏好模型**（人类偏好 $y_w \succ y_l$ 的概率 $\sigma(R(y_w)-R(y_l))$），配分函数 $Z(x)$ 在相减中抵消，得到 **DPO 损失**：

$$
\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]
$$

**这就是一个标准的二分类/排序损失**——输入是「一对回答 + 人类的偏好标签」，优化的对象是「策略」。**没有奖励模型、没有 PPO、没有采样**——只是对偏好对做监督学习。<span class="marginnote">DPO 损失的直觉：<strong>让「被偏好的回答」在策略里的相对概率（相对参考模型）高于「被拒绝的回答」</strong>——$\beta\log(\pi_\theta/\pi_{\text{ref}})$ 是「隐式奖励」，DPO 就是「把赢家的隐式奖励推高、把输家的压低」。σ 把它变成「排序正确率」的对数似然——<strong>「让模型更偏好人类偏好的」</strong>。</span>

## 3 为什么 DPO 好：轻、稳、快

DPO 相对 RLHF 的优势清晰：

| 维度 | RLHF（奖励模型 + PPO） | DPO |
| --- | --- | --- |
| 组件 | 奖励模型 + 策略 + critic | 只有策略（+ 参考模型） |
| 训练信号 | RL（采样 + 优势估计） | 监督（偏好对上的分类损失） |
| 稳定性 | PPO 一堆超参要调 | 接近监督学习、稳定 |
| 计算 | 采样生成 + 多步更新 | 一次前向/反向 |
| 数学等价 | — | 与 RLHF 目标等价（闭式解下） |

**数学等价性**：在「RLHF 的 KL 约束形式 + 偏好由 Bradley–Terry 生成」的假设下，DPO 优化的就是 RLHF 想优化的同一个目标——**它是 RLHF 的「解析解短路」**。<span class="marginnote">「等价但更简单」是 DPO 的卖点：RLHF 用 PPO 一步步逼近「带 KL 的最优策略」，DPO 直接算出「该最优策略长什么样」并一步到位地拟合它。<strong>优化问题的解是同一个，只是 DPO 走了解析捷径</strong>——这也是「后 RLHF」算法的共同主题：能不用 RL 就不用。</span>

**局限性**：DPO 的假设比 RLHF 强——它要求「偏好数据覆盖充分」「隐式奖励的配分函数可忽略（假设 $Z(x)$ 常数）」。数据分布偏、$Z(x)$ 随 $x$ 剧变时，DPO 会偏离 RLHF 的行为。**「简单」的代价是「假设更强」**。

## 4 公式解析：配分函数为什么「消失」

$$
\underbrace{R(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)}_{\text{隐式奖励}} \qquad\Rightarrow\qquad R(y_w) - R(y_l) = \beta \log\frac{\pi_\theta(y_w|x)\pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_l|x)\pi_{\text{ref}}(y_w|x)}
$$

- **第一步，认隐式奖励**：给定策略 $\pi_\theta$ 与参考 $\pi_{\text{ref}}$，奖励被「倒推」出来——$Z(x)$ 是只依赖提示 $x$ 的归一化常数。
- **第二步，认相减**：Bradley–Terry 里出现的是「两个回答的奖励差」$R(y_w) - R(y_l)$——**$\beta\log Z(x)$ 两项相同、直接抵消**。配分函数无需计算，这就是 DPO 不需要「求 $Z$」的原因。
- **第三步，认损失**：奖励差代入 $\log\sigma(\cdot)$ 就是 DPO 损失——**「赢家/参考 的比值 > 输家/参考 的比值」**。优化的全部信息来自「成对偏好」与「策略相对参考模型的偏离」。<span class="marginnote">对照第65课 MaxEnt IRL 的「配分函数是命门」：那里 $Z$ 无法解析、要靠采样；DPO 之所以优雅，是因为<strong>成对比较把 $Z(x)$ 消掉了</strong>——「相对偏好」不需要绝对归一化。<strong>「只要比较、不求绝对」</strong>让偏好优化免于配分函数的诅咒。</span>

## 5 后 DPO 时代：KTO、IPO 与对齐算法的谱系

DPO 打开了一扇门，后续对齐算法不断简化或修正：

**KTO（Kahneman-Tversky Optimization）**：不需要成对偏好，只需「单条回答 + 该不该被偏好」的标签——把「排序」简化成「分类」，更贴近真实标注（人们更容易说「这条还行/不行」而非「比较两条」）。
**IPO（Identity Preference Optimization）**：修正 DPO 的「正则强度随数据量漂移」问题——用「恒等」正则让优化更稳。
**SLiC / RPO 等**：从「排序损失」「相对偏好」的不同角度切入。

**共同主题**：**「能用监督/闭式解解决的，就不用 RL」**——DPO 及其后继把「对齐」从「复杂的 RL 工程」变成「简单的损失函数设计」，让对齐更可复现、更可扩展。<span class="marginnote">但「DPO 取代 RLHF」的争论远未定论：RLHF 的 PPO 在「训练时可采样、能在线优化」上仍不可替代（DPO 是离线、一次性拟合）；许多实践发现「PPO 在线 + 更高 KL」能超过「DPO 离线」。<strong>DPO 与 PPO 不是「谁更好」而是「离线解析 vs 在线迭代」的两条路</strong>——理解两者的数学联系，才不会被「取代论」带偏。</span>

## 6 易错点辨析

**辨析｜易错点：** 以为 DPO「不需要参考模型」。**需要**——$\pi_{\text{ref}}$ 出现在损失里（作为隐式奖励的基准）。参考模型通常取「SFT 后、对齐前」的模型，全程冻结。**「没有奖励模型」≠「没有参考模型」**。

**另一个易错点**：以为 DPO「完全等价于 RLHF」。它在**理想假设下**（偏好由 Bradley–Terry 生成、$Z(x)$ 近似常数、离线数据充分）等价——现实数据的偏态会让两者行为不同。**「数学等价」是理想情形，「实践差异」是常态**。

**第三个易错点**：把「隐式奖励」当「真奖励」。隐式奖励 $r = \beta\log(\pi_\theta/\pi_{\text{ref}})$ 是「策略偏离的另一种说法」，**不是独立于策略的「真实偏好」**——它随策略变。DPO 的「奖励」是策略的函数，不是「先学出来的东西」。

## 7 小结

- **DPO**：直接偏好优化——从 RLHF 目标的闭式解反解出隐式奖励，把偏好损失写成策略的函数。
- **隐式奖励**：$r(x,y) = \beta\log(\pi_\theta/\pi_{\text{ref}}) + \beta\log Z$——奖励藏在「策略与参考的比值」里。
- **无 RL**：不需要奖励模型、不需要 PPO、不需要采样——只是偏好对上的分类损失。
- **配分函数消失**：成对比较中 $Z(x)$ 抵消——「只要比较、不求绝对」。
- **后 DPO 谱系**：KTO（单条标签）、IPO（更稳正则）……共同主题是「能不用 RL 就不用」。

至此，强化学习专题的全部 70 篇已按计划写完。从多臂老虎机到 DPO，这条「从试错到对齐」的主线在此收官。
