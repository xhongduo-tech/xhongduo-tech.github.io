---
title: TRPO：信赖域策略优化
date: 2026-08-07
---

# TRPO：信赖域策略优化

<div class="epigraph">
<p>别走太远——每一步都别离开「旧策略还说得上话」的范围。</p>
<footer>—— 改编自约翰 · 舒尔曼（John Schulman）等，2015</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Schulman et al. 2015 ｜ 2026-08-07</p>
</div>

## 为什么策略梯度的「步长」是个致命问题

普通策略梯度（REINFORCE、AC）用步长 $\alpha$ 控制更新幅度——但策略梯度对步长**极其敏感**：$\alpha$ 大了，新策略一步跳出「旧策略还说得上话」的区域，价值估计失效、训练崩溃；$\alpha$ 小了，学习慢得让人绝望。更糟的是，**「参数空间的距离」与「策略空间的距离」不成正比**——参数挪一小步，softmax 可能把策略整个翻新。**TRPO（Trust Region Policy Optimization，信赖域策略优化）** 换掉「固定步长」：**每次更新都限制「新旧策略的 KL 散度」不超过一个信赖域半径**，并给出「单调改进」的理论保证。<span class="marginnote">TRPO 的名字来自优化理论里的「信赖域方法」：在每一步，用一个局部模型近似目标函数，但<strong>只在该模型可信的邻域内</strong>优化。TRPO 把「邻域」定义为「KL 散度球」——策略参数空间里的一个「信任半径」。</span>

## 1 代理目标：用重要性采样比改写策略梯度

TRPO 的目标函数用**重要性采样比** $\rho_t = \pi_\boldsymbol{\theta}(a|s)/\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)$ 写成「新旧策略的代理目标」：

$$
L(\boldsymbol{\theta}) \;=\; \mathbb{E}_{(s,a)\sim\pi_{\text{old}}}\Big[\frac{\pi_\boldsymbol{\theta}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)}\, \hat A(s,a)\Big]
$$

**用旧策略采集的数据，去评估新策略的表现**——这是「离策略」风格的代理：$L$ 对 $\boldsymbol{\theta}$ 可微（旧分布固定），梯度方向就是「新策略沿优势方向改进」的方向。**关键性质**：当 $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$ 时，$L$ 的梯度等于真实性能 $J$ 的梯度（一阶一致），且 $L(\boldsymbol{\theta}_{\text{old}}) = J(\boldsymbol{\theta}_{\text{old}})$。<span class="marginnote">重要性采样比 $\rho$ 是第5章的老朋友——这里它做的是「用旧数据近似新策略的目标」。代理目标 $L$ 是对「真实性能 $J$」的一阶局部近似：在旧策略处，$L$ 与 $J$ 值相等、梯度相等——所以在信赖域内优化 $L$ 就是在优化 $J$。</span>

## 2 信赖域约束：KL 散度球

TRPO 优化的问题是「带约束」的：

$$
\max_{\boldsymbol{\theta}}\; \mathbb{E}\Big[\frac{\pi_\boldsymbol{\theta}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} \hat A(s,a)\Big] \qquad \text{s.t.}\quad \mathbb{E}_{s\sim\pi_{\text{old}}}\big[\,D_{\text{KL}}\big(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot|s) \,\|\, \pi_{\boldsymbol{\theta}}(\cdot|s)\big)\big] \le \delta
$$

**约束是「平均 KL 散度不超过 $\delta$」**——新策略不能在「平均意义」上偏离旧策略太远。$\delta$ 是信赖域半径（通常 0.01），它把每次更新的「策略距离」钉在可控范围。

**为什么用 KL 而不是参数距离？** 因为**策略分布的距离才是真正要紧的**——参数 $\boldsymbol{\theta}$ 挪多少与「策略行为变多少」无关（参数化不同、缩放不同），KL 散度直接度量「新旧策略的行为差异」，与参数化无关。<span class="marginnote">「参数距离 ≠ 策略距离」是 TRPO 洞察的核心：同一个策略可以用不同参数表达，参数上的欧氏距离毫无意义；KL 是策略（分布）之间不变量化的距离。这正是「用 KL 球做信赖域」的理论动机。</span>

## 3 求解：自然梯度与共轭梯度

带约束的优化问题可以用**拉格朗日/KKT** 求解。把目标线性化（一阶泰勒）并用 KL 约束（二阶），得到：

$$
\boldsymbol{\theta}_{\text{new}} \;=\; \boldsymbol{\theta}_{\text{old}} + \alpha \,\mathbf{F}^{-1}\,\mathbf{g}
$$

其中 $\mathbf{g} = \nabla_\boldsymbol{\theta} L(\boldsymbol{\theta}_{\text{old}})$ 是目标梯度，$\mathbf{F}$ 是**费雪信息矩阵（Fisher information matrix）**——KL 散度的二阶近似（Hessian）：

$$
\mathbf{F} \;=\; \mathbb{E}\big[\nabla\ln\pi_\boldsymbol{\theta}(a|s)\, \nabla\ln\pi_\boldsymbol{\theta}(a|s)^\top\big]
$$

**$\mathbf{F}^{-1}\mathbf{g}$ 是自然梯度（natural gradient）**——「在策略分布空间里、而不是参数空间里的最陡下降方向」。参数空间的梯度经 $\mathbf{F}^{-1}$ 校正后，就不再被「参数缩放」干扰。<span class="marginnote">自然梯度是 Amari 的经典思想：普通梯度在「参数坐标系」里最陡，但参数坐标是任意的；自然梯度在「统计流形」里最陡——用费雪度量衡量「分布间的真实距离」。TRPO 把它带进深度 RL，让「步长」第一次有了几何意义。</span>

**工程实现**：$d$ 维参数的 $\mathbf{F}^{-1}$ 求逆是 $O(d^3)$，不可行。TRPO 用**共轭梯度法（conjugate gradient）** 解 $\mathbf{F}\mathbf{x} = \mathbf{g}$——只需矩阵-向量积（通过 Fisher-向量积技巧），复杂度降到 $O(d)$ 量级，让大网络可解。之后沿 $\mathbf{x}$ 方向做**线搜索**（backtracking），保证 KL 约束满足、且代理目标确有改进。<span class="marginnote">TRPO 的「Fisher-向量积」技巧：$\mathbf{F}\mathbf{v}$ 不需要显式构造 $\mathbf{F}$，只要对「分数函数向量积」求两次梯度即可——这让你在几百万参数的网络上也能用自然梯度。这个技巧后来被 PPO 简化（不再需要），但在许多依旧用 TRPO 的地方仍是关键。</span>

## 4 公式解析：自然梯度为什么「几何正确」

$$
\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} + \underbrace{\alpha\,\mathbf{F}^{-1}\,\mathbf{g}}_{\text{自然梯度方向}}
\qquad\text{对比}\qquad \boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} + \underbrace{\alpha\,\mathbf{g}}_{\text{普通梯度方向}}
$$

- **第一步，认普通梯度**：$\mathbf{g}$ 在参数坐标系里最陡——但参数坐标任意（缩放、旋转都会改变「最陡」方向），普通梯度对参数化敏感。
- **第二步，认费雪矩阵**：$\mathbf{F} = \mathbb{E}[\nabla\ln\pi\,\nabla\ln\pi^\top]$ 是「分数函数的外积期望」——它编码了「参数空间的度量」：哪个方向参数一挪、策略变得多。
- **第三步，认逆变换**：$\mathbf{F}^{-1}\mathbf{g}$ 把「参数坐标里的梯度」变换成「策略分布空间里的梯度」——**方向与参数化无关，步长对应真实的策略距离**。配合线搜索保证 KL 约束，每次更新都在信赖域内、保证代理目标单调不减。<span class="marginnote">对照：如果 $\mathbf{F}$ 是单位阵（参数正交、各向同性），自然梯度=普通梯度——普通梯度是自然梯度在「参数度量平凡」时的特例。理解这一点，就看懂了 TRPO 在「修正梯度方向」这件事上做了什么。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 TRPO 与 PPO 优化同一个目标。**PPO 是 TRPO 的「一阶近似」**：TRPO 严格约束 KL（用自然梯度 + 线搜索），PPO 用裁剪代理目标**软性**限制更新（更简单、更好实现）。两者目标函数同源（重要性采样比 × 优势），但约束机制不同——TRPO 更「严格」、PPO 更「实用」。

**另一个易错点**：把 $\rho_t = \pi_\theta/\pi_{\text{old}}$ 当「重要性采样的离策略学习」。这里它是「代理目标的权重」，只在**旧策略分布**下估计新策略——它不是「用任意行为数据学目标策略」的通用离策略，而是「局部更新的代理目标」。

**第三个易错点**：忽视 KL 约束的「平均」性质。约束是 $\mathbb{E}_s[D_{\text{KL}}] \le \delta$——**平均意义**，不是逐状态。某些「极端状态」的 KL 可以很大（只要平均被压住）；若任务对极端状态敏感，要换 max 约束（更难）。

## 6 小结

- **TRPO**：最大化「重要性采样比 × 优势」的代理目标，subject to **平均 KL 约束**。
- **信赖域**：KL 散度球定义「新旧策略的可信范围」，步长有了几何意义。
- **自然梯度**：$\mathbf{F}^{-1}\mathbf{g}$——参数无关的最陡方向；共轭梯度 + Fisher-向量积让大网络可解。
- **保证**：信赖域内优化代理目标 → 策略性能单调改进（理论）。
- TRPO 是 PPO 的前身——约束机制不同，目标同源。

在下一节，我们把 TRPO 的「昂贵约束」换成「裁剪」：**PPO**——近端策略优化，简单、稳、如今几乎是大模型时代 RLHF 的默认选择。
