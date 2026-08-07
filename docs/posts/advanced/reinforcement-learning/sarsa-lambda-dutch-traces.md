---
title: Sarsa(λ)与荷兰迹
date: 2026-08-07
---

# Sarsa(λ)与荷兰迹

<div class="epigraph">
<p>信用不止要分给「走过的状态」，更要分给「做过的动作」。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第12章 §12.5–12.6 ｜ 2026-08-07</p>
</div>

## 为什么资格迹要升级成「动作级」

上一课的状态价值资格迹回答「哪个状态该为未来负责」；但控制的决策单位是**动作**——「在 $S_t$ 选了 $A_t$」才是该被惩罚或奖励的主体。**Sarsa(λ)** 把资格迹搬到状态动作对上，让一次 TD 误差按「每个 $(s,a)$ 最近多活跃」分配责任。而**荷兰迹（Dutch trace）** 则是一个更精妙的重参数化：它在蒙特卡洛学习里把「在线更新的误差」变得与前向目标**精确一致**——是上一课「迹投影」思想的完整形态。<span class="marginnote">「荷兰迹」名字来自其提出者（van Seijen 等人，荷兰团队）——就像「荷兰式拍卖」一样，名字标记了出处，内容是一种「投影到特征空间的迹」：$\mathbf e_t = \gamma\lambda\mathbf e_{t-1} + \mathbf x_t - \gamma\lambda(\mathbf x_t^\top\mathbf e_{t-1})\mathbf x_t$。</span>

## 1 Sarsa(λ)：表格型的动作级迹

**Sarsa(λ)** 的资格迹对**每个状态动作对 $(s,a)$** 记一笔账，表格型下是：

$$
e_t(s,a) \;=\; \begin{cases}
\gamma\lambda\, e_{t-1}(s,a) + 1, & (s,a) = (S_t, A_t)\\
\gamma\lambda\, e_{t-1}(s,a), & \text{其他}
\end{cases}
$$

即：**当前执行的动作对迹 +1（频率律），所有动作对按 $\gamma\lambda$ 衰减（近因律）**。更新时，用一步 TD 误差 $\delta_t$ 乘整张迹表：

$$
Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\,\delta_t\, e_t(s,a), \qquad \delta_t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
$$

与 Sarsa(0) 只更新当前 $(S_t,A_t)$ 不同，**Sarsa(λ) 一次更新所有「迹不为零」的状态动作对**——按各自的迹深浅分配这次误差。<span class="marginnote">直觉：下棋时一步臭棋（TD 误差大），λ 让「最近走的那几步棋」（迹深）一起承担后果——λ 越大，回望的步数越多。λ=0 时只有当前动作被更新，Sarsa(λ) 退化为 Sarsa；λ=1 时整幕的动作都被平等记账，近似 MC。</span>

## 2 Sarsa(λ) 的函数逼近版

配函数逼近时，迹变成参数空间的向量——和状态价值版本一个套路，只是梯度换成了动作价值的梯度：

$$
\mathbf e_t \;=\; \gamma\lambda\,\mathbf e_{t-1} + \nabla \hat q(S_t, A_t, \mathbf w_t), \qquad
\mathbf w_{t+1} \;=\; \mathbf w_t + \alpha\,\delta_t\, \mathbf e_t
$$

**$\mathbf e$ 累积的是「每个参数对最近动作价值变化的贡献历史」**。线性情形 $\hat q = \mathbf w^\top\mathbf x(s,a)$ 下，迹就是「$\gamma\lambda$ 衰减的特征和」。Sarsa(λ) 配瓦片编码/神经网络，就是深度 RL 时代「带资格迹的 Actor-Critic」的祖先——PPO 的 GAE 本质上是「把 λ 思想用在策略梯度上」（第14篇会正面相见）。<span class="marginnote">教材用「网格世界」演示 Sarsa(λ)：配 $\varepsilon$-贪心与随机漫步，λ 取 0.5–0.9 时学习显著快于 Sarsa(0)。原因是迹让「一次奖励信号」沿最近的动作序列回传，等价于「多个 n步回报同时生效」——这正是 λ-回报的等价物在控制里的表现。</span>

## 3 荷兰迹：让蒙特卡洛学习「在线即精确」

**荷兰迹（Dutch trace）** 解决蒙特卡洛学习里的一个尴尬：MC 的回报 $G_t$ 要到幕末才完整，可我们希望**在线更新**。朴素做法是「边采边把部分回报往迹上记」，但这样记出的更新与「幕末一次性 MC 更新」**不一致**。荷兰迹修正了它——在线性逼近下用投影迹：

$$
\mathbf e_t \;=\; \gamma\lambda\,\mathbf e_{t-1} + \mathbf x_t - \gamma\lambda\big(\mathbf x_t^\top \mathbf e_{t-1}\big)\mathbf x_t
$$

**投影项 $-\gamma\lambda(\mathbf x_t^\top\mathbf e_{t-1})\mathbf x_t$ 把迹里「特征空间之外」的分量清掉**，保证迹始终是「$\mathbf w$ 能表达的形态」。配上特殊的更新方式，荷兰迹能让「在线累积的更新总量」与「幕末一次算清的总量」**逐项相等**——在线不再是 MC 的近似，而是精确的在线实现。<span class="marginnote">为什么叫「荷兰」？提出者 van Seijen 与 Sutton 是阿姆斯特丹学派；「Dutch」成了这种「投影 + 重参数化」风格的标签。它的核心价值是澄清一个长期困惑：<strong>资格迹到底该不该投影？</strong> 荷兰迹的回答是「该」——不投影，在线与离线的总量就对不上。</span>

## 4 公式解析：荷兰迹的投影项在做什么

$$
\underbrace{\mathbf e_t}_{\text{新迹}} = \underbrace{\gamma\lambda\mathbf e_{t-1}}_{\text{衰减旧迹}} + \underbrace{\mathbf x_t}_{\text{当前特征}} - \underbrace{\gamma\lambda\big(\mathbf x_t^\top\mathbf e_{t-1}\big)\mathbf x_t}_{\text{投影修正}}
$$

- **第一步，认朴素迹**：前两项是标准资格迹——衰减旧迹、加上当前特征。它累积的是「$\gamma\lambda$ 折扣下的历史特征和」。
- **第二步，认投影**：第三项把「旧迹 $\mathbf e_{t-1}$ 在当前方向 $\mathbf x_t$ 上的分量」按 $\gamma\lambda$ 减回去。净效果：**更新后 $\mathbf e_t$ 与 $\mathbf x_t$ 正交的那部分被保留，$\mathbf x_t$ 方向的分量被重算**——迹被「投影」到当前特征方向张成的空间。
- **第三步，认必要性**：当特征之间线性相关（如瓦片编码的重叠、贝尔德反例的特征），朴素迹会在「不同特征重叠方向」上重复记账，造成更新总量失真。投影项把重叠方向的账合并、清掉冗余，**让总量与「离线一次算清」精确对齐**。<span class="marginnote">记忆：朴素迹像「累加器」——所有历史各记各的；荷兰迹像「合并账本」——重叠部分只记一次，方向由当前特征重新标定。这 30 年来的「迹到底加不加投影」之争，荷兰迹给了「加」的精确理由。</span>

## 5 易错点辨析

**辨析｜易错点：** 把 Sarsa(λ) 当成「每步只更新当前 $(s,a)$ 的 Sarsa 的 λ 版」。**Sarsa(λ) 每步更新所有迹非零的 $(s,a)$**——这正是它与 Sarsa(0) 的本质区别。λ 越大，一次更新波及的「历史动作」越多，这是迹方法的全部意义。

**另一个易错点**：以为荷兰迹只能用于 MC。荷兰迹的投影机制对 TD(λ)、Sarsa(λ) 同样适用——上一课真在线 TD(λ) 用的正是这种投影迹。**「荷兰迹」是一种通用的迹重参数化，MC 只是它的一个舞台**。

**第三个易错点**：忽略 $\gamma\lambda$ 中的 $\gamma$。动作价值版本里，迹衰减仍要乘 $\gamma$——折扣与回望同步。若忘乘 $\gamma$（只写 $\lambda e_{t-1}$），迹衰减过快/过慢，等价性被破坏。**「迹的衰减 = 折扣 × λ」这条式子对状态价值、动作价值、函数逼近版一律成立**。

## 6 小结

- **Sarsa(λ)**：动作级资格迹，当前 $(s,a)$ 迹 +1、全体按 $\gamma\lambda$ 衰减；$\mathbf w \leftarrow \mathbf w + \alpha\delta_t\mathbf e_t$。
- λ=0 退化为 Sarsa，λ=1 近似 MC；λ 取中间值学习显著加速。
- **荷兰迹**：$\mathbf e_t = \gamma\lambda\mathbf e_{t-1} + \mathbf x_t - \gamma\lambda(\mathbf x_t^\top\mathbf e_{t-1})\mathbf x_t$——投影到特征空间。
- 荷兰迹让**在线 MC 更新与离线总量精确一致**；投影清掉线性相关特征上的重复记账。
- 迹衰减恒为「$\gamma \times \lambda$」；荷兰迹对 TD(λ)、Sarsa(λ) 通用。

在下一节，我们松开「λ 与 γ 恒定」的假定：**变量 λ 与变量 γ**——让回望深度与折扣随状态自适应，扩展资格迹的适用范围。
