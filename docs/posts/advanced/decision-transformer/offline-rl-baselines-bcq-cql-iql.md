---
title: 传统离线 RL 基线 BCQ/CQL/IQL
date: 2026-08-11
---

# 传统离线 RL 基线 BCQ/CQL/IQL

<div class="epigraph">
<p>过犹不及。</p>
<footer>—— 孔子，《论语 · 先进》（Excess is as bad as deficiency）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · Decision Transformer（序列建模 RL） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 BCQ/CQL/IQL 开始

上一课我们说「传统离线 Q-learning 脆弱」，但脆弱不等于坐以待毙——2019 到 2022 年间，一批方法给 Q-learning 打上了各不相同的补丁：BCQ 限制动作、CQL 压低价值、IQL 干脆回避查询。

它们构成 Decision Transformer 最直接的参照系：**DT 宣布「不踩分布外的雷」，这些方法则声明「我能踩，但穿上防雷靴」。**

认识这三双靴子，才能准确评价 DT 那条「不踩」的路到底省了什么。<span class="marginnote">三条基线的原始出处：BCQ（Fujimoto, Meger, Precup, 2019）、CQL（Kumar et al., NeurIPS 2020）、IQL（Kostrikov, Nair, Levine, ICLR 2022）。它们与 DT 在 D4RL 上的对比贯穿 Chen et al. §5 与后续各篇。</span>

## 1 BCQ：把动作关进数据的笼子

**Batch-Constrained Q-learning（BCQ）**：让策略只能选择「与数据集动作分布相近」的动作，从而把 $\max$ 的搜索空间限制在数据支持域内。

BCQ 的做法分两步。

第一步，训练一个条件生成模型（论文用条件 VAE）$G_\omega(s)$ 来模拟「数据里在状态 $s$ 下会出什么动作」。

第二步，在状态 $s$ 下采样 $n$ 个候选动作 $a_i = G_\omega(s) + \xi_\phi(s, a, \Phi)$，其中 $\xi_\phi$ 是一个有界的扰动网络，最后取价值最大的那个：

$$
\pi_{\text{BCQ}}(s) = \arg\max_{a_i} \; Q_\theta\!\left(s, a_i\right), \qquad a_i = G_\omega(s) + \xi_\phi(s, a_i, \Phi)
$$

**直觉：Q-learning 的病根是 $\max$ 会捡起分布外的胡猜值，那就让 $\max$ 只能在「数据说过的动作」附近挑。**

BCQ 用一个显式的动作空间约束，把外推误差挡在门外。<span class="marginnote">一个容易混淆的点：BCQ 仍然是 Q-learning——它有 $Q$ 网络、有贝尔曼自举，只是把「选动作」的搜索空间给裁剪了。它修的是「$\max$ 的病」，没修「自举的病」。</span>

BCQ 的实际表现有个微妙之处：它对**生成模型的忠实度**极其敏感——VAE 学得不准，候选动作偏离数据，约束形同虚设；VAE 学得太准，又可能丢失多样性、探索受限。

**「约束的松紧」成了 BCQ 的核心矛盾**，这也是它后来被 CQL 一类方法盖过风头的原因之一。

## 2 CQL：给价值函数装刹车

**Conservative Q-Learning（CQL）**：在贝尔曼更新之外，额外对「分布外的动作」施加价值惩罚，让 $Q$ 对没见过的 (状态, 动作) 保持悲观。

CQL 的损失函数在原有 TD 项上追加一个正则项：

$$
\min_Q \; \alpha\, \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp\big(Q(s, a)\big) - \mathbb{E}_{a \sim \pi_{\mathcal{D}}(a|s)} \big[ Q(s, a) \big] \right] + \mathcal{L}_{\text{TD}}(Q)
$$

方括号里第一项 $\log \sum_a \exp(Q)$ 相当于「对所有动作的价值求一个软 max」，它偏爱未被数据支持的动作；第二项是「数据动作的平均价值」。

**两项相减，等于给「数据没覆盖的动作」的价值兜头浇一盆冷水**，而 TD 项照常学数据内的关系。<span class="marginnote">CQL 的完整推导包含一个下界论证：最小化上述损失会得到一个「对策略真实价值的下界估计」——悲观不是拍脑袋，是保底。这是它与「打折动作空间」的 BCQ 在哲学上的分水岭。</span>

CQL 的工程手感是「**一个 $\alpha$ 走天下**」：$\alpha$ 太大，连数据内动作的价值都被压扁、策略变保守甚至退化；$\alpha$ 太小，分布外的冷水浇得不够，外推误差抬头。

**它把「离线难题」翻译成了一个标量超参的敏感度问题**——这在实践中既是便利，也是负担。

## 3 IQL：不查不存在的动作

**Implicit Q-learning（IQL）**：用分位数回归（expectile）拟合价值函数，只估计「数据内状态的价值」，绝不查询「数据外动作的 $Q$」。

IQL 的妙处在它绕开了 $\max$。

标准 Q-learning 更新需要 $Q(s', a')$，而 $a'$ 往往不在数据里。

IQL 换了个目标：学一个状态价值 $V(s)$，用 **expectile 回归**让它逼近「$r + \gamma V(s')$ 的分布中偏上的分位」：

$$
V(s) \leftarrow \arg\min_v \; \mathbb{E}\left[ L_2^{\tau}\!\left( r + \gamma V(s') - v \right) \right], \qquad L_2^{\tau}(u) = \left| \tau - \mathbf{1}\{u < 0\} \right| u^2
$$

$L_2^{\tau}$ 是个不对称的平方损失：对 $u>0$（实际目标高于当前估计）给权重 $\tau$，对 $u<0$ 给权重 $1-\tau$。

取 $\tau$ 靠近 1，就是在拟合「这个分布偏好的上分位」——**近似于「对数据内状态取乐观的 $\max$」，却从头到尾没对任何分布外动作算过 $Q$。**

策略提取则用行为克隆加优势加权（AWR），仍然只看数据内动作。<span class="marginnote">IQL 与 DT 有一个亲缘：两者都「回避分布外查询」。区别是 IQL 仍靠价值信号塑形策略，DT 直接把目标写进条件——这条亲缘与分野，正是下一课《Return-conditioned 监督学习》的铺垫。</span>

## 4 公式解析：三种补丁、一条主线

把三个目标函数摆在一起，能看出离线 Q-learning 的全部演化逻辑：

$$
\underbrace{\pi_{\text{BCQ}} = \arg\max_{\{G_\omega + \xi\}} Q}_{\text{约束动作空间}} \qquad \underbrace{\min_Q \; \alpha\left[\log\sum_a e^{Q(s,a)} - \mathbb{E}_{\mathcal{D}} Q\right]}_{\text{惩罚分布外价值}} \qquad \underbrace{V \leftarrow \text{expectile}\left(r + \gamma V(s')\right)}_{\text{不查询分布外}}
$$

三步拆解。

**第一步，找共同病根**：三者都在治「$\max$ 捡起分布外高估」这一个病。BCQ 从**动作**下手，CQL 从**价值**下手，IQL 从**查询**下手——同一病灶，三把手术刀。

**第二步，看各自的代价**：BCQ 要训好 VAE，生成质量决定上限；CQL 要调正则权重 $\alpha$，压狠了会拖累数据内价值；IQL 要调 expectile 参数 $\tau$ 与优势加权的温度，还得训练两次（价值、策略分开）。

**第三步，对照 DT 的哲学**：三条基线都是「**修好 Q-learning 再用**」；DT 是「**不用 Q-learning**」。

BCQ/CQL/IQL 的超参是「补丁的松紧」，DT 的超参是「条件与窗口」，后者更少、更直观。

**但代价也要记住：DT 放弃了价值信号带来的「显式最优性保证」，也放弃了对回报结构（如稀疏奖励）的精细利用。**

一张「失败模式」对照表，把三双靴子的各自盲区钉在墙上：

| 方法 | 补丁方式 | 典型失败模式 | 关键超参 |
| --- | --- | --- | --- |
| BCQ | 约束动作空间 | 生成模型失准 → 约束形同虚设 | VAE 容量、扰动幅度 $\Phi$ |
| CQL | 惩罚分布外价值 | $\alpha$ 过大 → 策略过保守 | 正则权重 $\alpha$ |
| IQL | 不查询分布外 | $\tau$ 或温度失配 → 价值/策略失衡 | expectile $\tau$、AWR 温度 |
| DT | 不用价值引擎 | RTG 信号弱（random 数据）→ 条件失效 | 初始目标 $g$、窗口 $K$ |

## 5 小结

- **BCQ**：生成模型圈定动作空间，$\max$ 只在数据支持域内挑——约束动作；盲区在生成模型失准。
- **CQL**：软 max 减数据均值，给分布外价值降温——惩罚价值；盲区在 $\alpha$ 的过保守/欠保守两难。
- **IQL**：expectile 回归学状态价值，从不查询分布外动作——回避查询；盲区在 $\tau$ 与温度。
- **共同病根**：离线 Q-learning 的 $\max$ 病；三种补丁 = 三种手术刀，各有超参与上限。
- **DT 的立场**：不修病，不用那套引擎——超参更少、哲学更简，但也放弃了价值信号的显式最优性。

在下一节，我们沿着「不用价值函数」这条路走到底，看看把序列建模简化到只剩监督学习的**Return-conditioned 监督学习范式（RvS）**。
