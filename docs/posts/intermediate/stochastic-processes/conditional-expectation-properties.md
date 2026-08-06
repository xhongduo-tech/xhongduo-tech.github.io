---
title: 条件期望的性质：塔性质、取己知量、独立性
date: 2026-08-07
---

# 条件期望的性质：塔性质、取己知量、独立性

<div class="epigraph">
<p>我不知道上帝是否掷骰子，但绝不该用比事实更简单的模型来误导。</p>
<footer>—— 约翰 · 冯·诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》第1章 ｜ 2026-08-07</p>
</div>

## 为什么要专门学性质

上一节我们用两条性质「定义」了条件期望，但定义只是入场券。随机过程真正靠的是条件期望的**运算规则**：全期望公式为什么能反复套用？
马尔可夫链的转移为什么能「一步到底」？鞅为什么是公平赌博？
——答案全藏在条件期望的几条性质里。<span class="marginnote">对标张波《应用随机过程》第1章：
他在定义条件期望后紧接着列出基本性质； 本专题第六篇的鞅，本质就是「条件期望的塔性质」在动态场景下的展开。</span>

本节是这一批四篇里最「工具化」的一篇：几条性质，条条都要用到顺手，才算真正掌握了条件期望。

## 1 取己知量：已知的东西可以「提出来」

设 $Z$ 是 $\mathcal{G}$-可测的随机变量，且可积性条件满足，则

$$
\mathbb{E}[Z X \mid \mathcal{G}] = Z \, \mathbb{E}[X \mid \mathcal{G}]
$$

直觉：$Z$ 的信息已经包含在 $\mathcal{G}$ 里，条件期望「知道」$Z$ 的值，于是 $Z$ 可以像常数一样从期望符号里提出来。<span class="marginnote">取己知量是「已知信息优先」的数学表达：$\mathcal{G}$ 已经告诉你 $Z$ 是什么，再条件化时 $Z$ 不再是随机变量，而是常数因子。英文名就叫 taking out what is known。</span>

两个特例值得记住：$Z \equiv c$（常数）时退化为线性性里的常数提取；若 $X$ 本身 $\mathcal{G}$-可测，则 $\mathbb{E}[X \mid \mathcal{G}] = X$——「知道了它，就原样保留它」。

**取己知量的一个数值例子**：沿用两次掷骰子，$X = Y_1 + Y_2$，取 $\mathcal{H} = \sigma(Y_1)$（只知道第一次）。求 $\mathbb{E}[Y_1 X \mid \mathcal{H}]$：因为 $Y_1$ 已经是 $\mathcal{H}$-可测的，取己知量给出 $\mathbb{E}[Y_1 X \mid \mathcal{H}] = Y_1 \mathbb{E}[X \mid \mathcal{H}] = Y_1 (Y_1 + 3.5)$。读者不妨把 $Y_1$ 可能取的值逐一代入直接验证：先固定 $Y_1 = y_1$ 算条件期望再乘以 $y_1$，与上式完全一致。这就是「己知因子提出」在数值上的自洽。

## 2 塔性质：信息只能越变越少

设 $\mathcal{H} \subset \mathcal{G}$，则

$$
\mathbb{E}\bigl[\mathbb{E}[X \mid \mathcal{G}] \mid \mathcal{H}\bigr] = \mathbb{E}[X \mid \mathcal{H}]
$$

这是条件期望**最重要的一条性质**，也叫迭代期望定律。<span class="marginnote">记忆口诀：「先按细信息 $\mathcal{G}$ 平均，再按粗信息 $\mathcal{H}$ 平均，等价于直接按粗信息 $\mathcal{H}$ 平均。」细平均里蕴含的信息，在粗平均时会被吸收掉。</span>上一节的全期望公式是它的特例：取 $\mathcal{H} = \{\emptyset, \Omega\}$，内层平均完再取普通期望，回到 $\mathbb{E}[X]$。

直觉解释：条件期望可以看作「信息压缩」——$\mathcal{G}$ 比 $\mathcal{H}$ 信息多，先把 $X$ 压到 $\mathcal{G}$ 层，再从 $\mathcal{G}$ 层压到 $\mathcal{H}$ 层，效果等价于一次性压到 $\mathcal{H}$ 层。**中间的 $\mathcal{G}$ 层被「过拟合」掉了。**

**塔性质的实战直觉**可以用两次掷骰子来体会。设 $Y_1, Y_2$ 独立同分布，$X = Y_1 + Y_2$，$\mathcal{G} = \sigma(Y_1, Y_2)$（知道两次结果），$\mathcal{H} = \sigma(Y_1)$（只知道第一次）。则 $\mathbb{E}[X \mid \mathcal{G}] = X$（信息齐全，原样保留），再对 $\mathcal{H}$ 取条件期望得 $\mathbb{E}[X \mid \mathcal{H}] = Y_1 + \mathbb{E}[Y_2] = Y_1 + 3.5$；而直接算 $\mathbb{E}[X \mid \mathcal{H}]$ 也等于 $Y_1 + 3.5$。第二次的结果 $Y_2$ 在从 $\mathcal{G}$ 压缩到 $\mathcal{H}$ 时被吸收成了它的期望——这正是「信息只能丢失」在数值上的体现。

## 3 独立性：没用的信息自动消失

若 $X$ 与 σ-代数 $\mathcal{G}$ **独立**，则

$$
\mathbb{E}[X \mid \mathcal{G}] = \mathbb{E}[X]
$$

直觉：$\mathcal{G}$ 对 $X$ 一无所知，条件化不能改变对 $X$ 的猜测，于是退回无条件期望。<span class="marginnote">独立性在测度论里有精确版：$X$ 与 $\mathcal{G}$ 独立，指对一切 $A \in \mathcal{G}$，$X$ 与 $\mathbf{1}_A$ 独立。注意这里要求的是 $X$ 与「整个 $\mathcal{G}$」独立，而非只与某个 $Y$ 独立。</span>

一个常用推论：若 $Y$ 与 $X$ 独立，则 $\mathbb{E}[X \mid Y] = \mathbb{E}[X]$。这正是上一节「信息越少，猜测越保守」的正式表达。

## 4 公式解析：塔性质为什么成立

用定义直接验证塔性质，是理解「为什么中间层会被吸收」的最好方式。要证 $\mathbb{E}[\mathbb{E}[X \mid \mathcal{G}] \mid \mathcal{H}] = \mathbb{E}[X \mid \mathcal{H}]$，记 $Z = \mathbb{E}[X \mid \mathcal{G}]$，$W = \mathbb{E}[Z \mid \mathcal{H}]$，需证 $W = \mathbb{E}[X \mid \mathcal{H}]$。

- **第一步，检查可测性**：$W$ 是 $\mathcal{H}$-可测的（因为它是 $\mathcal{H}$ 的条件期望），满足定义的第一条。
- **第二步，验证积分等式**：任取 $A \in \mathcal{H}$。因为 $\mathcal{H} \subset \mathcal{G}$，所以 $A \in \mathcal{G}$ 也成立。于是可以连续使用两次定义式：
$$\int_A W \, dP = \int_A Z \, dP = \int_A X \, dP$$
第一步用了 $W = \mathbb{E}[Z \mid \mathcal{H}]$ 在 $A$ 上的定义式；第二步用了 $Z = \mathbb{E}[X \mid \mathcal{G}]$ 在 $A$ 上的定义式——而这一步之所以合法，正因为 $A \in \mathcal{H} \subset \mathcal{G}$。
- **第三步，应用唯一性**：$W$ 满足「$\mathcal{H}$-可测 + 对所有 $A \in \mathcal{H}$ 积分等式」两条性质，这正是 $\mathbb{E}[X \mid \mathcal{H}]$ 的定义特征，由几乎处处唯一性得 $W = \mathbb{E}[X \mid \mathcal{H}]$。证毕。

**关键就在第二步**：$\mathcal{H} \subset \mathcal{G}$ 保证了「内层定义式可用的集合范围」包含「外层定义式可用的集合范围」，两条等式因此能在同一个 $A$ 上先后成立。这个「子集包含 ⇒ 定义式可用」的套娃结构，是整个证明的精髓。

## 5 其余常用性质一览

- **线性性**：$\mathbb{E}[aX + bY \mid \mathcal{G}] = a\mathbb{E}[X \mid \mathcal{G}] + b\mathbb{E}[Y \mid \mathcal{G}]$。
- **单调性**：$X \le Y$ 几乎处处 ⇒ $\mathbb{E}[X \mid \mathcal{G}] \le \mathbb{E}[Y \mid \mathcal{G}]$ 几乎处处。
- **条件 Jensen 不等式**：$\varphi$ 为凸函数时，$\varphi(\mathbb{E}[X \mid \mathcal{G}]) \le \mathbb{E}[\varphi(X) \mid \mathcal{G}]$。<span class="marginnote">条件 Jensen 不等式是「条件化不增加凸性失真」的保证，也是证明鞅收敛、Doob 不等式时反复出现的引擎——第六篇会用。</span>
- **保常数与保己知**：$\mathbb{E}[c \mid \mathcal{G}] = c$；$X$ 为 $\mathcal{G}$-可测时 $\mathbb{E}[X \mid \mathcal{G}] = X$。
- **条件方差公式**：$\operatorname{Var}(X \mid \mathcal{G}) = \mathbb{E}[X^2 \mid \mathcal{G}] - \bigl(\mathbb{E}[X \mid \mathcal{G}]\bigr)^2$，进而 $\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X \mid \mathcal{G})] + \operatorname{Var}(\mathbb{E}[X \mid \mathcal{G}])$。

把这五条核心性质收进一张速查表，方便随时对照：

| 性质 | 公式 | 一句话记忆 |
| --- | --- | --- |
| 线性性 | $\mathbb{E}[aX + bY \mid \mathcal{G}] = a\mathbb{E}[X \mid \mathcal{G}] + b\mathbb{E}[Y \mid \mathcal{G}]$ | 条件期望是线性算子 |
| 取己知量 | $\mathbb{E}[ZX \mid \mathcal{G}] = Z \mathbb{E}[X \mid \mathcal{G}]$（$Z$ 为 $\mathcal{G}$-可测） | 已知因子提出 |
| 塔性质 | $\mathcal{H} \subset \mathcal{G} \Rightarrow \mathbb{E}[\mathbb{E}[X \mid \mathcal{G}] \mid \mathcal{H}] = \mathbb{E}[X \mid \mathcal{H}]$ | 信息只能变少 |
| 独立性 | $X \perp \mathcal{G} \Rightarrow \mathbb{E}[X \mid \mathcal{G}] = \mathbb{E}[X]$ | 无关信息自动消失 |

### 塔性质的一次实战：贝尔曼期望方程

塔性质不是书斋里的把戏，它直接就是强化学习里**贝尔曼期望方程（Bellman expectation equation）**的骨架。设 $s_t$ 是时刻 $t$ 的状态，$r_t$ 是即时奖励，折扣因子为 $\gamma$，回报 $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$，状态价值函数 $V(s_t) = \mathbb{E}[G_t \mid s_t]$。把 $G_t$ 拆成即时奖励加未来回报，再用条件期望的线性性与塔性质：

$$
V(s_t) = \mathbb{E}[r_t + \gamma \mathbb{E}[G_{t+1} \mid s_{t+1}] \mid s_t] = \mathbb{E}[r_t + \gamma V(s_{t+1}) \mid s_t]
$$

中间那一层「先以 $s_{t+1}$ 条件化、再以 $s_t$ 条件化」正是塔性质 $\mathcal{H} = \sigma(s_t) \subset \sigma(s_t, s_{t+1}) = \mathcal{G}$ 的实例——把未来信息压缩到当前状态。这个方程是第三级《强化学习》与「从极限到大模型」主线反复使用的核心，而它的数学根源，就是本节这条看似抽象的性质。

## 6 辨析：塔性质的三个陷阱

**辨析｜易错点：** 第一，塔性质的方向**只能把信息变少**——$\mathcal{H} \subset \mathcal{G}$。若反过来先按粗信息平均再取细条件，$\mathbb{E}[\mathbb{E}[X \mid \mathcal{H}] \mid \mathcal{G}]$ 一般**不等于** $\mathbb{E}[X \mid \mathcal{G}]$，它只等于 $\mathbb{E}[X \mid \mathcal{H}]$：因为 $X$ 先被压到更粗的 $\mathcal{H}$，信息已经丢了，细信息 $\mathcal{G}$ 找不回来。<span class="marginnote">直觉：先按粗粒度平均，已经把细粒度信息抹掉了，之后再用细信息去条件化，也恢复不了丢失的信息。「信息只能丢失、不能凭空产生」，这条守恒律贯穿整个随机过程。</span>第二，$\mathbb{E}[\mathbb{E}[X \mid Y] \mid X]$ 与 $\mathbb{E}[\mathbb{E}[X \mid X] \mid Y]$ 完全不同：前者是「先知道 $Y$ 再知道 $X$」，后者内层直接退化成 $X$，整体等于 $\mathbb{E}[X \mid Y]$。第三，塔性质要求 $\mathcal{H} \subset \mathcal{G}$，写成反方向的包含关系是错误用法。

## 7 小结

- **取己知量**：$\mathcal{G}$-可测的 $Z$ 可从条件期望中提出：$\mathbb{E}[ZX \mid \mathcal{G}] = Z\mathbb{E}[X \mid \mathcal{G}]$。
- **塔性质**：$\mathcal{H} \subset \mathcal{G}$ 时 $\mathbb{E}[\mathbb{E}[X \mid \mathcal{G}] \mid \mathcal{H}] = \mathbb{E}[X \mid \mathcal{H}]$；全期望公式是它的特例，信息只能变少。
- **独立性**：$X$ 与 $\mathcal{G}$ 独立时 $\mathbb{E}[X \mid \mathcal{G}] = \mathbb{E}[X]$。
- 还有**线性性、单调性、条件 Jensen 不等式、保常数**等常用性质，组成条件期望的完整工具箱。
- 塔性质的证明是「定义验证」的样板：检查可测性、验证积分等式、应用几乎处处唯一性。

在下一节，我们将带着这个工具箱回到随机过程的舞台中央——用条件期望定义**鞅**：一个「条件期望等于当前值」的过程。塔性质将保证鞅的公平性在任意未来时刻都不被破坏，那正是「从极限到大模型」主线上强化学习收益期望的数学根基。




