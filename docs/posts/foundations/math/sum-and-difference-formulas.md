---
title: 两角和与差的正弦、余弦和正切公式
date: 2026-08-07
---

# 两角和与差的正弦、余弦和正切公式

<div class="epigraph">
<p>数学是给不同事物取相同名字的艺术。</p>
<footer>—— 昂利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.5.1 ｜ 2026-08-07</p>
</div>

## 为什么从两角和与差的正弦、余弦和正切公式开始

到目前为止，我们处理三角函数的方式很「单一」：要么研究同一个角 $\alpha$ 的各函数关系（同角基本关系），要么处理 $\alpha$ 与它的对称亲戚（诱导公式）。现在要迈出真正的第一步——**同时面对两个独立的角**：$\sin(\alpha + \beta)$ 到底等于什么？<span class="marginnote">这个问题在初中就埋下了：$\sin 75^\circ$ 能不能拆成 $\sin 45^\circ + \sin 30^\circ$？直觉会说「能」，但答案令人意外——<strong>不能</strong>。</span>

你或许会猜 $\sin(\alpha + \beta) = \sin\alpha + \sin\beta$，就像 $(a+b)^2 = a^2 + 2ab + b^2$ 一样天经地义。可惜**三角函数不满足分配律**。真正的关系要复杂、也优美得多。这一节推导出的六条公式，是整个三角恒等变换的「总发动机」：下一节的二倍角公式、辅助角公式、以及解三角形里的正弦定理余弦定理，全部从这里出发。在信号处理与 AI 领域，它们同样是「把两个波叠加」的理论基础——庞加莱说数学是「给不同事物取相同名字的艺术」，而这些公式正是给「两种角」找到同一个名字的杰作。

## 1 母公式：两角差的余弦公式

一切从一条公式开始，它是后面所有公式的源头，称为**两角差的余弦公式**：

$$\boxed{\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta}$$

为什么是它当「母公式」？因为只要把它证出来，其余五个（$\cos(\alpha+\beta)$、$\sin(\alpha\pm\beta)$、$\tan(\alpha\pm\beta)$）都可以**纯代数地**从它推出，不必再借助几何。这是数学里最漂亮的工作方式：**证一个最根本的，其余全部是推论。**<span class="marginnote">「少而真」的公理，「多而强」的定理——这个模式你已经见过一次（同角基本关系），现在又见一次。到第二级《抽象代数》，你会发现整个数学都这样运转。</span>

**重点：** 证明的舞台是单位圆。设角 $\alpha$ 与 $\beta$ 的终边分别与单位圆交于 $P(\cos\alpha, \sin\alpha)$ 与 $Q(\cos\beta, \sin\beta)$。点 $P$、$Q$ 之间的距离，可以用两种完全不同的方法计算，把两个结果一对照，公式就自己跳出来了。

## 2 公式解析：cos(α - β) 是怎么证出来的

$$
\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta
$$

把证明拆成四步，每一步都看得见摸得着：

- **第一步，用坐标算距离**：两点 $P(\cos\alpha, \sin\alpha)$ 与 $Q(\cos\beta, \sin\beta)$ 的距离平方为
  $$PQ^2 = (\cos\alpha - \cos\beta)^2 + (\sin\alpha - \sin\beta)^2$$
  展开并整理：首尾两项各自合并成 $\cos^2\alpha + \sin^2\alpha = 1$ 与 $\cos^2\beta + \sin^2\beta = 1$，中间留下交叉项，得
  $$PQ^2 = 2 - 2(\cos\alpha\cos\beta + \sin\alpha\sin\beta)$$
- **第二步，用几何算同一条距离**：$P$、$Q$ 都在单位圆上，它们对应的圆心角是 $|\alpha - \beta|$。单位圆上圆心角为 $\theta$ 的弦长是 $2\sin\frac{\theta}{2}$，于是
  $$PQ^2 = \left(2\sin\frac{|\alpha-\beta|}{2}\right)^2 = 4\sin^2\frac{\alpha-\beta}{2} = 2 - 2\cos(\alpha-\beta)$$
  最后一步用到了倍角关系 $2\sin^2\frac{\theta}{2} = 1 - \cos\theta$。
- **第三步，两个表达式相等**：同一条 $PQ^2$，两种算法结果必然一致，故
  $$2 - 2(\cos\alpha\cos\beta + \sin\alpha\sin\beta) = 2 - 2\cos(\alpha-\beta)$$
- **第四步，消去公因子**：两边同时除以 $-2$，得到 $\cos(\alpha-\beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$。

**要点：** 这个证明的核心思想是「**用两种方法算同一个量，然后令它们相等**」。弦长 $PQ$ 是那座桥梁：一边通往坐标代数，一边通往单位圆的几何。这种「双算法对照」的技巧，在数学里被称为一记重拳——后面证明余弦定理、推导距离公式时，你会不断见到它。<span class="marginnote">顺便，第四步里用到的 $\cos\theta = 1 - 2\sin^2\frac{\theta}{2}$ 正是下一节《二倍角公式》的内容——这提示我们：二倍角其实已经「埋伏」在母公式的证明里了。</span>

## 3 从母公式推出其余五个

母公式到手，其余五个全靠**换元与诱导公式**，一步都不需要新的几何。

**（1）$\cos(\alpha + \beta)$：把 $\beta$ 换成 $-\beta$。** 把 $\cos(\alpha-\beta)$ 公式里的 $\beta$ 换作 $-\beta$，再用 $\cos(-\beta) = \cos\beta$、$\sin(-\beta) = -\sin\beta$（诱导公式三）：

$$\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$

**（2）$\sin(\alpha + \beta)$：借 $\cos$ 的桥。** 用诱导公式把正弦翻译成余弦，再套 $\cos$ 的差公式：

$$\sin(\alpha+\beta) = \cos\left(\frac{\pi}{2} - \alpha - \beta\right) = \cos\left(\left(\frac{\pi}{2}-\alpha\right)-\beta\right) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$

**（3）$\sin(\alpha - \beta)$：再换一次 $\beta \to -\beta$：** $\sin(\alpha-\beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta$。

**（4）（5）$\tan(\alpha \pm \beta)$：商数关系 + 分子分母同除。** 例如
$$\tan(\alpha+\beta) = \frac{\sin(\alpha+\beta)}{\cos(\alpha+\beta)} = \frac{\sin\alpha\cos\beta + \cos\alpha\sin\beta}{\cos\alpha\cos\beta - \sin\alpha\sin\beta}$$
分子分母同除以 $\cos\alpha\cos\beta$（在 $\cos\alpha\cos\beta \neq 0$ 时），得
$$\tan(\alpha+\beta) = \frac{\tan\alpha + \tan\beta}{1 - \tan\alpha\tan\beta}, \qquad \tan(\alpha-\beta) = \frac{\tan\alpha - \tan\beta}{1 + \tan\alpha\tan\beta}$$

**重点：** 六条公式不是六个孤立的记忆点，而是**一棵树**：根是 $\cos(\alpha-\beta)$，一次换元长出 $\cos(\alpha+\beta)$，一次诱导公式长出 $\sin(\alpha+\beta)$，再换元长出 $\sin(\alpha-\beta)$，最后作商长出两条正切。**记一棵树，而不是记六片叶子。**

## 4 公式总表与记忆口诀

把六条公式集中陈列，方便对照：

| 公式 | 内容 |
| --- | --- |
| $\cos(\alpha - \beta)$ | $\cos\alpha\cos\beta + \sin\alpha\sin\beta$ |
| $\cos(\alpha + \beta)$ | $\cos\alpha\cos\beta - \sin\alpha\sin\beta$ |
| $\sin(\alpha + \beta)$ | $\sin\alpha\cos\beta + \cos\alpha\sin\beta$ |
| $\sin(\alpha - \beta)$ | $\sin\alpha\cos\beta - \cos\alpha\sin\beta$ |
| $\tan(\alpha + \beta)$ | $\dfrac{\tan\alpha + \tan\beta}{1 - \tan\alpha\tan\beta}$ |
| $\tan(\alpha - \beta)$ | $\dfrac{\tan\alpha - \tan\beta}{1 + \tan\alpha\tan\beta}$ |

记忆的抓手有两条：

- **余弦是「反号」的**：$\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$——外层是加号时，中间取减号；外层是减号时，中间取加号。**符号反着来。**
- **正弦是「同号」的**：$\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$——外层什么号，中间就是什么号。**符号顺着来。**
- 正切公式的分子是「$\tan\alpha \pm \tan\beta$」，分母是「$1 \mp \tan\alpha\tan\beta$」——**分子外层同号，分母外层反号**，与余弦的规律正好互补。

<span class="marginnote">与其死记口诀，不如每用一次就重推一遍：把 $\alpha,\beta$ 换成特殊角（如 $\alpha = \frac{\pi}{2}$）代入检验，错了立刻能察觉。公式是「用熟」的，不是「背熟」的。</span>

**辨析｜易错点：** 最经典、最致命的错误是**把分配律错误推广到三角函数**：
$$\cos(\alpha + \beta) \neq \cos\alpha + \cos\beta, \qquad \sin(\alpha+\beta) \neq \sin\alpha + \sin\beta$$
检验一句话就够：取 $\alpha = \beta = \frac{\pi}{3}$，左边 $\cos\frac{2\pi}{3} = -\frac{1}{2}$，右边 $1$，天差地别。**凡是函数，都不天然满足分配律；只有线性运算（如求和、求导、求积分）才有「先拆再算」的特权。**

## 5 应用：求值、化简、逆用

**例 1（求值）：** 求 $\sin 15^\circ$。$15^\circ = 45^\circ - 30^\circ$，用正弦差公式：
$$\sin 15^\circ = \sin 45^\circ\cos 30^\circ - \cos 45^\circ\sin 30^\circ = \frac{\sqrt{2}}{2}\cdot\frac{\sqrt{3}}{2} - \frac{\sqrt{2}}{2}\cdot\frac{1}{2} = \frac{\sqrt{6} - \sqrt{2}}{4}$$

**例 2（求值）：** 求 $\cos 75^\circ$。$75^\circ = 45^\circ + 30^\circ$，用余弦和公式：
$$\cos 75^\circ = \cos 45^\circ\cos 30^\circ - \sin 45^\circ\sin 30^\circ = \frac{\sqrt{2}}{2}\cdot\frac{\sqrt{3}}{2} - \frac{\sqrt{2}}{2}\cdot\frac{1}{2} = \frac{\sqrt{6} - \sqrt{2}}{4}$$

有意思的是 $\sin 15^\circ$ 与 $\cos 75^\circ$ 相等——它们本来互余，这从侧面验证了公式的正确性。<span class="marginnote">「把陌生角拆成两个特殊角」是求值的第一策略：$15^\circ = 45^\circ - 30^\circ$、$75^\circ = 45^\circ + 30^\circ$、$105^\circ = 60^\circ + 45^\circ$，拆开之后全部落在特殊角表内。</span>

**例 3（逆用）：** 化简 $\cos 20^\circ\cos 25^\circ - \sin 20^\circ\sin 25^\circ$。认出它是 $\cos(\alpha+\beta)$ 的展开式逆用：$\cos(20^\circ + 25^\circ) = \cos 45^\circ = \frac{\sqrt{2}}{2}$。**顺用是把公式展开，逆用是把展开式「收回」一个函数**——化简题的高阶姿势就是逆用。

**例 4（知一求余后相加）：** 已知 $\sin\alpha = \frac{3}{5}$，$\alpha$ 为第二象限角；$\cos\beta = \frac{12}{13}$，$\beta$ 为第四象限角，求 $\sin(\alpha - \beta)$。先分别补齐：$\cos\alpha = -\frac{4}{5}$（第二象限余弦为负），$\sin\beta = -\frac{5}{13}$（第四象限正弦为负）。代入差公式：
$$\sin(\alpha-\beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta = \frac{3}{5}\cdot\frac{12}{13} - \left(-\frac{4}{5}\right)\cdot\left(-\frac{5}{13}\right) = \frac{36}{65} - \frac{20}{65} = \frac{16}{65}$$
**关键步骤是先由象限定号、补齐缺失的 $\cos\alpha$ 与 $\sin\beta$，再代入**——这正是一节《同角三角函数的基本关系》与本节公式的接力。

## 6 辨析｜易错点汇总

- **符号规则**：余弦「外层与中间反号」，正弦「外层与中间同号」。写 $\cos(\alpha+\beta)$ 时中间是减号，写 $\sin(\alpha+\beta)$ 时中间是加号，两者最容易互相污染，务必对照公式表逐项核对。
- **正切公式的分母**：$\tan(\alpha+\beta)$ 要求 $1 - \tan\alpha\tan\beta \neq 0$，即 $\alpha+\beta \neq \frac{\pi}{2} + k\pi$（此时 $\tan$ 无定义）。分母恰好为零时，公式「失效」，但那个角本身是有意义的——它是竖直方向。
- **先定号，再代值**：任何「知一求余」类问题，都先由角所在象限确定 $\sin$、$\cos$ 的正负，再代入公式；符号错一位，整题归零。<span class="marginnote">四步走：<strong>先写公式骨架 → 补齐缺失值 → 定符号 → 代入计算</strong>。顺序越固定，出错率越低——这套「程序化」的解题流程，正是计算机能替你算三角的前提，也是后来算法思想的雏形。</span>
- **公式可以正着用，也可以反着用**：求值题顺用，化简题逆用。题目给的是「展开式」时，目标是把它们「收拢」成一个角的三角函数。

## 7 小结

- **母公式**：$\cos(\alpha-\beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$，用「双算法算同一条弦长」证明。
- **六个公式一棵树**：母公式经换元（$\beta \to -\beta$）得 $\cos(\alpha+\beta)$，经诱导公式得 $\sin(\alpha\pm\beta)$，再作商得 $\tan(\alpha\pm\beta)$。
- **记忆**：余弦「外层反号」、正弦「外层同号」、正切分子同号分母反号；三角函数**不满足分配律**。
- **应用**：求值（拆角）、化简（逆用收拢）、知一求余后相加（先定号再代入）。
- **地位**：一切三角恒等变换（二倍角、辅助角、解三角形）都从这里长出来。

在下一节，我们将让公式里的两个角**重合**——令 $\beta = \alpha$，看看 $\sin 2\alpha$、$\cos 2\alpha$、$\tan 2\alpha$ 会变成什么样子。这就是**二倍角的正弦、余弦、正切公式**。
