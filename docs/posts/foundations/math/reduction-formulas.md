---
title: 诱导公式
date: 2026-08-07
---

# 诱导公式

<div class="epigraph">
<p>对称，无论我们把它理解得多宽或多窄，都是人类世世代代用以理解和创造秩序、美与完美的一种观念。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl），《对称》（Symmetry，1952）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.3 ｜ 2026-08-07</p>
</div>

## 为什么从诱导公式开始

上一节，我们用同一个角 $\alpha$ 的三种三角函数之间的关系，建立起了整个三角恒等变换的公理。可现实中我们遇见的角常常不是 $\alpha$，而是 $\alpha$ 的「亲戚」：$-\alpha$、$\pi-\alpha$、$\pi+\alpha$、$\frac{\pi}{2}-\alpha$、$\frac{\pi}{2}+\alpha$。<span class="marginnote">这些角与 $\alpha$ 之间有一个共同点：它们终边上的点，都是由 $\alpha$ 终边上的点在单位圆上做了某种「对称运动」得到的。</span>

诱导公式要回答的问题很具体：**这些「亲戚角」的三角函数值，能不能只用 $\alpha$ 的三角函数值表示出来？** 答案藏在单位圆的对称性里——x 轴对称、y 轴对称、原点对称、旋转 $90^\circ$。外尔说对称是「秩序、美与完美」的源泉，而在这里，对称直接变成了可计算的公式。这也是全书第一次把「几何对称」翻译成「代数公式」，是后面研究周期性、图象性质、乃至傅里叶分析里一切「波」的起点。

## 1 回顾：终边相同的角

最朴素的一组「亲戚角」是 $\alpha + 2k\pi\ (k \in \mathbb{Z})$。它们与 $\alpha$ 的终边**完全重合**，交点坐标一模一样，三角函数值自然完全相同。这就是在《三角函数的概念》一节已经见过的：

$$\sin(\alpha + 2k\pi) = \sin\alpha, \qquad \cos(\alpha + 2k\pi) = \cos\alpha, \qquad \tan(\alpha + 2k\pi) = \tan\alpha$$

它说的是三角函数关于「$2\pi$ 的整数倍」不变，也就是后面要系统研究的**周期性**。不过它只处理「转整圈」的角，接下来要面对的是「转半圈」「翻个面」这类更硬的对称。

## 2 单位圆上的三面镜子

设角 $\alpha$ 的终边与单位圆交于点 $P(x, y)$，于是 $\sin\alpha = y$，$\cos\alpha = x$。现在让点 $P$ 在单位圆上做三种对称运动，观察新点的坐标，就得到三组诱导公式。

**镜面一：关于 $x$ 轴对称。** 点 $P(x,y)$ 关于 $x$ 轴对称后得到 $P'(x, -y)$。这个新点对应的角是 $-\alpha$。于是：

$$\sin(-\alpha) = -y = -\sin\alpha, \qquad \cos(-\alpha) = x = \cos\alpha, \qquad \tan(-\alpha) = -\tan\alpha$$

**镜面二：关于 $y$ 轴对称。** 点 $P(x,y)$ 关于 $y$ 轴对称后得到 $P'(-x, y)$。这个新点对应的角是 $\pi - \alpha$。于是：

$$\sin(\pi - \alpha) = y = \sin\alpha, \qquad \cos(\pi - \alpha) = -x = -\cos\alpha, \qquad \tan(\pi - \alpha) = -\tan\alpha$$

**镜面三：关于原点对称。** 点 $P(x,y)$ 关于原点对称后得到 $P'(-x, -y)$。这个新点对应的角是 $\pi + \alpha$。于是：

$$\sin(\pi + \alpha) = -y = -\sin\alpha, \qquad \cos(\pi + \alpha) = -x = -\cos\alpha, \qquad \tan(\pi + \alpha) = \tan\alpha$$

<span class="marginnote">这三面镜子把 $x,y$ 变成 $-x,-y$ 的四种组合，正好对应四个象限里的四种符号。所以诱导公式（二）（三）（四）的本质，就是「<strong>坐标的符号在对称下如何变化</strong>」——不看图，光想对称，也能推出符号。</span>

**重点：** 三组公式里，正弦与余弦都变号（或不变号），但**正切的变号规律与正弦余弦一致**——凡是「正弦变号」的，正切也变号；凡是「余弦不变号」的，正切也不变号。因为正切是两个坐标之比 $y/x$，分子分母同时变号时比值不变，这正是 $\tan(\pi+\alpha) = \tan\alpha$ 与 $\tan(-\alpha) = -\tan\alpha$ 看似矛盾、实则一致的根源。

## 3 公式解析：从对称到公式（二）

以最「硬」的一组——$\pi + \alpha$——为例，把几何翻译成公式的完整链条拆成四步：

$$
\sin(\pi + \alpha) = -\sin\alpha, \qquad \cos(\pi + \alpha) = -\cos\alpha, \qquad \tan(\pi + \alpha) = \tan\alpha
$$

- **第一步，定位对称操作**：$\pi + \alpha$ 就是把角 $\alpha$ 的终边再旋转 $180^\circ$。在单位圆上，旋转 $180^\circ$ 等价于关于原点对称。
- **第二步，写出坐标变化**：点 $P(x,y)$ 关于原点对称，坐标变为 $P'(-x,-y)$。这是解析几何的常识：关于原点对称，横纵坐标都取相反数。
- **第三步，读出三角函数**：新角 $\pi+\alpha$ 的正弦是 $P'$ 的纵坐标 $-y$，余弦是横坐标 $-x$，正切是 $(-y)/(-x)$。
- **第四步，代回 $\alpha$**：由 $y = \sin\alpha$、$x = \cos\alpha$，得 $\sin(\pi+\alpha) = -\sin\alpha$、$\cos(\pi+\alpha) = -\cos\alpha$、$\tan(\pi+\alpha) = \tan\alpha$。

**要点：** 整套推演只用了一个几何事实（原点对称坐标变号）和一个定义（三角函数是坐标），没有任何需要背的东西。同样地，把「关于原点对称」换成「关于 x 轴对称」「关于 y 轴对称」，立刻得到 $-\alpha$ 与 $\pi-\alpha$ 两组公式。**诱导公式不是六条孤立的式子，而是一棵从「坐标如何变号」长出来的树。**

**辨析｜易错点：** 许多同学只背公式，忘了它们从哪里来，结果在 $\sin(\pi - \alpha)$ 上翻车——直觉上「$\pi$ 减去一个角」，似乎该变号，实际却是 $\sin(\pi-\alpha) = \sin\alpha$（不变号）。看图最清楚：$\alpha$ 在第二象限附近的亲戚 $\pi-\alpha$ 在第二象限，正弦为正，与 $\alpha$ 同号。**判断正负永远回到「象限定号」，不要凭感觉。**

## 4 公式（五）（六）：把余弦变成正弦

前面三组镜子的对称轴都是坐标轴，还有一种更妙的对称：**关于直线 $y = x$ 对称**。点 $P(x,y)$ 关于直线 $y=x$ 对称，坐标互换为 $P'(y, x)$。这个新点对应的角，恰好是 $\frac{\pi}{2} - \alpha$。于是：

$$\sin\left(\frac{\pi}{2} - \alpha\right) = x = \cos\alpha, \qquad \cos\left(\frac{\pi}{2} - \alpha\right) = y = \sin\alpha$$

这就是公式（五）——它说出了一件初中就见过的事：**互余的两个角，正弦与余弦互换**。$\sin 30^\circ = \cos 60^\circ$，正是公式（五）的数值化身。<span class="marginnote">公式（五）另一个写法是 $\sin\theta = \cos(\frac{\pi}{2}-\theta)$。这条「把正弦换成余弦」的桥梁，在推导两角和差公式时是关键的引子，下一阶段会反复用到。</span>

再看 $\frac{\pi}{2} + \alpha$：它是 $P(x,y)$ 绕原点逆时针旋转 $90^\circ$，坐标变为 $P'(-y, x)$。于是：

$$\sin\left(\frac{\pi}{2} + \alpha\right) = x = \cos\alpha, \qquad \cos\left(\frac{\pi}{2} + \alpha\right) = -y = -\sin\alpha$$

公式（五）（六）的价值在于**沟通正弦与余弦两个函数**。有了它们，「求余弦值」可以转化为「求正弦值」，反之亦然；后面推导 $\sin(\alpha+\beta)$ 时，正是靠公式（五）把正弦一步步拆进余弦的框架。

## 5 记忆口诀：奇变偶不变，符号看象限

六组公式数量不少，但可以用一句口诀一网打尽：

**「奇变偶不变，符号看象限」**，它适用于形如 $\dfrac{k\pi}{2} \pm \alpha$ 的一类角（$k$ 为整数，诱导公式里常见的 $k = 0, 1, 2, 3, 4$）。

- **奇变偶不变**：看 $k$ 的奇偶。若 $k$ 为偶数（如 $k = 0, 2, 4$，即 $\pm\alpha,\ \pi\pm\alpha,\ 2\pi\pm\alpha$），函数名**不变**（正弦还是正弦）；若 $k$ 为奇数（如 $k = 1, 3$，即 $\frac{\pi}{2}\pm\alpha,\ \frac{3\pi}{2}\pm\alpha$），函数名**要变**——正弦变余弦、余弦变正弦、正切变余切。
- **符号看象限**：把 $\alpha$ 暂时当作锐角，看 $\dfrac{k\pi}{2}\pm\alpha$ 落在哪个象限，再按**原来**那个函数的符号定正负。

**例：** 求 $\sin\left(\frac{3\pi}{2} + \alpha\right)$。$k = 3$ 为奇数，函数名要变，先写 $\pm\cos\alpha$；把 $\alpha$ 当锐角，$\frac{3\pi}{2}+\alpha$ 是第四象限角，第四象限正弦为负，故 $\sin\left(\frac{3\pi}{2}+\alpha\right) = -\cos\alpha$。<span class="marginnote">口诀的每一步其实都可以追溯到「对称 + 象限定号」。口诀只是把两步合一的速记，前提是你能随时回到单位圆图验证，否则口诀就退化成无意义的咒语。</span>

**辨析｜易错点：** 「符号看象限」里要看的**始终是原函数的符号**，不是变名后函数的符号。比如 $\cos\left(\frac{\pi}{2}+\alpha\right)$，先变名为 $\pm\sin\alpha$，再把 $\frac{\pi}{2}+\alpha$ 当第二象限角、余弦为负，所以取负号，得 $-\sin\alpha$。**「看象限」用的是把 $\alpha$ 当锐角时新角的象限，定号用的却是原来的 $\cos$，两个「原」「新」别搞混。**

把六组公式放在一起看，它们其实整齐得惊人：

| 角 | 正弦 | 余弦 | 正切 |
| --- | --- | --- | --- |
| $\alpha + 2k\pi$ | $\sin\alpha$ | $\cos\alpha$ | $\tan\alpha$ |
| $-\alpha$ | $-\sin\alpha$ | $\cos\alpha$ | $-\tan\alpha$ |
| $\pi-\alpha$ | $\sin\alpha$ | $-\cos\alpha$ | $-\tan\alpha$ |
| $\pi+\alpha$ | $-\sin\alpha$ | $-\cos\alpha$ | $\tan\alpha$ |
| $\frac{\pi}{2}-\alpha$ | $\cos\alpha$ | $\sin\alpha$ | $\cot\alpha$ |
| $\frac{\pi}{2}+\alpha$ | $\cos\alpha$ | $-\sin\alpha$ | $-\cot\alpha$ |

这张表里藏着规律：**「符号的变化」永远只跟角所在象限有关，「函数名的变化」永远只跟旋转半圈还是四分之一圈有关**。看表容易，更值得做的是合上表，从单位圆对称出发把每一行重新推一遍——推一遍胜过背十遍。

## 6 化简求值实战

把 $\frac{11\pi}{6}$ 化简到锐角。$\frac{11\pi}{6} = 2\pi - \frac{\pi}{6}$，用公式（三）的逆向（等价于 $\sin(2\pi-\theta) = -\sin\theta$）：$\sin\frac{11\pi}{6} = \sin\left(-\frac{\pi}{6}\right) = -\frac{1}{2}$。

再化简 $\cos\left(\frac{\pi}{2} - \alpha\right) + \sin\left(\pi + \alpha\right)$。分别套公式：$\cos\left(\frac{\pi}{2}-\alpha\right) = \sin\alpha$，$\sin(\pi+\alpha) = -\sin\alpha$，两项相加得 $0$。<span class="marginnote">化简的目标永远是把「陌生的角」归到「锐角」，再把锐角的函数值从特殊角表里读出。诱导公式就是那张「把大角变小角」的通行证。</span>

**例：化简 $\dfrac{\sin(\pi-\alpha)\cos\left(\frac{\pi}{2}+\alpha\right)}{\cos(2\pi-\alpha)}$。** 逐项归约：$\sin(\pi-\alpha) = \sin\alpha$；$\cos\left(\frac{\pi}{2}+\alpha\right) = -\sin\alpha$；$\cos(2\pi-\alpha) = \cos(-\alpha) = \cos\alpha$。代入得 $\dfrac{\sin\alpha \cdot (-\sin\alpha)}{\cos\alpha} = -\dfrac{\sin^2\alpha}{\cos\alpha}$。可见整套操作就是「先归约成 $\alpha$ 的函数，再用同角关系收尾」——上一节的基本关系在最后一步登场。

**例：知一求余。** 已知 $\alpha$ 是第三象限角，且 $\cos\alpha = -\dfrac{4}{5}$，求 $\sin(\pi+\alpha)$、$\cos(\pi-\alpha)$、$\tan(-\alpha)$。先由平方关系求 $\sin\alpha$：$\sin^2\alpha = 1 - \cos^2\alpha = 1 - \dfrac{16}{25} = \dfrac{9}{25}$；第三象限正弦为负，$\sin\alpha = -\dfrac{3}{5}$。再逐项套诱导公式：$\sin(\pi+\alpha) = -\sin\alpha = \dfrac{3}{5}$；$\cos(\pi-\alpha) = -\cos\alpha = \dfrac{4}{5}$；$\tan(-\alpha) = -\tan\alpha = -\dfrac{\sin\alpha}{\cos\alpha} = -\dfrac{-3/5}{-4/5} = -\dfrac{3}{4}$。**这里诱导公式负责「变角」，同角关系负责「变名求值」，两者接力，正是后续所有三角计算的固定范式。**

## 7 小结

- 诱导公式的**源头是单位圆的对称性**：x 轴对称得 $-\alpha$，y 轴对称得 $\pi-\alpha$，原点对称得 $\pi+\alpha$，$y=x$ 对称得 $\frac{\pi}{2}-\alpha$，旋转 $90^\circ$ 得 $\frac{\pi}{2}+\alpha$。
- 每组公式都是「**坐标变号 + 三角函数定义**」的推论，无需死记；符号一律由象限定。
- 公式（五）（六）**沟通正弦与余弦**，是后续推导两角和差公式的桥梁。
- 记忆口诀「**奇变偶不变，符号看象限**」适用 $\frac{k\pi}{2}\pm\alpha$ 型角，使用时务必分清「变名看 $k$、定号看原函数、象限按 $\alpha$ 当锐角」。
- 化简流程：**大角归锐角 → 套公式 → 用同角关系收尾**。

在下一节，我们将把三角函数当成完整的「函数」来研究——画它的图象、讨论它的周期、单调与最值。诱导公式（一）所揭示的「$2\pi$ 的不变性」，将在那里升级为**周期性**的正式定义。
