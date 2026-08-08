---
title: 同角三角函数关系
date: 2026-08-07
---

# 同角三角函数关系

<div class="epigraph">
<p>同一个角的三个三角函数并非独立——它们被两条恒等式紧紧绑在一起。</p>
<footer>—— 欧几里得（Euclid，《几何原本》，勾股之裔）</footer>
</div>

<div class="article-byline">
<p>第一级 · 初等几何与三角 ｜ 人教A版 必修第一册 §5.2 ｜ 2026-08-07</p>
</div>

## 为什么从同角关系开始

同一个角 $\theta$ 的正弦、余弦、正切，并不是三个独立的量——它们之间被两条恒等式锁死：

$$\sin^2\theta + \cos^2\theta = 1, \qquad \tan\theta = \frac{\sin\theta}{\cos\theta}$$

这两条式子叫**同角三角函数关系**，是三角变形最重要的「转换器」：知道一个三角函数值，就能解出其余所有值；遇到含多种三角函数的式子，靠它们统一成单一函数。从「从极限到大模型」的主线看，这两条恒等式是后续一切三角恒等变换（诱导、和差角、二倍角）的地基，也是「用约束减少自由度」思想的第一个实例。

## 1 两条基本关系

**同角三角函数关系**包括两个部分：

- **平方关系**：$\sin^2\theta + \cos^2\theta = 1$（对一切 $\theta$ 成立）；
- **商数关系**：$\tan\theta = \frac{\sin\theta}{\cos\theta}$（$\cos\theta \neq 0$）。

<span class="marginnote">「同角」强调必须是对<strong>同一个角</strong>才成立：$\sin^2\theta + \cos^2\theta = 1$ 中的 $\theta$ 是同一个角。写成 $\sin^2\alpha + \cos^2\beta = 1$ 是错的（除非 $\alpha = \beta$）。这是同角关系最容易出错的地方。</span>

两条关系的几何来源都很清楚：在单位圆上，点 $(\cos\theta, \sin\theta)$ 满足 $x^2 + y^2 = 1$，即平方关系（勾股定理）；$\tan\theta = \frac{y}{x} = \frac{\sin\theta}{\cos\theta}$，即商数关系。

## 2 公式解析：知一求其余

同角关系最经典的应用是「**已知一个三角函数值，求其余三角函数值**」。设 $\sin\theta = \frac{3}{5}$，且 $\theta$ 在第二象限，求 $\cos\theta$ 与 $\tan\theta$。

完整步骤：

**第一步，平方关系求 $\cos$**：$\cos^2\theta = 1 - \sin^2\theta = 1 - \frac{9}{25} = \frac{16}{25}$，所以 $\cos\theta = \pm \frac{4}{5}$。
**第二步，符号由象限定**：$\theta$ 在第二象限，$\cos\theta < 0$，取 $\cos\theta = -\frac{4}{5}$。
**第三步，商数关系求 $\tan$**：$\tan\theta = \frac{\sin\theta}{\cos\theta} = \frac{3/5}{-4/5} = -\frac{3}{4}$。

**重点：** 「**先求绝对值，再定符号**」是这个流程的灵魂——平方关系给出两个解（正负），象限决定取舍。凡是「知一求余」题，都走这条「平方 → 定号 → 相除」的流程。<span class="marginnote">为什么开方会产生两个解？因为 $\cos^2\theta$ 相同的角有两个（关于 $x$ 轴对称的终边）。象限信息就是用来「二选一」的。若题目没给象限，则需按角所在象限分类讨论。</span>

## 3 恒等变形：统一函数与化简

同角关系的第二个大用途是「**化简与统一**」——把含多种三角函数的式子化成只含一种，或化成最简。常用技巧：

**「1」的代换**：把 1 换成 $\sin^2\theta + \cos^2\theta$，例如 $\frac{\cos^2\theta}{1 - \sin\theta} = \frac{1 - \sin^2\theta}{1 - \sin\theta} = 1 + \sin\theta$；
**弦切互化**：遇到 $\tan$，写成分式 $\frac{\sin}{\cos}$，通分合并；
**齐次式**：$\frac{a\sin\theta + b\cos\theta}{c\sin\theta + d\cos\theta}$ 上下同除 $\cos\theta$，变成 $\frac{a\tan\theta + b}{c\tan\theta + d}$——把多元问题变成一元。

<span class="marginnote">「齐次式同除 $\cos\theta$」是一个漂亮的技巧：式子分子分母的次数相同（都是关于 $\sin$、$\cos$ 的一次），同除 $\cos\theta$ 后全变成 $\tan\theta$。这种「降元」思想在后续所有三角恒等变换里反复出现。</span>

**辨析｜易错点：** 化简时「消去 $\sin\theta$」这样的操作要小心**定义域**：$\sin\theta$ 可能为 0，不能两边直接约去。三角化简讲究「等价变形」——每一步都要保证前后式子对同一自变量取值都成立，否则会「增根」或「漏根」。

## 4 公式解析：$\sin\theta \pm \cos\theta$ 与 $\sin\theta\cos\theta$ 的互推

同角关系还藏着一组「快捷换算」：设 $t = \sin\theta + \cos\theta$，则

$$
(\sin\theta + \cos\theta)^2 = 1 + 2\sin\theta\cos\theta
$$

所以 $\sin\theta\cos\theta = \frac{t^2 - 1}{2}$。对这条式子做三步拆解：

- **第一步，平方展开**：$(\sin\theta + \cos\theta)^2 = \sin^2\theta + 2\sin\theta\cos\theta + \cos^2\theta$。
- **第二步，代平方关系**：$\sin^2\theta + \cos^2\theta = 1$，于是 $(\sin\theta + \cos\theta)^2 = 1 + 2\sin\theta\cos\theta$。
- **第三步，反解**：移项得 $\sin\theta\cos\theta = \frac{(\sin\theta+\cos\theta)^2 - 1}{2}$。同理 $\sin\theta - \cos\theta$ 的平方是 $1 - 2\sin\theta\cos\theta$。

这组关系把「和」与「积」互相转化，在求值题里是「已知 $a+b$ 求 $ab$」的三角版本——与代数的韦达定理异曲同工。<span class="marginnote">注意范围约束：$\sin\theta + \cos\theta \in [-\sqrt{2}, \sqrt{2}]$，因为 $(\sin\theta+\cos\theta)^2 = 1 + \sin 2\theta \le 2$。这种「由恒等式反推范围」的约束，是后面《三角恒等变换》里求值、求最值的重要素材。</span>

## 5 同角关系的统一视角

同角关系本质上是在说：**$\sin$、$\cos$、$\tan$ 之间只有「两个自由度」**——知道其中一个，其他就都确定了。这对应单位圆上「一个点由角度唯一决定」的事实。

更进一步，用复数（欧拉公式）看同角关系会更统一：$e^{i\theta} = \cos\theta + i\sin\theta$，则 $|e^{i\theta}| = \sqrt{\cos^2\theta + \sin^2\theta} = 1$——平方关系就是「单位圆上点的模长为 1」的复数版本。<span class="marginnote">欧拉公式 $e^{i\theta} = \cos\theta + i\sin\theta$ 是三角学的「终极统一视角」：把三个函数塞进一个指数函数。到第二级《复变函数与积分变换》，你会看到三角恒等式几乎全部可以由指数运算推导——那时回头看同角关系，会格外亲切。</span>

同角关系还有一个「结构」层面的意义：它说明**正弦与余弦不是两个独立函数，而是「同一个点」的两个坐标**。知道了 $\sin\theta$，$\cos\theta$ 只剩「正负号」的自由度；知道了 $\tan\theta$，能同时解出 $\sin$ 与 $\cos$ 的比例关系。这种「自由度收缩」是「约束」的本质——两条恒等式把三个函数的自由度从 3 压到 1（角本身）。

| 已知 | 能求 | 求法 | 需要额外信息 |
| --- | --- | --- | --- |
| $\sin\theta$ | $\cos\theta, \tan\theta$ | 平方关系 + 商数关系 | 象限（定符号） |
| $\cos\theta$ | $\sin\theta, \tan\theta$ | 平方关系 + 商数关系 | 象限 |
| $\tan\theta$ | $\sin\theta, \cos\theta$ | 联立两关系 | 象限 |

这张表统一了「知一求二」的全部情形：**任意一个三角函数值都能推出其余两个，但符号必须由象限定**。这正是「同角关系 + 象限」这套组合拳的全部内涵。

<span class="marginnote">「已知一个量求其余」的流程里，象限是「最终裁判」——它决定开方后的正负号。这也是为什么解三角题的第一步往往是「先判断角在哪个象限」：象限信息看起来琐碎，却决定了答案的唯一性。在后续「给值求角」「解三角形」里，范围与象限判断永远是解题的第一道工序。</span>

同角关系在「证明恒等式」中的应用还有一个常用策略——**「1」的代换**：把式子里的常数 1 换成 $\sin^2\theta + \cos^2\theta$，往往能让分式通分后出现可约分的结构。例如 $\frac{1}{1 - \sin\theta}$ 乘上 $\frac{1 + \sin\theta}{1 + \sin\theta}$，用平方关系把分子化成 $\cos^2\theta$，就得到 $\frac{\cos^2\theta}{1-\sin\theta} = 1 + \sin\theta$。这种「主动造差平方」的手法，是三角化简中最常用的「魔术」。

最后，从「函数」的视角看同角关系：它们不是「方程」（不是只有在某些 $\theta$ 才成立），而是「恒等式」（对一切 $\theta$ 成立）。区分「方程」与「恒等式」是三角学的重要素养——解方程找「使等式成立的 $\theta$」，化简恒等式则把式子变成「对一切 $\theta$ 都成立的等价形式」。前者是「求解」，后者是「变形」。

同角关系还有一个「统一记忆」的角度：两条关系都可以从**同一个直角三角形**读出。在直角三角形中，设锐角 $\theta$，对边 $a$、邻边 $b$、斜边 $c$，则

$$
\sin\theta = \frac{a}{c}, \quad \cos\theta = \frac{b}{c}, \quad \tan\theta = \frac{a}{b}
$$

于是 $\sin^2\theta + \cos^2\theta = \frac{a^2 + b^2}{c^2} = \frac{c^2}{c^2} = 1$（勾股定理），而 $\frac{\sin\theta}{\cos\theta} = \frac{a/c}{b/c} = \frac{a}{b} = \tan\theta$——两条同角关系全部由勾股定理与边比定义直接得到。

| 关系 | 直角三角形来源 | 单位圆来源 |
| --- | --- | --- |
| $\sin^2+\cos^2=1$ | 勾股定理 | $x^2+y^2=1$ |
| $\tan=\frac{\sin}{\cos}$ | 边比定义 | 坐标之比 |

两条来源殊途同归：直角三角形给出「锐角」视角，单位圆给出「任意角」视角。**同一关系，两种视角**——这是「数形结合」的又一体现，也说明「同角关系」不是偶然的恒等式，而是几何事实的代数化。

<span class="marginnote">「一个公式，直角三角形与单位圆两种解释」的方法，在三角学里处处可用：理解任何三角恒等式，都问一句「它在直角三角形里意味着什么？在单位圆里呢？」——两个视角互相印证，记忆与理解都会加深。</span>

「知一求二」还有一个「便捷表」值得整理：已知 $\tan\theta = t$，则

$$
\sin\theta = \pm \frac{t}{\sqrt{1 + t^2}}, \qquad \cos\theta = \pm \frac{1}{\sqrt{1 + t^2}}
$$

（符号由象限决定）。这个「由 $\tan$ 直接得 $\sin$、$\cos$」的公式，是「知一求二」的高频加速器——已知斜率或坡度（都是 $\tan$）时尤其好用。它的推导也很干净：$\sin = \tan \cdot \cos$，再代入 $\sin^2 + \cos^2 = 1$ 解出。

## 6 小结

- **两条同角关系**：$\sin^2\theta + \cos^2\theta = 1$（平方）、$\tan\theta = \frac{\sin\theta}{\cos\theta}$（商数），都来自单位圆。
- **知一求余**流程：平方关系求绝对值 → 象限定符号 → 商数关系求 $\tan$。
- 化简技巧：「1」的代换、弦切互化、齐次式同除 $\cos\theta$；注意等价变形与定义域。
- $\sin\theta \pm \cos\theta$ 与 $\sin\theta\cos\theta$ 通过平方互相转化，且 $\sin\theta + \cos\theta$ 有界 $[-\sqrt2, \sqrt2]$。
- 同角关系 = 「一个角决定一切」，复数视角下是 $|e^{i\theta}| = 1$。

在下一节，我们将解决「不同角的三角函数如何互相转换」——学习**诱导公式**，把任意角化归到锐角。
