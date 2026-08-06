---
title: 同角三角函数的基本关系
date: 2026-08-07
---

# 同角三角函数的基本关系

<div class="epigraph">
<p>万物皆数。</p>
<footer>—— 毕达哥拉斯（Pythagoras，公元前六世纪，据传）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.2.2 ｜ 2026-08-07</p>
</div>

## 为什么从同角三角函数的基本关系开始

上一节我们用同一个交点 $P(x,y)$ 定义了同一个角 $\alpha$ 的三个三角函数：$\sin\alpha = y$，$\cos\alpha = x$，$\tan\alpha = \frac{y}{x}$。同一个点、同一组坐标，产出的三个函数之间**不可能彼此无关**——它们必然被单位圆方程 $x^2 + y^2 = 1$ 紧紧绑在一起。<span class="marginnote">这就是「基本关系」四个字的含义：不是推导出来的技巧，而是定义自带的、几乎无需验证的必然联系。</span>

这一节把这层绑定提炼成**两组公式**。它们是整个三角恒等变换世界的公理：后续的诱导公式、两角和与差、二倍角、辅助角公式，无一例外都要从这两组关系出发。<span class="marginnote">把「少而真的公理」推演成「多而强的定理」，是数学最经典的工作方式。这两组关系就是三角函数领域的第一批公理，到第二级《抽象代数》你还会看到同样的模式。</span>

## 1 平方关系：单位圆方程的直接翻译

设角 $\alpha$ 的终边与单位圆交于 $P(x,y)$，则 $P$ 在单位圆上，故 $x^2 + y^2 = 1$。把上一节的定义 $x = \cos\alpha$，$y = \sin\alpha$ 代入，立刻得到：

$$\boxed{\sin^2 \alpha + \cos^2 \alpha = 1}$$

这条公式对**一切实数 $\alpha$** 都成立，所以叫**平方关系（Pythagorean identity）**。<span class="marginnote">它本质上是勾股定理的化身：在单位圆里，$|\sin\alpha|$ 与 $|\cos\alpha|$ 是两条直角边，$1$ 是斜边——「万物皆数」与「万物皆几何」在这里合流了。</span>

**辨析｜易错点：** 记号 $\sin^2 \alpha$ 表示 $(\sin\alpha)^2$，即先求正弦再平方；它不是 $\sin(\alpha^2)$，也不是 $\sin \alpha \cdot \alpha$。这个记号约定是三角函数特有的，后面所有恒等式都建立在这条约定上。

## 2 商数关系：正切定义的重述

同理，把 $\tan\alpha = \frac{y}{x}$ 里的 $x, y$ 换成三角函数：

$$\boxed{\tan \alpha = \frac{\sin \alpha}{\cos \alpha}}$$

**这条公式要求 $\cos\alpha \neq 0$，即 $\alpha \neq \frac{\pi}{2} + k\pi\ (k \in \mathbb{Z})$。** 这与上一节讨论的 $\tan$ 定义域完全一致——终边落在 $y$ 轴上时，横坐标 $x = \cos\alpha = 0$，商无意义。<span class="marginnote">注意：商数关系不是「新」的知识，它只是把定义 $\tan\alpha = y/x$ 用 $\sin, \cos$ 的语言复述了一遍。真正的「关系」藏在平方关系里。</span>

## 3 公式解析：两组关系为什么必然成立

$$
\sin^2 \alpha + \cos^2 \alpha = 1, \qquad \tan \alpha = \frac{\sin \alpha}{\cos \alpha} \ (\cos\alpha \neq 0)
$$

把这两条公式从「记忆」变成「推导」，只需四步，每一步都对应上一节的一个动作：

- **第一步，回到单位圆**：角 $\alpha$ 的终边与单位圆交于 $P(x,y)$，且 $x^2 + y^2 = 1$。这里唯一用到的几何事实是「$P$ 在单位圆上」。
- **第二步，代换坐标**：把定义 $x = \cos\alpha,\ y = \sin\alpha$ 代入圆的方程，得到 $\cos^2\alpha + \sin^2\alpha = 1$——平方关系就此成立，对一切实数 $\alpha$。
- **第三步，作商**：由 $y = \sin\alpha$ 除以 $x = \cos\alpha$（在 $x \neq 0$ 时），得到 $\dfrac{\sin\alpha}{\cos\alpha} = \dfrac{y}{x}$。
- **第四步，认出正切**：等式右边正是上一节定义的 $\tan\alpha$，于是商数关系成立。

**要点：** 两条公式都不是「编出来的」，而是「单位圆方程 + 坐标定义」的组合推论。真正需要记住的新东西只有两条——圆方程 $x^2+y^2=1$ 与定义 $\sin = y, \cos = x, \tan = y/x$，其余全部可以现推。

## 4 应用一：知一求其余

知道 $\sin\alpha,\ \cos\alpha,\ \tan\alpha$ 中的任意一个，配合角 $\alpha$ 所在的象限，就能求出另外两个。以已知 $\sin\alpha$ 为例：

$$\cos\alpha = \pm \sqrt{1 - \sin^2\alpha}, \qquad \tan\alpha = \frac{\sin\alpha}{\cos\alpha}$$

**开平方的「$\pm$」怎么取舍？由象限定。** 这是本类题目的唯一关键步骤。<span class="marginnote">「知一求其余」是解三角形的经典题型：一个已知量加一个象限条件，恰好定死两个未知量的符号。少了象限条件，$\cos\alpha$ 会剩两个候选值。</span>

**例：** 已知 $\sin\alpha = \dfrac{3}{5}$，且 $\alpha$ 是第二象限角，求 $\cos\alpha$ 与 $\tan\alpha$。

- 第一步，代平方关系：$\cos^2\alpha = 1 - \sin^2\alpha = 1 - \dfrac{9}{25} = \dfrac{16}{25}$；
- 第二步，开方：$\cos\alpha = \pm \dfrac{4}{5}$；
- 第三步，定号：$\alpha$ 在第二象限，余弦为负，取 $\cos\alpha = -\dfrac{4}{5}$；
- 第四步，作商：$\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha} = \dfrac{3/5}{-4/5} = -\dfrac{3}{4}$。

**例 2：** 已知 $\cos\alpha = -\dfrac{12}{13}$，且 $\alpha$ 是第三象限角，求 $\sin\alpha$ 与 $\tan\alpha$。平方关系给出 $\sin^2\alpha = 1 - \cos^2\alpha = 1 - \dfrac{144}{169} = \dfrac{25}{169}$，开方得 $\sin\alpha = \pm\dfrac{5}{13}$；第三象限正弦为负，取 $\sin\alpha = -\dfrac{5}{13}$。再作商，$\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha} = \dfrac{-5/13}{-12/13} = \dfrac{5}{12}$。注意第三象限的正切为正——这与「一全正，二正弦，三正切，四余弦」的口诀完全吻合。

**辨析｜易错点：** 此类题错误率最高的一步是开方后**忘掉负号**，或把象限与符号对应错。记住口诀：$\sin$ 看 $y$、$\cos$ 看 $x$，第二象限 $x<0$、$y>0$，所以 $\cos\alpha$ 必为负、$\sin\alpha$ 必为正，没有商量余地。先定号，再代绝对值，顺序不要颠倒。

## 5 应用二：化简与证明

两组基本关系不仅是「求值工具」，更是「变形工具」。化简与证明恒等式，万变不离四招：

- **化切为弦**：把 $\tan$ 换成 $\dfrac{\sin}{\cos}$，统一成弦函数再处理；
- **平方公式**：$1 = \sin^2\alpha + \cos^2\alpha$，必要时把 $1$ 写成平方和；
- **因式分解**：遇到 $\sin^4\alpha - \cos^4\alpha$ 之类，用平方差公式拆开；
- **通分合并**：分式形式的恒等式，先通分再化简。

**例 1（化简）：** 化简 $(1+\sin\alpha)(1-\sin\alpha)$。

$$(1+\sin\alpha)(1-\sin\alpha) = 1 - \sin^2\alpha = \cos^2\alpha$$

用平方差公式展开，再用平方关系消掉 $1-\sin^2\alpha$——两步完成。

**例 2（求值）：** 已知 $\tan\alpha = 2$，求 $\dfrac{\sin\alpha + \cos\alpha}{\sin\alpha - \cos\alpha}$ 的值。分子分母同除以 $\cos\alpha$（在 $\cos\alpha \neq 0$ 时），得 $\dfrac{\tan\alpha + 1}{\tan\alpha - 1} = \dfrac{2+1}{2-1} = 3$。**这类题的关键是「化切为弦」的逆向——把分式整体除以 $\cos\alpha$，让分子分母只含 $\tan\alpha$。** 它比先求 $\sin,\cos$ 再代入快得多，也是后续处理齐次式的标准手法。

**例 3（证明）：** 证明 $\dfrac{\cos\alpha}{1-\sin\alpha} = \dfrac{1+\sin\alpha}{\cos\alpha}$。

- 第一步，观察两边都含弦，无需化切；
- 第二步，把左边分子分母同乘 $1+\sin\alpha$：$\dfrac{\cos\alpha(1+\sin\alpha)}{1-\sin^2\alpha}$；
- 第三步，分母用平方关系换成 $\cos^2\alpha$：$\dfrac{\cos\alpha(1+\sin\alpha)}{\cos^2\alpha}$；
- 第四步，约去 $\cos\alpha$（在 $\cos\alpha \neq 0$ 的范围内），得 $\dfrac{1+\sin\alpha}{\cos\alpha}$，与右边相等。证毕。

**例 4（化简）：** 化简 $\sin^4\alpha + \cos^4\alpha$。先补全平方：$\sin^4\alpha + \cos^4\alpha = (\sin^2\alpha + \cos^2\alpha)^2 - 2\sin^2\alpha\cos^2\alpha = 1 - 2\sin^2\alpha\cos^2\alpha$。**这里用到的技巧是「平方的平方」——把 $\sin^4 + \cos^4$ 看成 $(a+b)^2$ 去掉中间项**，再借平方关系消掉 $a+b$。

**例 5（化简）：** 化简 $\sin^2\alpha + \cos^4\alpha + \sin^2\alpha\cos^2\alpha$。后两项提公因式 $\cos^2\alpha$：$\sin^2\alpha + \cos^2\alpha(\cos^2\alpha + \sin^2\alpha) = \sin^2\alpha + \cos^2\alpha = 1$。这一连串化简的妙处在于反复利用「$1$ 可以写成 $\sin^2+\cos^2$」——**把 1 换成平方和，或把平方和换回 1，是恒等变形最常用的两个方向。**

**辨析｜易错点：** 证明恒等式时**不要「两边同乘」约分的前提**——只有在 $\cos\alpha \neq 0$ 时才可约分。严格的做法是注明「在 $\cos\alpha \neq 0$ 的条件下」；若条件未给，则这类约分须谨慎，考试中通常默认讨论其定义域内的恒等性。<span class="marginnote">「约分」背后的逻辑是「等式两边同除一个非零数」，分母为 0 的地方要单独说明。这种对定义域的警觉，是高中向大学过渡的重要一步，在第二级《数学分析》里会被反复强调。</span>

**辨析｜易错点：「同角」必须是同一个角。** $\sin^2\alpha + \cos^2\beta = 1$ 只在 $\alpha = \beta$ 时成立；$\tan\frac{\alpha}{2}$ 与 $\frac{\sin\alpha}{\cos\alpha}$ 也不是商数关系的直接对象。动笔之前，先确认式子里出现的所有三角函数对应的是**同一个记号**——「同角」两个字是整个「基本关系」成立的唯一前提。

**几个立刻有用的变式。** 把平方关系与商数关系各自「反着用」，能得到一组常用变式：

$$\sin^2\alpha = 1 - \cos^2\alpha, \qquad \cos^2\alpha = 1 - \sin^2\alpha, \qquad \sin\alpha = \tan\alpha \cdot \cos\alpha$$

化简时，这三条变式往往比原式更顺手。**把「平方关系」记成「三者知一求二」而不仅是「一个等式」，用起来才灵活**——这正是下一节诱导公式里反复使用的姿势。

## 6 小结

- **两组基本关系**：平方关系 $\sin^2\alpha + \cos^2\alpha = 1$（对一切实数 $\alpha$ 成立）；商数关系 $\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha}$（$\cos\alpha \neq 0$）。
- **记号的约定**：$\sin^2\alpha = (\sin\alpha)^2$，与 $\sin(\alpha^2)$ 完全不同。
- **知一求其余**：开方取 $\pm$ 后，由角所在象限定号；第二象限余弦为负，第三象限正切为正。
- **化简证明四招**：化切为弦、平方公式（把 $1$ 写成 $\sin^2+\cos^2$）、因式分解、通分合并；注意约分需保证分母非零。
- **地位**：两组关系是整个三角恒等变换的公理，后续所有公式都由它推演。

在下一节，我们要解决「不同角之间」的转化问题：$\sin(\alpha + \pi)$、$\sin(-\alpha)$、$\sin(\pi - \alpha)$ 这类角，与 $\sin\alpha$ 有什么关系？这是**诱导公式**——三角函数的又一组「对称操作」。
