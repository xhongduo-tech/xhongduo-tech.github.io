---
title: 二倍角的正弦、余弦、正切公式
date: 2026-08-07
---

# 二倍角的正弦、余弦、正切公式

<div class="epigraph">
<p>一切皆流，无物常住。</p>
<footer>—— 赫拉克利特（Heraclitus，πάντα ῥεῖ）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.5.2 ｜ 2026-08-07</p>
</div>

## 为什么从二倍角公式开始

上一节我们把「两个角之和」的正弦、余弦、正切写成了公式。今天只做一件看起来很小的改动：**让两个角相等**，令 $\beta = \alpha$。这一「折叠」，竟然把三角恒等变换的整个下半场打开了：降幂公式、半角公式、辅助角公式，乃至后面解三角形、物理里的简谐运动，全都挂在二倍角公式这条藤上。<span class="marginnote">「让变量取特殊值」是数学里最常用的一招：一个一般结论在特殊代入下会吐出新的结论。两角和公式代入 $\beta=\alpha$ 得到二倍角，二倍角再换元 $2\alpha \to \theta$ 又得到半角公式——同一条公式，换一个角度读，就换了身份。</span> 学会二倍角公式，等于学会了一种**看公式的方式**：不是背四条公式，而是「把一般公式折叠成特殊公式」这一套操作。

## 1 从两角和公式出发

回顾两角和公式（上一节 §5.5.1 的结论）：

$$
\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta
$$

$$
\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta
$$

$$
\tan(\alpha+\beta)=\frac{\tan\alpha+\tan\beta}{1-\tan\alpha\tan\beta}
$$

现在做唯一的一步操作：**令 $\beta=\alpha$**。于是 $\alpha+\beta$ 变成 $2\alpha$，右边出现 $\sin\alpha\cos\alpha$、$\cos^2\alpha$、$\sin^2\alpha$ 这些「平方项」。整理后得到二倍角公式的核心三式：

$$
\sin 2\alpha = 2\sin\alpha\cos\alpha, \qquad
\cos 2\alpha = \cos^2\alpha - \sin^2\alpha
$$

$$
\tan 2\alpha = \frac{2\tan\alpha}{1-\tan^2\alpha}
$$

这一步看起来平淡，却是整节的枢纽：**二倍角公式不是新东西，它只是两角和公式在 $\beta=\alpha$ 处的取值**。理解这一点，就不用死记——忘了公式时，从两角和公式推一遍即可。<span class="marginnote">推导本身就是记忆。数学里值得背的结论极少，真正值得练的是「从更一般的结论现推」的能力。二倍角、半角、诱导公式都可以在两角和公式这条根上重新长出来。</span>

## 2 余弦二倍角的三种写法

正弦和正切的二倍角公式只有一种写法，唯独**余弦**的二倍角公式长着三张脸：

$$
\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha
$$

为什么它有三种形式？因为余弦二倍角里同时混着 $\cos^2\alpha$ 与 $\sin^2\alpha$ 两个平方项，而**同角平方关系** $\sin^2\alpha+\cos^2\alpha=1$ 允许我们用其中一个替换另一个：把 $\sin^2\alpha=1-\cos^2\alpha$ 代入，得 $2\cos^2\alpha-1$；把 $\cos^2\alpha=1-\sin^2\alpha$ 代入，得 $1-2\sin^2\alpha$。

这三张脸各自有用武之地：在化简中，「哪个三角函数更方便」就选用哪个形式。更重要的是一旦**反着读**，立刻得到**降幂公式**：

$$
\cos^2\alpha=\frac{1+\cos 2\alpha}{2}, \qquad
\sin^2\alpha=\frac{1-\cos 2\alpha}{2}
$$

降幂公式是二倍角公式最隐蔽也最值钱的应用：把「平方」降成「一次」，从而把含 $\sin^2\alpha$、$\cos^2\alpha$ 的式子改写为只含 $\cos 2\alpha$ 的式子，为后续积分、求周期、研究最值扫清障碍。<span class="marginnote">「降幂」等价于「升角」：平方换成二倍角。这一思想在第二级《高等数学》求 $\int \sin^2 x\,dx$ 时是必用的手法——高一埋的种子，大学才开花。</span>

## 3 公式解析：把 $\cos 2\alpha = 2\cos^2\alpha - 1$ 拆开

选「三脸」中的 $2\cos^2\alpha-1$ 这一式做三步拆解，因为它最能说明二倍角公式的结构：

- **第一步，从哪里来**：由 $\cos 2\alpha=\cos^2\alpha-\sin^2\alpha$ 出发——这本身又是两角和公式令 $\beta=\alpha$ 的结果。
- **第二步，消去一个平方项**：利用 $\sin^2\alpha=1-\cos^2\alpha$，代入得 $\cos^2\alpha-(1-\cos^2\alpha)=2\cos^2\alpha-1$。注意这里**用掉了同角平方关系**，所以这条公式的有效范围仍是一切实数 $\alpha$。
- **第三步，反着读**：把等式两边交换次序并除以 2，得到 $\cos^2\alpha=\frac{1+\cos2\alpha}{2}$——「平方」被降到「一次」，这就是降幂公式。**同一条等式，正着读是升幂，反着读是降幂**，看你要往哪个方向化简。

## 4 应用与易错辨析

### 倍角的「相对性」

「二倍」不是针对某个固定数字，而是**相对关系**：$4\alpha$ 是 $2\alpha$ 的二倍，$\frac{\alpha}{2}$ 是 $\frac{\alpha}{4}$ 的二倍。于是

$$
\sin 4\alpha = 2\sin 2\alpha\cos 2\alpha, \qquad
\cos\frac{\alpha}{2}=\cos^2\frac{\alpha}{4}-\sin^2\frac{\alpha}{4}
$$

**辨析｜易错点：** 二倍角公式里的角**必须成二倍关系**，$\sin 2\alpha$ 不等于 $2\sin\alpha$。最经典的错误是把 $\sin 2\alpha$ 拆成 $2\sin\alpha$ 或把 $\cos 2\alpha$ 记成 $2\cos\alpha$——前者忘记系数 2 里还藏着一个「把 $\cos\alpha$ 配上去」的乘法，后者则根本漏掉了公式结构。另一个高频错误是**忘记 $\tan 2\alpha$ 的定义域**：分母 $1-\tan^2\alpha\neq 0$，即 $\tan\alpha\neq \pm1$，同时 $\alpha$ 本身还要使 $\cos\alpha\neq 0$、$\cos2\alpha\neq0$。

### 求值实例

已知 $\cos\alpha=\frac{3}{5}$，且 $\alpha$ 在第四象限，求 $\sin2\alpha$。先由平方关系得 $\sin\alpha=-\frac{4}{5}$（第四象限正弦为负），代入：

$$
\sin2\alpha=2\sin\alpha\cos\alpha=2\cdot\left(-\frac{4}{5}\right)\cdot\frac{3}{5}=-\frac{24}{25}
$$

注意这里必须先定 $\sin\alpha$ 的**符号**再代公式——二倍角公式本身不含符号信息，符号由 $\alpha$ 所在的象限决定。<span class="marginnote">解这类题有一条不变的顺序：先定位象限定符号，再代入公式。凡是不先定符号就代公式的，一半以上的错误都出在这里。</span>

## 5 例题精讲：二倍角公式的灵活运用

二倍角公式的考题考「正用、逆用、变用」。看两道综合题。

### 题一：正用与逆用

已知 $\cos\alpha=\frac35$（$0<\alpha<\frac\pi2$），求 $\sin2\alpha$ 与 $\cos2\alpha$。

**第一步，求 $\sin\alpha$**：$\sin\alpha=\sqrt{1-\cos^2\alpha}=\sqrt{1-\frac9{25}}=\frac45$（第一象限取正）。
**第二步，正用**：$\sin2\alpha=2\sin\alpha\cos\alpha=2\times\frac45\times\frac35=\frac{24}{25}$。
**第三步，逆用**：$\cos2\alpha=\cos^2\alpha-\sin^2\alpha=\frac9{25}-\frac{16}{25}=-\frac7{25}$——也可用 $2\cos^2\alpha-1=\frac{18}{25}-1=-\frac7{25}$，两种写法一致。

<span class="marginnote">二倍角公式的「正用」（$\alpha\to2\alpha$）与「逆用」（$2\alpha\to\alpha$）要灵活切换。<strong>「先定象限定符号，再代入公式」是不变纪律</strong>——$\sin\alpha=\frac45$ 的符号由 $\alpha$ 所在象限决定，本题第一象限取正。$\cos2\alpha$ 的三种形式选「已知哪个用哪个」：已知 $\cos\alpha$ 用 $2\cos^2\alpha-1$ 最直接。</span>

### 题二：降幂公式的应用

求函数 $y=\sin^2x$ 的最小正周期。

**第一步，降幂**：$\sin^2x=\frac{1-\cos2x}{2}$——平方降成一次，出现 $\cos2x$。
**第二步，看周期**：$\cos2x$ 的周期是 $\frac{2\pi}{2}=\pi$，故 $y=\frac{1-\cos2x}{2}$ 的周期为 $\pi$。
**第三步，对比**：若直接用 $\sin^2x$，不易看出周期；降幂后「$\sin^2x$ 的周期是 $\pi$」一目了然。

<span class="marginnote">「降幂求周期/最值」是二倍角公式最值钱的应用：<strong>把 $\sin^2x$、$\cos^2x$ 降成 $\cos2x$ 的一次式，周期、最值、单调性全变得清晰</strong>。$y=\sin^2x$ 的周期是 $\pi$（不是 $2\pi$），因为平方让波形「翻倍」——这个结论用降幂公式一眼看出。降幂公式与辅助角公式配合，能处理几乎所有「三角函数最值与周期」问题。</span>

**辨析｜易错点（补充）：** 一是**符号**——$\cos2\alpha=\cos^2\alpha-\sin^2\alpha=-\frac7{25}$，别把 $\cos^2-\sin^2$ 当成 $\sin^2-\cos^2$；二是**降幂公式记反**——$\sin^2x=\frac{1-\cos2x}{2}$（减号）、$\cos^2x=\frac{1+\cos2x}{2}$（加号），符号相反；三是**周期**——$\cos2x$ 的周期是 $\pi$，别写成 $2\pi$。

## 6 小结

- 二倍角公式是**两角和公式在 $\beta=\alpha$ 处**的特殊取值，可从两角和公式现推，不必死记。
- 核心三式：$\sin2\alpha=2\sin\alpha\cos\alpha$，$\cos2\alpha=\cos^2\alpha-\sin^2\alpha$，$\tan2\alpha=\dfrac{2\tan\alpha}{1-\tan^2\alpha}$。
- **余弦二倍角有三种形式**，靠同角平方关系互相转换；反着读即得降幂公式 $\cos^2\alpha=\frac{1+\cos2\alpha}{2}$、$\sin^2\alpha=\frac{1-\cos2\alpha}{2}$。
- 倍角是**相对关系**（$4\alpha$ 也是二倍角），代入前先定象限定符号，并留意 $\tan2\alpha$ 的定义域。

在下一节，我们将用二倍角与两角和公式做逆向工程：把高次、异名、异角的式子改写成单角一次式，这就是**简单的三角恒等变换**。
