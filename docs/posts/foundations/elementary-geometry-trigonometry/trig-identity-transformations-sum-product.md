---
title: 三角恒等变换（和差倍半公式、积化和差）
date: 2026-08-07
---

# 三角恒等变换（和差倍半公式、积化和差）

<div class="epigraph">
<p>三角恒等式是代数的体操：同一条式子，换一副眼镜看，就长成另一副模样。</p>
<footer>—— 自编（本文题旨）</footer>
</div>

<div class="article-byline">
<p>第一级 · 初等几何与三角 ｜ 人教A版 必修第一册 §5.5 ｜ 2026-08-07</p>
</div>

## 为什么从三角恒等变换开始

上一篇我们学会了「用三角比解三角形」——那是在**已知条件里直接用**三角函数。这一篇要学的是**改写**：同一个三角函数式，可以写成一族等价的形式，有时是为了化简、有时是为了求值、有时是为了证明。这种「恒等变形」的能力，是三角函数题的「内功」，也是代数的最高表现——**同一条真理，换了十副面孔，本质未变**。<span class="marginnote">「恒等变换」的枢纽是上一专题《勾股定理及其应用》导出的基本恒等式 $\sin^2\alpha + \cos^2\alpha = 1$——它是勾股定理在三角世界里的化身。所有和差倍半公式、积化和差公式，最终都能回溯到这条最简等式与加减法的代数展开。</span>

从「从极限到大模型」的主线看，三角恒等变换是**信号处理的算术**：调频广播、音频合成、傅里叶分析里的「乘积变和差」「和差变乘积」，都要用到积化和差与和差化积——把「两个频率相乘」写成「两个频率相加/相减」，正是频谱分析的核心操作。这一篇是「从初等几何到现代信号」的必经桥梁。

## 1 两角和与差公式

三角函数的第一组「组合拳」是**和差角公式**，它们把「$\alpha \pm \beta$ 的三角函数」展开成「$\alpha$ 与 $\beta$ 各自的三角函数」：

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$
$$\sin(\alpha - \beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta$$
$$\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$
$$\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$$

**重点：** 四条公式的记忆要抓住「**余弦号变号，正弦号不变**」：$\cos(\alpha+\beta)$ 中间是减号，$\sin(\alpha+\beta)$ 中间是加号；余弦公式是「同名相乘」，正弦公式是「异名相乘」。正切的两角和公式由正弦、余弦公式相除得到：

$$\tan(\alpha + \beta) = \frac{\tan\alpha + \tan\beta}{1 - \tan\alpha\tan\beta}$$

**辨析｜易错点：** 对初学者最大的坑是「$\sin(\alpha+\beta) = \sin\alpha + \sin\beta$」——这**不成立**。代入 $\alpha = \beta = 30°$ 检验：左边 $\sin 60° \approx 0.866$，右边 $0.5 + 0.5 = 1$，差得很远。**任何恒等式都用特殊角自检一遍**，能立刻戳穿记忆错误。

## 2 二倍角公式与降幂

在和差角公式里令 $\beta = \alpha$，就得到**二倍角公式**：

$$\sin 2\alpha = 2\sin\alpha\cos\alpha$$
$$\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha$$
$$\tan 2\alpha = \frac{2\tan\alpha}{1 - \tan^2\alpha}$$

**重点：** $\cos 2\alpha$ 的三个等价写法是「降幂」的钥匙。移项即得**降幂公式**：

$$\cos^2\alpha = \frac{1 + \cos 2\alpha}{2}, \qquad \sin^2\alpha = \frac{1 - \cos 2\alpha}{2}$$

降幂公式的价值在于：**把「平方」降成「一次」，把「二次的角」变成「二倍的角」**——积分、级数、最值问题里「平方」总是麻烦，降幂是化繁为简的标准动作。<span class="marginnote">二倍角公式还能反着用：$\sin 2\alpha = 2\sin\alpha\cos\alpha$ 让「乘积」变成了「二倍角的正弦」。这种「乘积与和差互相转化」正是第 4 节积化和差、和差化积的先声——代数上它们只是同一个公式的正用与逆用。</span>

## 3 公式解析：用向量数量积推导 $\cos(\alpha - \beta)$

与其死背四条和差公式，不如把最核心的一条**推导**出来——$\cos(\alpha-\beta)$ 是最有故事的一条，用向量数量积一次成型：

$$\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$$

对这条公式做三步拆解：

- **第一步，单位圆上取点**：在单位圆上取角 $\alpha$、$\beta$ 对应的点 $P(\cos\alpha, \sin\alpha)$、$Q(\cos\beta, \sin\beta)$。两向量的夹角恰为 $|\alpha - \beta|$。
- **第二步，数量积算两次**：按坐标定义，$\vec{OP} \cdot \vec{OQ} = \cos\alpha\cos\beta + \sin\alpha\sin\beta$；按「模 × 模 × 夹角的余弦」，又等于 $1 \times 1 \times \cos(\alpha-\beta)$。
- **第三步，两边相等**：两种算法结果必须一致，于是 $\cos(\alpha-\beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$。把 $\beta$ 换成 $-\beta$ 就得 $\cos(\alpha+\beta)$；把 $\alpha$ 换成 $90°-\alpha$ 就得正弦公式——**四条和差公式一条入口全部导出**。

<span class="marginnote">这个推导把「几何」（夹角、单位圆）与「代数」（数量积）焊接在一起，也再次点亮了「同一个量算两次」的思想（回顾第 5 篇《面积与体积》的面积法）。数量积 $\vec u \cdot \vec v = |\vec u||\vec v|\cos\theta$ 是今天的主角，它在第三篇《空间向量与立体几何》与第二级《线性代数》里都是度量空间的基石。</span>

## 4 积化和差与和差化积

把和差公式**倒过来用**，得到两族「大变身」公式。

**积化和差（把「乘积」变成「和差」）：**

$$\sin\alpha\cos\beta = \frac{1}{2}\big[\sin(\alpha+\beta) + \sin(\alpha-\beta)\big]$$
$$\cos\alpha\sin\beta = \frac{1}{2}\big[\sin(\alpha+\beta) - \sin(\alpha-\beta)\big]$$
$$\cos\alpha\cos\beta = \frac{1}{2}\big[\cos(\alpha+\beta) + \cos(\alpha-\beta)\big]$$
$$\sin\alpha\sin\beta = \frac{1}{2}\big[\cos(\alpha-\beta) - \cos(\alpha+\beta)\big]$$

**和差化积（把「和差」变成「乘积」）：** 令 $x = \alpha+\beta$、$y = \alpha-\beta$，反解出 $\alpha = \frac{x+y}{2}$、$\beta = \frac{x-y}{2}$，代入积化和差即得：

$$\sin x + \sin y = 2\sin\frac{x+y}{2}\cos\frac{x-y}{2}$$
$$\sin x - \sin y = 2\cos\frac{x+y}{2}\sin\frac{x-y}{2}$$
$$\cos x + \cos y = 2\cos\frac{x+y}{2}\cos\frac{x-y}{2}$$
$$\cos x - \cos y = -2\sin\frac{x+y}{2}\sin\frac{x-y}{2}$$

**重点：** 这八条公式**不需要背**——它们的来源只有一句话：**积化和差是「和差公式两式相加/相减除以 2」，和差化积是「设 $x=\alpha+\beta, y=\alpha-\beta$ 反解再代入」**。理解这条生成路径，比背下八条式子可靠得多。<span class="marginnote">积化和差在物理与工程里举足轻重：两列波的叠加、两个频率信号的混频（调幅广播把音频「搬」到载波上），都靠 $\cos A \cos B$ 的积化和差展开——「乘积转和差」在频谱上就是「频率的和与差」，这正是调制与解调的数学内核。</span>

## 5 半角公式与万能公式

再进一步，由降幂公式开平方，得**半角公式**（正负号由 $\frac{\alpha}{2}$ 所在的象限决定）：

$$\sin\frac{\alpha}{2} = \pm\sqrt{\frac{1-\cos\alpha}{2}}, \qquad \cos\frac{\alpha}{2} = \pm\sqrt{\frac{1+\cos\alpha}{2}}$$

以及**万能公式**——把一切三角函数都用 $\tan\frac{\alpha}{2}$ 表示：

$$\sin\alpha = \frac{2\tan\frac{\alpha}{2}}{1+\tan^2\frac{\alpha}{2}}, \qquad \cos\alpha = \frac{1-\tan^2\frac{\alpha}{2}}{1+\tan^2\frac{\alpha}{2}}, \qquad \tan\alpha = \frac{2\tan\frac{\alpha}{2}}{1-\tan^2\frac{\alpha}{2}}$$

**重点：** 万能公式的思想是「**换元一统**」：令 $t = \tan\frac{\alpha}{2}$，则 $\sin\alpha$、$\cos\alpha$、$\tan\alpha$ 全部变成关于 $t$ 的有理式——三角问题变成有理函数问题，三角函数方程变成代数方程。这个「化三角为代数」的换元，在积分学里更是标配（「万能代换」）。<span class="marginnote">「用有理式表示三角函数」看似炫技，实则是「消去周期性、保留代数的整式结构」的数学战略：处理无理式、分式、极限时，$t = \tan\frac{x}{2}$ 往往一招制敌。它与「换元消元」「降次」共同构成三角计算的三板斧。</span>

## 6 核心对比表：三角恒等变换公式族

| 公式族 | 代表公式 | 用途 | 生成方式 |
| --- | --- | --- | --- |
| 和差角 | $\sin(\alpha\pm\beta)$，$\cos(\alpha\pm\beta)$ | 展开、求值 | $\cos(\alpha-\beta)$ 用数量积推出 |
| 二倍角 | $\cos2\alpha = 2\cos^2\alpha-1$ | 倍角、降幂 | 和差公式令 $\beta=\alpha$ |
| 降幂 | $\cos^2\alpha=\frac{1+\cos2\alpha}{2}$ | 去平方 | 二倍角移项 |
| 积化和差 | $\sin\alpha\cos\beta = \frac{1}{2}[\sin(\alpha+\beta)+\sin(\alpha-\beta)]$ | 乘积→和差 | 和差公式相加除 2 |
| 和差化积 | $\sin x+\sin y = 2\sin\frac{x+y}{2}\cos\frac{x-y}{2}$ | 和差→乘积 | 积化和差换元反解 |
| 半角/万能 | $t=\tan\frac{\alpha}{2}$ | 化三角为代数 | 降幂开方 / 换元 |

**重点：** 这张表的精髓是「**一源多流**」——全部公式都从「基本恒等式 $\sin^2\alpha+\cos^2\alpha=1$ + 和差公式」长出来。做题时遇到陌生式子，先问「它属于哪一族、从哪条源公式生成」，比翻公式表更省力。

## 7 小结

- **和差角公式**四条 + 正切一条：余弦变号、正弦不变号、正余弦「同名/异名」。
- **二倍角**与**降幂**：$\cos2\alpha$ 的三个写法是降幂钥匙；「去平方」是化简常客。
- **$\cos(\alpha-\beta)$ 用向量数量积推导**：单位圆取点 + 数量积算两次，一条入口导出四式和差。
- **积化和差、和差化积**：不用背，记住「和差公式相加除 2」与「换元反解」两条生成路径。
- **半角、万能公式**：$t=\tan\frac{\alpha}{2}$ 把三角问题化为有理函数问题。
- 全部恒等式都是「基本恒等式 + 代数变形」的孩子；恒等变形的本质是**同一事实的不同写法**。

在下一节，我们将从「公式」转向「图像与方程」——**三角函数的图像与性质**：周期性、奇偶性、单调性，以及反三角函数与三角方程，把三角函数从「算数值」推向「看结构」。
