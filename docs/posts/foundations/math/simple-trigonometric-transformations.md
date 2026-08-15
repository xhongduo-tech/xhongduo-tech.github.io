---
title: 简单的三角恒等变换
date: 2026-08-07
---

# 简单的三角恒等变换

<div class="epigraph">
<p>数学就是给不同的东西以相同的名字。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.5.3 ｜ 2026-08-07</p>
</div>

## 为什么从三角恒等变换开始

前两节我们手里已经攒下两角和差公式与二倍角公式。本节只做一件事：**把已有公式当工具，把一个式子改写成另一个更「好用」的式子**。改写的方向千变万化——平方降成一次、单角换半角、$\sin$ 与 $\cos$ 合成一个、积化成和、和化成积。学会三角恒等变换，核心不是背下所有公式，而是掌握两个最基本的「手艺」：**换元**（令 $\theta=2\alpha$）与**逆用**（把公式反过来读）。<span class="marginnote">庞加莱这句话点破了恒等变换的本质：公式两边是同一个量的两种写法。变换不是「算出新东西」，而是「换个更顺手的外衣」。化简、求值、证明，全都是在外衣之间来回换。</span> 这门手艺不只在三角函数里有用——代数、数列、导数里「换元 + 逆用公式」的组合会反复出现。

## 1 半角公式：二倍角的逆向

把二倍角的降幂公式**反着用**：降幂公式是「平方 → 一次」，半角公式是「一次 → 半角的平方」。设 $\theta = 2\alpha$，则 $\alpha=\frac{\theta}{2}$，代入降幂公式：

$$
\cos^2\frac{\theta}{2}=\frac{1+\cos\theta}{2}, \qquad
\sin^2\frac{\theta}{2}=\frac{1-\cos\theta}{2}
$$

开平方，就得到半角公式的常用形式（符号由 $\frac{\theta}{2}$ 所在象限决定）：

$$
\cos\frac{\theta}{2}=\pm\sqrt{\frac{1+\cos\theta}{2}}, \qquad
\sin\frac{\theta}{2}=\pm\sqrt{\frac{1-\cos\theta}{2}}
$$

正切的半角公式更妙，它不用根号，可以写成「无理式」的等价形式：

$$
\tan\frac{\theta}{2}=\frac{\sin\theta}{1+\cos\theta}=\frac{1-\cos\theta}{\sin\theta}
$$

这后两种写法没有符号之争——右边由已知的 $\sin\theta$、$\cos\theta$ 直接确定。<span class="marginnote">同一个量有带根号和不带根号两种表达，是恒等变换的典型风景：带根号者直观但要定符号，不带根号者绕开了符号判断却多一条分母非零的限制。解题时按需选用。</span>

## 2 辅助角公式：把两个同名函数并成一个

形如 $a\sin x+b\cos x$ 的式子经常出现——简谐运动、物理学里的叠加、求最值。我们的目标是把它写成**单一的**正弦函数：

$$
a\sin x + b\cos x = \sqrt{a^2+b^2}\,\sin(x+\varphi)
$$

其中 $\varphi$ 由 $\cos\varphi=\dfrac{a}{\sqrt{a^2+b^2}}$、$\sin\varphi=\dfrac{b}{\sqrt{a^2+b^2}}$ 确定。<span class="marginnote">几何直觉：把 $(a,b)$ 看成平面上的一个点，$\sqrt{a^2+b^2}$ 是它到原点的距离，$\varphi$ 是它与 $x$ 轴正方向的夹角。把 $a,b$ 换成「距离 × 余弦」「距离 × 正弦」，正好凑出 $\sin(x+\varphi)$ 的展开式——这已经是「向量」思想的雏形，下一章《平面向量》会正式登场。</span>

**重点：辅助角公式的本质是把两项并成一项**，从而把「两个三角函数的和」化简为「一个振幅为 $\sqrt{a^2+b^2}$、初相为 $\varphi$ 的正弦」。这样一来，求最值、求周期、画图象全都退化为研究单一正弦函数。

## 3 公式解析：$\sin x+\cos x=\sqrt{2}\sin\left(x+\frac{\pi}{4}\right)$

以一个具体例子做三步拆解，看清辅助角公式的运作机制：

**第一步，提出振幅**：$\sin x+\cos x=\sqrt{2}\left(\frac{1}{\sqrt{2}}\sin x+\frac{1}{\sqrt{2}}\cos x\right)$。这里 $a=b=1$，故 $\sqrt{a^2+b^2}=\sqrt{2}$。提出 $\sqrt{2}$ 后，括号里两个系数 $\frac{1}{\sqrt{2}}$ 恰好满足「一个的平方加另一个的平方等于 1」，这正是 $\cos\varphi$ 与 $\sin\varphi$ 的资格。
**第二步，凑出两角和公式**：把 $\frac{1}{\sqrt{2}}$ 认成 $\cos\frac{\pi}{4}=\sin\frac{\pi}{4}$，于是括号里是 $\cos\frac{\pi}{4}\sin x+\sin\frac{\pi}{4}\cos x=\sin\left(x+\frac{\pi}{4}\right)$。
**第三步，收拢**：$\sin x+\cos x=\sqrt{2}\sin\left(x+\frac{\pi}{4}\right)$。由此立刻读出最大值 $\sqrt{2}$、周期 $2\pi$、以及图象相对 $\sin x$ 左移了 $\frac{\pi}{4}$。

**辨析｜易错点：** 辅助角公式里 $\varphi$ 的位置与符号极易出错。若写成 $a\cos x\pm b\sin x$，要先把式子整理成「正弦在前」的标准形再做变换；且 $\varphi$ 的取值必须落在 $\cos\varphi$、$\sin\varphi$ 同时满足的那个象限——只写 $\tan\varphi=\frac{b}{a}$ 会丢掉象限信息，导致 $\varphi$ 相差 $\pi$。<span class="marginnote">tan 只告诉比例，不告诉方向：$\frac{1}{1}$ 与 $\frac{-1}{-1}$ 的 tan 相同，象限却相反。凡由 tan 反求角，必须先定点 $(a,b)$ 所在的象限。</span>

## 4 积化和差与和差化积

这一组公式同样可以由两角和差公式推导，它们的价值在于**改变函数之间的运算级别**：

$$
\sin\alpha\cos\beta=\frac{1}{2}\big[\sin(\alpha+\beta)+\sin(\alpha-\beta)\big]
$$

$$
\sin\alpha+\sin\beta=2\sin\frac{\alpha+\beta}{2}\cos\frac{\alpha-\beta}{2}
$$

第一条把「积」化成「和」，第二条把「和」化成「积」。名字对称、方向相反，是一对互逆操作。教材把这两组列为选学内容，但它们在物理（波的叠加）与后续高等数学（定积分、三角级数）中地位很高——**「把乘积变求和」恰好是取对数、取积分背后同一种「降级」冲动**。现在不必死记推导细节，但要认识它们的存在：见到 $\sin\alpha\cos\beta$ 时，知道它还有「拆成两项和」的另一种写法，就够了。

## 5 例题精讲：恒等变换的实战

三角恒等变换的考题，考「选公式」与「化目标形」。看两道题。

### 题一：化简求值

化简 $\sin^2\frac\theta2$，并求 $\cos15^\circ$ 的精确值。

- **第一步，半角公式**：$\sin^2\frac\theta2=\frac{1-\cos\theta}{2}$。
- **第二步，求 $\cos15^\circ$**：$\cos15^\circ=\cos\frac{30^\circ}{2}=\sqrt{\frac{1+\cos30^\circ}{2}}=\sqrt{\frac{1+\frac{\sqrt3}{2}}{2}}=\frac{\sqrt{2+\sqrt3}}{2}$（取正，因为 $15^\circ$ 在第一象限）。
- **第三步，验证**：$15^\circ$ 也可用两角差公式 $\cos(45^\circ-30^\circ)=\frac{\sqrt6+\sqrt2}{4}$，两种结果相等（$\sqrt{2+\sqrt3}=2\sqrt{\frac{2+\sqrt3}{4}}$ 与 $\frac{\sqrt6+\sqrt2}{2}$ 平方后一致）——半角与两角差殊途同归。

<span class="marginnote">「半角公式求精确值」的关键是<strong>开方后的符号选择</strong>：$\cos\frac\theta2=\pm\sqrt{\frac{1+\cos\theta}{2}}$，正负由 $\frac\theta2$ 所在象限决定——$15^\circ$ 在第一象限取正。与两角差公式 $\cos(45^\circ-30^\circ)$ 对比验证，是「算得对不对」的好方法。</span>

### 题二：辅助角公式的应用

求 $y=3\sin x+4\cos x$ 的最大值。

**第一步，辅助角**：$y=\sqrt{3^2+4^2}\sin(x+\varphi)=5\sin(x+\varphi)$，其中 $\cos\varphi=\frac35$、$\sin\varphi=\frac45$。
**第二步，求最大值**：$\sin(x+\varphi)\le1$，$y_{\max}=5$。
**第三步，何时取到**：$x+\varphi=\frac\pi2+2k\pi$，即 $x=\frac\pi2-\varphi+2k\pi$ 时取最大值。

<span class="marginnote">辅助角公式求最值的套路：<strong>$a\sin x+b\cos x=\sqrt{a^2+b^2}\sin(x+\varphi)$，最大值就是 $\sqrt{a^2+b^2}$</strong>。$3\sin x+4\cos x$ 的最大值 5、最小值 $-5$——「直角三角形的斜边」。这类题不用展开，直接提 $\sqrt{a^2+b^2}$ 一步到位。$\varphi$ 的具体值通常不必求出，只需知道「存在这样一个相位」。</span>

**辨析｜易错点（补充）：** 一是**半角公式开方忘定符号**——$\sqrt{\frac{1+\cos\theta}{2}}$ 的正负由 $\frac\theta2$ 的象限定，不是永远取正；二是**辅助角提出来的系数**——$\sqrt{a^2+b^2}$ 是「斜边」，$3\sin x+4\cos x$ 提 5 不是提 3 或 4；三是**$\sin^2\frac\theta2$ 与 $\cos^2\frac\theta2$ 记混**——一个 $\frac{1-\cos\theta}{2}$、一个 $\frac{1+\cos\theta}{2}$，符号相反。
### 恒等变换工具箱速查

本节用到的公式虽然多，但都从两角和差与二倍角「逆用 / 换元」而来。放一张速查表按「功能」分类：

| 功能 | 工具 | 关键式 |
| --- | --- | --- |
| 降幂 | 二倍角逆用 | $\cos^2\theta=\frac{1+\cos2\theta}{2}$ |
| 半角 | 降幂公式换元 | $\sin\frac\theta2=\pm\sqrt{\frac{1-\cos\theta}{2}}$ |
| 合成 | 辅助角 | $a\sin x+b\cos x=\sqrt{a^2+b^2}\sin(x+\varphi)$ |
| 积化积 | 积化和差 | $\sin\alpha\cos\beta=\frac12[\sin(\alpha+\beta)+\sin(\alpha-\beta)]$ |
| 和化和 | 和差化积 | $\sin\alpha+\sin\beta=2\sin\frac{\alpha+\beta}{2}\cos\frac{\alpha-\beta}{2}$ |

**核心思想：一切变换都是「把同一个量换一种写法」——降幂、半角、合成、积化和差，无非在「平方与一次」「单角与半角」「两项与一项」「积与和」之间来回切换。** 遇到化简题，先问「目标形态是什么」，再选对应的工具，比死记硬背公式高效得多。

**辨析｜易错点：** 辅助角公式的 $\varphi$ 由点 $(a,b)$ 定象限；若式子出现「$\cos x$ 在前」，先整理成「$a\sin x+b\cos x$」的标准形再提振幅，否则相位会错。

## 6 小结

- 三角恒等变换的**两大手艺**：换元（令 $\theta=2\alpha$）与逆用公式（反着读）。
- **半角公式**由降幂公式换元而来，开方形式要定符号；正切半角有无根号的形式可绕开符号之争。
- **辅助角公式** $a\sin x+b\cos x=\sqrt{a^2+b^2}\sin(x+\varphi)$，把两项并成一项，是研究最值与周期的利器；$\varphi$ 的象限由点 $(a,b)$ 决定。
- 积化和差、和差化积把乘法与加法互相转化，是「降级」思想的预演。

在下一节，我们把三角函数的图象做一次「组装」：给正弦波加上振幅、周期与相位，得到**函数 $y=A\sin(\omega x+\varphi)$ 的图象**——这是描述一切周期现象的统一语言。
