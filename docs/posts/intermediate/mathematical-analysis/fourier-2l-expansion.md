---
title: 以 2l 为周期的函数的傅里叶级数
date: 2026-08-07
---

# 以 2l 为周期的函数的傅里叶级数

<div class="epigraph">
<p>把周期从 $2\pi$ 推广到任意 $2l$，只需一次「伸缩」——傅里叶级数的公式在 $[-l,l]$ 上长得几乎一模一样，只是频率变成了 $\frac{n\pi}{l}$。</p>
<footer>—— 约瑟夫·傅里叶（Joseph Fourier），《热的解析理论》（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§15.3 ｜ 2026-08-07</p>
</div>

## 为什么「伸缩」就够了

现实中很少有函数恰好周期 $2\pi$：声波的周期是 $\frac1{\text{频率}}$ 秒，潮汐是 12 小时，心电图是心跳间隔。但**任何周期 $2l$ 的函数，经过伸缩变换都能「标准化」成周期 $2\pi$ 的函数**——把 $x$ 换成 $\frac{l}{\pi}t$（或反向）。傅里叶级数公式随之「翻译」到 $[-l,l]$。

这一步推广看似机械，却是从「数学理论」走向「物理应用」的必经之路：偏微分方程里的区间是 $[0,l]$（弦长、杆长），信号处理里的周期是任意 $T$。**「归一化到 $2\pi$」是万能技巧**——公式只需记一套，其余靠伸缩。<span class="marginnote">伸缩变换的本质是「换标尺」：物理量周期 $T$ 时，用角频率 $\omega=\frac{2\pi}{T}$ 描述，谐波写成 $\cos(n\omega t)$ 与 $\sin(n\omega t)$。在 $[-l,l]$ 上 $\omega=\frac{\pi}{l}$，谐波是 $\cos\frac{n\pi x}{l}$。<strong>傅里叶级数公式里的「$\pi/l$」就是频率刻度</strong>——它让「第 $n$ 个谐波的波长 = $\frac{2l}{n}$」保持正确。工程书里把 $\cos\frac{n\pi x}{l}$ 写成 $\cos(n\omega x)$，只是记号的换装。</span>

## 1 一般周期的傅里叶级数

设 $f$ 是周期 $2l$ 的函数（$l>0$）。做伸缩变换 $t=\frac{\pi x}{l}$（把 $[-l,l]$ 映到 $[-\pi,\pi]$），$g(t)=f(\frac{lt}{\pi})$ 是周期 $2\pi$ 的函数。把 $g$ 的傅里叶级数（§15.2）换回 $x$：

**定理：周期 $2l$ 的函数 $f$ 的傅里叶级数**

$$f(x)=\frac{a_0}{2}+\sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{l}+b_n\sin\frac{n\pi x}{l}\right),$$

**其中**

$$a_n=\frac1l\int_{-l}^{l}f(x)\cos\frac{n\pi x}{l}\,dx\qquad(n=0,1,2,\dots),$$

$$b_n=\frac1l\int_{-l}^{l}f(x)\sin\frac{n\pi x}{l}\,dx\qquad(n=1,2,\dots).$$

**公式解析：三步拆解**

**第一步，伸缩**。令 $t=\frac{\pi x}{l}$，则 $x=\frac{l}{\pi}t$，$dx=\frac{l}{\pi}dt$，$[-\pi,\pi]\leftrightarrow[-l,l]$；

**第二步，代换**。$a_n=\frac1\pi\int_{-\pi}^{\pi}g(t)\cos nt\,dt=\frac1\pi\int_{-l}^lf(x)\cos\frac{n\pi x}{l}\cdot\frac{\pi}{l}dx=\frac1l\int_{-l}^lf\cos\frac{n\pi x}{l}dx$；

**第三步，**$\frac1\pi\cdot\frac\pi l=\frac1l$**——系数分母从 $\pi$ 变成 $l$，谐波从 $\cos nx$ 变成 $\cos\frac{n\pi x}{l}$**。∎

**要点**：**唯一的改动是「$\pi\to l$」与「$nx\to\frac{n\pi x}{l}$」**——这就是「把 $2\pi$ 换成 $2l$」的全部。公式结构零变化。

## 2 奇偶性与半区间展开（一般周期）

与 $2\pi$ 情形完全平行：

**奇函数（周期 $2l$）** ⇒ 正弦级数：

$$f(x)=\sum_{n=1}^\infty b_n\sin\frac{n\pi x}{l},\qquad b_n=\frac2l\int_0^lf(x)\sin\frac{n\pi x}{l}dx.$$

**偶函数** ⇒ 余弦级数：

$$f(x)=\frac{a_0}{2}+\sum_{n=1}^\infty a_n\cos\frac{n\pi x}{l},\qquad a_n=\frac2l\int_0^lf(x)\cos\frac{n\pi x}{l}dx.$$

**半区间展开**：$[0,l]$ 上的 $f$ 可奇延拓得正弦级数（端点收敛到 0）或偶延拓得余弦级数（端点保持值）——**一切与 $[0,\pi]$ 情形同构，只是 $l$ 代替 $\pi$**。

**示范**：$f(x)=x$ 在 $[0,l]$ 上。

**正弦级数**：$b_n=\frac2l\int_0^lx\sin\frac{n\pi x}{l}dx$，分部积分：

$$b_n=\frac2l\left[-\frac{lx}{n\pi}\cos\frac{n\pi x}{l}\right]_0^l+\frac2l\int_0^l\frac l{n\pi}\cos\frac{n\pi x}{l}dx=\frac{2l(-1)^{n+1}}{n\pi}.$$

$$x=\frac{2l}{\pi}\sum_{n=1}^\infty\frac{(-1)^{n+1}}{n}\sin\frac{n\pi x}{l}\qquad(0<x<l).$$

**余弦级数**：$a_0=l$，$a_n=\frac{2l}{\pi^2n^2}((-1)^n-1)$：

$$x=\frac l2-\frac{4l}{\pi^2}\sum_{k=0}^\infty\frac{\cos\frac{(2k+1)\pi x}{l}}{(2k+1)^2}.$$

**同一个函数在 $[0,l]$ 上的两种展开**——选择哪种取决于问题的边界条件（§15.2 已述）。

## 3 公式解析：傅里叶系数的物理解读

把一般周期公式读成「物理语言」：

$$f(x)=\frac{a_0}{2}+\sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{l}+b_n\sin\frac{n\pi x}{l}\right).$$

**第一步，直流分量**。$\frac{a_0}{2}$ 是函数的平均值（$a_0=\frac1l\int f$，除 2 后是均值）——「零频成分」；

**第二步，基波与谐波**。$n=1$ 是**基波**（周期 $2l$），$n\ge2$ 是**谐波**（周期 $\frac{2l}{n}$，频率 $n$ 倍）——「第 $n$ 谐波的振幅与相位由 $a_n,b_n$ 决定」；

**第三步，振幅与相位**。$a_n\cos\frac{n\pi x}{l}+b_n\sin\frac{n\pi x}{l}=A_n\cos(\frac{n\pi x}{l}-\varphi_n)$，其中 $A_n=\sqrt{a_n^2+b_n^2}$（振幅谱）、$\varphi_n=\arctan\frac{b_n}{a_n}$（相位谱）——**傅里叶级数把函数分解成「不同频率的余弦波」**，每个频率有自己的振幅与相位。

**要点**：傅里叶级数 = **频谱分解**——$(a_n,b_n)$ 或 $(A_n,\varphi_n)$ 就是「频率 $\frac{n\pi}{l}$ 的成分」。这个解读让傅里叶级数成为信号分析的语言：**一个波形的全部信息 = 它的频谱**。

## 4 应用：方波的傅里叶级数

**示范**：方波 $f(x)=\begin{cases}1,&0<x<l\\-1,&-l<x<0\end{cases}$（周期 $2l$，奇函数）。

$a_n=0$（奇函数）；$b_n=\frac2l\int_0^l1\cdot\sin\frac{n\pi x}{l}dx=\frac2{n\pi}(1-\cos n\pi)=\begin{cases}\frac4{n\pi},&n\text{ 奇}\\0,&n\text{ 偶}\end{cases}$。

$$f(x)=\frac4\pi\sum_{k=0}^\infty\frac{\sin\frac{(2k+1)\pi x}{l}}{2k+1}.$$

**方波只含奇次谐波**（$n=1,3,5,\dots$），振幅按 $\frac1n$ 衰减——**「方波 = 无穷多奇次正弦波的叠加」是信号处理最经典的图像**。<span class="marginnote">方波的傅里叶级数 $\frac4\pi\sum\frac{\sin(2k+1)x}{2k+1}$ 揭示了「吉布斯现象」：部分和在跳变点 $x=0$ 附近会「过冲」约 9%（不随 $n$ 减小）——即使 $n\to\infty$，跳变点旁的过冲依然存在。这是傅里叶级数「逐点收敛但不一致收敛」的直接后果（§13.1 的 $x^n$ 式教训在傅里叶世界重演）。图像压缩里的「振铃伪影」、滤波器的「过冲」，都是吉布斯现象的工程化身。</span>

## 5 一般周期公式总览

| | 周期 $2\pi$ | 周期 $2l$ |
| --- | --- | --- |
| 谐波 | $\cos nx,\sin nx$ | $\cos\frac{n\pi x}{l},\sin\frac{n\pi x}{l}$ |
| 系数分母 | $\frac1\pi$ | $\frac1l$ |
| 半区间系数 | $\frac2\pi\int_0^\pi$ | $\frac2l\int_0^l$ |
| 端点收敛 | 正弦级数端点 → 0 | 同左 |

**「$\pi\to l$」一条规则，$2\pi$ 的全部结论自动移植到 $2l$**。这套「归一化伸缩」的思想在傅里叶变换、拉普拉斯变换里同样适用。

**「$\pi/l$」的工程读法**：物理上周期 $T$ 的谐波角频率 $\omega_n=\frac{2\pi n}{T}=\frac{n\pi}{l}$（$T=2l$）。傅里叶系数公式里的 $\frac1l\int_{-l}^l$ 正是「一个周期内的平均 × 谐波」——**周期只是刻度，公式的「形状」是普适的**。这就是为什么工程书一套公式吃遍所有周期信号：把任意周期先归一化，套公式，再伸缩回去。

## 6 小结

- **一般周期公式**：$f=\frac{a_0}{2}+\sum(a_n\cos\frac{n\pi x}{l}+b_n\sin\frac{n\pi x}{l})$；$a_n,b_n$ 分母 $\frac1l$。
- **推导**：伸缩变换 $t=\frac{\pi x}{l}$ 把 $[-l,l]$ 映到 $[-\pi,\pi]$，公式自动翻译。
- **奇偶简化**：奇函数正弦级数、偶函数余弦级数，系数分母 $\frac2l$，积分区间 $[0,l]$。
- **物理解读**：$\frac{a_0}2$ 直流、$A_n=\sqrt{a_n^2+b_n^2}$ 振幅谱、$\varphi_n$ 相位谱——频谱分解。
- **应用**：方波只含奇次谐波、振幅 $\frac1n$ 衰减；吉布斯现象（9% 过冲）。

在下一节，我们进入傅里叶级数理论的收官：**收敛定理及其证明**。为什么傅里叶级数收敛、收敛到什么值、如何证明——狄利克雷条件与傅里叶级数的逐点收敛将给出答案。
