---
title: 复数的三角表示
date: 2026-08-07
---

# 复数的三角表示

<div class="epigraph">
<p>乘法就是旋转，旋转就是乘法——这就是复数的魔法。</p>
<footer>—— 对欧拉公式的通俗转述（Euler's formula）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第二册 §7.3 ｜ 2026-08-07</p>
</div>

## 为什么从三角表示开始

同一个复数，代数形式 $a+bi$ 描述的是「横纵坐标」，而三角形式 $r(\cos\theta+i\sin\theta)$ 描述的是「距离 + 角度」——就像描述一个点，既可以用直角坐标，也可以用极坐标。三角形式的威力在于它**把复数的乘法翻译成角度的加法**：原来展开多项式的繁琐乘法，换成模相乘、辐角相加，一步到位。<span class="marginnote">极坐标（r, θ）与直角坐标（a, b）的互换公式是 $a=r\cos\theta$、$b=r\sin\theta$——这就是「圆的参数方程」，也是三角形式与代数形式之间的翻译机。</span> 三角表示是复数的「极坐标视角」，它让复数的威力（旋转、振动、波形）彻底显形，也为第二级《复变函数》与信号处理里的傅里叶分析埋下伏笔。

## 1 辐角与三角形式

设复数 $z=a+bi\neq 0$ 对应复平面上的点 $Z(a,b)$。记模 $r=|z|=\sqrt{a^2+b^2}$，以实轴正方向为始边、射线 $OZ$ 为终边的角 $\theta$ 叫作 $z$ 的**辐角（argument）**。

**复数 $z$ 的三角形式**：

$$
z=r(\cos\theta+i\sin\theta)
$$

其中 $r>0$，$\theta$ 是 $z$ 的一个辐角。<span class="marginnote">辐角不唯一：转一整圈还是同一个点，所以 $\theta+2k\pi$（$k\in\mathbb{Z}$）都是辐角。习惯上取 $\theta\in(-\pi,\pi]$ 的那个值，叫<strong>辐角主值</strong>，记作 $\operatorname{arg} z$。做题时先明确「辐角」还是「辐角主值」，否则答案会因周期而显得「不对」。</span>

**重点：三角形式有标准结构**——模 $r>0$ 写在括号外，括号内必须是 $\cos\theta+i\sin\theta$ 的形式（先余弦后正弦，中间加号）。$z=-r(\cos\theta+i\sin\theta)$ 或 $z=r(\cos\theta-i\sin\theta)$ 都不是标准形式，需要先「提取符号」：

$$
-r(\cos\theta+i\sin\theta)=r\big(\cos(\theta+\pi)+i\sin(\theta+\pi)\big)
$$

## 2 三角形式的乘法：模相乘、辐角相加

设 $z_1=r_1(\cos\theta_1+i\sin\theta_1)$，$z_2=r_2(\cos\theta_2+i\sin\theta_2)$，则

$$
z_1z_2=r_1r_2\big[\cos(\theta_1+\theta_2)+i\sin(\theta_1+\theta_2)\big]
$$

**乘法的三角规则：模相乘，辐角相加**。这条结论直接由两角和公式推出：

$$
(\cos\theta_1+i\sin\theta_1)(\cos\theta_2+i\sin\theta_2)=\cos(\theta_1+\theta_2)+i\sin(\theta_1+\theta_2)
$$

展开左边四项，实部恰好是 $\cos\theta_1\cos\theta_2-\sin\theta_1\sin\theta_2$，虚部是 $\sin\theta_1\cos\theta_2+\cos\theta_1\sin\theta_2$——正是余弦、正弦的两角和公式。<span class="marginnote">上一节说的「乘 $i$ 是转 $90^\circ$」现在有了严格解释：$i=\cos\frac{\pi}{2}+i\sin\frac{\pi}{2}$，模为 1，辐角 $\frac{\pi}{2}$。乘 $i$ 就是把辐角加上 $\frac{\pi}{2}$——纯旋转。</span> 乘法的几何意义由此完全清晰：**先伸缩（模相乘），再旋转（辐角相加）**。两个复数相乘，就是两个变换的叠加。

## 3 公式解析：除法的三角形式

三角形式下的除法同样优雅，拆三步：

- **第一步，写出除法规则**：$\dfrac{z_1}{z_2}=\dfrac{r_1}{r_2}\big[\cos(\theta_1-\theta_2)+i\sin(\theta_1-\theta_2)\big]$——**模相除，辐角相减**。这可由乘法规则反推：$z_1=z_2\cdot\frac{z_1}{z_2}$，模和辐角分别相乘相加，解出 $\frac{z_1}{z_2}$ 的模为 $\frac{r_1}{r_2}$、辐角为 $\theta_1-\theta_2$。
- **第二步，几何直觉**：除以 $z_2$ 是乘 $z_2$ 的逆操作——模收缩 $r_2$ 倍，辐角减去 $\theta_2$。乘法转几圈，除法就转回去几圈，方向相反。
- **第三步，特例验证**：$\dfrac{1}{i}=\dfrac{\cos 0+i\sin 0}{\cos\frac{\pi}{2}+i\sin\frac{\pi}{2}}=\cos\left(-\frac{\pi}{2}\right)+i\sin\left(-\frac{\pi}{2}\right)=-i$——与上一节 $i^{-1}=-i$ 的结果一致，两套方法互相印证。

除法的三角形式说明：**复数的四则运算在三角表示下，全部变成「模的代数运算 + 辐角的加减」**——指数级的简洁。<span class="marginnote">这就是为什么工程上做相量计算都用三角或指数形式：乘除变成加减，极大简化了交流电路、振动叠加的运算。欧拉公式 $e^{i\theta}=\cos\theta+i\sin\theta$ 把三角形式升级为指数形式后，这套简洁达到极致——那是大学复变函数的内容。</span>

## 4 棣莫弗公式：幂与开方

反复用乘法规则，得到**棣莫弗公式（de Moivre's formula）**：对正整数 $n$，

$$
\big[r(\cos\theta+i\sin\theta)\big]^n=r^n(\cos n\theta+i\sin n\theta)
$$

即：**幂的模是模的幂，幂的辐角是辐角的 $n$ 倍**。特殊地，$r=1$ 时 $(\cos\theta+i\sin\theta)^n=\cos n\theta+i\sin n\theta$，这就是**三倍角等公式的统一来源**。<span class="marginnote">棣莫弗公式把「多次旋转」压缩成「一次大旋转」：转 $n$ 次 $\theta$，等价于转一次 $n\theta$。物理里「波程差累积成相位差」、信号里「N 点旋转矢量」，都是这条公式的现实化身。</span>

棣莫弗公式还是「开方」的钥匙：$z^n=w$ 的根有 $n$ 个，模都取 $\sqrt[n]{|w|}$，辐角在 $\frac{\theta}{n}$ 的基础上每次加上 $\frac{2\pi}{n}$——$n$ 个根均匀分布在圆周上。<span class="marginnote">「$n$ 次方程有 $n$ 个根」的几何图景在此显形：根就是圆周上的 $n$ 等分点。前面代数基本定理的抽象结论，用三角形式一看就活了。</span>

**辨析｜易错点：** 用三角形式前，务必把复数写成**标准三角形式**——模必须为正、括号内必须是 $\cos\theta+i\sin\theta$。一个常见错误是直接把 $r(\cos\theta-i\sin\theta)$ 当标准形式代入乘法公式，导致辐角符号全乱。正确的做法是先把 $-i\sin\theta$ 改写：$\cos\theta-i\sin\theta=\cos(-\theta)+i\sin(-\theta)$。

## 5 例题精讲：三角形式的运算

复数的三角形式把乘除变成「模的运算 + 辐角的加减」。看两道题。

### 题一：三角形式的乘法

$z_1=\sqrt2\left(\cos\frac\pi4+i\sin\frac\pi4\right)$，$z_2=2\left(\cos\frac\pi6+i\sin\frac\pi6\right)$，求 $z_1z_2$ 与 $\frac{z_1}{z_2}$。

- **第一步，乘法**：模相乘 $\sqrt2\times2=2\sqrt2$，辐角相加 $\frac\pi4+\frac\pi6=\frac{5\pi}{12}$，$z_1z_2=2\sqrt2\left(\cos\frac{5\pi}{12}+i\sin\frac{5\pi}{12}\right)$。
- **第二步，除法**：模相除 $\frac{\sqrt2}{2}$，辐角相减 $\frac\pi4-\frac\pi6=\frac{\pi}{12}$，$\frac{z_1}{z_2}=\frac{\sqrt2}{2}\left(\cos\frac{\pi}{12}+i\sin\frac{\pi}{12}\right)$。
- **第三步，验证代数形式**：$z_1=1+i$，$z_2=\sqrt3+i$，$z_1z_2=(1+i)(\sqrt3+i)=(\sqrt3-1)+(\sqrt3+1)i$——用三角形式算出的模与辐角与之吻合。

<span class="marginnote">三角形式乘除的优越性：<strong>不用展开多项式，模相乘除、辐角相加减，一步写出结果</strong>。特别是「开方」——$z^3=w$ 的根用三角形式求：模取 $\sqrt[3]{|w|}$，辐角取 $\frac{\theta+2k\pi}{3}$，$k=0,1,2$ 三个根均匀分布在圆周上。<strong>三角形式是「旋转与缩放」的语言</strong>，凡涉及乘除、幂、开方，它都比代数形式简洁得多。</span>

### 题二：代数形式转三角形式

把 $z=-1+\sqrt3\,i$ 写成三角形式。

- **第一步，求模**：$r=|z|=\sqrt{1+3}=2$。
- **第二步，求辐角**：$\cos\theta=-\frac12$，$\sin\theta=\frac{\sqrt3}{2}$，点 $(-1,\sqrt3)$ 在第二象限，$\theta=\frac{2\pi}{3}$。
- **第三步，写三角形式**：$z=2\left(\cos\frac{2\pi}{3}+i\sin\frac{2\pi}{3}\right)$。

<span class="marginnote">「代数转三角」的三步：<strong>求模 $r=\sqrt{a^2+b^2}$ → 由 $\cos\theta=\frac ar$、$\sin\theta=\frac br$ 定辐角（注意象限）→ 写 $r(\cos\theta+i\sin\theta)$</strong>。由 $\tan\theta=\frac ba$ 反求角时要看点 $(a,b)$ 所在象限，否则辐角可能差 $\pi$。</span>

**辨析｜易错点（补充）：** 一是**辐角象限判错**——$z=-1+\sqrt3 i$ 在第二象限，$\theta=\frac{2\pi}{3}$，别用 $\tan$ 直除得 $-\frac\pi3$；二是**乘除辐角的符号**——乘法辐角相加、除法辐角相减，别颠倒；三是**开方漏根**——$n$ 次方程在复数里有 $n$ 个根，用三角形式求根时 $k=0,1,\dots,n-1$ 要取全。

## 6 小结

- **三角形式**：$z=r(\cos\theta+i\sin\theta)$，$r=|z|$ 为模，$\theta$ 为辐角；辐角主值 $\operatorname{arg} z\in(-\pi,\pi]$。
- **乘法**：模相乘、辐角相加；**除法**：模相除、辐角相减。
- **棣莫弗公式**：$z^n=r^n(\cos n\theta+i\sin n\theta)$；开 $n$ 次方得 $n$ 个根，均匀分布在圆周上。
- 三角形式把复数运算变成「模的代数 + 辐角的加减」，是旋转与振动问题的最优语言。

在下一节，我们告别平面，进入三维空间：从最基本的立体图形讲起——**基本立体图形（棱柱、棱锥、棱台）**，空间几何的起点。
