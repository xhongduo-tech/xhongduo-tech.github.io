---
title: 麦克斯韦方程组与电磁边界条件
date: 2026-08-07
---

# 麦克斯韦方程组与电磁边界条件

<div class="epigraph">
<p>从这一大堆数据中，我推导出电磁场传播的方程，其传播速度与光速如此接近，以致我无法抗拒这样的结论：光本身是一种电磁扰动。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell, *A Dynamical Theory of the Electromagnetic Field\*, 1865）</footer>
</div>

<div class="article-byline">
<p>第四级 · 无线电物理 ｜ J. A. Stratton, *Electromagnetic Theory\*, 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从麦克斯韦方程组开始

无线电物理的全部内容——波的传播、天线的辐射、波导里的导波模式——都是麦克斯韦方程组的推论。这部发表于 1865 年的方程组，把电、磁、光统一成同一件事：电磁波，并预告了它的存在。<span class="marginnote">麦克斯韦做了一件前无古人的事：他不是在解释已知实验，而是在预言新现象。1887 年赫兹用火花隙实验证实电磁波存在时，距理论发表已过 22 年。</span>

这套方程组之所以是无线电物理的起点，在于它是**一切电磁问题的「宪法」**：给定了源（电荷与电流）和边界（介质分界面），解方程组就得到场；天线的辐射、波导的约束、波的折射反射，全部从这四条方程与边界条件里长出来。不把这一章立住，后面每一篇都像在沙地上盖楼。

## 1 积分形式与微分形式

麦克斯韦方程组有四条，先写**积分形式**，它是物理直觉最直接的载体：

$$
\begin{aligned}
\oint_S \mathbf{D} \cdot d\mathbf{S} &= Q_{\mathrm{enc}}, \qquad &
\oint_S \mathbf{B} \cdot d\mathbf{S} &= 0, \\
\oint_C \mathbf{E} \cdot d\mathbf{l} &= -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{S}, \qquad &
\oint_C \mathbf{H} \cdot d\mathbf{l} &= I_{\mathrm{enc}} + \frac{d}{dt}\int_S \mathbf{D} \cdot d\mathbf{S}.
\end{aligned}
$$

四条方程从左到右、从上到下依次是**高斯定律、磁通连续性、法拉第定律、安培–麦克斯韦定律**。前两条管「通量」（散度），后两条管「环流」（旋度）。<span class="marginnote">把电荷放进闭合面包围的空间，电场就「向外流」；把磁通放进一个环，就感应出电场的环流。积分形式的好处是物理图景直接：每一条都是一个「守恒」或「感应」的陈述。</span>

用矢量分析把积分形式压缩到一点，就得到**微分形式**：

$$
\nabla \cdot \mathbf{D} = \rho, \qquad \nabla \cdot \mathbf{B} = 0, \qquad \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \qquad \nabla \times \mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}.
$$

微分形式把「一块区域内通量的净流出」变成「每一点的散度」，把「沿环路的环流」变成「每一点的旋度」——它是偏微分方程理论的语言，也是后面解波动方程的出发点。

## 2 位移电流：麦克斯韦的革命性添加

安培定律原本只有 $\nabla \times \mathbf{H} = \mathbf{J}$。对它两边取散度，左边恒为零，右边 $\nabla \cdot \mathbf{J}$ 却不一定为零——电荷守恒要求 $\nabla \cdot \mathbf{J} = -\partial \rho / \partial t$，两者冲突。麦克斯韦看到这个矛盾，补上一项，使右端变为 $\mathbf{J} + \partial \mathbf{D}/\partial t$。

这一项就是**位移电流密度（displacement current density）** $\mathbf{J}_d = \partial \mathbf{D}/\partial t$。它没有实物粒子流动，却与传导电流一样产生磁场。<span class="marginnote">插入位移电流不只是数学修补：它预言了「变化的电场产生磁场」，与法拉第的「变化的磁场产生电场」对称互补。两条对称的感应定律一合拢，电磁波就从方程里自然涌现——没有这一项，就没有无线电。</span>

取旋度消元可以立刻看到这一点。对 $\nabla \times \mathbf{H} = \mathbf{J} + \partial \mathbf{D}/\partial t$ 取旋度，再代入法拉第定律，在无源、线性、均匀介质中得：

$$
\nabla^2 \mathbf{E} = \mu\varepsilon\frac{\partial^2 \mathbf{E}}{\partial t^2}
$$

这是一个波动方程。波的传播速度 $v = 1/\sqrt{\mu\varepsilon}$，真空中正是 $c = 1/\sqrt{\mu_0\varepsilon_0} \approx 3 \times 10^8$ m/s——**光速**。这是麦克斯韦方程组最震撼的推论，也是「光是一种电磁波」的判据。

## 3 本构关系与媒质分类

方程组自身有 8 个未知量（$\mathbf{E},\mathbf{D},\mathbf{B},\mathbf{H}$ 各 3 分量），四条方程不足定解，需要**本构关系（constitutive relations）**把 $\mathbf{D}$ 与 $\mathbf{E}$、$\mathbf{B}$ 与 $\mathbf{H}$ 联系起来：

$$
\mathbf{D} = \varepsilon \mathbf{E}, \qquad \mathbf{B} = \mu \mathbf{H}, \qquad \mathbf{J} = \sigma \mathbf{E}
$$

其中 $\varepsilon$ 为介电常数，$\mu$ 为磁导率，$\sigma$ 为电导率。媒质按性质可分为：**线性/非线性**（本构关系是否与场强成正比）、**均匀/非均匀**（参数是否随位置变）、**各向同性/各向异性**（参数是否随方向变）、**色散/无色散**（参数是否随频率变）。<span class="marginnote">「无色散」只在理想化模型里成立。真实媒质的 $\varepsilon(\omega)$ 随频率变化，导致不同频率的波速不同——这就是后面色散与相速/群速分离的根源，见第4篇的传播章节。</span>

把本构关系代入并约定**时谐场**（下一章专门展开），麦克斯韦方程组就化简成可解的形式；而几乎所有无线电系统的工作，都在这个化简后的世界里进行。

## 4 边界条件：不连续面上的麦克斯韦方程组

两种介质的分界面，或理想导体表面，是电磁波频繁遭遇的「不连续面」。在面上微分形式失效（场发生跳变），必须回到积分形式，取一个压扁的高斯面或细长矩形回路取极限，得到**边界条件（boundary conditions）**：

$$
\hat{\mathbf{n}} \times (\mathbf{E}_1 - \mathbf{E}_2) = 0, \qquad \hat{\mathbf{n}} \cdot (\mathbf{D}_1 - \mathbf{D}_2) = \rho_s, \qquad
\hat{\mathbf{n}} \times (\mathbf{H}_1 - \mathbf{H}_2) = \mathbf{J}_s, \qquad \hat{\mathbf{n}} \cdot (\mathbf{B}_1 - \mathbf{B}_2) = 0.
$$

其中 $\hat{\mathbf{n}}$ 为界面法向（由介质 2 指向介质 1），$\rho_s$ 与 $\mathbf{J}_s$ 为面电荷与面电流密度。<span class="marginnote">规律很好记：<strong>电场的切向分量连续，磁场的法向分量连续</strong>；而磁场切向被面电流打断，电场法向被面电荷打断。四条中只有「无源侧」的两条连续是铁律。</span>

**辨析｜易错点：** 初学者最容易把「切向/法向」弄反，或误以为所有场量都连续。记住物理来源就不会错：切向电场必须连续，否则沿界面走一圈的环流不为零、法拉第定律被违反；法向磁场必须连续，否则高斯面内的磁通有净流出、磁单极出现。边界条件不是额外假设，而是麦克斯韦方程组在界面上的必然结果。

## 5 理想导体的边界

理想导体内部 $\mathbf{E} = \mathbf{0}$，且时变场下内部 $\mathbf{B} = \mathbf{0}$（趋肤效应把场排挤出去）。于是界面上的边界条件退化为：

$$
\hat{\mathbf{n}} \times \mathbf{E} = 0, \qquad \hat{\mathbf{n}} \cdot \mathbf{B} = 0, \qquad
\hat{\mathbf{n}} \times \mathbf{H} = \mathbf{J}_s, \qquad \hat{\mathbf{n}} \cdot \mathbf{D} = \rho_s
$$

它说的是：电场只允许有法向分量（在导体表面终止于面电荷），磁场只允许有切向分量（在表面感应出面电流）。<span class="marginnote">这两条是<strong>金属天线与波导的全部物理</strong>：波导内壁近似理想导体，边界条件决定了允许的场结构（模式）离散化；天线表面的电流分布由切向磁场直接给出。第2篇的导波模式、第3篇的天线分析，都将反复回到这一页。</span>

理想导体边界的简洁性还体现在电磁场的「镜像法」里：一个在理想导电平面附近的电荷，其场等于去掉平面、在原平面另一侧对称放一个异号电荷的场。这种「用虚源替代边界」的思维，是后面求解天线上方场、波导内场时的常用技巧。

## 6 公式解析：由积分形式推导切向电场连续

取两种介质的界面，构造一个「压扁的长方形回路」：回路两长边分别在介质 1 与介质 2 中，平行于界面；两短边穿越界面，长度 $h \to 0$。对法拉第定律的积分形式取该回路的环流：

$$
\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{S}
$$

逐项拆解：

- **第一步，看左端环流**：两长边的贡献分别为 $\mathbf{E}_1 \cdot \hat{\mathbf{t}}\, \Delta l$ 与 $-\mathbf{E}_2 \cdot \hat{\mathbf{t}}\, \Delta l$（方向相反）；短边贡献随 $h \to 0$ 消失。
- **第二步，看右端磁通**：面积 $S = \Delta l \cdot h \to 0$，只要 $\mathbf{B}$ 有限，右端趋近于零。
- **第三步，令等式成立**：$(\mathbf{E}_1 - \mathbf{E}_2)\cdot \hat{\mathbf{t}} = 0$，对任意切向 $\hat{\mathbf{t}}$ 成立，即切向分量连续，$\hat{\mathbf{n}} \times (\mathbf{E}_1 - \mathbf{E}_2) = 0$。

同样的「压扁回路」手段用于安培–麦克斯韦定律，就得到磁场切向的边界条件；「压扁高斯面」用于两条散度定律，就得到法向边界条件。<span class="marginnote">这套「取极限压扁」的方法是电磁边值问题的通用工具箱。日后处理薄介质层、带孔屏、微带线基片时，你都会回到这个动作：把三维的积分定律压到一个二维面上，读出跳变关系。</span>

## 7 小结

- 麦克斯韦方程组四条：**高斯定律、磁通连续性、法拉第定律、安培–麦克斯韦定律**，各有积分与微分两种形式。
- **位移电流** $\partial \mathbf{D}/\partial t$ 让方程组自洽，并推出波动方程与光速，统一了光与电磁波。
- 本构关系 $\mathbf{D} = \varepsilon\mathbf{E}$、$\mathbf{B} = \mu\mathbf{H}$ 补全方程组；媒质按线性、均匀、各向同性、无色散四维分类。
- 边界条件：**电场切向连续、磁场法向连续**；面电荷打断电场法向、面电流打断磁场切向。
- 理想导体表面：电场只留法向分量，磁场只留切向分量——波导与天线的分析基石。

在下一节，我们将把麦克斯韦方程组放进**时谐场**的框架，用复矢量（相量）方法把对时间的偏导变成代数的 $j\omega$