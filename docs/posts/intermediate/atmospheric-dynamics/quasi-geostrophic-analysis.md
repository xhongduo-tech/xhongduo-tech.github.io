---
title: 准地转分析与天气尺度动力学
date: 2026-08-07
---

# 准地转分析与天气尺度动力学

<div class="epigraph">
<p>大气的天气系统活在两个世界里：它几乎地转，却靠那一点非地转活过来。</p>
<footer>—— 化用朱尔 · 沙尼（Jule Charney），1948 年</footer>
</div>

<div class="article-byline">
<p>第二级 · 大气动力学 ｜ Holton &amp; Hakim《An Introduction to Dynamic Meteorology》§6–7 ｜ 2026-08-07</p>
</div>

## 为什么从准地转开始

前几节我们反复说：中纬度大尺度运动**几乎**处于地转平衡，而那 1/10 的非地转偏差（年龄风）才真正驱动天气。问题来了：怎么把「几乎平衡」本身变成一套**自洽的、可解的**动力学？直接解完整方程太复杂，直接把年龄风扔掉又丢了天气。**准地转（quasi-geostrophic, QG）近似**就是答案：把运动分成「地转主项 + 年龄风小修正」，对两者都做一致的处理，得到一套只含两个未知函数的闭合方程。它是动力气象学最经典的「解析引擎」——从它推出**ω 方程**与**高度倾向方程**，预报员就能从一张天气图上回答「这个低压会不会发展、槽会不会加深」。今天所有数值模式的诊断工具里，QG 的身影依然无处不在。

## 1 准地转近似的思想：用地转风做骨架

QG 近似的起点是罗斯贝数 $Ro \ll 1$（中纬度天气尺度约 0.1）。它做的三件事：

1. **水平速度分解**：$\mathbf{V} = \mathbf{V}_g + \mathbf{V}_a$，其中 $\mathbf{V}_a$ 是年龄风，比地转风小一个量级。
2. **地转风作为「骨架」**：所有**平流项**里的速度都用 $\mathbf{V}_g$ 近似（地转风做输送）；但**散度项**里必须保留 $\mathbf{V}_a$——因为地转风本身无辐散（$\nabla\cdot\mathbf{V}_g=0$），辐合辐散全部来自年龄风，而辐合辐散是垂直运动的来源。
3. **科里奥利参数取 $f_0$**（参考纬度常数），但它在涡度平流里保留随纬度的变化（$\beta$ 效应）——这正是罗斯贝波得以进入方程组的方式。

一句话总结：**QG 近似是「用地转风运送涡度与温度，用年龄风输送散度与垂直速度」的近似**。<span class="marginnote">这看似自相矛盾（输送靠主项、散度靠小量），但正是它让方程组既保留全部天气动力学、又滤掉了声波和大部分重力波——数值预报里「过滤掉高频波」的经典思路就来自这里。</span>

由此可证一个漂亮的守恒律：**QG 位势涡度（QGPV）守恒**——在无摩擦绝热近似下，跟随地转气流，$q$ 不变：

$$
q = \frac{1}{f_0}\nabla^2\Phi + f + \frac{\partial}{\partial p}\left(\frac{f_0}{\sigma}\frac{\partial\Phi}{\partial p}\right)
$$

它把「旋转（涡度项）+ 行星效应（$f$）+ 静力稳定度（第三项）」统一成一个量。给定 $q$ 的分布和边界条件，可以唯一地**反演**出风场与位势场——这就是现代「PV 思维」（Hoskins 1985）的数学核心：**位势涡度反演原理**。<span class="marginnote">反演（inversion）与守恒（conservation）是 PV 的双翼：守恒告诉你 q 怎么随时间变，反演告诉你 q 长什么样对应什么天气。现代天气诊断把「PV 异常」当作天气系统的身份证——气旋就是一个正的 PV 异常柱。这个概念是《斜压不稳定性》与《中层大气》的公共语言。</span>

## 2 准地转方程组：涡度方程 + 热力学方程

由动量方程与热力学第一定律做一致近似，得 QG 系统两个核心方程：

**准地转涡度方程**（相对涡度以地转涡度 $\zeta_g = \frac{1}{f_0}\nabla^2\Phi$ 表示）：

$$
\frac{\partial\zeta_g}{\partial t} = -\mathbf{V}_g\cdot\nabla(\zeta_g + f) + f_0\frac{\partial\omega}{\partial p}
$$

右边第一项是涡度平流（含 $\beta$ 效应），第二项是**拉伸项**——气柱在等压面之间的拉伸（$\partial\omega/\partial p$ 辐散辐合）改变涡度。

**准地转热力学方程**（厚度倾向，$- \partial\Phi/\partial p$ 正比于温度）：

$$
\frac{\partial}{\partial t}\left(-\frac{\partial\Phi}{\partial p}\right) = -\mathbf{V}_g\cdot\nabla\left(-\frac{\partial\Phi}{\partial p}\right) - \sigma\omega
$$

右边是温度平流与绝热升降（$\sigma\omega$）。联合消去 $\partial\omega/\partial p$ 与 $\partial\Phi/\partial t$，分别得到两个诊断方程——这就是本节的明星。

## 3 高度倾向方程与 ω 方程

**高度倾向方程（height tendency equation）**（$\chi = \partial\Phi/\partial t$ 即位势高度变化率）：

$$
\left(\nabla^2 + \frac{f_0^2}{\sigma}\frac{\partial^2}{\partial p^2}\right)\chi
= -f_0\,\mathbf{V}_g\cdot\nabla(\zeta_g+f)
-\frac{f_0^2}{\sigma}\frac{\partial}{\partial p}\left(\mathbf{V}_g\cdot\nabla\frac{\partial\Phi}{\partial p}\right)
$$

左边是椭圆型算子（一个「三维拉普拉斯」），右边是两项强迫：**涡度平流**与**温度平流随高度的变化**。数学上，对椭圆算子做逆运算，强迫项的正负决定高度升降。

**ω 方程（omega equation）**（$\omega = Dp/Dt$，上升运动为负）：

$$
\left(\nabla^2 + \frac{f_0^2}{\sigma}\frac{\partial^2}{\partial p^2}\right)\omega
= \underbrace{\frac{f_0}{\sigma}\frac{\partial}{\partial p}\left[\mathbf{V}_g\cdot\nabla(\zeta_g+f)\right]}_{\text{差动涡度平流}}
+ \underbrace{\frac{1}{\sigma}\nabla^2\left[\mathbf{V}_g\cdot\nabla\left(-\frac{\partial\Phi}{\partial p}\right)\right]}_{\text{厚度（温度）平流}}
$$

这是天气尺度动力学最著名的方程。它不随时间积分——给定某时刻的位势场，解一次椭圆方程就得到**此刻的垂直运动场**。<span class="marginnote">ω 方程的诊断地位类似静电学里的泊松方程：电荷分布（强迫项）一给，电势场（ω）立刻可解。它不预言未来，但把「当前场里藏着的上升/下沉」完整地挖出来。历史上 Charney、Eliassen、Sutcliffe 在 1940 年代各自独立推出它，是三股独立研究殊途同归的佳话。</span>

## 4 公式解析：准地转 ω 方程

$$

\left(\nabla^2 + \frac{f_0^2}{\sigma}\frac{\partial^2}{\partial p^2}\right)\omega
= \frac{f_0}{\sigma}\frac{\partial}{\partial p}\left[\mathbf{V}_g\cdot\nabla(\zeta_g+f)\right]
+ \frac{1}{\sigma}\nabla^2\left[\mathbf{V}_g\cdot\nabla\left(-\frac{\partial\Phi}{\partial p}\right)\right]
$$

逐项拆解这条「天气诊断第一方程」：

- **第一步，左边算子**：$\nabla^2$（水平）+ $\frac{f_0^2}{\sigma}\frac{\partial^2}{\partial p^2}$（垂直）。这是一个椭圆算子：$\omega$ 的「弯曲」由强迫项决定，边界处 $\omega=0$。逆算子对强迫项的作用相当于「把源抹平成场」——局部源被摊开成大范围的小上升/下沉。
- **第二步，差动涡度平流**：$\frac{f_0}{\sigma}\frac{\partial}{\partial p}[\mathbf{V}_g\cdot\nabla(\zeta_g+f)]$ 是「涡度平流随高度的变化率」。正涡度平流（PVA）随高度增强 → 强迫为正 → 逆算子后 $\omega<0$，**上升**。这就是「槽前上升、槽后下沉」的数学出处——槽前（下游）是正涡度平流最强处。
- **第三步，厚度（温度）平流**：$\frac{1}{\sigma}\nabla^2[\mathbf{V}_g\cdot\nabla(-\frac{\partial\Phi}{\partial p})]$。$-\partial\Phi/\partial p\propto T$，$\mathbf{V}_g\cdot\nabla T>0$ 是暖平流。暖平流中心附近强迫为正 → **上升**。物理上：暖平流让气柱「变暖膨胀」，低层必然辐合抬升。
- **第四步，读出天气**：**上升区（$\omega<0$）＝ 差动正涡度平流 + 暖平流**，对应云、降水、锋面抬升；下沉区反之。预报员口头禅「涡度平流随高度增强 + 暖平流 → 上升」，就是这条方程的译文。

## 5 用 ω 方程读天气：槽脊发展与气旋加深

把 ω 方程与高度倾向方程合起来，就能读一张 500 hPa 天气图：

- **槽前（下游）**：正涡度平流最强，且随高度增强 → **高度下降 + 上升运动** → 槽加深、云雨发展。
- **槽后（上游）**：负涡度平流 → 高度上升 + 下沉 → 槽被「填平」、天气转晴。
- **气旋加深（cyclogenesis）**：中心高度持续下降。Sutcliffe 的发展理论把它归结为**差动涡度平流的强度**——而差动涡度平流又近似正比于**温度平流**（因为涡度平流随高度的变化来自温度场的斜压结构）。于是得到一个简洁的诊断链：**水平温差（斜压性）→ 暖/冷平流 → 差动涡度平流 → 上升/下降与高度倾向 → 气旋加深或减弱**。<span class="marginnote">这条链的终点就是下一节《斜压不稳定性》：当温差与扰动位相配合得当（暖湿气流抬升、冷气流下沉、释放有效位能），扰动就能「自养」地增长——不是被强迫，而是靠斜压能量机制放大的不稳定模态。</span>

**垂直运动的「强迫」与「响应」分离**：ω 方程是**诊断**的（当场反演），不描述时间演变——天气系统的演变藏在高度倾向方程的强迫里，两者耦合才构成 QG 闭环。

## 6 急流与年龄风环流

QG 方程还能解释急流附近的次级环流。把 ω 方程应用到**急流核（jet streak）**，会得到**四象限模型**：急流入口与出口区，由于地转风加速/减速，柯里奥利力不再平衡，出现跨流线的年龄风——入口左前与出口右后是上升区，入口右前与出口左后是下沉区。<span class="marginnote">四象限模型是航空气象里「颠簸区」预报的经典工具：急流入口/出口的上升区对应卷云与对流，下沉区对应晴空。它也是理解「急流如何自己修正自己」——年龄风环流把动量和热量重新分配、把急流维持在近地转状态——的入口。</span>

这套「地转骨架 + 年龄风响应」的架构，把上一节的罗斯贝波与下一节的斜压不稳定性串成了同一根链条：QG 涡度方程给出波动的相速（罗斯贝波），QG 系统在斜压基本态上的不稳定解给出气旋的增长——两个问题，一套方程。

## 7 小结

- **QG 近似**：$Ro\ll1$，平流用 $\mathbf{V}_g$、散度保留年龄风 $\mathbf{V}_a$；滤掉声波、保留罗斯贝波。
- **QGPV 守恒** $q=\dfrac{1}{f_0}\nabla^2\Phi+f+\dfrac{\partial}{\partial p}\left(\dfrac{f_0}{\sigma}\dfrac{\partial\Phi}{\partial p}\right)$，配合反演原理构成 PV 思维。
- **QG 系统**：涡度方程 + 热力学方程，消元得高度倾向方程与 **ω 方程**两个诊断方程。
- **ω 方程判据**：差动正涡度平流与暖平流 → 上升；槽前上升、槽后下沉。
- **气旋加深**：斜压温差 → 暖/冷平流 → 差动涡度平流 → 中心高度下降（Sutcliffe 发展理论）。
- **急流四象限**：入口/出口的年龄风次级环流决定上升与下沉区。

在下一节，我们把「扰动会发展」这句话变成严格的数学：**斜压不稳定性**。Eady 与 Charney 的两个经典模型将告诉我们：什么样的波长长得最快、什么条件必生不稳定——中纬度气旋的「出生证」就握在这里。
