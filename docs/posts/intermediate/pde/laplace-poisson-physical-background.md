---
title: 拉普拉斯方程与泊松方程的物理背景（引力场、静电场）
date: 2026-08-08
---

# 拉普拉斯方程与泊松方程的物理背景（引力场、静电场）

<div class="epigraph">
<p>位势函数的拉普拉斯，是场的「源」密度的测量仪。</p>
<footer>—— 位势理论（potential theory）的出发点</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第三章 ｜ 2026-08-08</p>
</div>

## 为什么从物理背景开始

波动方程描述「动」，热传导描述「耗散」，而拉普拉斯方程描述「静」——引力场、静电场、稳态温度场在平衡状态下全都满足它。椭圆型方程的哲学在物理背景里一目了然：**没有时间、没有传播，场由边界和源完全决定。** 这一节从引力与静电两个最经典的例子导出拉普拉斯方程与泊松方程，为第六篇的调和函数理论铺好物理地基。学完这一节，你会明白「位势」（potential）这个名字从何而来。

## 1 静电场：泊松方程的导出

真空中电荷密度 $\rho(x)$ 产生的静电场 $\boldsymbol{E}$，满足 Maxwell 方程组的两个关键式：

$$
\nabla \cdot \boldsymbol{E} = \frac{\rho}{\varepsilon_0} \quad\text{（Gauss 定律）}, \qquad \nabla \times \boldsymbol{E} = 0 \quad\text{（无旋）}
$$

**无旋场**保证存在**电势（electric potential）**$\varphi$，使 $\boldsymbol{E} = -\nabla\varphi$（静电场保守，做功与路径无关）。代入 Gauss 定律：

$$
-\nabla\cdot(\nabla\varphi) = \frac{\rho}{\varepsilon_0} \quad\Longrightarrow\quad \Delta\varphi = -\frac{\rho}{\varepsilon_0}
$$

这就是**泊松方程（Poisson equation）**。在无电荷区域 $\rho = 0$，得到**拉普拉斯方程**

$$
\boxed{\;\Delta u = 0\;}
$$

<span class="marginnote">「拉普拉斯算子」$\Delta = \partial_{xx} + \partial_{yy} + \partial_{zz}$ 在物理上就是「一点的值与邻域平均的偏差」的度量（二阶差商）。$\Delta u = 0$ 说：$u$ 处处等于邻域平均值——这直观上就是「势场里没有峰谷、处处均衡」的意思，下一节的平均值定理会让它精确化。</span>

**泊松方程与拉普拉斯方程的关系：拉普拉斯是无源情形的泊松。** 源密度 $\rho$ 是「泊松方程的右端」，源为零则降级为拉普拉斯。

## 2 引力场：同一个方程

牛顿引力场 $\boldsymbol{g}$ 满足

$$
\nabla\cdot\boldsymbol{g} = -4\pi G\rho_m, \qquad \nabla\times\boldsymbol{g} = 0
$$

$\rho_m$ 是质量密度，$G$ 是万有引力常数。无旋性给出引力势 $\Phi$：$\boldsymbol{g} = -\nabla\Phi$。代入得

$$
\Delta\Phi = 4\pi G\rho_m
$$

**引力场与静电场的方程结构完全一样，只是常数换了。** 这是「一方程多物理」的又一例——位势理论（potential theory）这个名字正是为了统一处理这类「源决定势、势的梯度是场」的问题。

| 物理系统 | 势 | 源密度 | 方程 |
| --- | --- | --- | --- |
| 静电场 | 电势 $\varphi$ | $-\rho/\varepsilon_0$ | $\Delta\varphi = -\rho/\varepsilon_0$ |
| 引力场 | 引力势 $\Phi$ | $4\pi G\rho_m$ | $\Delta\Phi = 4\pi G\rho_m$ |
| 稳态温度 | 温度 $u$ | 热源密度 $f$ | $\Delta u = -f/k$ |

## 3 稳态热传导：另一个入口

热传导方程 $u_t = a^2\Delta u$ 的**稳态**（$\partial u/\partial t = 0$）给出

$$
\Delta u = 0
$$

一个物体长时间置于稳定环境下，内部温度不再随时间变化，就是调和场。稳态热传导、静电场、引力场在数学上共用一个方程——**这是「椭圆型 = 平衡态」的物理根源**：方程不含时间，解由边界条件整体决定，没有「演化」只有「平衡」。<span class="marginnote">对比双曲（有 $u_{tt}$，惯性主导）与抛物（有 $u_t$，耗散主导），椭圆型两者皆无——它是「时间不存在」的方程。这也是为什么椭圆型问题通常提边界条件而非初始条件：没有时间轴，就没有「初始」可言。</span>

## 4 公式解析：从高斯定律到点源势

用点电荷验证方程，把「泊松方程」和「位势」两个概念钉死。真空中位于原点的点电荷 $Q$，电势是库仑势

$$
\varphi(x) = \frac{Q}{4\pi\varepsilon_0}\frac{1}{r}, \qquad r = |x|
$$

- **第一步，算拉普拉斯（$r \ne 0$）。** 直接求导：
  $$ \frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\frac{1}{r}\right) = \frac{\partial}{\partial r}(-1) = 0 $$
  故 $\Delta\frac{1}{r} = 0$（$r\neq0$）——库仑势在无源处满足拉普拉斯方程。
- **第二步，算原点处的「源」大小。** 用 Gauss 定理（散度定理）包围原点：$\int_{|x|=\epsilon}\nabla\varphi\cdot\boldsymbol{n}\,dS$ 恰好给出 $-Q/\varepsilon_0$。
- **第三步，用 δ 函数统一表达。** 综合两步：
  $$ \Delta\left(\frac{1}{4\pi\varepsilon_0}\frac{Q}{r}\right) = -\frac{Q}{\varepsilon_0}\delta(x) $$
  拉普拉斯作用在 $1/r$ 上，在原点产生一个 δ 函数「源」。
- **第四步，结论。** 泊松方程 $\Delta\varphi = -\rho/\varepsilon_0$ 对点源 $\rho = Q\delta$ 成立——**$1/r$ 是泊松方程的基本解**（第六篇专节）。

**「基本解」$1/r$ 是位势理论的原子：** 任意电荷分布的势 = 无数点源的势的叠加（积分）。这个「用基本解做卷积」的思想将在第六篇、第九篇反复使用，是格林函数方法的种子。

## 5 物理背景的启发

拉普拉斯方程的物理出身给出三条方法论启示：

1. **为什么叫「位势」**：$u$ 的梯度是物理场（力场/电场），$u$ 本身是「势」——它只有相对意义，整体加减常数不影响场。这决定了拉普拉斯方程的解「差一个常数仍可」——Neumann 问题需要相容性条件的原因（第六篇专节）。
2. **为什么是椭圆型**：没有时间、能量没有传播方向，解由边界整体决定——与波动、热传导的「信息沿特征线走」完全不同。
3. **为什么要研究调和函数**：位势（温度、电势、引力势）在无源区都是调和函数，研究调和函数就是研究「无源场」的全部性质——平均值、极值、解析性，都是位势理论的核心内容。<span class="marginnote">从物理背景出发的研究路径，在数学物理方程这门课里一以贯之：先建方程（本篇），再解方程（波动/热传导/拉普拉斯三篇），最后抽象理论（分类、广义函数、变分）。拉普拉斯方程站在「平衡态」这一端，是整条路径的收官一环。</span>

## 6 小结

- 静电场无旋 ⇒ 存在电势 $\varphi$，Gauss 定律给出泊松方程 $\Delta\varphi = -\rho/\varepsilon_0$。
- 无源区域（$\rho=0$）得到拉普拉斯方程 $\Delta u = 0$。
- 引力场、稳态温度场满足同一类方程——位势理论的统一对象。
- 点源势 $1/r$ 在 $r\ne0$ 处调和、原点处产生 δ 源，是泊松方程的基本解。
- 椭圆型 = 平衡态，解由边界整体决定，无时间演化。

在下一节，我们正式定义调和函数，并看它的基本例子。
