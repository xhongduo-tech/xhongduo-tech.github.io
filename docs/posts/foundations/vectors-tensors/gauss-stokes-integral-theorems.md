---
title: 积分定理：Gauss 散度定理与 Stokes 旋度定理
date: 2026-08-11
---

# 积分定理：Gauss 散度定理与 Stokes 旋度定理

<div class="epigraph">
<p>物理定律的两种表述，往往一种是局部微分式，一种是整体积分式，而它们在数学上等价。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础科学 · 向量与张量初步 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么把「局部」与「整体」焊接起来

前两讲分别交付了两套工具：**微分算子**（梯度、散度、旋度）给出逐点的局部信息，**三种积分**给出区域上的整体累计。这一讲把它们焊成两座桥——**Gauss 散度定理**与 **Stokes 旋度定理**。<span class="marginnote">本讲对应 Arfken 第一章与第二章末节、Boas §6.5–§6.6。两条定理是「微积分基本定理」的多元推广——单变量里「导数的积分 = 端点的差」，多元里「算子的体积/面积分 = 边界的积」。</span>

这两条定理的物理地位怎么强调都不过分：**电磁学的 Maxwell 方程组**（Gauss 定律、Faraday 定律）本质上就是它们在微分形式下的改写；流体力学里「边界流量 = 内部散度之和」正是 Gauss 定理；而「环路上的线积分 = 包围面上旋度之和」是 Stokes 定理。理解它们，等于拿到了整个经典场论的钥匙。

## 1 Gauss 散度定理：流出边界的总量 = 内部散度的体积分

**Gauss 散度定理（divergence theorem）**：设 $V$ 是空间中的有界闭区域，边界为闭曲面 $S$（法向朝外），$\mathbf F$ 在其上连续可微，则

$$
\oint_S \mathbf F \cdot d\mathbf S = \int_V (\nabla\cdot\mathbf F)\, dV
$$

左侧是**穿过闭合曲面的总通量**，右侧是**内部散度的体积分**。<span class="marginnote">理解法：把 $V$ 切成无数小立方体，相邻立方体共享的面上，通量一个出、一个进，恰好抵消；剩下来的只有最外层面上的通量。内部净流出量 = 各点散度之和——逐项相加后边界项独存。</span>

用「水」来记忆：河流某区域的入水量减出水量，等于该区域内部「涌出的水总量」。散度是**单位体积的净流出密度**，Gauss 定理把「密度」还原成「总量」。

## 2 Stokes 旋度定理：环路上的环量 = 所张曲面上的旋度通量

**Stokes 定理**：设 $S$ 是以闭合曲线 $C$ 为边界的光滑曲面，$\hat{\mathbf n}$ 与其边界 $C$ 满足右手定则，则

$$
\oint_C \mathbf F \cdot d\mathbf l = \int_S (\nabla\times\mathbf F)\cdot d\mathbf S
$$

左侧是**绕边界的环量**（环路上的线积分），右侧是**旋度穿过曲面的通量**。<span class="marginnote">曲面选哪张都行——只要边界同为 $C$。这是 Stokes 定理最反直觉的地方：旋度通量居然不依赖曲面形状，只依赖边界！这正是它对称美的来源。</span>

两个直观检验：

若 $\nabla\times\mathbf F = \mathbf 0$（无旋场），则 $\oint \mathbf F\cdot d\mathbf l = 0$——上一讲「保守场环路积分为零」在此获得了完整证明。
- 若把 $S$ 缩得无穷小，则 $\nabla\times\mathbf F \cdot \hat{\mathbf n} \approx \dfrac{\oint \mathbf F\cdot d\mathbf l}{\text{面积}}$——**旋度 = 单位面积上的环量密度**，与散度的解释完美对偶。

## 3 两条定理是同一个故事的两面

把 Gauss 与 Stokes 并排看，它们共享同一骨架：

| | 积分形式 | 微分形式 | 维数 |
| --- | --- | --- | --- |
| 微积分基本定理 | $\int_a^b f'(x)\,dx = f(b)-f(a)$ | $f'$ | 1 维 |
| Gauss 定理 | $\oint_S \mathbf F\cdot d\mathbf S = \int_V \nabla\cdot\mathbf F\, dV$ | $\nabla\cdot\mathbf F$ | 3 维（体积 ↔ 闭面） |
| Stokes 定理 | $\oint_C \mathbf F\cdot d\mathbf l = \int_S \nabla\times\mathbf F\cdot d\mathbf S$ | $\nabla\times\mathbf F$ | 2 维（曲面 ↔ 边界） |

共同结构一句话：**「某个量的边界积分 = 这个量的微分算子作用后的内部积分」**。<span class="marginnote">现代微分几何把它们统一为一条 Stokes 广义定理 $\int_{\partial\Omega}\omega = \int_\Omega d\omega$：边界上的积分等于内部外微分后的积分。这一条公式囊括上面三行——第三级《微分几何》会展开。</span> 微分形式记号的威力，正在于让「边界算子 $\partial$」与「外微分 $d$」看起来像一对对偶。

**辨析｜易错点：** 方向与朝向约定是两定理的高频翻车点。Gauss 定理闭合面法向**朝外**；Stokes 定理中曲面的法向与边界绕行方向必须满足**右手定则**。方向错了，等式两边差一个负号。

## 4 应用：把方程改写成两种形式

两条定理的实用价值在于**积分形式 ↔ 微分形式互译**。以 Maxwell 方程组为例：

**Gauss 定律**：$\oint_S \mathbf E\cdot d\mathbf S = \dfrac{Q_{\text{内}}}{\varepsilon_0} \iff \nabla\cdot\mathbf E = \dfrac{\rho}{\varepsilon_0}$。通量积分式（整体、好测）与散度微分式（局部、好推演）由 Gauss 定理同化。
- **Faraday 定律**：$\oint_C \mathbf E\cdot d\mathbf l = -\dfrac{d\Phi_B}{dt} \iff \nabla\times\mathbf E = -\dfrac{\partial\mathbf B}{\partial t}$。左边是环量，Stokes 定理把它变成旋度的面通量，再对任意曲面成立即得微分式。<span class="marginnote">“对任意曲面成立”是关键一步：若一个积分对任意积分区域都为零，则被积函数本身必须处处为零。这是“积分定理把整体等式化成逐点等式”的标准手法。</span>

这套「整体 $\iff$ 局部」的翻译术，也是偏微分方程、流体力学、电磁学反复出现的通用技能。

## 5 公式解析：Gauss 定理为什么两边是同一个量

$$

\oint_S \mathbf F \cdot d\mathbf S = \int_V (\nabla\cdot\mathbf F)\, dV

$$

拆成四步看：

- **第一步，区域切成小立方体**：把 $V$ 剖成大量小体积元 $dV_i$。每个小立方体有自己的边界面，其通量 $\oint_{\partial V_i}\mathbf F\cdot d\mathbf S$ 可近似为 $\nabla\cdot\mathbf F\,\big|_i \,dV_i$——这是散度的定义（单位体积净流量）在小尺度上的直接使用。
- **第二步，相邻面互相抵消**：相邻两立方体共享一个面，法向相反（一个朝外必朝另一个朝外），通量 $\mathbf F\cdot d\mathbf S$ 恰好一正一负，相加为零。内部面全部成对消失。
- **第三步，只剩外边界**：所有小立方体的边界求和中，唯一不成对的是最外层——它们合成整个 $V$ 的边界 $S$。于是左端出现 $\oint_S$。
- **第四步，连起来**：$\sum_i \nabla\cdot\mathbf F\big|_i\,dV_i \to \int_V \nabla\cdot\mathbf F\, dV$（取极限），而边界求和就是 $\oint_S$。等式成立。

同样的论证换个方向读，就得到 Stokes 定理——只是把「立方体」换成「小面片」，把「面通量」换成「边环绕量」。**两条定理不是两个孤立结论，而是同一个「切分-抵消-取极限」方案在两种几何形状上的执行**。

## 6 应用：从散度定理到点电荷

看一个经典而极端的例子。点电荷 $q$ 的电场在球坐标下为

$$
\mathbf E = \frac{q}{4\pi\varepsilon_0}\,\frac{\hat{\mathbf r}}{r^2}
$$

在球坐标里算它的散度（用到《正交曲线坐标》的公式）：

$$
\nabla\cdot\mathbf E = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 E_r\right) = \frac{1}{r^2}\frac{\partial}{\partial r}\left(\frac{q}{4\pi\varepsilon_0}\right) = 0 \quad (r \ne 0)
$$

除原点外散度处处为零——**点电荷的空间里没有「源」**。可是 Gauss 定律又说，穿过包围电荷任意球面的通量是 $q/\varepsilon_0$，非零。矛盾吗？

不矛盾。矛盾恰好指出了散度定理的**边界细节**：原点处 $\mathbf E$ 无穷大，$r=0$ 这一点不在「连续可微」的适用条件内。通量「凭空」从原点漏出来，数学上对应一个尖峰——**Dirac delta 函数**：

$$
\nabla\cdot\mathbf E = \frac{q}{\varepsilon_0}\,\delta(\mathbf r)
$$

于是「电荷是场的源」有了精确的数学表述：**源的强度集中在一点，散度只在这一点非零，而 Gauss 定理负责把点源翻译成整体通量**。<span class="marginnote">这个思想直接导出 Green 函数法与 Poisson 方程 $\nabla^2\phi = -\rho/\varepsilon_0$ 的解——「点源 + 叠加」是场论里最通用的求解策略。见第三级《偏微分方程》与第二级《数学物理方法》。</span>

这条链子值得记住：**散度定理 + 点源模型 = 全场论的方法论**。它也在大模型的语境里出现——attention 里每个 token 都像一个「点源」，对整体信息场做加权叠加。

## 7 小结

- **Gauss 定理** $\oint_S\mathbf F\cdot d\mathbf S = \int_V\nabla\cdot\mathbf F\,dV$：闭面通量 = 内部散度的体积分。
- **Stokes 定理** $\oint_C\mathbf F\cdot d\mathbf l = \int_S\nabla\times\mathbf F\cdot d\mathbf S$：边界环量 = 张面上旋度的通量。
- **统一骨架**：边界积分 = 微分算子作用的内部积分；微分几何里浓缩为一条广义 Stokes 定理 $\int_{\partial\Omega}\omega=\int_\Omega d\omega$。
- **应用核心**：积分式与微分式互译，是 Maxwell 方程组的两种等价写法。

在下一节，我们从笛卡尔坐标的温室里走出来——当坐标系变成球面、柱面甚至更一般的曲线坐标，梯度、散度、旋度该长成什么样子？这就是**正交曲线坐标与度量系数**。
