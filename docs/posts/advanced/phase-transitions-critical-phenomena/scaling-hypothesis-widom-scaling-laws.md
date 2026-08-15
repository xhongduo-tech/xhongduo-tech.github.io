---
title: 标度假设与 Widom 标度律
date: 2026-08-07
---

# 标度假设与 Widom 标度律

<div class="epigraph">
<p>我提出：在临界点附近，自由能是温度与场（或密度）的一个齐次函数。</p>
<footer>—— 本杰明 · 维登（Benjamin Widom, "Equation of State in the Neighborhood of the Critical Point", J. Chem. Phys. 43, 3898, 1965）</footer>
</div>

<div class="article-byline">
<p>第四级 · 相变与临界现象 ｜ Cardy Ch.3、Goldenfeld Ch.5 ｜ 2026-08-07</p>
</div>

## 为什么需要标度假说

Onsager 解给出了 2D Ising 模型那串精确的「怪分数」：$\beta = 1/8$、$\gamma = 7/4$。实验给出了三维世界另一串接近的数值。这些数字各不相同，却共享同一组**关系**：$\alpha + 2\beta + \gamma = 2$、$\gamma = \beta(\delta-1)$、$\gamma = \nu(2-\eta)$。巧合不可能如此系统。**一定有一条更深的原则把这些指数绑在一起**——这就是本节的主题：**标度假说（scaling hypothesis）**。

1965 年，本杰明 · 维登（Benjamin Widom）提出一个惊人的简单假设：**临界点附近自由能的奇异部分，是 $t$ 与 $h$ 的齐次函数。** 一个齐次函数只有一个自由参数（「齐次度」），却同时约束着所有临界指数——于是六个指数之间的标度关系，不再是经验巧合，而是一条假设的逻辑推论。本节将完整走一遍这条推理，并检验它在 Onsager 解与实验中的成色。

## 1 齐次函数：标度不变性的代数语言

先建立工具。**齐次函数（homogeneous function）**：若对任意 $\lambda > 0$ 成立

$$
g(\lambda x, \lambda y) = \lambda^p g(x, y)
$$

则称 $g$ 是 $(x,y)$ 的 $p$ 次齐次函数。<span class="marginnote">齐次函数的最简单例子是单项式 $x^k y^{p-k}$——每个变量贡献一个幂，合计幂次为 $p$。齐次性意味着「把两个变量同时放大 $\lambda$ 倍，函数整体放大 $\lambda^p$ 倍」，即函数的形状在缩放下<strong>不变</strong>。这正是临界点处「没有特征尺度」的代数翻译。</span>

齐次函数最实用的推论是：可以把它写成「一个变量整体提出来，其余变量以比值出现」的形式：

$$
g(x, y) = x^{p} \, \tilde{g}\!\left(\frac{y}{x}\right)
$$

例如 $p=1$ 时 $g(x,y) = x\, \tilde g(y/x)$。这个「提幂次、留比值」的结构，就是标度律的引擎：**两个变量的缩放关系，被压缩成一个无量纲比值的函数**。

**Widom 标度假设**：临界点附近自由能的奇异部分满足齐次性，对 $T > T_c$ 与 $T \lt  T_c$ 两侧分别有

$$
f_{\text{sing}}(t, h) = |t|^{2-\alpha} \, \Phi_{\pm}\!\left(\frac{h}{|t|^{\Delta}}\right)
$$

其中 $t = (T-T_c)/T_c$，$h$ 是外场（或化学势偏离），$\Phi_{\pm}$ 是 $t>0$、$t\lt 0$ 两侧各一个的普通函数，$\Delta$ 称**缺口指数（gap exponent）**。这个式子是本节的中心对象。

## 2 从假设推出指数关系：Rushbrooke 与 Widom

现在让这个假设接受检验：看它能推出多少已知的指数关系。记 $x = |t|$，热力学量由自由能对 $h$ 求导得到。

**磁化强度**是自由能对场的一阶导：

$$
m = -\frac{\partial f_{\text{sing}}}{\partial h} = |t|^{2-\alpha-\Delta} \left[-\Phi_\pm'\!\left(\frac{h}{|t|^\Delta}\right)\right]
$$

零场（$h = 0$）下 $\Phi_\pm'(0)$ 是常数，于是 $m \propto |t|^{2-\alpha-\Delta}$。与定义 $m \propto |t|^\beta$ 比较：

$$
\beta = 2 - \alpha - \Delta
$$

**磁化率**是二阶导。零场下 $\Phi_\pm''(0)$ 为常数，$h^2/|t|^{2\Delta}$ 贡献主导项，得 $\chi \propto |t|^{2-\alpha-2\Delta}$，与 $\chi \propto |t|^{-\gamma}$ 比较：

$$
\gamma = 2\Delta + \alpha - 2
$$

两式相加立即得到 **$\beta + \gamma = \Delta$**，即缺口指数就是 $\beta+\gamma$。<span class="marginnote">把 $\beta = 2-\alpha-\Delta$ 代入 $\gamma = 2\Delta+\alpha-2$，两式联立消去 $\Delta$，就得到 $\alpha + 2\beta + \gamma = 2$——这就是 Rushbrooke 关系。它不依赖任何特定模型的细节，纯粹是齐次假设的代数推论。</span>

**Widom 关系 $\gamma = \beta(\delta-1)$** 需要临界等温线（$t = 0$）。在标度式里，$t=0$ 时 $h/|t|^\Delta$ 的比值必须保持有限以给出非零 $m$，故 $h \sim |t|^\Delta$；又 $m \sim |t|^\beta$，消去 $|t|$ 得 $h \sim m^{\Delta/\beta}$。与临界等温线定义 $h \sim m^\delta$ 比较：$\delta = \Delta/\beta$，再利用 $\Delta = \beta+\gamma$，得到 $\gamma = \beta(\delta-1)$。

## 3 公式解析：一条假设如何同时管住三个指数

把上面的推导浓缩成一个「推导瀑布」，看齐次假设怎么一口气推出全部静态指数关系：

$$
f_{\text{sing}}(t,h) = |t|^{2-\alpha}\Phi_\pm\!\left(\frac{h}{|t|^\Delta}\right)
\quad \xrightarrow{\ \partial_h\ } \quad
\begin{cases}
m \sim |t|^{2-\alpha-\Delta} & \Rightarrow\ \beta = 2-\alpha-\Delta \\
\chi \sim |t|^{2-\alpha-2\Delta} & \Rightarrow\ \gamma = 2\Delta+\alpha-2
\end{cases}
$$

- **第一步，认清单一未知数**：整个假设只有一个待定指数 $\Delta$（连同 $2-\alpha$ 的幂）。其余一切由求导决定。
- **第二步，读出 $\beta$ 与 $\gamma$**：如上，一阶导给出 $\beta$，二阶导给出 $\gamma$。
- **第三步，消去 $\Delta$ 得 Rushbrooke**：$\alpha + 2\beta + \gamma = 2$。
- **第四步，利用临界等温线得 Widom**：$t=0$ 处 $h \sim m^\delta$ 要求 $\Delta = \beta\delta$，从而 $\gamma = \beta(\delta-1)$。

这样，**实验观测到的三个看似独立的指数，被压缩成两个独立自由度**（$\alpha$ 与 $\Delta$，或等价地 $\beta$ 与 $\gamma$）。这是标度假说最可检验的预言：测出任两个指数，其余所有指数都应能由关系式预言——而实验与精确解都一一证实。

## 4 关联函数标度：Fisher 与超标度关系

自由能齐次性只能约束**热力学**（静态）指数。关联长度、关联函数属于**空间**信息，需要另一条标度假设：**关联函数在临界点附近的标度形式**。

在 $T \neq T_c$、$r \gg a$ 处，假设

$$
G(\boldsymbol{r}, t) \approx \frac{1}{r^{d-2+\eta}} \, g\!\left(\frac{r}{\xi}\right)
$$

其中 $g(r/\xi)$ 在 $r \ll \xi$ 时趋于常数，在 $r \gg \xi$ 时指数衰减。<span class="marginnote">这个形式把「临界点幂次尾巴」$r^{-(d-2+\eta)}$ 与「有限关联长度截断」$e^{-r/\xi}$ 合二为一：只要 $\xi$ 有限，关联在远处就被切断；$\xi \to \infty$ 时只剩幂次。`cutoff` 与 `power law` 的并存，是所有临界关联函数标度形式的通用骨架。</span>

**磁化率是关联函数的空间积分**（涨落—耗散定理）：$\chi = \int d^d r\, G(\boldsymbol{r})$。把标度形式代入，积分主要来自 $r \lesssim \xi$ 区域，量纲分析给出 $\chi \sim \xi^{2-\eta}$。再代 $\xi \sim |t|^{-\nu}$、$\chi \sim |t|^{-\gamma}$，立即得

$$
\gamma = \nu(2-\eta)
$$

这就是 **Fisher 关系**。<span class="marginnote">注意这条关系的层次：它由「关联函数标度」＋「$\chi$ 是 $G$ 的积分」两条事实推出，属于<strong>空间</strong>标度，而不是自由能齐次性的纯代数推论。这提醒我们：临界现象至少需要两套标度信息——热力学量（自由能）与关联函数。</span>

**超标度（hyperscaling）关系 $2 - \alpha = d\nu$** 则把两者连起来。临界点附近，每个关联体积 $\xi^d$ 内的自由能奇异贡献约为 $k_B T_c$（一个热单位的量级），故自由能密度 $f_{\text{sing}} \sim \xi^{-d} \sim |t|^{d\nu}$；与 $f_{\text{sing}} \sim |t|^{2-\alpha}$ 比较：

$$
2 - \alpha = d\nu
$$

**辨析｜易错点：** 超标度关系在平均场指数上**不成立**：平均场给出 $2-\alpha = 2$、$d\nu = 3/2$（三维），两者矛盾。这并非标度假说错了，而是**平均场根本不在标度框架内**——因为 $d > d_c = 4$ 时涨落被抑制，超标度失效。所以超标度关系本身是一个「维数探测器」：它成立，说明系统处于临界涨落主导的标度区；它失效，说明你面对的是平均场区。

## 5 检验：精确解、实验与数值

标度假说不是空谈，它经受了最严格的检验：

| 检验对象 | 关系式 | 检验结果 |
| --- | --- | --- |
| 2D Ising 精确解 | $\alpha+2\beta+\gamma = 0+1/4+7/4 = 2$ | 精确成立 |
| 2D Ising 精确解 | $\gamma = \beta(\delta-1) = 14/8 = 7/4$ | 精确成立 |
| 2D Ising 精确解 | $\gamma = \nu(2-\eta) = 1\times 7/4$ | 精确成立 |
| 3D 实验（液气、铁磁） | 各标度关系 | 误差内成立 |
| 3D 数值（Ising、$n$-vector） | 标度关系 + 超标度 | 误差内成立 |

<span class="marginnote">标度假说后来被重整化群<strong>证明</strong>（而非假设）：Wilson 的重整化群从第一性原理导出了自由能的齐次形式，并把 $\Phi_\pm$、$\Delta$ 从「假设输入」升级为「理论输出」。这正是下一节的主题——标度关系是重整化群给我们的第一份红利。</span>

**重点是**：标度假说把「无穷多个临界指数」压缩成「两个独立指数」。整个临界现象不再是一堆孤立数字，而是一张由齐次性编织成的网——这张网在后来的重整化群中获得了动力学解释。

## 6 小结

- **Widom 标度假设**：$f_{\text{sing}}(t,h) = |t|^{2-\alpha}\Phi_\pm(h/|t|^\Delta)$，临界自由能是 $t$、$h$ 的齐次函数。
- 由假设推出**静态指数关系**：Rushbrooke $\alpha+2\beta+\gamma = 2$ 与 Widom $\gamma = \beta(\delta-1)$；缺口指数 $\Delta = \beta+\gamma = \beta\delta$。
- 由**关联函数标度**推出 Fisher 关系 $\gamma = \nu(2-\eta)$ 与**超标度** $2-\alpha = d\nu$。
- 标度关系在 Onsager 精确解与三维实验/数值中全部成立，是临界现象最坚实的经验规律之一。
- 超标度关系在 $d > d_c = 4$