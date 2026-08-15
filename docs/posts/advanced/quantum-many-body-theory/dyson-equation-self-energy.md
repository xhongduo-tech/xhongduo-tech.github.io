---
title: Dyson方程与自能
date: 2026-08-07
---

# Dyson方程与自能

<div class="epigraph">
<p>Dyson 方程的美妙在于它用一个自洽的方程取代了无穷级数：你不再需要逐项画出所有高阶图，只需把「基本自能」代进一条方程，级数就自动求和完毕。</p>
<footer>—— F. J. Dyson（1950s 经典工作）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么需要 Dyson 方程

上一节我们看到，微扰级数的每一项是一个 Feynman 图。但真正的物理 Green 函数要求**把所有阶的图全部加起来**——这是个无穷级数，不能靠逐项画图来完成。好消息是，这个级数有非常强的结构：它几乎就是一条**几何级数**。

Dyson 方程（Dyson equation）把「无穷阶图之和」坍缩成一条**代数自洽方程**。它不只是一个求和技巧——它揭示了一个深刻的物理图景：**多体相互作用的一切效应，都被打包进一个对象——自能 $\Sigma$——而 Green 函数只是自能驱动下的一条「重正化传播子」**。这条方程，加上前面学的谱表示，构成了整个零温多体理论的运算闭环。<span class="marginnote">Dyson 1951 年提出的原始方程针对 QED 的电子传播子；Feynman、Schwinger 等人在同一时期独立得到等价结果。在凝聚态多体理论里，这条方程用于所有准粒子（电子、声子、磁振子）的重正化。</span>

## 1 无穷级数的几何结构

回忆上一节的图分类：传播子 $G(\mathbf{k})$ 的所有修正，可看作「不可约自能块 $\Sigma$」沿一根线串接的组合。把第零阶（无自能）记为 $G^{(0)}$，一阶记为 $G^{(0)}\Sigma G^{(0)}$，二阶为 $G^{(0)}\Sigma G^{(0)}\Sigma G^{(0)}$，依次类推：

$$G = G^{(0)} + G^{(0)}\Sigma G^{(0)} + G^{(0)}\Sigma G^{(0)}\Sigma G^{(0)} + \cdots$$

如果把这个级数看成「几何级数 $1 + x + x^2 + \cdots$」的形式，其中 $x = G^{(0)}\Sigma$，那么形式上可以求和：

$$G = G^{(0)} + G^{(0)}\Sigma\,G \quad\Longleftrightarrow\quad G = \frac{G^{(0)}}{1 - G^{(0)}\Sigma}$$

**重点：Dyson 方程** $G = G^{(0)} + G^{(0)}\Sigma\,G$ 是自洽的——右边的 $G$ 又出现在求和里。它是一条「图式不动点方程」：把全 Green 函数看作「自由传播子 + 一次自能修正后的全传播子」。几何级数求和在这里是合法的，因为每个图仅被算一次（不可约分解保证无重复计数）。<span class="marginnote">这与费曼路径积分里「把无限个散射串起来」是同一个结构。在数学上，几何级数求和要保证收敛（或至少是渐近级数）；物理上，我们通常不纠结收敛性，而把 Dyson 方程当作自洽定义——重正化场论的做法就是反过来用方程定义 $\Sigma$。</span>

## 2 Dyson 方程的两种写法

Dyson 方程在动量-频率空间有等价的两种代数形式：

**求和形式**：
$$G(\mathbf{k},\omega) = G^{(0)}(\mathbf{k},\omega) + G^{(0)}(\mathbf{k},\omega)\,\Sigma(\mathbf{k},\omega)\,G(\mathbf{k},\omega)$$

**逆形式**（更常用）：
$$G^{-1}(\mathbf{k},\omega) = G^{(0)-1}(\mathbf{k},\omega) - \Sigma(\mathbf{k},\omega)$$

把自由传播子 $G^{(0)-1} = \omega - \varepsilon_{\mathbf{k}} + i\eta$ 代入逆形式，得到**重正化传播子的显式解**：

$$G(\mathbf{k},\omega) = \frac{1}{\omega - \varepsilon_{\mathbf{k}} - \Sigma(\mathbf{k},\omega)}$$

**重点：自能 $\Sigma(\mathbf{k},\omega)$ 在 Dyson 方程里以「分母里被减去的项」出现。** 它把自由电子的能量 $\varepsilon_{\mathbf{k}}$ 修正为 $\varepsilon_{\mathbf{k}} + \text{Re}\,\Sigma$（能量重正化），并给谱函数一个有限宽度 $-\text{Im}\,\Sigma$（寿命修正）。自能因此是「多体效应如何改变单粒子」的唯一入口。<span class="marginnote">注意 $\Sigma$ 本身依赖 $\omega$：如果忽略这个依赖（把 $\Sigma$ 当常数），Green 函数只有一个极点，谱函数仍是 δ 峰——只是峰位移了；若保留 $\omega$ 依赖并取虚部，峰才展宽。频率依赖的自能是费米液体理论的关键细节。</span>

## 3 谱函数与准粒子

把 Dyson 解代进谱函数 $A(\mathbf{k},\omega) = -\text{Im}\,G^R/\pi$，可以得到带相互作用时的谱函数结构。设自能可展开为（在准粒子极点附近）：

$$\Sigma(\mathbf{k},\omega) \approx \Sigma(\mathbf{k},\varepsilon_{\mathbf{k}}) + (\omega - \varepsilon_{\mathbf{k}})\frac{\partial \Sigma}{\partial \omega}\Big|_{\omega=\varepsilon_{\mathbf{k}}} + i\,\text{Im}\,\Sigma(\mathbf{k},\omega)$$

定义**重正化因子** $Z_{\mathbf{k}} = \big[1 - \partial\Sigma/\partial\omega\big]^{-1}$，谱函数在准粒子极点附近呈 Lorentzian：

$$A(\mathbf{k},\omega) \approx Z_{\mathbf{k}}\,\frac{\Gamma_{\mathbf{k}}/2\pi}{(\omega - E_{\mathbf{k}})^2 + (\Gamma_{\mathbf{k}}/2)^2} + A_{\text{inc}}(\mathbf{k},\omega)$$

其中 $E_{\mathbf{k}} = \varepsilon_{\mathbf{k}} + \text{Re}\,\Sigma$ 是**准粒子能量**，$\Gamma_{\mathbf{k}} = -2Z_{\mathbf{k}}\text{Im}\,\Sigma$ 是**衰变率**（寿命 $\tau = 1/\Gamma$），$A_{\text{inc}}$ 是平滑的本底（incoherent background）。

**重点：准粒子（quasiparticle）的概念由此诞生。** 相互作用后的体系里，单粒子激发不再是 δ 峰，而是一根有宽度、有位移、有权重的 Lorentzian 峰。当 $\Gamma \ll E$（峰比能量窄得多）时，这个峰仍像一个「近似粒子」——能量 $E_{\mathbf{k}}$、寿命 $\tau$、谱权重 $Z_{\mathbf{k}}$——这就是 Landau 费米液体理论的基本图像（本专题第 2 篇详述）。<span class="marginnote">$Z_{\mathbf{k}}$ 是「裸粒子有多少成分存活在准粒子中」的度量，满足 $0 \le Z \le 1$。费米面处 $Z$ 降到 1 以下意味着动量分布的跳跃变小；若 $Z \to 0$，准粒子消失，体系进入强关联区——这正是 Mott 转变（本专题第 4 篇）的判据之一。</span>

## 4 公式解析：从 Dyson 方程读出准粒子

用自由电子气 + 一个**近似的、频率无关的自能** $\Sigma_0$ 演示全程：

**第一步，写 Dyson 方程逆形式**：$G^{-1} = \omega - \varepsilon_{\mathbf{k}} - \Sigma_0$，故 $G(\mathbf{k},\omega) = \frac{1}{\omega - \varepsilon_{\mathbf{k}} - \Sigma_0}$。
**第二步，找极点**：分母为零处 $\omega_* = \varepsilon_{\mathbf{k}} + \Sigma_0$。这就是重正化能量：自能的实部把准粒子能量整体抬升/压低，正如电子气的交换能 $\Sigma_{F}$ 压低能量一样。
- **第三步，加虚部看寿命**：若 $\Sigma = \Sigma_0 + i\Gamma_0$（$\Gamma_0\lt 0$），极点下移到 $\omega_* = \varepsilon_{\mathbf{k}}+\Sigma_0 + i\Gamma_0$，谱函数变成 Lorentzian，峰宽 $2|\Gamma_0|$，寿命 $\tau = \hbar/|\Gamma_0|$——粒子在有限时间内衰变。
- **第四步，读出求和规则**：把 $A(\mathbf{k},\omega)$ 对 $\omega$ 积分仍为 1，但其中 $Z = [1-\partial\Sigma/\partial\omega]^{-1}$ 的部分是峰，其余部分进入非相干本底——能量守恒依然严格，只是谱权重在「峰」与「本底」之间重新分配。

**辨析｜易错点：** 初学者常见两个误读。其一，**把 $\Sigma$ 当成「能量修正量」就够**——错了，$\Sigma$ 是频率依赖的复函数，它的虚部给出寿命，两者缺一不可；其二，**认为准粒子图像对任何相互作用都成立**——错了，当 $\text{Im}\,\Sigma$ 大到与 $\omega$ 同量级（强关联、近 Mott 绝缘体），Lorentzian 峰展宽到无法辨认，「准粒子」概念失效，需要全新的语言（自旋液体、分数化等，见本专题第 4 篇）。

## 5 Dyson 方程与「模型降维」

Dyson 方程是「从极限到大模型」主线的绝佳隐喻：**面对一个无法直接求解的无穷维问题，先找到它的「基本自能」，再用一条自洽方程把无穷级数封闭**。这正是一维问题的核心策略——把「算不尽的级数」变成「解一条方程」。<span class="marginnote">在大模型的世界里，类似的「封闭」无处不在：不动点迭代、共轭梯度、EM 算法、自注意力里的「层内几何级数」，乃至强化学习里的 Bellman 方程——全部是「自洽方程代替穷举」的实例。Dyson 方程是多体物理给这个通用策略起的名字。</span>

同时，Dyson 方程还点明了一条重要的认识论原则：**「基本单元」（自能）不是天然的，而是我们选择的**。不同的问题选不同的不可约块（Hartree-Fock、GW、RPA），就得到不同的近似理论——多体理论的大部分功力，都在于选择正确的自能近似。

## 6 小结

- **Dyson 方程** $G = G^{(0)} + G^{(0)}\Sigma G$ 把无穷阶图级数坍缩成一条自洽代数方程；逆形式 $G^{-1} = G^{(0)-1} - \Sigma$ 更常用。
- 解为 $G(\mathbf{k},\omega) = 1/(\omega-\varepsilon_{\mathbf{k}}-\Sigma(\mathbf{k},\omega))$：自能实部重正化能量，虚部给出寿命。
- **准粒子**是相互作用后的 Lorentzian 峰：能量 $E_{\mathbf{k}}$、寿命 $1/\Gamma$、权重 $Z_{\mathbf{k}}$；当峰足够窄时仍是好近似。
- 重正化因子 $Z = [1-\partial\Sigma/\partial\omega]^{-1}$ 度量准粒子的存活比例，$Z\to0$ 标志强关联区。
- 谱权重在峰与本底间重分配，但求和规则 $\int d\omega\, A=1$ 始终成立。
- 自能是「我们选择的基本单元」，不同选择给出不同近似理论。

在下一节，我们将把 Green 函数方法推广到有限温度：**有限温度 Green 函数与 Matsubara 求和**——虚时（imaginary time）技巧如何让热平衡多体问题同样享受 Wick 定理与 Dyson 方程的全部便利。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| Dyson 方程 | $G = G^{(0)} + G^{(0)}\Sigma\,G$ | 无穷阶图级数坍缩成自洽方程 |
| 逆形式 | $G^{-1} = G^{(0)-1} - \Sigma$ | 自能以「被减项」进入分母 |
| 重正化传播子 | $G(\mathbf{k},\omega) = \frac{1}{\omega - \varepsilon_{\mathbf{k}} - \Sigma(\mathbf{k},\omega)}$ | 实部重正化能量，虚部给寿命 |
| 重正化因子 | $Z = [1-\partial\Sigma/\partial\omega]^{-1}$ | 准粒子谱权重，$Z\to0$ 标志强关联 |
| 准粒子谱 | $A(\mathbf{k},\omega) \approx Z_{\mathbf{k}}\frac{\Gamma/2\pi}{(\omega-E_{\mathbf{k}})^2+(\Gamma/2)^2}$ | Lorentzian 峰：峰位、峰宽、权重 |

**易错复盘**：两点要盯住。其一，$\Sigma$ 是频率依赖的复函数——只取实部会丢掉寿命，只当常数会丢掉 $Z$；其二，准粒子图像有适用范围——当 $\text{Im}\Sigma$ 大到与 $\omega$ 同量级（强关联、近 Mott 绝缘体），Lorentzian 峰展宽到无法辨认，准粒子概念失效。

**知识连线**：Dyson 方程把第 1 篇的微扰图级数封闭成自洽方程；它服务的谱函数与占据数回接第 1 篇《单粒子 Green 函数与谱表示》。「几何级数求和代替无穷展开」的策略，在第 2 篇 RPA（$\chi=\chi_0/(1-V\chi_0)$）与第 3 篇 Bogoliubov 理论里以同一模式反复出现。

**延伸思考**：若 $\Sigma$ 与频率无关，Green 函数只有一个极点，谱函数仍是 δ 峰——为什么费米液体的准粒子峰必须依赖 $\partial\Sigma/\partial\omega$？提示：没有频率依赖就没有 $Z\lt 1$，也就没有谱权重从峰到本底的转移。自能的虚部为什么必须是负的（$\text{Im}\Sigma\lt 0$）？提示：推迟函数的谱函数正定性要求。


**实践与辨析**：一道综合题：已知自能 $\Sigma(\mathbf{k},\omega) = a + b\omega + i\Gamma_0$（$a,b,\Gamma_0$ 为常数），求谱函数的峰位、峰宽与重正化因子。提示：先解 Dyson 方程得 $G$，再读 $Z=[1-b]^{-1}$、$E_{\mathbf{k}}=\varepsilon_{\mathbf{k}}+a$、$\Gamma = -2Z\Gamma_0$。这道题浓缩了本章全部核心：Dyson 方程 → 谱函数 → 准粒子三要素。易错提醒：不要忘了 $\omega$ 依赖的自能才会压低 $Z$，常数自能只能移动峰位。

**延伸阅读**：Dyson 方程与自能的系统论述见 Mahan《Many-Particle Physics》第 2 章；准粒子概念的物理内涵见文小刚《量子多体理论》第 5 章费米液体部分。


**延伸阅读**：Dyson 方程与自能见 Mahan《Many-Particle Physics》第 2 章；准粒子概念的物理内涵见文小刚《量子多体理论》第 5 章；Green 函数微扰技术的系统推导见 Negele & Orland《Quantum Many-Particle Systems》。


**关键术语**：自能 $\Sigma$、Dyson 方程、重正化因子 $Z$、准粒子、非相干本底 $A_{\text{inc}}$、不可约自能块、几何级数求和。

**小结一句话**：多体相互作用的一切单粒子效应都收进自能 $\Sigma$，Dyson 方程把无穷阶图级数封闭成一条方程 $G=G^{(0)}+G^{(0)}\Sigma G$——准粒子的峰位、寿命与权重，都由这一条方程读出；而「选对不可约块」正是多体理论最需要功力的地方。