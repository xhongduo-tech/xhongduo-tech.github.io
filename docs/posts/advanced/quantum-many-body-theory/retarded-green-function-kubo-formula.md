---
title: 推迟Green函数与Kubo公式
date: 2026-08-07
---

# 推迟Green函数与Kubo公式

<div class="epigraph">
<p>测量一个系统，总会扰动它。Kubo 公式告诉我们：只要扰动足够小，系统对扰动的响应完全由它内部的涨落决定——这就是线性响应理论，也是「外场如何暴露内幕」的数学表达。</p>
<footer>—— R. Kubo（*Journal of the Physical Society of Japan\*, 1957）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 3 ｜ 2026-08-07</p>
</div>

## 为什么需要线性响应理论

Green 函数给我们的是一套「微观关联」的语言，但实验测的是「宏观响应」：加一个电场，测电导率；加一个磁场，测磁化率；加一个光场，测光学常数。宏观响应与微观关联之间，需要一座桥——**线性响应理论（linear response theory）**。

思路极其朴素：给体系加一个**弱外场**，比如电场 $\mathbf{E}$，把它看成对哈密顿量的微扰 $\delta H = -\mathbf{j}\cdot\mathbf{A}$（或 $-\mathbf{P}\cdot\mathbf{E}$）。在一阶微扰论内，外场引发的物理量变化正比于场强，比例系数就是响应函数。**Kubo 公式**的深刻之处在于：这些响应函数（电导率、磁化率、极化率）**全部可以写成推迟 Green 函数**——也就是体系内部「关联函数」的谱函数。<span class="marginnote">Kubo 公式的发现（1957）把输运系数从玻尔兹曼方程的经验框架里解放出来：不再需要假设「散射时间」等唯象量，一切从微观哈密顿量原则上可算。这使 Kubo 公式成为凝聚态理论从唯象走向第一性的分水岭。</span>

## 1 线性响应的一般结构

设体系哈密顿量 $H = H_0 + H_{\text{ext}}$，外场耦合到算符 $\hat{B}$：$H_{\text{ext}}(t) = -\hat{B}\,f(t)$，其中 $f(t)$ 是外场。要计算另一个算符 $\hat{A}$ 的期望值变化 $\langle\delta \hat{A}(t)\rangle$。在一阶微扰论下（用相互作用绘景，$U$ 展开到一阶）：

$$\langle\delta \hat{A}(t)\rangle = i\int_{-\infty}^{t} dt'\, \langle[\hat{A}_I(t), \hat{B}_I(t')]\rangle\, f(t')$$

定义**响应函数（response function）**为对易子的推迟平均：

$$\chi_{AB}(t-t') = -i\theta(t-t')\,\langle[\hat{A}(t), \hat{B}(t')]\rangle$$

于是 $\langle\delta\hat{A}(t)\rangle = \int_{-\infty}^{\infty} dt'\, \chi_{AB}(t-t')\, f(t')$——**响应 = 响应函数与驱动场的卷积**。这正是因果的线性系统理论：响应函数就是「脉冲响应」，其 Fourier 变换 $\chi_{AB}(\omega)$ 就是频率依赖的传递函数。<span class="marginnote">注意响应函数用的是<strong>对易子</strong>的平均（反对易子也可以，取决于算符统计），且必须是推迟形式（带 $\theta(t-t')$）——因果律是硬约束。这就是为什么推迟 Green 函数是线性响应理论的主角，而时序函数不是：时序函数在复平面上没有干净的因果边界条件。</span>

## 2 Kubo 公式：电导率 = 电流-电流关联

把一般框架用到电导。外电场 $\mathbf{E}$ 耦合到电流算符，哈密顿量微扰为 $\delta H = -\mathbf{j}\cdot\mathbf{A}$。电子气体系统对外电场的**电导率张量**由 Kubo 公式给出：

$$\sigma_{\alpha\beta}(\omega) = \frac{i}{\omega}\Big[\,i\int_0^\infty dt\, e^{i\omega t}\langle[j_\alpha(t), j_\beta(0)]\rangle\Big] = \frac{i}{\omega}\,\chi_{j_\alpha j_\beta}^R(\omega)$$

其中 $\chi_{j_\alpha j_\beta}^R(\omega)$ 是电流-电流推迟关联函数：

$$\chi_{j_\alpha j_\beta}^R(\omega) = \int_0^\infty dt\, e^{i\omega t}\, \big(-i\big)\,\langle [j_\alpha(t), j_\beta(0)]\rangle$$

**重点：Kubo 公式把宏观电导率还原成微观电流-电流关联函数。** 电流算符 $j_\alpha = \frac{e}{m}\sum_{\mathbf{k}} k_\alpha\, c^\dagger_{\mathbf{k}}c_{\mathbf{k}}$ 由二次量子化直接给出，关联函数用前面学的 Green 函数技术可算——于是电导率原则上从第一性原理出发完全确定。<span class="marginnote">$1/\omega$ 前的 $i$ 与 $i/\omega$ 因子是「导数」算符的 Fourier 表示：$j = \dot{P}$（电流是极化率的导数），所以 $\sigma = i\omega\chi_P$ 与 $\sigma = (i/\omega)\chi_{jj}$ 是同一条链的两端。</span>

## 3 涨落-耗散定理：内幕的另一面

Kubo 公式的姊妹定理是**涨落-耗散定理（fluctuation-dissipation theorem）**。它把「响应」（耗散）与「系统内部自发涨落」（关联）联系起来：

$$\text{Im}\,\chi_{AA}^R(\omega) = \pi\,\big(1 - e^{-\beta\hbar\omega}\big)\, S_{AA}(\omega)$$

其中 $S_{AA}(\omega) = \int_{-\infty}^{\infty} dt\, e^{i\omega t}\langle \hat{A}(t)\hat{A}(0)\rangle$ 是涨落谱密度（非对称关联函数）。**定理的物理内容**：系统在平衡态下的自发涨落（左端）决定了它对外场吸收能量的能力（右端）——「一个系统对扰动越敏感，它自身的涨落就越大」。

<span class="marginnote">一个重要的推论是<strong>零频磁化率（静态响应）与涨落的联系</strong>：$\chi_{AA}^R(0) = \beta\langle(\delta\hat{A})^2\rangle$。例：磁化率正比于磁矩涨落（居里定律），密度响应正比于粒子数涨落（压缩率）。这套「响应 ↔ 涨落」对偶是整个统计力学响应理论的统一核心。</span>

**辨析｜易错点：** 初学者常把响应函数里的对易子与关联函数 $\langle \hat{A}\hat{B}\rangle$ 混为一谈。区别很关键：对易子 $\langle[\hat{A},\hat{B}]\rangle$ 在经典极限（$\hbar\to0$）下对应泊松括号，反映**动力学的因果响应**；而关联函数 $\langle \hat{A}\hat{B}\rangle$ 反映**静态涨落**。两者由涨落-耗散定理联系，但绝不相等——这正是「响应」与「涨落」两个概念必须分开记的原因。

## 4 公式解析：推迟 Green 函数与谱函数

Kubo 公式里的一切最终都落到推迟 Green 函数 $G^R$。用单粒子推迟 Green 函数演示它与谱函数、占据数的自洽链条：

- **第一步，推迟函数定义**：$G^R(\mathbf{k},t) = -i\theta(t)\langle\{c_{\mathbf{k}}(t), c_{\mathbf{k}}^\dagger(0)\}\rangle$。它的虚部给出谱函数 $A(\mathbf{k},\omega) = -\frac{1}{\pi}\text{Im}\,G^R(\mathbf{k},\omega)$。
- **第二步，涨落-耗散联系占据数**：$\langle c_{\mathbf{k}}^\dagger c_{\mathbf{k}}\rangle = \int_{-\infty}^{\infty} d\omega\, A(\mathbf{k},\omega)\, n_F(\omega)$。谱函数既决定占据（统计），又决定响应（动力学）——一个函数承载双重信息。
- **第三步，电导的谱函数表达**：把电流-电流关联展开成单粒子谱的卷积：
  $$\sigma(\omega) \propto \int \frac{d^3k}{(2\pi)^3}\int d\omega'\, A(\mathbf{k},\omega')\,A(\mathbf{k},\omega'+\omega)\,\frac{n_F(\omega')-n_F(\omega'+\omega)}{\omega}$$
  这表示电导率 = 「粒子从动量态 $\mathbf{k}$ 跃迁到能量差 $\omega$ 的另一态」的速率之和——谱函数越宽、可跃迁的末态越多，电导越大。
- **第四步，读出物理**：在无散射的自由电子气极限，$A^{(0)}$ 是 δ 峰，上述积分给出熟悉的 Drude 结果 $\sigma = ne^2\tau/m$；相互作用把 δ 峰展宽成有限宽度 $\Gamma$，等价于散射时间 $\tau = \hbar/\Gamma$。**Kubo 公式自动包含了 Drude 公式，而不需要预设散射机制**——这是它超越唯象理论的地方。

**重点：推迟 Green 函数是「微观关联 → 宏观响应」的唯一转接站。** 学会把各种响应系数（电导、磁化率、极化率、热导）翻译成推迟关联函数，是多体理论连通实验的核心技能。

## 5 线性响应与「从极限到大模型」

线性响应理论的方法论在 AI 领域有直接的回响。**梯度就是「响应函数」**：损失函数对参数的敏感性 $-\partial L/\partial\theta$ 是体系对「参数扰动」的线性响应；**Fisher 信息矩阵**正是这个响应的二阶度量，等价于统计力学里的广义磁化率。<span class="marginnote">更精确的对应在<strong>隐式正则化与敏感性分析</strong>：一个训练好的大模型对输入的微小扰动（对抗样本）有多敏感，由输入-输出的 Jacobian 描述——这几乎是 $\chi_{AB}$ 的机器学习翻版。涨落-耗散式的思想（「对扰动敏感 ↔ 内部涨落大」）也解释了为什么表征越丰富的模型越容易被对抗样本扰动。</span>

对多体物理本身，Kubo 公式是通往无数具体理论的入口：光导纳谱、核磁共振自旋-晶格弛豫（$1/T_1$ 由自旋-自旋关联函数的谱决定）、量子霍尔电导、超流的第二声波……本专题后面谈超导、磁性、输运时，Kubo 公式会反复作为「从 Green 函数到实验」的最后一步出现。

## 6 小结

- **线性响应理论**：弱外场下一阶微扰，物理量变化 = 响应函数 × 外场（卷积）；响应函数是推迟对易子平均。
- **Kubo 公式**：电导率 $\sigma(\omega) = (i/\omega)\chi_{jj}^R(\omega)$，把宏观输运还原为电流-电流关联函数。
- **涨落-耗散定理**联系响应（耗散）与自发涨落（关联），是「敏感 ↔ 涨落」对偶的严格表述。
- 推迟 Green 函数是微观关联与宏观响应的**转接站**；谱函数同时承载占据（统计）与响应（动力学）双重信息。
- Drude 公式从 Kubo 公式自动涌现：散射时间 $\tau = \hbar/\Gamma$ 由谱函数展宽读出，无需预设散射机制。
- 响应函数用对易子而非关联函数，经典极限下对应泊松括号——「响应」与「涨落」必须分开记。

在下一节，我们将换一套语言重新推导整个多体理论：**相干态路径积分与配分函数**——用泛函积分把 Green 函数、配分函数与相互作用全部重写一遍，这套语言在后面的对称破缺、重正化群与强关联理论中将无可替代。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 响应函数 | $\chi_{AB}(t-t') = -i\theta(t-t')\langle[A(t),B(t')]\rangle$ | 推迟对易子，因果律内建 |
| 线性响应 | $\langle\delta A(t)\rangle = \int dt'\,\chi_{AB}(t-t')\,f(t')$ | 响应 = 响应函数与驱动场的卷积 |
| Kubo 电导 | $\sigma(\omega) = \frac{i}{\omega}\,\chi_{jj}^R(\omega)$ | 宏观电导 = 电流-电流关联 |
| 涨落-耗散 | $\text{Im}\,\chi^R(\omega) = \pi(1-e^{-\beta\hbar\omega})\,S(\omega)$ | 响应（耗散）与自发涨落（关联）对偶 |
| 谱函数 | $A(\mathbf{k},\omega) = -\frac{1}{\pi}\text{Im}\,G^R(\mathbf{k},\omega)$ | 推迟函数虚部给出谱，连接占据与响应 |

**易错复盘**：三处高频失误。其一，把响应函数中的对易子 $\langle[A,B]\rangle$ 当成关联函数 $\langle AB\rangle$——前者是动力学因果响应，后者是静态涨落，两者靠涨落-耗散定理联系但绝不相等；其二，忘记响应函数必须是推迟形式（带 $\theta(t-t')$）——因果律是硬约束，时序函数没有干净的因果边界；其三，把 Drude 公式当作独立假设——它其实是 Kubo 公式在「谱函数为 Lorentzian 展宽 $\Gamma=\hbar/\tau$」时的推论。

**知识连线**：Kubo 公式把第 2 篇的推迟 Green 函数与第 2 篇的输运理论（Boltzmann 方程）统一起来；涨落-耗散定理则是第 1 篇谱表示（$G^R$ 由 $A$ 决定）的「输运版」。统计力学里「响应 ↔ 涨落」的对偶，在机器学习里对应「模型对扰动的敏感性 ↔ 表征的多样性」。

**思考题**：为什么电导率里会出现 $1/\omega$ 因子？提示：电流是极化率的导数，$\sigma = i\omega\chi_P$。若谱函数是 δ 峰（自由电子、无散射），Kubo 公式会给出什么电导？提示：δ 峰意味着 $\tau\to\infty$，电导发散——需要有限展宽。


## 延伸思考：自查与学习路径

**自查清单**：学完本篇，你应该能不加参考地回答：

- 从推迟 Green 函数的定义出发，说明为什么它比时序函数更适合线性响应。
- 把 Kubo 电导公式拆成三步：电流-电流关联 → 推迟函数 → 谱函数表达，说出每步的物理。
- 用涨落-耗散定理的一句话版本解释「为什么越灵敏的系统噪声越大」。
- 推导 Drude 公式从 Kubo 公式涌现的条件：谱函数须为 Lorentzian，$\tau = \hbar/\Gamma$。

## 7 数值算例：从谱函数读出 Drude 电导

把 Kubo 公式用在最简单的模型上，看它如何自动给出 Drude 电导。设谱函数取 Lorentzian 形式：

$$
A(\mathbf{k},\omega) = \frac{1}{\pi}\frac{\Gamma}{(\omega - \varepsilon_{\mathbf{k}} + \mu)^2 + \Gamma^2}
$$

- **第一步，代进电导的谱表达**：$\sigma(\omega)\propto\int A(\mathbf{k},\omega')A(\mathbf{k},\omega'+\omega)\,[n_F(\omega')-n_F(\omega'+\omega)]/\omega$。取 $T=0$，$n_F(\omega')-n_F(\omega'+\omega)$ 只在宽度为 $\omega$ 的带内有贡献。
- **第二步，取长波与窄展宽极限**：$q\to0$、$\Gamma\to0$ 但 $\tau=\hbar/2\Gamma$ 保持有限。积分给出 Drude 电导 $\sigma(\omega)=ne^2\tau/m(1-i\omega\tau)$——零频极限 $\sigma=ne^2\tau/m$，正是 Drude。
- **第三步，读出物理**：Kubo 公式不需要「假设」Drude——只要谱函数有 Lorentzian 展宽，电导自动是 Drude 型。展宽来源（电子-声子、杂质、电子-电子）只进入 $\Gamma$ 的大小，不改变公式结构。

**重点：Kubo 公式是「脚手架」而非「假设」——它把电导率的全部复杂性压缩进谱函数，而谱函数正是 Green 函数理论要算的对象。** 谱函数越宽（散射越强），$\Gamma$ 越大、$\tau$ 越小、电导越低——「散射压低电导」这句话在 Kubo 语言里就是「展宽压低谱函数峰值」。