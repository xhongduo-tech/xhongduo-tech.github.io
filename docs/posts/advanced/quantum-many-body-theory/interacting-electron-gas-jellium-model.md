---
title: 相互作用电子气（凝胶模型、RPA、屏蔽与等离激元）
date: 2026-08-07
---

# 相互作用电子气（凝胶模型、RPA、屏蔽与等离激元）

<div class="epigraph">
<p>一块金属几乎就是一个被正电荷背景泡着的电子海。看懂这片海，就看懂了固体的一半。</p>
<footer>—— 延斯 · 林哈德（Jens Lindhard）1954 年极化论文的当代注脚（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 5 ｜ 2026-08-07</p>
</div>

## 为什么从相互作用电子气开始

前面几篇把 Green 函数、Feynman 图、自能的工具箱都备齐了。现在把它们用在最朴素、却最深刻的真实系统上：**均匀电子气（uniform electron gas）**。<span class="marginnote"><strong>为什么是它</strong>：一块金属里 $10^{23}$ 个自由电子被均匀正电荷背景（离子实的 jelly）中和，这是第一个能完整走通「多体理论全过程」的系统。交换关联能、屏蔽、等离子体激元三个概念全从这里诞生。</span>这一篇我们以**凝胶模型（jellium model）**为舞台，把交换关联、RPA、屏蔽、等离激元四条线串起来——它们构成现代电子结构理论的原始模板。

## 1 凝胶模型

**凝胶模型（jellium model）**把离子实替换成均匀正电荷背景，电子在这个背景里靠库仑相互作用彼此耦合。哈密顿量为

$$\hat{H} = \sum_{\mathbf{k}\sigma}\frac{\hbar^2 k^2}{2m}c_{\mathbf{k}\sigma}^\dagger c_{\mathbf{k}\sigma} + \frac{1}{2V}\sum_{\mathbf{k}\mathbf{k}'\mathbf{q}\atop\sigma\sigma'}\frac{4\pi e^2}{q^2}\,c_{\mathbf{k}+\mathbf{q},\sigma}^\dagger c_{\mathbf{k}'-\mathbf{q},\sigma'}^\dagger c_{\mathbf{k}',\sigma'} c_{\mathbf{k},\sigma}$$

背景电荷保证系统电中性。系统由密度 $n$（或无量纲的 $r_s$，电子球平均半径与玻尔半径之比，$n^{-1}=4\pi r_s^3 a_B^3/3$）刻画：$r_s$ 小是稠密金属（如铝，$r_s\approx2$），$r_s$ 大是稀薄极限。费米动量由 $k_F=(3\pi^2n)^{1/3}$ 给出，费米能量 $\varepsilon_F=\hbar^2 k_F^2/2m$。

## 2 交换关联能：经典结果

无相互作用的基态是费米球，动能 $E_{kin}/N = 3\varepsilon_F/5$。加入库仑相互作用的**一阶修正**（Hartree-Fock 交换项）给出交换能：

$$\frac{E_x}{N} = -\frac{3}{4\pi}\,\frac{e^2 k_F}{\varepsilon_0}\cdot\frac{1}{r_s}\quad(\text{数量级上 } E_x \sim -\frac{0.916}{r_s}\,\mathrm{Ry})$$

交换能是负的——电子排斥彼此，交换空穴（exchange hole）让每个电子周围出现「少一个电子」的区域，平均拉近效应被抑制，能量下降。<span class="marginnote"><strong>交换空穴</strong>：费米子反对称性使同自旋电子不能靠得太近，每个电子周围的自旋平行空穴密度正好积分掉一个电子。交换能就是「自能修正的经典平均值」在均匀气体里的显式解。</span>

**关联能**（超越 HF 的全部修正）没有初等闭式，但有一个著名的不等式区间：高密度极限下 Wigner 与 Seitz 给出 $E_c \approx 0.0622\ln r_s - 0.094$（Ry），低密度极限趋向经典 Wigner 晶格。这些结果共同构成局域密度近似（LDA）的输入，是第一性原理计算的发源点。

## 3 RPA：动态介电函数

真正让电子气理论变得「多体」的是**随机相位近似（Random Phase Approximation, RPA）**。它把极化过程的无穷多环图求和，得到介电函数

$$\epsilon(\mathbf{q},\omega) = 1 - V_q\,\Pi_0(\mathbf{q},\omega)$$

其中 $\Pi_0$ 是**自由极化率（Lindhard 函数）**，对有限温度由第 4 篇的频率求和得到。RPA 的两个著名后果是**动态屏蔽**与**等离激元**，下面两节分别展开。<span class="marginnote"><strong>RPA 名字由来</strong>：对 $\Pi_0$ 里的两个传播子取相同相位（相位「随机平均」抵消），只剩密度-密度通道的无穷环求和。它之所以重要，是因为长波极限下库仑相互作用是强的、不能被有限阶微扰捕获——只有无穷求和才能屏蔽掉它。</span>

## 4 屏蔽：库仑势被电子海削弱

**屏蔽（screening）**是电子气最直观的集体效应：外来电荷被周围电子重新分布所部分抵消。静态极限 $\omega=0$ 下，介电函数在小 $q$ 时行为为

$$\epsilon(q,0) \approx 1 + \frac{\kappa^2}{q^2}$$

**托马斯-费米（Thomas-Fermi）屏蔽波矢** $\kappa^2 = 4\pi e^2 N(0)$，$N(0)$ 是费米面态密度。屏蔽后有效相互作用

$$V_{\mathrm{eff}}(q) = \frac{4\pi e^2}{q^2\epsilon(q,0)} \to \frac{4\pi e^2}{q^2+\kappa^2}$$

在实空间对应 Yukawa 势 $V_{\mathrm{eff}}(r)\sim e^{-\kappa r}/r$：库仑长程势被砍成短程。<span class="marginnote"><strong>动态效应</strong>：有限频率下屏蔽减弱，高频时电子来不及响应，相互作用恢复为裸库仑。这就是为什么 RPA 必须保留 $\omega$ 依赖——等离激元正是从「屏蔽跟不上高频」里长出来的集体模式。</span>

## 5 等离激元：集体密度振荡

介电函数 $\epsilon(\mathbf{q},\omega)=0$ 的零点给出纵向集体激发的色散。长波极限下这给出著名的**等离子体频率**：

$$\omega_p = \sqrt{\frac{4\pi n e^2}{m}}$$

对典型金属密度，$\omega_p\sim 10\,\mathrm{eV}$，是 X 射线/紫外区间的特征能量。这个模式就是**等离激元（plasmon）**：全体电子以同一相位作长波集体振荡，频率与波矢几乎无关（色散平坦），因此只能靠电子束损失谱等外源激发。<span class="marginnote"><strong>为何是集体模</strong>：单粒子激发的能量在 $\omega \sim v_F q$ 一线（电子-空穴对连续区），而等离激元在长波极限把能量抬高到 $\omega_p$，脱离单粒子连续区。它不依赖单个电子，是密度场的共振——多体理论「整体大于部分之和」的教科书例子。</span>

## 6 公式解析：RPA 介电函数如何给出等离激元

把两个结果连起来，看 RPA 的一根完整逻辑链：

$$
\epsilon(\mathbf{q},\omega) = 1 - \frac{4\pi e^2}{q^2}\sum_{\mathbf{k}}\frac{n_{\mathbf{k}}-n_{\mathbf{k}+\mathbf{q}}}{\omega - (\varepsilon_{\mathbf{k}+\mathbf{q}}-\varepsilon_{\mathbf{k}}) + i0^+}
$$

- **第一步，看自由极化率**：$\Pi_0$ 的分子是占据差 $n_\mathbf{k}-n_{\mathbf{k}+\mathbf{q}}$（只有跨费米面的跃迁才贡献），分母是激发能量。它在 $\omega=v_F q$ 附近有强结构，即电子-空穴对连续区。
- **第二步，取长波极限**：$q\to0$ 时按 $q^2$ 展开，介电函数分母出现 $\omega^2 - \omega_p^2$ 的结构，零点在 $\omega=\omega_p$。
- **第三步，读出模式**：$\epsilon=0$ 对应纵向集体振荡能量 $\omega_p$。对中等 $q$，色散为 $\omega^2 = \omega_p^2 + \frac{3}{5}v_F^2 q^2$，小幅向上弯——这是实验上可测的等离激元色散关系。

**重点：等离激元不是某一阶微扰的产物，而是 RPA 无穷求和后介电函数极点的体现。** 用有限阶微扰永远得不到它——这是「必须重求和不裸展开」的最有力论据。

## 7 具体例子：估算铝的等离激元能量

把理论落到具体数字上，才算真正掌握了。以金属铝为例，自由电子密度约为 $n \approx 1.8\times10^{23}\,\mathrm{cm}^{-3}$（每个铝原子贡献约 3 个价电子）。代入等离激元频率：

$$\omega_p = \sqrt{\frac{n e^2}{\varepsilon_0 m}} \approx \sqrt{\frac{(1.8\times10^{29}\,\mathrm{m}^{-3})(1.6\times10^{-19}\,\mathrm{C})^2}{(8.85\times10^{-12}\,\mathrm{F/m})(9.11\times10^{-31}\,\mathrm{kg})}} \approx 2.4\times10^{16}\,\mathrm{s}^{-1}$$

换算成能量 $\hbar\omega_p \approx 15\,\mathrm{eV}$。这个数字与实验（电子能量损失谱，EELS）测得的铝体等离激元约 15 eV 的峰一致。

把这个数字放进物理图景：

- **单粒子连续区**的能量尺度是 $v_F q$，在 $q\to0$ 时趋于零——电子-空穴对激发在长波下几乎不花钱。
- **等离激元**却稳定地坐在 $\sim 15$ eV 的高位，远离单粒子区，因此寿命长、容易被 EELS 观测。
- **对比费米能**：铝的费米能约 11.7 eV，等离激元能量与它同量级，说明集体模并非「小修正」，而是电子气里与单粒子自由度平起平坐的重要分支。

**重点：一个理论数字直接对上一个实验峰，这就是多体理论可证伪性的最佳演示。** 从纯参数（密度、电荷、质量）出发预言一个 15 eV 的集体模，不需要任何可调参数——RPA 的有效性在这里得到了最干净的确认。

## 8 小结

- **凝胶模型**用均匀正电背景 + 库仑相互作用电子气描述金属电子海，$r_s$ 标度密度。
- **交换能** $E_x\sim-0.916/r_s$ Ry 来自 HF 交换空穴，**关联能**无闭式但高/低密度极限已知。
- **RPA** 对无穷环图求和，介电函数 $\epsilon=1-V_q\Pi_0$，捕捉动态屏蔽。
- **托马斯-费米屏蔽**把库仑势变成 $e^{-\kappa r}/r$，长程变短程；$\kappa^2=4\pi e^2N(0)$。
- **等离激元**是 $\epsilon(\mathbf{q},\omega)=0$ 的集体振荡，$\omega_p=\sqrt{4\pi ne^2/m}$，色散平坦、脱离单粒子区。
- RPA 是 LDA/GW 等第一性原理近似的共同祖先，本专题由此通向真实材料计算。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 费米动量 | $k_F=(3\pi^2n)^{1/3}$ | 密度决定 |
| 交换能 | $E_x=-0.916/r_s$ Ry | 交换空穴 |
| RPA 介电 | $\epsilon=1-V_q\Pi_0$ | 无穷环求和 |
| TF 屏蔽 | $V_{\mathrm{eff}}=4\pi e^2/(q^2+\kappa^2)$ | 长程变短程 |
| 等离激元 | $\omega_p=\sqrt{4\pi ne^2/m}$ | 集体振荡 |
| 色散 | $\omega^2=\omega_p^2+\frac35 v_F^2q^2$ | 小 $q$ 略上弯 |

**易错复盘**：交换空穴积分贡献一个单位负电荷（自旋平行）；$\epsilon=0$ 是纵向模条件；RPA 只在长波小 $q$ 严格成立，$q$ 大到 Landau 阻尼区要小心。

**知识连线**：本篇把第 3–5 篇的 Green 函数/图论全部应用到真实系统，是第 8 篇《线性响应与输运》与第 9 篇《玻色系统》之间承上启下的一环；RPA 也是理解超导里库仑赝势、以及大模型里「集体模式 vs 单粒子噪声」类比的对象。

在下一节，我们将用微扰与自能的语言，重新审视金属中电子的量子态：费米液体理论——准粒子、朗道参数与非费米液体。
