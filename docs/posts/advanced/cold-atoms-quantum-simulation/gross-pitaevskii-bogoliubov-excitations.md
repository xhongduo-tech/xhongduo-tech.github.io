---
title: Gross-Pitaevskii 方程与 Bogoliubov 元激发
date: 2026-08-11
---

# Gross-Pitaevskii 方程与 Bogoliubov 元激发

<div class="epigraph">
<p>数学语言在表述物理定律时的适当性，是一件我们既无法理解也不配得到的神秘礼物。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner），《数学在自然科学中不可理喻的有效性》，1960</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 冷原子与量子模拟 ｜ Pethick &amp; Smith《Bose-Einstein Condensation in Dilute Gases》第 4–6 章 ｜ 2026-08-11</p>
</div>

## 为什么从平均场开始

上一篇我们建立了 BEC 的统计图像：宏观数量的玻色子挤进同一个量子态。但「占据数」不是一幅运动图像——凝聚体在外势里怎么流动？涡旋怎么形成？两个凝聚体相撞会怎样？回答这些问题需要一个**演化的方程**。这一篇给出冷原子物理的工作母机：Gross-Pitaevskii（GP）方程，以及在它之上线性化得到的 Bogoliubov 元激发理论。前者是平均场（把 $N$ 体问题压成单粒子方程），后者是把「宏观波函数」当成经典背景、在上面添加量子涨落的起点——这两步合起来，构成了从「一堆原子」到「一个可计算的量子场」的完整桥梁。

## 1 序参量：宏观波函数

在 $T=0$ 且相互作用很弱时，几乎全部 $N$ 个原子占据同一轨道。我们把凝聚体的集体状态写成一个复函数：

$$\Psi(\mathbf r, t) = \sqrt{n(\mathbf r, t)}\,e^{i\phi(\mathbf r, t)}, \qquad \int |\Psi|^2\,d^3\mathbf r = N$$

**宏观波函数 / 凝聚体序参量（order parameter）**：$\Psi$ 的模平方是凝聚体密度，其相位 $\phi$ 给出全域一致的相干性。<span class="marginnote">相位是凝聚体的命根子：相位梯度 $\nabla\phi$ 直接决定超流速度 $\mathbf v_s = (\hbar/m)\nabla\phi$，涡旋、声子、干涉条纹全部由 $\phi$ 支配。相位一旦在宏观范围确定，系统就丧失了原来的 U(1) 对称性——这是一个自发的对称性破缺。</span>在凝聚体物理里，$\Psi$ 经常被直接称作「物质波的波函数」。与线性薛定谔方程里归一化为 1 的波函数不同，它归一化为 $N$，因此它承载的是「一个宏观物体的量子态」，而不是单粒子的概率幅。

## 2 从二次量子化到 Gross-Pitaevskii 方程

$N$ 体玻色子的哈密顿量中，两体相互作用在稀释极限下可以用接触势 $V(\mathbf r-\mathbf r') = g\,\delta(\mathbf r-\mathbf r')$ 近似，其中耦合常数

$$g = \frac{4\pi\hbar^2 a}{m}$$

完全由两个原子间的 **s 波散射长度（scattering length）$a$** 决定——这正是「稀释」的含义：平均间距远大于相互作用半径，一切相互作用细节都被压缩进单个参数 $a$。从二次量子化的场算符出发，把凝聚体占据数很大的场算符 $\hat\psi$ 近似为 c 数 $\Psi$（即 $\langle\hat\psi\rangle \approx \Psi$），就得到时间依赖的 Gross-Pitaevskii 方程：

$$i\hbar\frac{\partial\Psi}{\partial t} = \left(-\frac{\hbar^2}{2m}\nabla^2 + V_{\rm ext}(\mathbf r) + g|\Psi|^2\right)\Psi$$

<span class="marginnote">散射长度 $a$ 可正可负：$a>0$ 对应净排斥、$a<0$ 对应净吸引。负散射长度的凝聚体会自发坍缩（实验上在 $^{7}$Li、$^{85}$Rb 中都见过「Bose 新星」式的坍缩-爆炸）；如何用磁场实时调节 $a$，是后面《Feshbach 共振》一篇的主角。</span>这个方程长得像非线性薛定谔方程：除了动能和外势，多出一项 $g|\Psi|^2$，它表示每个原子感受到其余原子给它的平均排斥势——这正是「平均场」三个字的出处。

## 3 定态解与 Thomas-Fermi 近似

把 $\Psi(\mathbf r,t) = \psi(\mathbf r)e^{-i\mu t/\hbar}$ 代入 GP 方程，得到定态方程，$\mu$ 是化学势：

$$\left(-\frac{\hbar^2}{2m}\nabla^2 + V_{\rm ext}(\mathbf r) + g|\psi(\mathbf r)|^2\right)\psi(\mathbf r) = \mu\,\psi(\mathbf r)$$

当粒子数很大、排斥作用很强时，动能项可以忽略（Thomas-Fermi 近似），密度分布变成一条「倒扣的抛物线」：

$$n(\mathbf r) = |\psi(\mathbf r)|^2 = \frac{\mu - V_{\rm ext}(\mathbf r)}{g}$$

在谐波陷阱中这就是一个椭球；密度在边界处平滑地跌到零，过渡层的宽度由**愈合长度（healing length）**控制：

$$\xi = \frac{\hbar}{\sqrt{2mgn}} = \frac{1}{\sqrt{8\pi n a}}$$

对 $^{87}$Rb 典型参数（$n \approx 10^{14}\ \mathrm{cm}^{-3}$，$a \approx 5\ \mathrm{nm}$），$\xi \approx 0.2\text{–}0.3\ \mu\mathrm{m}$——它是凝聚体「自我修复密度不均匀」的特征长度，也是后面讨论涡旋核心尺寸、晶格标度时的基本尺度。<span class="marginnote">Thomas-Fermi 近似成立的条件是 $Na/a_{\rm ho} \gg 1$，其中 $a_{\rm ho} = \sqrt{\hbar/m\bar\omega}$ 是谐振子长度（对 $^{87}$Rb 典型约 1 $\mu$m）。稀释玻色气体条件 $na^3 \ll 1$ 与这个条件并不矛盾：前者说「势能远小于间距尺度」，后者说「粒子足够多、相互作用足够大以致动能可忽略」。</span>

## 4 Bogoliubov 元激发

把 GP 背景当成「真空」，在其上叠加小扰动 $\Psi = \Phi + \delta\psi$，把能量展开到 $\delta\psi$ 的二次项并用 Bogoliubov 变换（$u$、$v$ 系数）对角化，就得到凝聚体上的准粒子——元激发。均匀气体中元激发的能量-动量色散关系是 Bogoliubov 谱：

$$E_{\bf q} = \sqrt{\varepsilon_{\bf q}^2 + 2gn\,\varepsilon_{\bf q}}, \qquad \varepsilon_{\bf q} = \frac{\hbar^2 q^2}{2m}$$

**辨析｜易错点：** 长波极限（$q \to 0$）给出 $E_{\bf q} \approx \hbar c q$，其中声速 $c = \sqrt{gn/m}$——这是线性色散，元激发是**声子**，它们可以无耗散地传播，这正是超流的微观机制（Landau 判据：物体在液体中运动的速度小于声速 $c$ 就不会产生激发、没有摩擦）。短波极限（$q$ 大）则回到 $E_{\bf q} \approx \varepsilon_{\bf q} + gn$，即自由粒子色散加一个平均场平移。中间是平滑过渡。<span class="marginnote">Bogoliubov 变换与线性代数里「把二次型对角化」是同一件事的量子化版本：先对二次量子化哈密顿量做 $u,v$ 混合，再保证新算符满足玻色对易关系。它给出的「准粒子 = 声子」图像，是第二级统计物理里德拜模型、以及后面量子模拟里「量子涨落」的语言。</span>基态并非「所有原子都在 $\Psi$ 里」：量子涨落会把一小部分粒子踢出凝聚体，基态损耗 $N_{\rm ex}/N \propto \sqrt{na^3}$——对稀释气体这是一个很小的数（典型百分之几），所以平均场图像自洽。

## 5 公式解析：Gross-Pitaevskii 方程

把 GP 方程逐项拆开看：

$$i\hbar\frac{\partial\Psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\Psi + V_{\rm ext}(\mathbf r)\Psi + g|\Psi|^2\Psi$$

- **第一项，动能算符**：$-\frac{\hbar^2}{2m}\nabla^2$ 与线性薛定谔方程完全一致，它给物质波「铺展」的倾向。之所以是 $- \frac{\hbar^2}{2m}\nabla^2$ 而不是 $+\cdots$，来自动量算符 $\hat{\mathbf p} = -i\hbar\nabla$ 代入自由哈密顿量 $\hat p^2/2m$。
- **第二项，外势**：$V_{\rm ext}(\mathbf r)$ 是磁阱、光晶格或任何实验室陷阱的作用势，对单粒子有效，不随凝聚体状态改变。
- **第三项，平均场相互作用**：$g|\Psi|^2 = 4\pi\hbar^2 a\,n/m$，正比于局域密度 $n$ 与散射长度 $a$。它把 $N$ 体相互作用「投影」成每个粒子感受到的密度依赖势——非线性就藏在这里。
- **整体结构**：右边是哈密顿量作用在 $\Psi$ 上，左边是演化。因为是**非线性**项，GP 方程不满足叠加原理：$\Psi_1 + \Psi_2$ 不再是解。这是它与「普通」薛定谔方程最深刻的差别，也是「物质波场」与「概率幅」的又一个分界线。

注意 GP 方程里平均场势实际是 $2gn$ 而非 $gn$（每个原子既受 Hartree 直接项又受交换对称性修正），细节来自对称化波函数；这个「2 因子」是初学最容易漏的坑。

### 元激发的实验观测：把 Bogoliubov 谱「拍」下来

Bogoliubov 谱不只是纸面公式，几个关键实验把它逐点「拍」了下来：

- **布拉格谱学（Bragg spectroscopy）**：用两束大失谐激光在凝聚体上刻下一个可控的动量转移 $q$，扫描失谐测共振频率，就得到 $E_q$ 随 $q$ 的色散曲线——长波端的线性段直接读出声速 $c$；
- **声速测量**：$c = \sqrt{gn/m}$ 对 $^{87}$Rb 典型约 1–2 mm/s，实验测量与理论吻合到百分之几；
- **Landau 临界速度**：在凝聚体里拖动障碍物，只要速度低于声速就无阻力——这是「超流」的操作性定义，BEC 是第一个在宏观尺度验证它的中性原子系统。

| 观测量 | 对应公式 | 典型值（$^{87}$Rb） |
| --- | --- | --- |
| 声速 $c$ | $\sqrt{gn/m}$ | 约 1–2 mm/s |
| 愈合长度 $\xi$ | $1/\sqrt{8\pi na}$ | 约 0.2–0.3 $\mu$m |
| 基态损耗 $N_{\rm ex}/N$ | $\propto \sqrt{na^3}$ | 约 1%–5% |
| 化学势 $\mu$ | $gn$ | 约 20–50 nK·$k_B$ |

这些量同时是 BEC 实验的「体检指标」：一个实验室自报做出了 BEC，就得能测出这几个量落在合理区间内——这也让理论公式在冷原子里扮演了「标准尺」的角色。

### 平均场什么时候失效：迈向强关联

GP 方程与 Bogoliubov 理论是平均场的杰作，但它们的适用范围有清晰的边界。失效的标志是涨落不再小：

- **温度升高**：热声子占据数增多、序参量幅值涨落变大，GP 方程退化为「两流体」描述；
- **接近临界点**：$T \to T_c$ 附近临界涨落主导，平均场的临界指数失真（需要重标度分析，呼应第一级里相变临界行为的讨论）；
- **低维系统**：一维/二维里长波涨落破坏真长程序（Mermin-Wagner 定理），平均场在 $d \le 2$ 不能自洽；
- **强相互作用**：$na^3$ 不再小（如 Feshbach 共振附近），「散射长度编码一切细节」的前提失效。

| 失效方式 | 典型场景 | 替代工具 |
| --- | --- | --- |
| 温度效应 | $T \sim T_c$ | 有限温度 GP / 两流体 |
| 临界涨落 | 相变点附近 | 标度理论 / 蒙特卡洛 |
| 低维涨落 | 一维晶格 | Luttinger 液体 / 精确解 |
| 强关联 | $na^3 \sim 1$ | 量子蒙特卡洛 / 量子模拟 |

这最后一行把我们从 BEC 理论直接引向本专题的终点——量子模拟：当平均场失灵、经典计算又撞上符号问题，把问题「实物化」交给冷原子，就是最自然的出路。

## 6 小结

- 凝聚体由**宏观波函数** $\Psi = \sqrt{n}e^{i\phi}$ 描述，归一化为粒子数 $N$；相位 $\phi$ 编码超流速度。
- 稀释极限下相互作用全部由**散射长度 $a$** 编码，耦合常数 $g = 4\pi\hbar^2a/m$。
- **Gross-Pitaevskii 方程**是凝聚体的平均场运动方程；非线性项 $g|\Psi|^2$ 使其不满足叠加原理。
- Thomas-Fermi 近似给出倒扣抛物线的密度分布；**愈合长度** $\xi = 1/\sqrt{8\pi na}$ 是密度不均匀的特征尺度。
- Bogoliubov 谱 $E_q = \sqrt{\varepsilon_q^2 + 2gn\varepsilon_q}$：长波是声子（超流），短波回到自由粒子；基态有 $\propto\sqrt{na^3}$ 的量子损耗。

在下一节，我们把镜头拉回实验台：GP 方程描述的这么冷的原子，究竟是怎么造出来的？答案是激光——先讲 Doppler 与 Sisyphus 激光冷却。
