---
title: 费曼图计算
date: 2026-08-07
---

# 费曼图计算

<div class="epigraph">
<p>场论计算的一半是画对图，另一半是把旋量代数干净利落地化成迹。</p>
<footer>—— 自 P&S 教学传统（为本文所作）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §5.1–5.2, §6.1 ｜ 2026-08-07</p>
</div>

## 为什么需要「会算」

费曼规则与截面公式都是「语法」，本节是第一次「写作」：把一套真实过程算到底。
典范是 **$e^+e^- \to \mu^+\mu^-$**（正负电子对撞产生正反缪子）——它是 QED 的「hello world」，结构简单却包含全部关键技巧：旋量链的书写、**自旋求和**、**迹运算（Casimir 技巧）**、相空间积分。
做完这一步，你就能独立算任何 QED 树级过程，也第一次看到「圈图」的引子。<span class="marginnote">选 $e^+e^-\to\mu^+\mu^-$ 的理由：初末态都是「无质量极限下好处理」的费米子、只有一张树级图（$s$ 通道光子）、结果简洁到能背下来：$\sigma = \frac{4\pi\alpha^2}{3s}$。P&S 把它作为第 5 章的第一道完整例题。</span>

## 1 从拉格朗日量到振幅

QED 拉格朗日量 $\mathcal{L} = \bar\psi(i\not\partial - m)\psi - \frac14F^2 + \text{int}$，相互作用 $=-e\bar\psi\gamma^\mu\psi A_\mu$。
对 $e^+(p_1)e^-(p_2)\to\mu^+(p_3)\mu^-(p_4)$，树级只有一张 $s$ 通道图：电子线从外到外、交换一个光子、接到缪子线上。用费曼规则直接写：

$$i\mathcal{M} = (-ie)^2\,\bar v(p_1)\gamma^\mu u(p_2)\cdot\frac{-ig_{\mu\nu}}{(p_1+p_2)^2}\cdot\bar u(p_4)\gamma^\nu v(p_3)$$

其中 $u, v, \bar u, \bar v$ 是旋量波函数，$-ie\gamma^\mu$ 是电子与缪子各一个顶点，$\frac{-ig_{\mu\nu}}{q^2}$ 是光子传播子（$q = p_1+p_2$）。<span class="marginnote">符号检查：$v(p_1)$ 是入射正电子的外腿（$\bar v$ 在入射正电子一侧），$\bar u(p_4)$ 是出射 $\mu^-$。费米子线方向：从正电子的「反时向」延续到缪子——写旋量链时沿箭头方向读矩阵乘积。</span>

这一步是「符号的书法」：四个旋量、一个 $\gamma^\mu$ 链、一根光子线，机械照规则即可。真正的手艺在下一步——把它平方并求和自旋。

## 2 自旋求和与 Casimir 技巧

实验不区分初末态的自旋极化，所以要**对所有自旋取平均/求和**：

$$\overline{|\mathcal{M}|^2} = \frac{1}{4}\sum_{s_1,s_2,s_3,s_4}|\mathcal{M}|^2$$

$\frac14$ 是对两个入射粒子自旋取平均。直接算四重自旋和是灾难，**Casimir 技巧**救场：把 $|\mathcal{M}|^2 = \mathcal{M}^*\mathcal{M}$ 的共轭用 $\gamma$ 矩阵性质反转，再交换求和顺序，把「旋量 × 旋量」配成**外积之和**：

$$\sum_s u^s(p)\bar u^s(p) = \not p + m, \qquad \sum_s v^s(p)\bar v^s(p) = \not p - m$$

于是自旋和变成**完整 $\gamma$ 矩阵链的迹**。对 $e^+e^-\to\mu^+\mu^-$，化简为两个迹的乘积：

$$\overline{|\mathcal{M}|^2} = \frac{e^4}{4 q^4}\, \text{Tr}\!\left[(\not p_1 - m_e)\gamma^\mu(\not p_2 + m_e)\gamma^\nu\right]\cdot \text{Tr}\!\left[(\not p_3 - m_\mu)\gamma^\nu(\not p_4 + m_\mu)\gamma^\mu\right]$$

两条迹各属一个费米子对，电子部分与缪子部分**完全解耦**——这就是 Casimir 技巧的威力：四重旋量求和塌缩成两条独立迹。<span class="marginnote">迹恒等式是唯一的「数学武器库」：$\text{Tr}(\text{奇个 }\gamma) = 0$、$\text{Tr}(\not a\not b) = 4a\cdot b$、$\text{Tr}(\not a\not b\not c\not d) = 4[(a\cdot b)(c\cdot d) - (a\cdot c)(b\cdot d) + (a\cdot d)(b\cdot c)]$。奇偶性规则让大量项自动消失。</span>

## 3 算到底：截面

忽略电子/缪子质量（高能极限），用迹恒等式展开、做洛伦兹缩并，得到：

$$\overline{|\mathcal{M}|^2} = \frac{8e^4}{q^4}\left[(p_1\cdot p_3)(p_2\cdot p_4) + (p_1\cdot p_4)(p_2\cdot p_3)\right]$$

在质心系里用角度参数化（$q^2 = s$），代入截面公式：

$$\frac{d\sigma}{d\Omega} = \frac{\alpha^2}{4s}\left(1 + \cos^2\theta\right), \qquad \alpha = \frac{e^2}{4\pi} \approx \frac{1}{137}$$

对立体角积分得到**总截面**：

$$\sigma(e^+e^- \to \mu^+\mu^-) = \frac{4\pi\alpha^2}{3s}$$

这个结果的三个亮点：$\sigma \propto 1/s$（高能下随质心能量平方衰减）；角分布 $1 + \cos^2\theta$（光子自旋 1 的签名——交换矢量玻色子的典型角分布）；数值与实验精确吻合，是 QED 的奠基性验证。<span class="marginnote">$1+\cos^2\theta$ 与 Rutherford 的 $1/\sin^4(\theta/2)$ 天差地别：前者来自「自旋 1 的 $s$ 通道交换」，后者来自「无自旋库仑散射」。<strong>测角分布能直接读出中间交换粒子的自旋</strong>——这是 1970 年代验证标准模型的手段。</span>

## 4 公式解析：Casimir 技巧

**Casimir 技巧把「旋量求和」换成「迹」，是整个费曼图计算的枢纽。** 拆解三步：

$$
\overline{|\mathcal{M}|^2} = \frac{1}{4}\sum_{\text{spins}}|\mathcal{M}|^2 = \frac{e^4}{4q^4}\,\text{Tr}[\cdots]\cdot\text{Tr}[\cdots]
$$

- **第一步，共轭的翻转**：$\mathcal{M}^*$ 里 $\gamma$ 矩阵变成 $\gamma^{0}\gamma^{\mu\dagger}\gamma^0 = \gamma^\mu$（对 $\gamma^\mu$ 厄米性），旋量 $\bar v\gamma^\mu u$ 的共轭是 $\bar u\gamma^\mu v$，但**乘积顺序要整体反转**——所以 $\mathcal{M}^*\mathcal{M}$ 变成「$\bar u(\cdots)\cdot u\,\bar u(\cdots)\cdot u$」的链，相邻旋量自然配成外积。
- **第二步，外积换成 $\not p \pm m$**：$\sum_s u^s\bar u^s = \not p + m$ 是投影到正能旋量子空间的完备性关系（P&S §3.3 的投影算符 $\Lambda_+$）。把四重求和里的相邻外积逐个替换，四重求和塌成迹。
- **第三步，迹的分裂**：电子链与缪子链之间只有光子传播子（纯数）相连，无 $\gamma$ 耦合，所以迹**因式分解**成两条独立迹的乘积。这是「电子与缪子各自独立演化、仅在光子处相遇」的结构在代数上的镜像。

## 5 辨析｜易错点

- **共轭顺序错误**：$(\bar v\gamma^\mu u)^* = \bar u\gamma^\mu v$，不是 $(\bar v\gamma^\mu u)$ 直接取共轭。整体链的顺序要反转，这是 Casimir 技巧最容易写错的地方。<span class="marginnote"><strong>外腿自旋平均</strong>：入射有 2 个粒子、各 2 个自旋，分母 $2\times2 = 4$；出射粒子不自旋平均（实验测得到它们）。「平均初态、求和末态」是铁律。</span>
**质量忽略的边界**：$m_e \to 0, m_\mu \to 0$ 的近似让迹恒等式大幅简化，但**低能时不能丢质量**——$m_\mu$ 会出现在阈值 $s \ge 4m_\mu^2$ 附近。做近似前先问「这个能量区间的质量项重不重要」。
**把 $\alpha$ 当常数**：$\alpha = 1/137$ 只在低能标度成立；高能下它「跑动」（第三章）。$e^+e^-$ 对撞能量越高，有效 $\alpha$ 越大——这是「跑动耦合常数」的伏笔。

## 6 延伸：更高阶过程的计算套路

$e^+e^-\to\mu^+\mu^-$ 是「开胃菜」，但它暴露了一套可复用的**套路**，写进任何 QED/QCD 树级计算：

1. **画图 + 定通道**：识别所有树级拓扑（$s/t/u$ 通道），每个顶点一个 $-ie\gamma^\mu$。
2. **写旋量链**：沿费米子线方向读矩阵乘积，外腿按「粒子/反粒子 + 入射/出射」放 $u,\bar u,v,\bar v$。
3. **共轭 + 自旋求和**：$\mathcal{M}^*\mathcal{M}$ 反转链序，用 $\sum u\bar u = \not p + m$ 化迹。
4. **迹恒等式化简**：奇个 $\gamma$ 的迹为零，偶个的用四迹公式，把四动量内积展开。
5. **代入运动学**：质心系、无质量极限等，化简成角度依赖。

这条流水线对 $e^-\mu^-\to e^-\mu^-$、$\mu$ 衰变、康普顿散射完全通用。**区别只在初末态组合与通道数量**，套路不变。

记住几个「量级感」：$\alpha \approx 1/137$ 让每个顶点压低约两个数量级；树级截面 $\sigma \propto \alpha^2/s$。这些数字让你在算之前就对手头结果有预期——对撞机物理「先猜后算」是常态。

### 自测清单

[ ] 能写出 $e^+e^-\to\mu^+\mu^-$ 的振幅与两条迹。
[ ] 能默写自旋求和公式 $\sum_s u\bar u = \not p + m$。
[ ] 能背出两条关键迹恒等式。
[ ] 能说出 $1+\cos^2\theta$ 角分布来自自旋 1 交换。

<span class="marginnote">套路的价值在于「知其然也知其所以然」：<strong>为什么化迹？因为自旋求和被完备性关系吸收；为什么因式分解？因为两个费米子对只在光子处耦合</strong>。</span>

### 延伸阅读指引

- 深化推导：P&S §5.1 的 Casimir 技巧、§5.2 的轨迹恒等式全集；想练手可做 §5 的全部习题。
- 实践：独立重算 $e^+e^-\to\mu^+\mu^-$ 全程并对照 P&S §5.1 的结果；再挑战 $e^-\mu^-\to e^-\mu^-$。
- 联系主线：「五步套路」是「可复用的计算流水线」——与《软件工程》里的「代码模板」同理：会背套路，才能自由变形。

## 7 小结

- $e^+e^-\to\mu^+\mu^-$：单张 $s$ 通道树图，振幅是两条旋量链 × 光子传播子。
- 自旋求和用完备性 $\sum_s u\bar u = \not p + m$，**Casimir 技巧**把它化成迹。
- 迹恒等式（奇个 $\gamma$ 迹为零等）让计算机械化。
- 高能极限截面 $\sigma = \frac{4\pi\alpha^2}{3s}$，角分布 $1+\cos^2\theta$（自旋 1 的签名）。
- 测量角分布能读出中间交换粒子的自旋。

在下一节，我们从树级进入**圈图**——第一次遇到动量积分发散。紫外发散如何出现、如何定义、又如何被重正化驯服，是第三章的全部内容。


