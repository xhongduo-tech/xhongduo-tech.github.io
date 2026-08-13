---
title: Z₄ 上的码：Galois 环与格理论联系
date: 2026-08-07
---

# Z₄ 上的码：Galois 环与格理论联系

<div class="epigraph">
<p>真实往往不在二元的对立里，而在四元的层次中。</p>
<footer>—— 现代数学格言</footer>
</div>

<div class="article-byline">
<p>第二级 · 编码理论（纠错编码） ｜ van Lint 第8章；Roth 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 $\mathbb{Z}_4$ 开始

上一节的结尾留了一个悬念：非线性 Kerdock 码「其实是线性码的映射像」——但映射到哪个线性码？答案震惊了整个编码理论界：**$\mathbb{Z}_4$（整数模 4）上的线性码**。1994 年，Hammons、Kumar、Calderbank、Sloane 与 Solé 证明了 Kerdock 码、Preparata 码这些「著名非线性码」全是 $\mathbb{Z}_4$ 线性码经 **Gray 映射**的像。一个十年的困惑（非线性为什么超过线性）瞬间被消解：**非线性是「看错了坐标」的幻觉，换个字母表就线性了**。

这件事的深远意义在于：$\mathbb{Z}_4$ 不是域（$2 \cdot 2 = 0$，有零因子），编码理论需要从「有限域」升级到「有限环」。环上的线性代数（模论）比域上的更微妙，但也更丰富。<span class="marginnote">在第二级《抽象代数》里，$\mathbb{Z}_4$ 是「非整环的交换环」的标准例子：$2$ 是零因子、不是域、也不够「干净」。编码理论一直依赖域的四则运算，$\mathbb{Z}_4$ 的登场迫使我们重新审视：哪些结论用到了「可除」，哪些只用到了「线性」。</span>

## 1 $\mathbb{Z}_4$ 与 Gray 映射

**$\mathbb{Z}_4$**：模 4 的剩余类 $\{0, 1, 2, 3\}$，加法模 4、乘法模 4。它与 $\mathbb{F}_2$ 最大的差别是 $2 \cdot 2 = 0$ 但 $2 \neq 0$——存在零因子，所以不是域，不能做除法。

**Gray 映射** $\phi : \mathbb{Z}_4 \to \mathbb{F}_2^2$：

$$0 \mapsto 00, \quad 1 \mapsto 01, \quad 2 \mapsto 11, \quad 3 \mapsto 10$$

即把 $a \in \mathbb{Z}_4$ 写成「最高位 = $a \bmod 2$，最低位 = 某种相位」。<span class="marginnote">Gray 映射的名字来自通信里的 Gray 编码：相邻数字只有一位不同（$01 \to 11$ 只有第 2 位变，$11 \to 10$ 只有第 1 位变）。这里的 $0,1,2,3$ 按 $00,01,11,10$ 排列，正是「循环顺序」上每步只翻一位的 Gray 码。</span>

**关键性质：Gray 映射是「距离保持」的，但不是线性的。** 定义 $\mathbb{Z}_4$ 上的 **Lee 重量**：

$$\mathrm{wt}_L(a) = \min\{a, 4-a\}, \qquad \text{即 } \mathrm{wt}_L(0)=0, \mathrm{wt}_L(1)=1, \mathrm{wt}_L(2)=2, \mathrm{wt}_L(3)=1$$

则 **Lee 距离 = Gray 像的 Hamming 距离**：$d_L(a, b) = d_H(\phi(a), \phi(b))$。例如 $1$ 与 $2$ 的 Lee 距离 $= |1-2|$ 取环上最短路径 $= 1$，Gray 像 $01$ 与 $11$ 的 Hamming 距离也是 1。

## 2 $\mathbb{Z}_4$ 线性码：把线性搬到环上

**$\mathbb{Z}_4$ 线性码**：$\mathbb{Z}_4^n$ 的一个**子模**（对 $\mathbb{Z}_4$ 线性组合封闭的子集），不要求是子空间（$\mathbb{Z}_4$ 没有域结构）。

这样的码可以用生成矩阵描述，但结构比域上复杂：它有「自由部分」（同构于 $\mathbb{Z}_4^a$）和「挠部分」（同构于 $\mathbb{Z}_2^b$ 或 $\mathbb{Z}_2^{2b}$），由**不变因子分解**刻画。<span class="marginnote">模论里「自由 + 挠」的分解（第二级《抽象代数》的 PIR/主理想环结构定理）在这里有了实际用途：$\mathbb{Z}_4$ 线性码的大小是 $4^a 2^b$，不再是 $q^k$ 这种纯指数形式。「维数」的概念被「类型」$(a, b)$ 取代。</span>

**重点：把 $\mathbb{Z}_4$ 线性码逐位 Gray 映射，得到二元码——通常是非线性的。** 这正是 Kerdock 码的现代定义：Kerdock 码 = $\mathbb{Z}_4$ 线性码的 Gray 像。二元世界里的「非线性奇迹」，在 $\mathbb{Z}_4$ 世界里是平淡无奇的「线性」——**非线性只是映射造成的投影失真**。

## 3 一个小例子：从 $\mathbb{Z}_4$ 码到 Gray 像

用 $\mathbb{Z}_4$ 上由单个生成元 $(1, 2)$ 张成的码走一遍全套机制。码字是 $a \cdot (1, 2)$（$a \in \mathbb{Z}_4$）的四个：

| $a$ | $\mathbb{Z}_4$ 码字 | Gray 像 |
| --- | --- | --- |
| 0 | $(0, 0)$ | `0000` |
| 1 | $(1, 2)$ | `0111` |
| 2 | $(2, 0)$ | `1100` |
| 3 | $(3, 2)$ | `1011` |

验证距离保持：$(0,0)$ 与 $(2,0)$ 的 Lee 距离 $= 2 + 0 = 2$；Gray 像 `0000` 与 `1100` 的 Hamming 距离 $= 2$。$(1,2)$ 与 $(3,2)$ 的 Lee 距离 $= 2$；`0111` 与 `1011` 的 Hamming 距离 $= 2$。**Lee = Hamming，逐一成立**。<span class="marginnote">注意这个例子的 Gray 像碰巧仍是线性的（四个向量是 $0000, 0111, 1100, 1011 = $ 以 $0111, 1100$ 张成的二维子空间）。要看到「非线性」的 Gray 像，需要更复杂的挠结构——最小的经典例子是长度 8 的 <strong>Octacode</strong>（$\mathbb{Z}_4$ 版本的扩展 Golay 码），它的 Gray 像是 $(8, 256, 6)$ 的二元码，<strong>不是</strong>线性子空间。非线性不是「随便就有」，而是「挠结构足够丰富才出现」。</span>

这个例子还演示了「类型」：四个码字 = $4^1 \cdot 2^0$，类型 $(1, 0)$（自由秩 1）。若再加一个独立生成元 $(2, 0)$（二阶），码就变成 8 个码字，类型 $(1, 1)$——「自由 + 挠」的分解在实例里看得清清楚楚。

## 4 Galois 环：把 $\mathbb{F}_{2^m}$ 升到环上

要造 $\mathbb{Z}_4$ 上的 BCH/RS 类码，需要像有限域那样的扩环。**Galois 环** $\mathrm{GR}(4, m)$ 定义为

$$\mathrm{GR}(4, m) = \mathbb{Z}_4[x] / (h(x))$$

其中 $h(x)$ 是 $\mathbb{Z}_4[x]$ 里「模 2 后为 $\mathbb{F}_2$ 上的 $m$ 次本原多项式」的某个首一提升。<span class="marginnote">例如 $m = 2$：$\mathbb{F}_2$ 上取 $x^2 + x + 1$，提升到 $\mathbb{Z}_4$ 可取 $h(x) = x^2 + x + 1$（系数 0/1 时提升唯一）。$\mathrm{GR}(4, 2) = \mathbb{Z}_4[x]/(x^2+x+1)$ 有 16 个元素。</span>

**结构定理：** $\mathrm{GR}(4, m)$ 有 $4^m$ 个元素，且唯一。它的乘法群 $\mathrm{GR}(4,m)^\times$ 形如「一个 $2^m-1$ 阶循环群 × 一个 $2^m$ 阶群」——比有限域 $\mathbb{F}_{2^m}$ 的「纯循环」多出一块挠部分。<span class="marginnote">有限域乘法群纯循环（第 2 篇）；Galois 环的乘法群多出一个 $2^m$ 阶「单位群内核」。这块挠结构正是 Kerdock/Preparata 码的丰富性的来源——它提供了「域里没有的自由度」。</span>Galois 环上可以定义 BCH 界、循环码理论、以及 RS 类的构造——整个有限域编码理论的框架在环上「重新上演」，只是每个结论都要重新检查「可除性用到哪一步」。

## 5 公式解析：为什么 Gray 映射「恰好」保持距离

Gray 映射不是线性的（$\phi(1) + \phi(1) = 00 + 00 \ne \phi(2) = 11$ 等等），却能保持距离。拆开看这并不矛盾。

**第一步，Lee 重量的几何**：$\mathbb{Z}_4$ 的四个元素按循环排成「环」：$0 \to 1 \to 2 \to 3 \to 0$。Lee 距离是「环上最短路」，$0$ 到 $2$ 的两条路长都是 2，$0$ 到 $1$、$0$ 到 $3$ 都是 1。
**第二步，Gray 映射把环展开成方阵**：$00, 01, 11, 10$ 按顺序排，相邻差 1 位，首尾 $10$ 与 $00$ 也差 1 位——Gray 序列本身是「循环」的，$2^n$ 个二进制串首尾相连，每步翻一位。
- **第三步，距离一致**：环上最短路 = 序列上步数 = Hamming 距离。因为 Gray 码的「相邻翻一位」性质，环上距离 $k$ 的元素，Gray 像恰好 Hamming 距离 $k$。

**直觉：** Gray 映射是一个「等距嵌入」——它把环 $\mathbb{Z}_4$ 等距地装进超立方体 $\mathbb{F}_2^2$，保持距离结构。**线性被牺牲，距离被保存**——而对纠错码来说，距离才是命根子，线性只是手段。这就是「$\mathbb{Z}_4$ 上的线性 + Gray 的等距 = 二元世界的非线性好码」的全部秘密。<span class="marginnote">术语对照：域上的 Hamming 重量在环上对应 Lee 重量；$\mathbb{Z}_4$ 线性码的 Lee 距离 = Gray 像的 Hamming 距离。设计 $\mathbb{Z}_4$ 码保证 Lee 距离，就等于设计二元码保证 Hamming 距离——「距离设计」完全不受「线性丢失」影响。</span>

## 6 从 $\mathbb{Z}_4$ 码到格：Construction A 与 Leech 格

$\mathbb{Z}_4$ 码的另一个惊人去向是**格（lattice）理论**。格是 $\mathbb{R}^n$ 里的离散加法子群，是最密的球堆积、也是模形式与数论的核心对象。

**Construction A（$\mathbb{Z}_4$ 版本）**：给 $\mathbb{Z}_4$ 线性码 $\mathcal{C} \subseteq \mathbb{Z}_4^n$，造一个 $\mathbb{R}^n$ 里的格

$$\Lambda(\mathcal{C}) = \frac{1}{2}\{\boldsymbol{c} + 4\boldsymbol{z} : \boldsymbol{c} \in \mathcal{C}, \boldsymbol{z} \in \mathbb{Z}^n\}$$

即「码字除 2 + 整数点阵」的并。格的最小范数平方由码的 Lee 距离决定。<span class="marginnote">经典对应：把 $\mathbb{F}_2$ 上的 Golay 码 $\overline{G}_{24}$ 用「二元 Construction A」（$\Lambda = \{\boldsymbol{c} + 2\boldsymbol{z}\}$）造格，得到的正是 <strong>Leech 格</strong> $\Lambda_{24}$——24 维里最密的球堆积，也是「魔群月光」猜想的主角。$\mathbb{Z}_4$ 版本让这个构造更精细：$\mathbb{Z}_4$ 码的挠结构对应格的「不同深度」。</span>

一个具体对应让人印象深刻：**Octacode（$\mathbb{Z}_4$ 上的长度 8 码）经 Construction A 造出的格是 $E_8$**——8 维里最密的球堆积，与 $\mathbb{F}_2$ 上扩展 Hamming 码 $[8,4,4]$ 经二元 Construction A 造出的格是同一个 $E_8$。两条完全不同的路（$\mathbb{Z}_4$ 与 $\mathbb{F}_2$）通向同一个格，说明「码 → 格」的构造有深刻的普遍性，而 $\mathbb{Z}_4$ 只是其中更精细的一条。

**格 ↔ 码的翻译表**：

| 格理论 | $\mathbb{Z}_4$ 码理论 |
| --- | --- |
| 格 $\Lambda$ | 码 $\mathcal{C}$ |
| 最小范数平方 | 最小 Lee 距离 |
| 对偶格 $\Lambda^*$ | 对偶码 $\mathcal{C}^\perp$（某些定义下） |
| 模形式/θ 函数 | 重量枚举器 |
| 格堆积密度 | 码的纠错能力 |

这不仅是类比：**模形式理论与 MacWilliams 恒等式在深处是同一条数学**（第 5 篇的 Krawtchouk 多项式与模形式的 Hecke 算子共享谱论结构）。$\mathbb{Z}_4$ 码站在格理论与编码理论的交汇点，是「一个对象、两套语言」的活标本。<span class="marginnote">如果走得更远：Leech 格在弦论里对应「24 维格点上的紧化」，而 Golay 码 $\to$ Leech 格 $\to$ Mathieu 群 $M_{24}$ 这条链，把编码理论、格论、群论、数论串成一条完整的珠链——「$\mathbb{Z}_4$」在其中是承上启下的关键一环。</span>

## 7 小结

- $\mathbb{Z}_4$ 有零因子（$2\cdot 2 = 0$），不是域——环上编码用「模论」而非「向量空间」。
- **Gray 映射** $0,1,2,3 \mapsto 00,01,11,10$：非线性但**等距**（Lee 距离 = Hamming 距离）。
- **$\mathbb{Z}_4$ 线性码**：子模，类型 $(a,b)$；Gray 像是二元码，通常是**非线性**的。
- Kerdock/Preparata 等「非线性奇迹」= $\mathbb{Z}_4$ 线性码的 Gray 像——非线性是映射投影的假象。
- **Galois 环** $\mathrm{GR}(4,m) = \mathbb{Z}_4[x]/(h(x))$：$4^m$ 个元素，乘法群「循环 × 挠」，支撑环上 BCH/RS 理论。
- **Construction A** 把 $\mathbb{Z}_4$ 码变成格：Golay → Leech 格，格论与模形式同编码理论在此合流。
- 实例：$(1,2)$ 张成的 $\mathbb{Z}_4$ 码类型 $(1,0)$，Gray 像恰为线性；非线性需要更丰富的挠结构（Octacode 是最小例子）。
- Octacode 与 $[8,4,4]$ 码经不同 Construction A 都造出 $E_8$ 格——码到格的构造有普遍性。
- 记住主线：环上线性 + Gray 等距 = 二元世界的好码；非线性只是「看错坐标系」的投影假象。
- Galois 环 $\mathrm{GR}(4,m)$