---
title: 三比特相位翻转码（phase-flip code）
date: 2026-08-07
---

# 三比特相位翻转码（phase-flip code）

<div class="epigraph">
<p>把相位翻转看成比特翻转，只需要换一个基。</p>
<footer>—— 尼尔森（Michael Nielsen）与庄（Isaac Chuang）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§10.2 ｜ 2026-08-07</p>
</div>

## 为什么从相位翻转码开始

上一节的比特翻转码只能处理 $X$ 错误。但量子世界里还有一类同样基本的错误——**相位翻转** $Z$：它把 $\lvert+\rangle$ 翻成 $\lvert-\rangle$（即把 $\lvert1\rangle$ 的相位加 $\pi$），对计算基 $\lvert0\rangle$、$\lvert1\rangle$ 却「看不见」。相位翻转码正是为它而生的三比特码。<span class="marginnote">相位翻转码的价值远不止「又一种码」：它揭示了量子纠错里一条深刻的对称性——<strong>在 Hadamard 基下，相位翻转就是比特翻转</strong>。这条「$H$ 共轭」关系是后续 CSS 码、Shor 码设计的核心工具。</span>本节先造码，再讲对称性。

## 1 相位错误长什么样

**相位翻转（phase flip）** 是 $Z$ 门的作用：$Z\lvert0\rangle = \lvert0\rangle$、$Z\lvert1\rangle = -\lvert1\rangle$。对叠加态 $\lvert\psi\rangle = \alpha\lvert0\rangle + \beta\lvert1\rangle$，$Z\lvert\psi\rangle = \alpha\lvert0\rangle - \beta\lvert1\rangle$——两个分量的相对相位反转。<span class="marginnote">为什么计算基测量「看不见」它？因为 $\lvert0\rangle$、$\lvert1\rangle$ 都是 $Z$ 的本征态，$Z$ 只是乘一个整体相位（对 $\lvert1\rangle$ 是 $-1$），测量概率 $\lvert\alpha\rvert^2$、$\lvert\beta\rvert^2$ 不变。相位错误只有在<strong>相位敏感</strong>的基（如 $\lvert+\rangle/\lvert-\rangle$）或<strong>相干运算</strong>里才显现。</span>它对应物理上的「退相位」（dephasing）噪声——量子比特和环境的能量耦合造成的相对相位漂移。

## 2 相位翻转码的编码：转到 $X$ 基

在 $X$ 基下，$Z$ 的作用是「翻转」：$Z\lvert+\rangle = \lvert-\rangle$、$Z\lvert-\rangle = \lvert+\rangle$。于是「相位翻转」在 $X$ 基下就是「比特翻转」！把比特翻转码的编码整个搬到 $X$ 基，就得到相位翻转码：

$$
\lvert0_L\rangle = \lvert+++\rangle, \qquad \lvert1_L\rangle = \lvert---\rangle
$$

编码线路 = 比特翻转码线路 + 每比特前后夹 $H$：$H^{\otimes 3}$ 作用把 $Z$ 错误变成 $X$ 错误，$X$ 基的重复码照常工作，再 $H^{\otimes 3}$ 转回计算基。<span class="marginnote">这条「$H^{\otimes 3}$ 包裹」的线路是本节最重要的一张图：它把一个「相位翻转码」变成一个「带上 $H$ 的比特翻转码」。物理实现里你不需要新的校验逻辑——只要在编码前转基、解码后转回，复用比特翻转码的全部纠错机制。</span>

## 3 公式解析：为什么 $H^{\otimes 3}$ 能把相位错误转成比特错误

设编码态为 $\lvert\psi_L\rangle$，对它作用相位错误 $Z_i$，再作用 $H^{\otimes 3}$：

$$
H^{\otimes 3}\, Z_i\, H^{\otimes 3} = X_i
$$

- **第一步，共轭关系**：由 $HZH = X$（单比特恒等式，可验证：$HZH$ 作用在 $\lvert0\rangle$ 上得 $\lvert1\rangle$，作用在 $\lvert1\rangle$ 上得 $\lvert0\rangle$），张量积给出 $H^{\otimes3}Z_iH^{\otimes3} = X_i$。
- **第二步，翻译**：若在编码线路里**先** $H^{\otimes3}$ **再**让噪声进来，等效于「噪声直接以 $X$ 形式作用在比特翻转码上」。于是相位错误被「降级」为比特翻转码能处理的错误。
- **第三步，代价**：付出的额外成本是两个 $H^{\otimes3}$ 层——「转基」的开销。<span class="marginnote">这个「共轭换基」技巧是量子纠错设计的基本功：任何「在 $U$ 基下的错误」都能用「$U$ 包裹的编码」转成「计算基下的错误」。Shor 码、CSS 码正是靠同时处理「$X$ 类」与「$Z$ 类」两组共轭错误而构建的。</span>

## 4 相位翻转码的综合征

相位翻转码的校验算符从比特翻转码的 $Z_i Z_j$ 变成 $X_i X_j$：

$$
X_1 X_2, \qquad X_2 X_3
$$

综合征测量同样用两个辅助比特读出。错误与综合征的对照表：

| 错误 | 态（$X$ 基下） | 综合征 $(X_1X_2, X_2X_3)$ | 修复 |
| --- | --- | --- | --- |
| 无 | $\alpha\lvert+++\rangle+\beta\lvert---\rangle$ | $(+1,+1)$ | 无 |
| 第 1 位相位翻转 $Z_1$ | 等价 $X_1$ | $(-1,+1)$ | $Z_1$ |
| 第 2 位相位翻转 $Z_2$ | 等价 $X_2$ | $(-1,-1)$ | $Z_2$ |
| 第 3 位相位翻转 $Z_3$ | 等价 $X_3$ | $(+1,-1)$ | $Z_3$ |

修复门是 $Z_i$（把翻掉的相位翻回来）。<span class="marginnote">整张表与比特翻转码的表同构——只要把「$Z$ 校验、$X$ 修复」换成「$X$ 校验、$Z$ 修复」。这正是「换基等价」的另一个体现：两张码互为 $H$ 共轭。</span>

**辨析｜易错点：** 相位翻转码**不能**纠比特翻转 $X$。$X$ 在 $X$ 基下是「相位错误」——$X\lvert+\rangle = \lvert+\rangle$（整体相位不变），$X\lvert-\rangle = -\lvert-\rangle$，对码字只是整体相位，检测不到。所以「比特翻转码 + 相位翻转码」各管一类错误，缺一不可。同时防两类错误，需要把两者**嵌套**——这就是下一节的 Shor 九比特码。

## 5 从两张码到 Shor 码

相位翻转码与比特翻转码的组合逻辑已经清晰：

比特翻转码：防 $X$，校验 $Z_iZ_j$。
相位翻转码：防 $Z$，校验 $X_iX_j$。
**两者无法直接合并**：只用 3 个比特，要么测 $Z_iZ_j$ 要么测 $X_iX_j$，不能同时防两类。

Shor 的洞察是**嵌套**：把每个「逻辑比特」先用相位翻转码（3 组），每组内部再用比特翻转码（3 个比特）——$3\times3 = 9$ 个物理比特保护 1 个逻辑比特，同时防 $X$ 与 $Z$。<span class="marginnote">Shor 码是「CSS 结构」的第一次现身：$X$ 错误与 $Z$ 错误分别由内层、外层码处理，互不干扰。下一节《Shor 九比特码》详细展开它的构造与纠错流程。</span>

## 6 小结

- **相位翻转** $Z$：翻转 $\lvert+\rangle\leftrightarrow\lvert-\rangle$，计算基测量看不见，需相位敏感基才能察觉。
- **相位翻转码**：$\lvert0_L\rangle=\lvert+++\rangle$、$\lvert1_L\rangle=\lvert---\rangle$，校验 $X_1X_2, X_2X_3$，修复门 $Z_i$。
- **对称性**：$H^{\otimes3}$ 共轭把相位码变成比特码（$HZH=X$），两张码互为镜像。
- **局限**：只防 $Z$；同时防 $X$ 与 $Z$ 需要把两张码嵌套——引出 Shor 九比特码。

在下一节，我们把两个三比特码嵌套起来——**Shor 九比特码**，史上第一个能同时纠正比特翻转与相位翻转的量子纠错码。
