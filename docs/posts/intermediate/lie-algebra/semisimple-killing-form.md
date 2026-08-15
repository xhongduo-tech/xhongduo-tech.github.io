---
title: 半单李代数与 Killing 型
date: 2026-08-07
---

# 半单李代数与 Killing 型

<div class="epigraph">
<p>我见过许多将对称性奉为圭臬的人，但真正的力量在于对称性背后的不变量。</p>
<footer>—— 埃米 · 诺特（Emmy Noether，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 李代数与李群 ｜ Humphreys §5 ｜ 2026-08-07</p>
</div>

## 为什么引入 Killing 型

可解与幂零描述了「退化」的一端，这一端比较容易处理。另一半则深藏结构：**半单（semisimple）**李代数——不含非零可解理想的李代数。半单李代数的结构极为丰富，而进入它的钥匙是一把对称双线性型：**Killing 型**。<span class="marginnote">Killing 型是 Wilhelm Killing 在 1888 年前后独立于 Sophus Lie 完成的李代数分类工作的一部分——他正是通过这把「度量」成功刻画了复半单李代数，比 Cartan 的分类更早奠定根基。Killing 本人未能完全完成证明，Cartan 的博士论文补全了它。</span>

这套「退化 ⟶ 半单」的过渡之所以重要，在于它把「坏的部分」完全剥离：任意有限维李代数都有一个唯一的极大可解理想（根），商掉它得到半单部分，这就是 Levi 分解。半单部分承载结构的精华，可解部分是可控的噪声——本节先立住「非退化」这把尺子。

历史上 Killing 正是借助 Killing 型完成了复半单李代数的分类草案，而 Cartan 的博士论文用更清晰的方式重做了证明。一个双线性型能撑起一整套分类理论，这本身就说明「度量」在代数结构中的分量——本节是这套古典智慧的最小展示。

## 1 定义：从伴随表示生成的不变量

**Killing 型（Killing form）**：$L$ 上的对称双线性型 $\kappa$ 定义为

$$\kappa(x, y) = \operatorname{tr}(\operatorname{ad}x \circ \operatorname{ad}y)$$

即取伴随算子复合后的迹。<span class="marginnote">迹 $tr$ 本身是最基本的「不变量」：相似变换下不变。因此 Killing 型对同构、同态都有良好的自然性——它从李代数「自身作用」的痕迹中提取信息，不依赖任何外部的选择。</span>

它自动具备两个关键性质：

**对称性**：$\kappa(x,y) = \kappa(y,x)$，来自迹的对称性 $\operatorname{tr}(AB) = \operatorname{tr}(BA)$。
- **不变性**：$\kappa([x,y], z) = \kappa(x, [y,z])$，即括号对 Killing 型可「移动」。这使 $\kappa$ 成为「不变双线性型」，相当于李代数的度量结构。<span class="marginnote">不变性可从循环性 $\operatorname{tr}([x,y]w) = \operatorname{tr}(x[y,w])$ 推出；它对后文「正交补」结构、以及根空间分解的几何化至关重要。</span>

**半单（semisimple）**：若 $L$ 不含非零可解理想（等价地，不含非零阿贝尔理想），则称 $L$ **半单**。<span class="marginnote">「半单」直译即「一半是单的」——严格说它要求的是「无平凡可解理想」，比「无平凡理想」弱。全体单李代数的理想直和恰是半单李代数（见第 5 节）。</span>

**单（simple）**：$L$ 无平凡理想且 $[L, L] \neq 0$，则称 $L$ **单**。半单包含「若干单的代数和」；单是「不能再分解」的构件。

「半单」与「单」的差别常用一句话区分：半单是「无退化」，单是「无杂质」。严格说，若干单理想的理想直和构成半单；反过来半单的每个理想直和因子都是单理想。初学者最容易在「半单是否等价于单」上犯晕——答案是否定的，除非直和只有一个因子。

对称性 $\kappa(x,y) = \kappa(y,x)$ 的证明只用一条恒等式 $\operatorname{tr}(AB) = \operatorname{tr}(BA)$，它是「迹可循环」的特例；不变性则来自循环性更深的一层：$\operatorname{tr}([x,y]w) = \operatorname{tr}(x[y,w])$。这两条性质一个保证 $\kappa$ 可对角化，一个保证括号与度量相容——缺一不可。

把三个关键概念放进一张速查表：

| 概念 | 定义要点 | Killing 型表现 | 例子 |
| --- | --- | --- | --- |
| 可解 | 导出序列有限步归零 | $\kappa(L,[L,L]) = 0$ | $\mathfrak{t}(n)$ |
| 单 | 无非零真理想且 $[L,L]\neq 0$ | 非退化 | $\mathfrak{sl}(n,\mathbb{C})$ |
| 半单 | 无非零可解理想 | 非退化 | 单代数的理想直和 |

不变性 $\kappa([x,y], z) = \kappa(x, [y,z])$ 的几何意义是：括号像「协变导数」一样与度量相容——把 $z$ 换成沿 $y$ 方向的「平移」，内积不变。这让 Killing 型成为李代数上的黎曼结构，而 Cartan 子代数正是这个黎曼结构下的「极大平坦子代数」。

## 2 Cartan 判别法：退化性把两类分开

**Cartan 判别法（Cartan's criterion）** 是全书第一个「判别结构」的利器：

> $\mathbb{F}$ 特征零，$L$ 是 $\mathbb{F}$ 上有限维李代数。
> 1. $L$ **可解** ⟺ $\kappa(L, [L, L]) = 0$（即 $\kappa$ 在 $L$ 与 $[L,L]$ 上退化为零）。
> 2. $L$ **半单** ⟺ $\kappa$ **非退化**（即对每个非零 $x$ 存在 $y$ 使 $\kappa(x, y) \neq 0$）。

**辨析｜易错点：** 「$\kappa(L, [L,L]) = 0$」不等于「$\kappa$ 对 $[L,L]$ 限制后为零」——它是说 $\kappa(x, y) = 0$ 对所有 $x \in L, y \in [L,L]$。初学者常误读为「Killing 型平凡」。判别法第二句才是关键应用：**半单 ⟺ 非退化**，这把「无退化理想」的代数条件翻译成了「非退化度量」的线性条件，也直接让正交补、对偶空间等工具进场。

辨析里那句「$\kappa(L,[L,L]) = 0$ 不是限制为零」值得用例子固化：对 $\mathfrak{t}(3)$，$\kappa(\mathfrak{t}(3), \mathfrak{n}(3)) = 0$，但 $\kappa$ 在 $[L,L] = \mathfrak{n}(3)$ 上的限制（作为 $\mathfrak{n}(3)$ 上的双线性型）却非平凡。初学者若把两者混为一谈，就会在判定可解性时得到错误的退化结论。

把判别法读成操作流程：第一步算 $\kappa$ 的矩阵（需要选基），第二步看 $\kappa(L,[L,L])$ 是否全为零，第三步看 $\kappa$ 是否非退化。三步全是线性代数计算，不涉及任何抽象概念——这正是结构理论的「可计算入口」。

Cartan 判别法的两条合起来还给出一个漂亮的对偶：可解对应「$\kappa$ 在某个方向上塌缩」，半单对应「$\kappa$ 满秩」。中间情形（既不半单也不可解，如 $\mathfrak{sl}(2) \oplus \mathfrak{t}(1)$）则介于两者之间——退化部分与半单部分并存，这正是 Levi 分解要处理的对象。

$\mathfrak{sl}(n, \mathbb{C})$ 的 Killing 型可以直接算出：$\kappa(x, y) = 2n \operatorname{tr}(xy)$（在 $\mathfrak{sl}(n)$ 上）。当 $n \geq 2$ 时非退化，故 $\mathfrak{sl}(n,\mathbb{C})$ 半单——实际上它是单的。<span class="marginnote">计算细节：$\operatorname{ad}x$ 在标准基上的矩阵迹需要较长推导，Humphreys 习题提供了组合证明。此处记住公式即可——它是后续 $\mathfrak{sl}(2)$ 表示论中归一化的基础。</span>

比例系数 $2n$ 不是凭空来的：$\mathfrak{sl}(n)$ 的维数是 $n^2 - 1$，而伴随表示把 $x$ 映成作用在 $n^2-1$ 维空间上的算子，迹的比例因子正是从这个维度关系里长出来的。这个系数在后文 Casimir 元素（第 13 篇）的归一化里会再次出现。

## 3 结构定理：半单的代数和与分解

**半单结构定理（第一形式）**：特征零有限维半单李代数 $L$ 可写成其单理想 $L_i$ 的**理想直和**：

这一条定理回答了上一节遗留的问题：任意李代数的「杂质」（非零可解理想）可以像筛掉沙子一样被剥离，剩下的核心部分恰好是半单的。Weyl 完全可约性（第 5 篇）的全部内容都建立在这个「核心可以干净切出」的事实之上。

$$L = L_1 \oplus L_2 \oplus \cdots \oplus L_t$$

其中每个 $L_i$ 是单理想，且这种分解在排列次序意义下**唯一**。<span class="marginnote">这相当于半单李代数的「素数分解」：单理想是素因子，直和是相乘。与第 1 篇的「理想直和」概念精确对上——$[L_i, L_j] = 0$ 当 $i \neq j$。</span>

这个「素数分解」类比还可以再推进一步：半单李代数的单理想直和不仅唯一，而且各块之间由 Killing 型正交。于是研究任意半单李代数，本质上就是研究单李代数这「有限张素卡片」，而单李代数的分类由第 9 篇的 Dynkin 图清单给出——分类问题的规模被压到了「有限清单」。

**半单结构定理（第二形式）**：特征零半单李代数 $L$ 满足

$$L = \operatorname{Der}(L), \qquad [L, L] = L$$

即每个导子都是内导子 $\operatorname{ad}x$，且 $L$ 由括号自生成。这正是「半单 = 完整 + 自我充足」的代数含义。<span class="marginnote">后文 Weyl 定理（第 5 篇）将在此基础上推出完全可约性；根空间分解（第 7 篇）则把半单的「度量」进一步翻译成「格」——几何化由此展开。</span>

「$L = \operatorname{Der}(L)$」这句话在计算上极有用：要判断一个线性映射是否保持括号，只需验证它是否等于某个 $\operatorname{ad}x$。对半单李代数这是自动成立的；而一般李代数的导子代数可能比 $L$ 大得多，那是可解/幂零理论里的另一套话题。

半单性与非退化还直接给出「正交补」结构：对子空间 $S$，$S^\perp = \{x \mid \kappa(x, s) = 0 \ \forall s \in S\}$，非退化保证 $\dim S^\perp = \dim L - \dim S$ 且 $(S^\perp)^\perp = S$。当 $S$ 是理想时 $S^\perp$ 也是理想，于是理想分解与正交分解重合——这正是第 7 篇根空间分解里「根空间彼此正交」的先声。

第 5 篇的 Weyl 完全可约性定理有一个「Killing 型版本」的证明：半单 ⇒ 非退化 ⇒ 可在表示空间上构造等变内积，进而把任意不变子空间的正交补也变成不变子空间。这条路线比我们下一节要讲的「取迹消去」更几何，两者殊途同归。

## 4 公式解析：非退化性的判定

用 $\mathfrak{sl}(2, \mathbb{C})$ 验证 Cartan 判别法第二句。基为 $e, f, h$，伴随算子按第 3 篇的矩阵计算，得到 Killing 型矩阵：

$$\kappa = \begin{pmatrix} 0 & 4 & 0 \\ 4 & 0 & 0 \\ 0 & 0 & 8 \end{pmatrix} \qquad \text{(在基 } e, f, h \text{ 下)}$$

三步拆解：

- **第一步，算 $\operatorname{ad}x \operatorname{ad}y$ 的迹**：如 $\kappa(e, f) = \operatorname{tr}(\operatorname{ad}e \operatorname{ad}f)$。已知 $\operatorname{ad}e(f) = -h$，$\operatorname{ad}f(h) = 2f$，$\operatorname{ad}f(e) = -h$ 等，逐项取迹得 4。
- **第二步，排矩阵**：对角元 $\kappa(h,h) = \operatorname{tr}((\operatorname{ad}h)^2) = 2^2 + (-2)^2 = 8$，其余由对称性填齐。
- **第三步，判行列式**：$\det \kappa = 4 \cdot 4 \cdot 8 \neq 0$（主行列式非零），故非退化——$\mathfrak{sl}(2,\mathbb{C})$ 半单。

具体算一遍更踏实：第 4 节的矩阵 $\kappa$ 是分块对角的（$\{e,f\}$ 块与 $\{h\}$ 块），主对角元依次为 $0, 4, 8$，交叉项为 $4$。二阶子式 $\det\begin{pmatrix}0 & 4 \\ 4 & 0\end{pmatrix} = -16 \neq 0$，再乘对角元 $8$ 得 $\det\kappa = -128 \neq 0$——非退化成立。判别时只看「行列式是否为零」，不看符号：符号与基的定向选择有关，非退化性则与基无关。

**核心要点**（判别法两方向）：

| 结构 | Killing 型表现 | 含义 |
| --- | --- | --- |
| 可解 | $\kappa(L, [L,L]) = 0$ | 度量在交换子方向上「塌缩」 |
| 半单 | $\kappa$ 非退化 | 度量满秩，无零方向 |

这张表的第二行值得再读一遍：半单 ⟺ 非退化。非退化意味着 $\kappa$ 能给出 $L$ 与 $L^*$ 的典范同构，于是「权」$\lambda \in H^*$ 可以换成「对偶权」$t_\lambda \in H$——这正是第 7 篇根空间分解里把线性泛函当向量用的代数前提。

再做一个算术自检：对 $\mathfrak{sl}(2)$，已知 $\operatorname{ad}e$ 与 $\operatorname{ad}f$ 在基 $\{e,f,h\}$ 下的矩阵，可直接验证 $\kappa(e,f) = \operatorname{tr}(\operatorname{ad}e\operatorname{ad}f) = 4$、$\kappa(h,h) = \operatorname{tr}((\operatorname{ad}h)^2) = 8$，其余分量按对称性补齐即得第 4 节的矩阵。若读者手算出的对角元与这里不一致，多半是把 $\operatorname{ad}$ 的矩阵写错了——值得反复核对这一处，它是后面所有归一化的基准。

## 5 小结

- **Killing 型** $\kappa(x,y) = \operatorname{tr}(\operatorname{ad}x \operatorname{ad}y)$ 是天然的对称不变双线性型，天然地携带结构信息。
- **Cartan 判别法**：可解 ⟺ $\kappa(L,[L,L])=0$；**半单 ⟺ $\kappa$ 非退化**。
- **半单结构定理**：特征零半单 = 单理想的唯一理想直和；且 $L = \operatorname{Der}(L) = [L, L]$。
- 半单是「无退化」与「自充足」的同义语；它是本专题后半所有深结构的舞台。
- $\mathfrak{sl}(n,\mathbb{C})$（$n \ge 2$）是单而非半单原型，其 Killing 型 $\kappa(x,y) = 2n\operatorname{tr}(xy)$。
- 判定流程：先算 $\kappa$ 的矩阵，再检查其退化性——退化 ⇒ 可解方向，非退化 ⇒ 半单方向；$\det\kappa$ 只关心是否为零。

在下一节，我们将问：半单李代数的表示能否「拆开」——这引出本专题的中心定理之一，**Weyl 完全可约性定理**。
