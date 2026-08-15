
---

title: Tangles 与 skein 理论

date: 2026-08-07

---



# Tangles 与 skein 理论



<div class="epigraph">

<p>把结切碎成缠结，缠结组织成代数，代数又拼回结——这就是 skein 理论的循环。</p>

<footer>—— 本文作者按</footer>

</div>



<div class="article-byline">

<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第11–12章 ｜ 2026-08-07</p>

</div>



## 为什么从「缠结」开始



把一条结切一刀，得到两条有端点的「半截绳」；把链环在平面上划开几个口，得到一块带端点的局部图。这类「有端点的结图局部」叫 **tangle（缠结）**。单独看 tangle 毫无意义（有端点、没封闭），但把 tangle 当作「积木」，通过**拼接**组合，就能重新组装出完整的结——于是「结的构造」变成了「代数的运算」。



**skein 理论（skein theory）**正是把这个想法系统化：取所有 tangle 图，按 skein 关系（交叉展开、圈消去）模掉，得到的商空间带自然乘法——这就是 **skein 模（skein module）**。Tangle 与 skein 理论是量子不变量（第3篇之四）的组合引擎：Jones 多项式、Temperley-Lieb 代数、量子群表示，全部可以放进这个框架。<span class="marginnote">Tangle 的思想源头是 Conway 1969 年的「tangle」记号——他用带两个端点的 tangle 分析有理链环。Turaev 1988 年把它系统化为代数理论。到 1990 年代，Kauffman 括号 skein 模（KBSM）成为三维流形量子不变量（Witten 不变量）的离散骨架——第4篇会看到它的流形版本。</span>



## 1 什么是 tangle



**Tangle（缠结）**：三维空间中若干条互不相交的曲线（闭曲线与开弧段）构成的嵌入，其中开弧段的端点在边界上，且端点配置成「标准位置」。实践中常用平面图表示。



最常用的是 **$(m, n)$-tangle**：有 $m$ 个端点在上方、$n$ 个端点在下方，内部是弧段与圈的并。几个基本例子：



**平凡 $(0, 0)$-tangle**：空图（没有曲线）。

**$(0, 0)$-tangle 的圈**：一个闭圈，单独存在。

- **$(1, 1)$-tangle**：一条从上方到下方的弧（平凡弧）。

- **$(2, 2)$-tangle**：可含交叉——例如一个正交叉、一个负交叉、或两种平滑。



**辨析｜tangle 与结图的区别**：结图没有端点；tangle 有端点。端点的存在让 tangle 可以「接合」（对着端点连），而结图不能。从 tangle 到结只需「封口」（closure）：把上方端点与下方端点配对连接，就得到链环。



## 2 Tangle 的拼接与积



Tangle 的威力在于它可以**拼接**。定义两种基本操作：



**垂直拼接（vertical composition）**：把两个 tangle 上下堆叠，对齐端点连起来。若 $\alpha$ 是 $(n, m)$-tangle、$\beta$ 是 $(m, k)$-tangle，则 $\alpha \cdot \beta$ 是 $(n, k)$-tangle。

**水平拼接（horizontal composition）**：把两个 tangle 并排摆放，端点数相加。$\alpha \otimes \beta$ 把 $(m, n)$ 与 $(m', n')$ 拼成 $(m+m', n+n')$。



- 垂直拼接让「$(n, n)$-tangle 的集合」带上**乘法**——这是通向代数结构的入口。

- 水平拼接是「直和」型的运算，把不相干的部分并置。



**关键观察**：闭包 + 拼接把「构造结」变成「代数运算」。例如，任何结 = 某条 $(0, 0)$-tangle 的闭包（本身就是封闭的）；而辫子（第1篇之五）正是一类特殊的 $(n, n)$-tangle——辫群 $B_n$ 是「$(n, n)$-tangle 在保端点同痕下的等价类」配垂直拼接。<span class="marginnote">辫群是 tangle 理论最古老的例子：$B_n$ = 单调 $(n, n)$-tangle 的等价类。从辫群到一般 tangle，等于允许「回头」（绳子不必单调向下）——这正对应从「辫子」推广到「一般缠绕」，Alexander 定理（任何结是闭辫）在此获得 tangle 语言下的统一解读。</span>



## 3 skein 模：商掉局部关系



把所有 tangle（固定端点配置）的**线性组合**收集起来，得到一个自由 $\mathbb{Z}[A^{\pm 1}]$-模（或 $\mathbb{C}$-向量空间）。再**模掉 skein 关系**（交叉展开 + 圈消去），得到商空间：



**Kauffman 括号 skein 模（KBSM）**：取未定向 tangle 图的 $\mathbb{Z}[A^{\pm 1}]$-线性组合，模掉关系



$$

\langle L_+ \rangle = A \langle L_0 \rangle + A^{-1} \langle L_\infty \rangle, \qquad

\langle L \sqcup \bigcirc \rangle = (-A^{-2} - A^2)\, \langle L \rangle,

$$



得到的商空间，记为 $\mathcal{S}(M)$（$M$ 为承载的流形）。



- 这正是 Kauffman 括号的「公理化」：括号定义时我们断言「存在这样的函数」，现在我们把「函数」换成「商空间」——把计算规则升格为代数结构。

- 对结图，$\mathcal{S}$ 的每个元素是「满足 skein 关系的图类」；闭包后就回到括号与 Jones 多项式。



**易错点｜skein 模 ≠ 多项式**：skein 模是一个**空间/代数**（含所有 tangle 的组合），不是单个多项式。Jones 多项式是「对 skein 模施加一个线性泛函（迹）后得到的值」——模是结构，多项式是读出。混淆两者等于混淆「空间」与「坐标」。



## 4 公式解析：为什么商空间承载代数



设 $V$ 是「$(n, n)$-tangle 图」的自由模（基为所有 $(n,n)$-tangle 图），$R$ 是 skein 关系生成的子模，则



$$

\mathcal{S}_n = V / R

$$



是商模，且带诱导的乘法（垂直拼接）：



$$

[\alpha] \cdot [\beta] = [\alpha \cdot \beta].

$$



- **第一步，商模的含义**：把满足 skein 关系的「不同图」视为同一元素——$L_+$ 与 $A L_0 + A^{-1} L_\infty$ 在商空间里相等。这与「把 $x^2 + 1 = 0$ 作为关系构造复数域」同构。

- **第二步，乘法为何良定义**：若 $\alpha \sim \alpha'$（差一个 skein 关系），则 $\alpha \cdot \beta \sim \alpha' \cdot \beta$——因为关系只改局部、不碰拼接点。所以商空间上的乘法自动良定义。

- **第三步，有限维性**：对固定的 $n$，$\mathcal{S}_n$ 往往**有限维**——skein 关系把无穷多个图压缩进有限维空间。这正是 Temperley-Lieb 代数（第3篇之二）的雏形：$\mathcal{S}_n$ 就是 $TL_n$ 的「图表示」。



**辨析｜skein 关系为什么是「减法」**：商掉关系 $x = y$ 等价于商掉 $x - y = 0$——「令两个对象相等」与「令它们的差为零」是同一件事。skein 关系 $L_+ = A L_0 + A^{-1}L_\infty$ 被理解为「$L_+ - A L_0 - A^{-1}L_\infty = 0$ 是关系」，商掉它之后，任何包含 $L_+$ 的组合都可以「重写」成 $L_0, L_\infty$ 的组合——递归计算由此而来。



## 5 Tangle 与 skein 理论的应用



- **结多项式**：skein 模上的线性泛函（迹）给出 Jones、HOMFLY、Kauffman 多项式——一个框架，三个不变量。

- **Temperley-Lieb 代数**：$(n, n)$-tangle 的 KBSM 商给出 $TL_n$ 的图基（第3篇之二）。

- **量子群表示**：skein 关系是量子群 $\mathcal{R}$-矩阵的「图解版本」——R 矩阵满足 Yang-Baxter 方程，正是 skein 关系 R3 的代数化身（第3篇之四）。<span class="marginnote">skein 理论最深刻的统一：<strong>skein 关系 ⟷ Yang-Baxter 方程</strong>。R3 移动（第三根绳滑越交叉）在代数语言里就是 $R_1R_2R_1 = R_2R_1R_2$（Yang-Baxter 方程）。量子群提供 R 矩阵，skein 理论把它画成图，结不变量由此诞生——「画图」与「矩阵」在此合流。</span>

- **三维流形不变量**：把 skein 模推广到三维流形（在流形里考虑 tangle），得到 Witten 不变量与 Reshetikhin-Turaev 不变量的组合基础（第3篇之四、五）。



### 有理 tangle 与 Conway 记号



Tangle 理论最古老也最实用的部分是 **$(2, 2)$-tangle**——两个端点在上、两个在下。特殊的 $(2, 2)$-tangle 叫**有理 tangle（rational tangle）**：由「一排整数交叉」按连分数结构组成，可用有理数 $\frac{p}{q}$ 编码。



**Conway 记号（Conway notation）**：把链环按「有理 tangle 的拼接」编码为整数序列。例如：



- $3_1$ 记作 $(3)$：一个「3 交叉」的有理 tangle 闭包。

- $4_1$ 记作 $(2, 2)$：两个「2 交叉」tangle 的拼接。

- 一般记法 $(\ldots)$ 中的整数串对应「一串有理 tangle 的交替拼接」。



**为什么有理数编码有效**：$(2, 2)$-tangle 的「缠绕方式」由连分数 $\frac{p}{q}$ 确定，而连分数与有理数一一对应——**整数串 ↔ 连分数 ↔ 有理数 ↔ tangle**。这是「组合对象被数论编码」的漂亮例子，也是 Conway 记号能压缩结表的基础。



### skein 模的流形版本



把 skein 模的定义从「$\mathbb{R}^3$ 中的 tangle」推广到「任意三维流形 $M$ 中的 tangle」，得到 **Kauffman 括号 skein 模 $\mathcal{S}(M)$**：



- $\mathcal{S}(M)$ 对每个三维流形给一个代数——它是「$M$ 的 Kauffman 括号版本的代数」。

- 对 $M = S^3$，$\mathcal{S}(S^3) = \mathbb{Z}[A^{\pm 1}]$（一维：括号给出唯一的「总不变式」）。

- 对 $M = D^2 \times I$（圆盘柱），$\mathcal{S}(D^2 \times I) = TL_n$（Temperley-Lieb 代数，第3篇之二）。



**流形 skein 模是「TQFT 前的 TQFT」**：它给每个流形配代数、给拼接配乘法——正是第3篇之五 TQFT 公理（流形 → 向量空间）的雏形。从 tangle 到流形，skein 理论一步跨进三维流形的量子世界。



## 6 小结



- **Tangle** 是有端点的结图局部；垂直/水平拼接赋予其代数结构。

- 辫群 $B_n$ 是单调 $(n, n)$-tangle 的等价类——tangle 理论最古老的成员。

- **skein 模** $V/R$ 是「tangle 图的线性组合模掉 skein 关系」的商空间：模是结构，多项式只是从模上读出的坐标——别把「空间」与「坐标」混为一谈。

- 辫群 $B_n$ 是单调 $(n,n)$-tangle 的等价类；Temperley-Lieb 代数 $TL_n$ 是 $(n,n)$-tangle 的 KBSM 商——tangle 理论是量子不变量（Reshetikhin-Turaev、Witten）的组合引擎。

- **有理 tangle** 用连分数编码（整数串 ↔ 有理数 ↔ tangle），Conway 记号因此能压缩结表；流形版 skein 模 $\mathcal{S}(M)$ 给每个三维流形配一个代数，是「TQFT 前的 TQFT」。



在下一节，我们深入 skein 模背后的代数——**Temperley-Lieb 代数**：它是 tangle 图代数的正式化身，也是 Jones 多项式与量子群表示论的公共舞台。