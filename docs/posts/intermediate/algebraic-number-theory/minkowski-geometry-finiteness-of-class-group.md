---
title: Minkowski 几何与类数有限性定理
date: 2026-08-11
---

# Minkowski 几何与类数有限性定理

<div class="epigraph">
<p>如果我比别人看得更远，那是因为我站在巨人的肩膀上。</p>
<footer>—— 艾萨克 · 牛顿（Isaac Newton，If I have seen further it is by standing on the shoulders of giants）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Minkowski 几何开始

前几节反复提到「Minkowski 界」「类数有限」，却一直没给出证明。本节把这笔账结清，而且要付出一笔美妙的几何定金：**把数域 $K$ 塞进 $\mathbb{R}^n$，让代数整数变成一个「格子」，再用凸体的体积来「数」格子里的点**。这就是 **几何数论（geometry of numbers）**——闵可夫斯基在 19 世纪末创立的方法。它不仅给出类数有限的干净证明，还批量给出判别式下界、类数算法与二次型理论，是代数数论从「代数」走向「几何」的关键一跳。

## 1 Minkowski 嵌入与格

**Minkowski 嵌入（Minkowski embedding）**：把 $K$ 的所有嵌入合体，映到实空间

$$
\sigma: K \longrightarrow K \otimes_{\mathbb{Q}} \mathbb{R} \;\cong\; \mathbb{R}^{r_1} \times \mathbb{C}^{r_2} \;\cong\; \mathbb{R}^{n}
$$

每个复共轭对取其一，$\mathbb{C}$ 的 2 个实维度正好补足 $n = r_1 + 2r_2$。<span class="marginnote">这是「把抽象的域变成具体坐标」：$\sqrt{2} \mapsto (\sqrt2, -\sqrt2) \in \mathbb{R}^2$，$i \mapsto (i, -i) \in \mathbb{C}$。加法和乘法都变成坐标运算，域的一切算术被翻译成 $\mathbb{R}^n$ 的几何。</span>

**关键引理（格与协体积）**：$\sigma(\mathcal{O}_K)$ 是 $\mathbb{R}^n$ 中的**满格（lattice）**（秩 $n$ 的离散 $\mathbb{Z}$-子模），且协体积为

$$
\mathrm{vol}\big(\mathbb{R}^n / \sigma(\mathcal{O}_K)\big) = 2^{-r_2} \sqrt{|d_K|}
$$

更一般地，**任意非零分式理想 $\mathfrak{a}$ 的像也是满格**，协体积 $= 2^{-r_2}\sqrt{|d_K|}\, \mathrm{N}(\mathfrak{a})$。<span class="marginnote">协体积 = 基本平行体（fundamental domain）的体积。判别式 $d_K$ 在这里第一次获得几何含义：<strong>它是格子的体积平方（差一个常数）</strong>——代数不变量与几何体积在 Minkowski 嵌入下合二为一。</span>

## 2 Minkowski 格点定理

**定理（Minkowski 格点定理）：** 设 $\Lambda \subset \mathbb{R}^n$ 是满格，$X \subset \mathbb{R}^n$ 是**中心对称**（$X = -X$）的**凸**体。若 $\mathrm{vol}(X) > 2^n\, \mathrm{vol}(\Lambda)$，则 $X$ 含非零格点。

![对称凸体绕原点放置，体积超过 2^n·vol(Λ) 时必吞进非零格点](/images/algebraic-number-theory/minkowski-geometry-finiteness-of-class-group-1.svg)

**直觉与证明核心**：关键引理是 **Blichfeldt 定理**——若可测集体积大于格子协体积，则它含两个点差一个非零格向量（鸽子笼原理的连续版）。再把 $X$ 放大一倍：$\mathrm{vol}(\tfrac12 X) > 2^{n-?}\dots$——标准论证是把 $\tfrac12 X$ 与它的平移 $\tfrac12 X + \lambda$ 对拼，用中心对称与凸性推出差向量 $\in X \cap (\Lambda \setminus \{0\})$。<span class="marginnote">中心对称（$X=-X$）保证「差向量」重回 $X$，凸性保证「中点」还在 $X$——两条假设缺一不可。这个纯几何引理，是「代数对象的长度/体积控制」的万能钥匙。</span>**维度提醒**：$n = 1$ 时条件退化为「对称区间长度 $> 2$ 倍步长」，格点定理即一维鸽笼原理——一切从一维直觉出发，逐维推广到 $n$ 维。

## 3 Minkowski 界与类数有限

现在选择「体」：$X_t = \{(x_1,\dots,x_{r_1}, z_1,\dots,z_{r_2}) : \sum_i |x_i| + 2\sum_j |z_j| \le t\}$。它是中心对称凸体，体积可精确算出：

$$
\mathrm{vol}(X_t) = \frac{2^n}{n!}\left(\frac{\pi}{2}\right)^{r_2} t^n
$$

**关键不等式**：$X_t$ 中任一元素 $\alpha$ 满足 $|\mathrm{N}(\alpha)| \le \big(\tfrac{t}{n}\big)^n$（算术-几何平均）。于是取 $\mathrm{vol}(X_t) > 2^n \cdot 2^{-r_2}\sqrt{|d_K|}$，即 $t = \sqrt[n]{\big(\tfrac{4}{\pi}\big)^{r_2} n!\, \sqrt{|d_K|}}$ 时，格点定理给出非零 $\alpha \in \mathfrak{a}$ 使

$$
|\mathrm{N}(\alpha)| \le \frac{t^n}{n^n} = \left(\frac{4}{\pi}\right)^{r_2} \frac{n!}{n^n} \sqrt{|d_K|} \cdot \frac{1}{\,n^n\,} \;\Rightarrow\; M_K = \left(\frac{4}{\pi}\right)^{r_2}\frac{n!}{n^n}\sqrt{|d_K|}
$$

整理成标准形式：**每个非零分式理想 $\mathfrak{a}$ 含非零元 $\alpha$，使 $|\mathrm{N}(\alpha)| \le M_K \cdot \mathrm{N}(\mathfrak{a})$**。

**类数有限性定理**：任意理想类含整理想 $\mathfrak{a}$ 满足 $\mathrm{N}(\mathfrak{a}) \le M_K$（取该类里的分式理想，用上面的 $\alpha$ 消掉分母）。而范 $\le M_K$ 的整理想只有有限多个（$\mathcal{O}_K$ 中范为定值的元素有限），故**类群有限**。$\blacksquare$<span class="marginnote">这一行收束了前几节欠下的证明：类数有限性不是代数技巧，而是「体积超过格协体积」的几何事实的推论。同一论证顺带给出：若 $M_K < 2$，则 $h_K = 1$——用判别式大小直接判唯一分解。</span>

## 4 公式解析：Minkowski 界从哪来

$$
M_K = \left(\frac{4}{\pi}\right)^{r_2} \frac{n!}{n^n} \sqrt{|d_K|}
$$

- **第一步，锁体积**：$\mathrm{vol}(X_t) = \frac{2^n}{n!}(\frac{\pi}{2})^{r_2} t^n$，来自 $\mathbb{R}$ 上 $\sum|x_i|\le t$ 的超八面体体积 $\frac{(2t)^n}{n!}$ 乘上复方向带来的 $(\pi/2)^{r_2}$。
- **第二步，并体积与协体积**：格点定理要求 $\mathrm{vol}(X_t) > 2^n \cdot 2^{-r_2}\sqrt{|d_K|}$。右边把 $2^n$ 抵消进 $X_t$ 的体积系数，剩下 $\frac{2^n}{n!}(\frac{\pi}{2})^{r_2} t^n > \frac{2^n}{2^{r_2}}\sqrt{|d_K|}$。
- **第三步，解出 $t$ 并算范**：得到 $t^n = \frac{n!\,2^{r_2}}{(\pi/2)^{r_2}} \cdot \frac{\sqrt{|d_K|}}{2^{r_2}} = \big(\frac{4}{\pi}\big)^{r_2} n!\sqrt{|d_K|}$，代入 $\big(\frac{t}{n}\big)^n$ 得 $M_K$。
- **第四步，回到理想类**：对任意类取分式理想 $\mathfrak{a}$，格点定理在 $\mathfrak{a}$ 的格子里找到小范元素 $\alpha$，$\alpha^{-1}\mathfrak{a}$ 即所求整理想，范 $\le M_K$。

## 5 Hermite 定理与判别式下界

**Hermite 定理**：给定 $X > 0$，判别式 $|d_K| \le X$ 的（扩张）数域 $K/\mathbb{Q}$ 只有**有限多个**。<span class="marginnote">证明是 $M_K$ 界的推论之一：判别式有界 ⟹ 次数 $n$ 有界 ⟹ 嵌入空间维数有界，而单位格、类群都被控制住，剩下的组合只有有限种。这是「数域分类」理论的开端——代数数论关心「全部数域」这个集合的结构。</span>

由 $M_K \ge 1$ 反解出**判别式下界**

$$
|d_K| \;\ge\; \left(\frac{\pi}{4}\right)^{2r_2} \left(\frac{n^n}{n!}\right)^2
$$

对 $n$ 较大时近似 $\big(\frac{\pi e^2}{4}\big)^{n}$，指数增长——**判别式（因而域的「大小」）随次数爆炸**。这是「次数大的域必须有大判别式」的精确陈述，也是类域论里「存在无限多互不相交的域」等结论的燃料。

**辨析｜易错点：** 别把「Minkowski 界」与「判别式下界」混为一个对象。前者是**类的范上界**（用于数类群），后者是**判别式的体积下界**（用于数域分类），两者互为对偶的几何事实。另外 $M_K$ 常**非整数**（如 $\mathbb{Q}(\sqrt{-5})$ 得 $2.85$），取「$\le M_K$」时按实数比较即可，不需取整。

## 6 实例：Minkowski 界的算术

**例 1（$K = \mathbb{Q}(\sqrt{-5})$）**：$n = 2, r_1 = 0, r_2 = 1, |d_K| = 20$，故

$$
M_K = \frac{4}{\pi}\cdot\frac{2!}{2^2}\cdot\sqrt{20} = \frac{4\sqrt{5}}{\pi} \approx 2.85
$$

范 $\le 2.85$ 的整理想只有 $\mathcal{O}_K$（范 $1$）与范 $2$ 的素理想 $\mathfrak{p}_2 = (2, 1+\sqrt{-5})$。检验：$\mathfrak{p}_2$ 非主（$a^2 + 5b^2 = 2$ 无解）、$\mathfrak{p}_2^2 = (2)$ 主，故 $h_K = 2$，类群由 $[\mathfrak{p}_2]$ 生成。

**例 2（$K = \mathbb{Q}(\sqrt{-14})$）**：$|d_K| = 56$，

$$
M_K = \frac{2}{\pi}\sqrt{56} \approx \frac{2}{\pi}\cdot 7.48 \approx 4.76
$$

要检查范 $= 2, 3$ 的素理想。$a^2 + 14b^2 = 2$ 与 $= 3$ 均无解，故 $\mathfrak{p}_2, \mathfrak{p}_3$ 都非主；再算 $\mathfrak{p}_2^2 = (2)$ 主、$\mathfrak{p}_3^2$ 的类、$\mathfrak{p}_2\mathfrak{p}_3$ 的类……逐类归并得 $h_K = 4$。**这就是类群算法的完整节奏**：算界 → 枚举范 $\le M_K$ 的素理想 → 判断主不主 → 理想间的类归并。

**快速再算（$\mathbb{Q}(\sqrt{6})$）**：$M_K = \frac{2}{\pi}\sqrt{24} \approx 3.12$，检查范 $2, 3$ 的素理想可得 $h_K = 2$——同一套流程的第三次演练。<span class="marginnote">这就是「类数算法」：<strong>Minkowski 界把无限类群化成有限次验算</strong>——算 $M_K$、枚举范 $\le M_K$ 的素理想、判断主不主与类关系。它是手算、也是现代 SAGE/Magma 类群命令背后的原始逻辑。</span>

**辨析｜易错点：** Minkowski 界只给「每个类含小范理想」的**上界**，它不直接告诉你类数——还要逐个判定主不主、以及理想之间的类等价。**「上界够小」不等于「类数小」**；但上界小到 $M_K < 2$ 时可直接断言 $h_K = 1$，这是唯一零成本判定（如 $\mathbb{Q}(\sqrt{-3})$、$\mathbb{Q}(\sqrt{-7})$ 等）。

## 7 小结

- **Minkowski 嵌入** $\sigma: K \to \mathbb{R}^{r_1} \times \mathbb{C}^{r_2} \cong \mathbb{R}^n$：域变成坐标，整数环变成格。
- **协体积** $\mathrm{vol}(\mathbb{R}^n/\sigma(\mathcal{O}_K)) = 2^{-r_2}\sqrt{|d_K|}$：判别式的几何身份。
- **格点定理**：中心对称凸体体积 $> 2^n\,\mathrm{vol}(\Lambda)$ 必含非零格点；Blichfeldt 是它的「连续鸽子笼」。
- **Minkowski 界** $M_K = (\frac{4}{\pi})^{r_2}\frac{n!}{n^n}\sqrt{|d_K|}$ ⟹ 类数有限、$h_K = 1$ 的判别式判据、Hermite 定理。
- 判别式下界 $|d_K| \ge (\frac{\pi}{4})^{2r_2}(\frac{n^n}{n!})^2$：域的「大小」随次数指数增长。

在下一节，我们转入「分歧的度量」：如何用一个理想精确记录扩张里每个素理想分歧的强度——**差积（different）与判别式（discriminant）**，它们把分歧理论的高阶群滤过写成一条算术等式。
