---
title: 对数 Sobolev 不等式引论
date: 2026-08-07
---

# 对数 Sobolev 不等式引论

<div class="epigraph">
<p>信息的耗散，是物理与概率共同的语法。</p>
<footer>—— 埃利奥特 · 利布（Elliott Lieb）</footer>
</div>

<div class="article-byline">
<p>第二级 · 马尔可夫链与混合时间 ｜ Levin, Peres &amp; Wilmer《Markov Chains and Mixing Times》Ch. 13 与 Diaconis–Saloff-Coste 系列 ｜ 2026-08-07</p>
</div>

## 为什么从对数 Sobolev 不等式开始

谱隙给出了混合时间的一个好上界，但它有一个瑕疵：对某些链（如顶随机洗牌），谱隙极小、松弛时间极大，混合却很快——**谱隙低估了真实的混合速度**。原因是谱隙对应的 $L^2$ 范数放大因子 $\log(1/\pi_{\min})$ 在状态空间巨大时过于保守。<span class="marginnote">顶随机洗牌的谱隙约为 $1/52$，松弛时间 52，但混合时间只有 $52\log 52$ 而非 $52\cdot\log(4/\pi_{\min})$（$\pi_{\min} = 1/52!$ 天文数字般小）。谱隙上界完全失效，而真实混合要快得多。</span>**对数 Sobolev 不等式（log-Sobolev inequality）**是谱隙的精细替代：它把 $L^2$ 换成带对数的熵，给出**无 $\pi_{\min}$ 项**、直接以熵衰减刻画收敛的速率。它源自 Gross 1975 年的超压缩性研究，由 Diaconis 与 Saloff-Coste 引入马尔可夫链理论，如今是证明「均匀混合」的利器。

## 1 从 Poincaré 到对数 Sobolev

先摆出两类不等式并排对照。**Poincaré 不等式**（即谱隙的变分定义，上一节）：

$$
\operatorname{Var}_\pi(f) \leq \frac{1}{\gamma} \, \mathcal{E}(f, f)
$$

等价于谱隙 $\gamma$。**对数 Sobolev 不等式**则把方差换成**熵**：

$$
\operatorname{Ent}_\pi(f^2) \leq \frac{2}{\rho} \, \mathcal{E}(f, f)
$$

其中 $\operatorname{Ent}_\pi(g) = \mathbb{E}_\pi[g \log g] - \mathbb{E}_\pi[g]\log\mathbb{E}_\pi[g]$ 是 $g$ 关于 $\pi$ 的熵。使得不等式成立的最大常数 $\rho$ 称为**对数 Sobolev 常数（log-Sobolev constant）**。<span class="marginnote">为什么用 $f^2$？因为熵对非负函数定义自然，而 $f^2$ 保证非负；这也是为什么推导收敛时熵在 $L^2$ 的「平方」层面工作。</span>

对数 Sobolev 常数与谱隙有确定的大小关系：

$$
\rho \leq \gamma
$$

即对数 Sobolev 不等式**比 Poincaré 更紧**——更大的分母 $\rho$ 意味着更慢的衰减保证，但它摆脱了对 $\pi_{\min}$ 的依赖。<span class="marginnote">关系 $\rho \leq \gamma$ 的直觉：取 $f \equiv 1 + \varepsilon \phi_1$（接近特征函数的小扰动），熵与方差在同一量级，但熵额外的对数项放大较小扰动，所以常数更小。物理上，熵比方差更敏感，自然给出更慢的保证。</span>

## 2 熵衰减与混合时间

对数 Sobolev 不等式的威力在于它给出**熵的直接指数衰减**：

$$
\operatorname{Ent}_\pi(P^n f) \leq e^{-\rho n} \operatorname{Ent}_\pi(f)
$$

推导与谱方法同构：把熵替换成 Dirichlet 形式（用对数 Sobolev 不等式），再对 $n$ 递推求和，就得到几何衰减。<span class="marginnote">这个「熵 → Dirichlet → 递推」三步是标准模板：每推进一步，熵至少乘 $e^{-\rho}$。对比谱方法的「$L^2$ → 逐模式」，熵方法避开特征值分解，直接作用在概率密度上。</span>

**关键优势**：熵衰减可以直接翻译成全变差距离上界，且**不需要 $\pi_{\min}$**：

$$
\lVert \mu_n - \pi \rVert_{\mathrm{TV}}^2 \leq \frac{1}{2}\operatorname{Ent}_\pi\left(\frac{\mu_0}{\pi}\right) e^{-\rho n}
$$

这里用到 Pinsker 不等式 $\lVert\mu-\nu\rVert_{\mathrm{TV}}^2 \leq \tfrac12 \mathrm{KL}(\mu \| \nu)$。右边没有 $\pi_{\min}$，所以即使状态空间有 $52!$ 个元素，常数也不爆炸——这正是顶随机洗牌那种「谱隙失效但对数 Sobolev 有效」场景的解药。

## 3 公式解析：从熵衰减到混合时间

把「对数 Sobolev 常数如何变成混合时间上界」拆成四步：

$$
\lVert \mu_n - \pi \rVert_{\mathrm{TV}}^2 \leq \frac{1}{2}\operatorname{Ent}_\pi\left(\frac{\mu_0}{\pi}\right) e^{-\rho n}, \qquad \operatorname{Ent}_\pi\left(\frac{\mu_0}{\pi}\right) \leq \log\frac{1}{\pi_{\min}}
$$

- **第一步，Pinsker 不等式**：$\lVert\mu-\pi\rVert_{\mathrm{TV}}^2 \leq \tfrac12 \mathrm{KL}(\mu \| \pi)$。而 KL 散度 $\mathrm{KL}(\mu\|\pi) = \operatorname{Ent}_\pi(\mu/\pi)$，于是距离被熵控制。
- **第二步，熵的指数衰减**：用对数 Sobolev 不等式对每一步做递推，得 $\operatorname{Ent}_\pi(f P^n) \leq e^{-\rho n}\operatorname{Ent}_\pi(f)$，其中 $f = \mu_0/\pi$。
- **第三步，熵的初值上界**：$\operatorname{Ent}_\pi(f) \leq \log(1/\pi_{\min})$——最坏情况是 $\mu_0$ 集中在概率最小的状态上。这一步出现 $\pi_{\min}$，但它只在**初值**里出现一次，不像谱隙那样乘在整个 $n$ 上。
- **第四步，解 $n$**：令右边 $\leq \varepsilon^2$，得

$$
t_{\mathrm{mix}}(\varepsilon) \leq \frac{1}{\rho}\left(\log\log\frac{1}{\pi_{\min}} + \log\frac{1}{2\varepsilon^2}\right)
$$

注意 $\log\log(1/\pi_{\min})$ 是**双重对数**——即使 $\pi_{\min}$ 小到 $10^{-100}$，它也只有 $\approx 5$ 量级。这是对数 Sobolev 方法压倒谱隙方法的决定性优势。

## 4 超压缩性与 Tensorization

对数 Sobolev 不等式还有两个深刻的性质，使它成为「乘积空间」的天然工具。

**Tensorization（张量化）**：若每个因子 $i$ 上的链满足对数 Sobolev 常数 $\rho_i$，则**乘积链**的对数 Sobolev 常数满足

$$
\rho_{\mathrm{prod}} \geq \min_i \rho_i
$$

换句话说，对数 Sobolev 常数**不会因张量积而退化**。<span class="marginnote">对比谱隙：乘积链的谱隙是 $\min_i \gamma_i$，同样取最小；但对数 Sobolev 的常数在需要时可以通过「逐坐标更新」的 Glauber 动力学保持。这个性质让「高维乘积空间上的链」可以逐坐标分析再拼合——Ising 模型高维分析的入口。</span>

**超压缩性（hypercontractivity）**：对数 Sobolev 不等式等价于半群 $T_t = e^{-t(I-P)}$ 的范数压缩性质：对 $t \geq \tfrac12\log(q-1)$，$\lVert T_t f \rVert_q \leq \lVert f \rVert_2$。这是 Gross 原始问题的形式化——它说明「信息随时间扩散、$L^p$ 范数互相压缩」的速度恰由 $\rho$ 控制。

## 5 例子：双点空间与顶随机洗牌

**例一：双点空间。** $\Omega = \{0,1\}$，$\pi = (1/2, 1/2)$，转移为等概率翻转或不动。此时对数 Sobolev 常数 $\rho$ 与谱隙 $\gamma$ 相同（$\rho = \gamma = 1$），因为熵与方差在双点空间上等价——这是**最紧的情形**。

**例二：顶随机洗牌。** 谱隙 $\gamma \approx 1/52$，但已知 $\rho \geq c/52$，两者同阶。<span class="marginnote">这看似没改善，但关键在于上界公式：谱方法要乘 $\log(4/\pi_{\min}) \sim \log(4 \cdot 52!)$ 巨大，对数 Sobolev 只乘 $\log\log(52!) \approx 3$，最终给出与真实混合时间 $52\log 52$ 同阶的结果。方法学价值远大于常数优化。</span>

**例三：二项分布与洗牌族。** 对 $\mathrm{Bin}(n, 1/2)$ 对应的双点乘积，张量化给出 $\rho \geq c/n$，从而 $t_{\mathrm{mix}} = O(n\log n)$——正是随机游走混合时间 $n \log n$ 的谱系来源。

## 6 辨析｜易错点：$\rho$ 小不代表混合慢

**辨析｜易错点：** 初学者容易认为「$\rho$ 小 ⇒ 混合慢」。实际上 $\rho$ 只进入上界公式的**系数**，而谱方法的上界还要乘 $\log(1/\pi_{\min})$。$\rho$ 略小于 $\gamma$ 换来的是公式里去掉灾难性的对数项——这是「用稍弱的不等式换掉坏常数」的经典交换。

**辨析｜易错点：** 对数 Sobolev 不等式的验证是**所有 $f$** 都要成立，不能只验特征函数。它本质上是对全函数空间的约束。实践中常用 Tensorization + 双点空间的验证逐坐标拼装，而不是直接对整条链证明。

## 7 小结

- **对数 Sobolev 不等式** $\operatorname{Ent}_\pi(f^2) \leq \tfrac{2}{\rho}\mathcal{E}(f,f)$ 是谱隙的精细化，常数满足 $\rho \leq \gamma$。
- 它给出**熵的指数衰减**，进而给出无 $\pi_{\min}$ 灾难因子的混合时间上界（只需 $\log\log(1/\pi_{\min})$）。
- **Tensorization** 让乘积链的对数 Sobolev 常数取各因子最小，是高维分析的入口；**超压缩性**是它的函数分析等价形式。
- 顶随机洗牌等「谱隙失效」场景，对数 Sobolev 方法仍然给出正确量级。
- **Pinsker 不等式** $\lVert\mu-\nu\rVert_{\mathrm{TV}}^2 \leq \tfrac12 \mathrm{KL}(\mu\|\nu)$ 是「熵 → 全变差」的桥梁，把熵衰减翻译成混合时间上界。
- 求 $\rho$ 通常比求 $\gamma$ 更难，但**张量化（tensorization）**把乘积链化归为各因子 $\rho_i$ 取最小，是高维分析的入口。
- **超压缩性**是对数 Sobolev 不等式的函数分析等价形式，把概率结论与 $L^p$ 范数压缩相连。
- **易错点**：$\rho$ 小不直接等于混合慢——它只进入上界公式的系数，换掉的是 $\log(1/\pi_{\min})$ 灾难因子。
- **应用链**：Pinsker 不等式把熵衰减翻译成全变差距离，$t_{\mathrm{mix}}$ 上界只需 $\log\log(1/\pi_{\min})$，状态空间 $52!$ 也不爆炸。
- **对照谱隙**：谱方法要乘 $\log(4/\pi_{\min})$，对数 Sobolev 只乘双重对数——这是顶随机洗牌场景的决定性优势。

在下一节，我们将把这些抽象常数放回具体模型——用**洗牌模型**检验耦合、谱隙与对数 Sobolev 三类工具各自的用武之地。