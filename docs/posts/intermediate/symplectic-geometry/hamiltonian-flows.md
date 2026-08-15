---
title: 哈密顿流与辛同痕
date: 2026-08-07
---

# 哈密顿流与辛同痕

<div class="epigraph">
<p>时间不逝去，是我们在流逝；在相空间里，时间就是一个哈密顿向量场的积分。</p>
<footer>—— 让 · 迪厄多内（Jean Dieudonné，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ Cannas 第8章；McDuff & Salamon 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从哈密顿流开始

上一篇的哈密顿向量场是「无穷小」的：它告诉系统瞬间往哪走。这一篇把它**积分**成有限时间的运动——哈密顿流。把「无穷小保辛」积分起来，得到的每一步映射都自动是**辛同胚**。这带来一个漂亮的对应：从函数（哈密顿量）到一族保辛同胚（流），从「李代数」到「李群」。更深刻的是，如果我们允许哈密顿量依赖时间，就得到**哈密顿同痕**——它是辛几何里「同伦」的精确版本，也是后面 Hofer 几何、Floer 同调、Arnold 猜想里「考虑所有哈密顿函数」的舞台。<span class="marginnote">在课程地图上：这一篇把上一节的向量场「积」成流，下一篇《可积系统》研究哈密顿流特别规则的情形，而第3篇《辛同态群》研究全体辛同胚/哈密顿同胚构成的无穷维群。</span>

## 1 哈密顿流与保辛

**哈密顿流（Hamiltonian flow）**：设 $H \in C^\infty(M)$，其哈密顿向量场 $X_H$ 的流记作 $\phi_H^t$，即

$$
\frac{d}{dt}\phi_H^t(p) = X_H(\phi_H^t(p)), \qquad \phi_H^0 = \mathrm{id}
$$

**定理（流保持辛结构）**：对每个固定时刻 $t$，$\phi_H^t$ 是辛同胚：$(\phi_H^t)^*\omega = \omega$。

**证明（Cartan 公式三行）**：

$$
\frac{d}{dt}(\phi_H^t)^*\omega = (\phi_H^t)^* \mathcal{L}_{X_H}\omega = (\phi_H^t)^*(d\iota_{X_H}\omega + \iota_{X_H}d\omega) = (\phi_H^t)^* d(dH) = 0
$$

第一等号是李导数与拉回的链式法则；第二等号是 **Cartan 公式** $\mathcal{L}_X = d\iota_X + \iota_X d$；第三等号用了 $\iota_{X_H}\omega = dH$（哈密顿向量场定义）与 $d\omega = 0$（辛形式闭）；末等号 $d^2 = 0$。<span class="marginnote">这大概是辛几何里最重要的一行计算：<strong>闭性 + 非退化性 + Cartan 公式 = 流保辛</strong>。它把「力学演化」与「几何保持」绑成同一个结论。</span>

**推论（Liouville 定理）**：$\phi_H^t$ 保辛体积 $\omega^n/n!$，即保相空间体积——统计力学里相体积守恒的几何证明。

## 2 哈密顿同痕

真实物理的哈密顿量常依赖时间（如含时外力），所以更自然的对象是：

**哈密顿同痕（Hamiltonian isotopy）**：一族微分同胚 $\phi_t: M \to M$（$0 \le t \le 1$，$\phi_0 = \mathrm{id}$），存在依赖时间的函数 $H_t$ 使

$$
\frac{d}{dt}\phi_t = X_{H_t} \circ \phi_t
$$

其中 $X_{H_t}$ 由 $\iota_{X_{H_t}}\omega = dH_t$ 定义。称 $\phi_t$ 由哈密顿量族 $H_t$ **生成**，$\phi_1$ 是**哈密顿同胚（Hamiltonian diffeomorphism）**。全体哈密顿同胚构成**哈密顿同态群**

$$
\mathrm{Ham}(M, \omega) \subset \mathrm{Symp}(M, \omega)
$$

的**辛同态群**（symplectomorphism group）是 $\mathrm{Symp}(M,\omega)$，哈密顿同胚是其中「能被哈密顿流连到恒等」的那部分。<span class="marginnote">类比：$\mathrm{Symp}(M,\omega)$ 里的元素像「所有保持度量的映射」，$\mathrm{Ham}(M,\omega)$ 像「能通过连续形变从恒等到达」的子群。在紧致辛流形上，$\mathrm{Ham}$ 是 $\mathrm{Symp}$ 的<strong>正规子群</strong>，且通常严格小——辛同胚未必哈密顿。</span>

**关键事实**：
- $\mathrm{Ham}(M,\omega)$ 是 $\mathrm{Symp}(M,\omega)$ 的**正规子群**（对 $\mathrm{Symp}$ 的共轭封闭）。
- 每个 $\mathrm{Ham}$ 元素都由某个 $H_t$ 生成（$H_t$ 可整体定义），这是「哈密顿」与「一般辛」的分界线。
- 若 $H^1(M; \mathbb{R}) = 0$（如 $S^{2n}$、$\mathbb{CP}^n$），则 $\mathrm{Ham} = \mathrm{Symp}^0$（恒等分支全体）；若有非平凡一维上同调（如环面 $T^{2n}$），则 $\mathrm{Symp}^0$ 比 $\mathrm{Ham}$ 大——存在非哈密顿的辛同痕。

## 3 Flux 同态：度量「离哈密顿有多远」

如何精确判定一个辛同痕是否哈密顿？答案是 **flux 同态**。

给定辛同痕 $\phi_t$（$\phi_0 = \mathrm{id}$），定义**flux**

$$
\mathrm{Flux}(\{\phi_t\}) = \int_0^1 [\iota_{X_t}\omega]\, dt \in H^1(M; \mathbb{R})
$$

其中 $X_t = \frac{d}{dt}\phi_t \circ \phi_t^{-1}$ 是生成向量场。<span class="marginnote">直觉：$\iota_{X_t}\omega$ 是一个 1-形式，沿时间积分给出一个上同调类。若 $\phi_t$ 是哈密顿的，则 $\iota_{X_t}\omega = dH_t$ 精确，flux = 0。所以 <strong>flux 度量「非哈密顿性」</strong>——它是「沿同痕累积的拓扑障碍」。</span>

**定理**：辛同痕 $\phi_t$ 是哈密顿同痕当且仅当 $\mathrm{Flux}(\{\phi_t\}) = 0$。且 flux 诱导一个同态

$$
\mathrm{Flux}: \pi_1(\mathrm{Symp}(M,\omega)) \longrightarrow H^1(M; \mathbb{R})
$$

**推论**：若 $\mathrm{Flux}$ 的核是 $\pi_1(\mathrm{Ham})$，则 $\mathrm{Symp}^0/\mathrm{Ham} \cong \mathrm{im}(\mathrm{Flux}) / \Gamma$（$\Gamma$ 是某个格）。这个商群叫**辛映射类群/周期**，是辛拓扑的精细不变量。

**辨析｜易错点：** 不要把「辛同痕」与「哈密顿同痕」混用。辛同痕是每步都保辛的同痕（等价地 $X_t$ 使 $\mathcal{L}_{X_t}\omega = 0$，即 $\iota_{X_t}\omega$ 是**闭**的）；哈密顿同痕额外要求 $\iota_{X_t}\omega$ 是**精确**的。闭 ⟺ 精确 的差异就是 $H^1$——所以「非哈密顿的辛同痕」恰在 $H^1(M;\mathbb{R}) \neq 0$ 时出现。

## 4 公式解析：流的保辛计算

**核心公式（保辛的完整链条）：**

$$
\frac{d}{dt}(\phi_H^t)^*\omega = (\phi_H^t)^* \left( d\iota_{X_H}\omega + \iota_{X_H}d\omega \right) = 0
$$

拆解：

- **第一步，拉回的时间导数**：对固定 $\omega$，$\frac{d}{dt}(\phi^t)^*\omega = (\phi^t)^*\mathcal{L}_X\omega$。这是「拉回与流互换」的标准公式——把时间导数穿过拉回，变成李导数。
- **第二步，Cartan 公式**：$\mathcal{L}_X\omega = d\iota_X\omega + \iota_X d\omega$。这是微分形式的「魔术公式」，把李导数拆成「外微分 × 内乘」的两项。
- **第三步，两项分别消失**：$\iota_X\omega = dH$（哈密顿向量场定义），所以 $d\iota_X\omega = d^2H = 0$；$\iota_X d\omega = \iota_X 0 = 0$（辛形式闭）。
- **第四步，结论**：李导数为零，拉回不随时间变化；在 $t=0$ 时 $(\phi^0)^*\omega = \omega$，故恒等。**流保辛。**

**直觉总结：** 这条证明把「动力系统保几何」归结为「$d^2 = 0$」——哈密顿向量场定义让第一项为零，闭性让第二项为零。**辛几何的动力学完全由「闭形式的精确性」控制**，这个主题会贯穿到 Floer 同调。

## 5 可观察量随时间的演化

哈密顿流的另一个视角：可观察量 $f$ 沿流如何变？记 $f_t = f \circ \phi_H^t$，

$$
\frac{d}{dt} f_t = \phi_H^{t*} (\mathcal{L}_{X_H} f) = \phi_H^{t*} \{H, f\}
$$

所以在初始时刻 $\frac{df}{dt} = \{H, f\}$——**$f$ 守恒当且仅当 $\{H, f\} = 0$**（$f$ 与哈密顿量 Poisson 对易）。这统一了「守恒量」的概念：守恒量就是「与 $H$ 对易的函数」。Noether 定理的辛版本随之而来：每个单参数辛对称群生成一个守恒量。<span class="marginnote">注意符号：运动方程通常写作 $\dot{f} = \{f, H\}$（某些教材用 $\{H, f\}$）。两种约定差一个整体符号，只要 $X_H$ 定义一致即可。物理书惯用 $\dot{f} = \{f,H\}$，几何书惯用 $X_H(f) = \{H,f\}$——阅读时注意作者约定。</span>

## 6 小结

- **哈密顿流** $\phi_H^t$ 积分哈密顿向量场；**Cartan 公式一行证明它保辛**——闭性 + 非退化 + $d^2 = 0$。
- **Liouville 定理**：哈密顿流保相体积（辛体积）——统计力学的几何基础。
- **哈密顿同痕**：允许含时 $H_t$，生成元为 $\phi_1$；$\mathrm{Ham}(M,\omega) \subset \mathrm{Symp}(M,\omega)$ 是正规子群。
- **Flux 同态**：$\mathrm{Flux}(\phi_t) = \int_0^1 [\iota_{X_t}\omega]\,dt \in H^1$；为零当且仅当同痕是哈密顿的。
- **守恒量 = 与 $H$ Poisson 对易的函数**：$\{H, f\} = 0 \iff f$ 沿哈密顿流不变；Noether 定理的辛版本。
- **Flux 的意义**：$\mathrm{Flux} = 0$ 区分哈密顿同痕与一般辛同痕——「非哈密顿性」由 $H^1$ 检测。

在下一节，我们将研究哈密顿流特别规整的情形：**可积系统与 Liouville-Arnold 定理**——当守恒量足够多，相空间被纤维化成环面，哈密顿流变成环面上的直线运动。从「流保辛」到「环面纤维化」，可积性正是「流几何最规整」的名字。