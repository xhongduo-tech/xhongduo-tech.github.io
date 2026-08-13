---
title: Darboux 定理与 Moser 稳定性
date: 2026-08-07
---

# Darboux 定理与 Moser 稳定性

<div class="epigraph">
<p>上帝在辛几何里放了一个定理，让所有辛流形局部上都一模一样——这既是恩赐，也是诅咒。</p>
<footer>—— 佚名讲义（作者不可考）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第3章；Cannas 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从 Darboux 定理开始

黎曼几何里，每条曲线、每张曲面都有曲率，曲率在每一点「拒绝被坐标变换消掉」——这是黎曼几何的局部不变量。辛几何正好相反：**Darboux 定理说，辛流形每一点附近都能找到正则坐标，使辛形式变成标准形式 $\omega_0$。** 辛结构在局部没有任何不变量！这个「平凡」的局部结论，却是整个学科的转折点：它意味着辛几何的问题必须在**整体**层面提出。Moser 稳定性是 Darboux 定理的发动机——它把「找到坐标系」翻译成「构造一个依赖时间的向量场」，用流形上的常微分方程来解决几何问题。这套「Moser 技巧」日后会反复出现（同痕、嵌入、容量），值得彻底吃透。<span class="marginnote">哲学上对比：黎曼几何像「每个点都有纹理的布」，辛几何像「每个点都一样光滑的台球桌」——差异不在局部而在整体。这也是为什么辛几何天生靠近拓扑与代数，而不是分析。</span>

## 1 Darboux 定理的陈述

**Darboux 定理**：设 $(M, \omega)$ 是 $2n$ 维辛流形，$p \in M$ 是任意一点。则存在 $p$ 的开邻域 $U$ 与坐标映射 $\varphi: U \to \mathbb{R}^{2n}$，使得

$$
\varphi(p) = 0, \qquad \varphi^* \omega_0 = \omega|_U
$$

即 $U$ 与 $(\mathbb{R}^{2n}, \omega_0)$ 的一个开子集**辛同胚**。坐标 $(\varphi_1, \dots, \varphi_{2n}) = (q_1, \dots, q_n, p_1, \dots, p_n)$ 满足

$$
\omega|_U = \sum_{i=1}^{n} dq_i \wedge dp_i
$$

这样的坐标正是上一篇说的**正则坐标**。<span class="marginnote">对比微分几何里的「黎曼正规坐标」：黎曼几何只能把度量化成欧氏的「到一阶」，曲率（二阶量）永远留在那里；Darboux 定理能把辛形式化成标准的<strong>精确到零阶</strong>——不需要近似，是完全相等。辛形式没有「曲率」，这就是它的含义。</span>

**推论：辛流形没有局部不变量。** 两个辛流形在任何局部都是辛同胚的；辛几何的全部结构信息都编码在整体拓扑（$[\omega]$ 的上同调类、辛同胚的刚性、嵌入的容量）之中。这是理解整门学科的钥匙。

## 2 为什么 Darboux 能成立：线性层面

Darboux 定理的证明思路分两步：**先线性化，再非线性修正**。

第一步（线性）：在 $p$ 点，切空间 $T_p M$ 是 $2n$ 维辛向量空间。由上一篇的结论，存在辛基 $e_1, \dots, f_n$ 使 $\omega_p$ 化为 $J_0$。选坐标 $x$ 使 $dx_i|_p$ 对偶于这个基，则在 $p$ 处

$$
\omega = \omega_0 + O(|x|)
$$

即在 $p$ 点 $\omega$ 与 $\omega_0$ 的差是一阶小量。**这一步用到的就是辛线性代数——任何辛形式在一点都能化成标准型**，因为反对称双线性形式没有不变量（不像对称形式有特征值）。

第二步（非线性）：如何把「一阶小量」消掉？如果 $\omega$ 和 $\omega_0$ 相差一个「闭的」小量，就可以用 Moser 技巧沿一条形变路径把它推平。这就引出 Moser 稳定性。

## 3 Moser 稳定性与 Moser 技巧

**Moser 稳定性定理**：设 $M$ 是紧致流形，$\omega_t = \omega_0 + d\alpha_t$（$0 \le t \le 1$）是一族**上同调相同**的辛形式。则存在一族光滑同痕 $\varphi_t: M \to M$（$\varphi_0 = \mathrm{id}$），使得

$$
\varphi_t^* \omega_t = \omega_0, \qquad \text{对所有 } t
$$

换句话说，**在同一个上同调类里，辛形式彼此都可以通过同痕互相拉回**——辛结构在紧流形上对上同调类内的小扰动是稳定的。<span class="marginnote">这正是 Darboux 定理的非线性引擎。注意条件 $[\omega_t] = [\omega_0]$（上同调类相同）不可少：辛体积 $\int \omega_t^n$ 是辛同胚不变量，若上同调类变了就不可能拉回。</span>

**Moser 技巧（Moser's trick）** 的证明只要几步，是「用流生成形变」的样板：

- **设定目标**：设 $\varphi_t$ 是由依赖时间的向量场 $X_t$ 生成的流，即 $\frac{d}{dt}\varphi_t = X_t \circ \varphi_t$。拉回的时间导数满足

$$
\frac{d}{dt} \varphi_t^* \omega_t = \varphi_t^* \big( \mathcal{L}_{X_t} \omega_t + \dot{\omega}_t \big)
$$

- **Cartan 公式**：$\mathcal{L}_{X_t} \omega_t = d\iota_{X_t}\omega_t + \iota_{X_t} d\omega_t$。由于 $\omega_t$ 闭（$d\omega_t = 0$），李导数项简化为 $d\iota_{X_t}\omega_t$。
- **目标方程**：我们想要 $\varphi_t^*\omega_t$ 恒等于 $\omega_0$，即其时间导数为零：

$$
d\iota_{X_t}\omega_t + \dot{\omega}_t = 0
$$

- **非退化性救场**：$\dot{\omega}_t = d\alpha_t$（假设），于是需要 $d\iota_{X_t}\omega_t = -d\alpha_t$。取 $\iota_{X_t}\omega_t = -\alpha_t$——这是**代数方程**！由于 $\omega_t$ 非退化，映射 $X \mapsto \iota_X \omega_t$ 是同构，$X_t$ 唯一存在。
- **收官**：积分这个向量场得 $\varphi_t$，取 $t=1$ 完成拉回。

**核心洞见：非退化性把「解偏微分方程」变成「解代数方程」。** 这是辛几何反复出现的魔法——要找向量场，先找 1-形式，再用 $\omega$ 的配对把它变成向量场。哈密顿向量场正是同一个机制。

## 4 公式解析：Moser 技巧的核心方程

**核心公式：**

$$
\frac{d}{dt} \varphi_t^* \omega_t = \varphi_t^* \big( d\iota_{X_t}\omega_t + \dot{\omega}_t \big)
$$

逐项拆解：

- **第一步，拉回的时间导数**：对依赖时间的 $\omega_t$ 求拉回的时间导数，得到 $\varphi_t^*(\mathcal{L}_{X_t}\omega_t) + \varphi_t^*(\dot{\omega}_t)$。这是莱布尼茨法则的流版本——两个来源：$\varphi_t$ 在动（李导数项）、$\omega_t$ 本身在变（$\dot{\omega}_t$ 项）。
- **第二步，Cartan 公式**：$\mathcal{L}_{X}\omega = d(\iota_X \omega) + \iota_X(d\omega)$。对闭形式 $d\omega_t = 0$，第二项消失，只剩 $d\iota_X\omega_t$。**闭性在这里第一次派上用场。**
- **第三步，目标方程**：设 $\varphi_t^*\omega_t = \omega_0$（不依赖 $t$），则左边导数为零，得 $d\iota_X\omega_t + \dot{\omega}_t = 0$。
- **第四步，代数化**：代入 $\dot{\omega}_t = d\alpha_t$，得 $d(\iota_X\omega_t + \alpha_t) = 0$。取 $\iota_X\omega_t = -\alpha_t$ 是充分条件——**这里 $\omega_t$ 的非退化性保证解存在且唯一**。$X_t$ 就从「解一个线性方程组」直接得到，无需解任何微分方程。

**直觉总结：** Moser 技巧 = 把「找同痕」转化为「找 1-形式 $\alpha_t$，再用辛配对翻成向量场 $X_t$，最后积分」。闭性消去李导数中的二阶项，非退化性保证可逆——两个条件各司其职，缺一不可。

## 5 Darboux 定理的证明骨架

用 Moser 稳定性证明 Darboux：<span class="marginnote">完整证明见 McDuff & Salamon 第3章或 Cannas 第8章。这里给出骨架，重点是看清「线性化 + 形变路径」的套路。</span>

1. **取线性坐标**：在 $p$ 附近选坐标 $x$，使 $\omega_p = (\omega_0)_p$。在 $p$ 的凸邻域 $U$ 上定义 $\omega_t = (1-t)\omega_0 + t\omega$（凸组合）。
2. **验证闭与非退化**：$\omega_t$ 闭（两个闭形式组合）。在 $p$ 点 $\omega_t = \omega_0$ 非退化，由紧邻域的连续性，$U$ 缩小后 $\omega_t$ 处处非退化。
3. **上同调相同**：$\omega_t - \omega_0 = t(\omega - \omega_0)$ 在凸邻域 $U$ 上是闭形式，因此**精确**（庞加莱引理：凸开集上的闭形式都是某形式的微分）。存在 $\alpha_t$ 使 $\omega_t - \omega_0 = d\alpha_t$。
4. **Moser 收官**：应用 Moser 技巧，得 $\varphi_t$ 使 $\varphi_1^*\omega_1 = \omega_0$，即 $\varphi_1^*\omega = \omega_0$。把 $\varphi_1^{-1}$ 当作坐标映射就得到 Darboux 坐标。

**庞加莱引理在这里的作用**：凸邻域上没有拓扑障碍，闭形式必精确——这是「局部」能成立的根本原因。一旦离开凸邻域，闭形式未必精确，Darboux 就失效，必须考虑整体问题。

**对比：为什么黎曼几何做不到**。黎曼度量在每点可以化为欧氏的（正规坐标），但**二阶项（曲率张量）无法消去**——曲率是局部不变量。而辛形式在 Darboux 坐标下被化到「精确等于标准型」，一阶、二阶乃至所有阶都被消光。**「Darboux 定理成立」与「曲率不存在」是同一件事的两种说法**——辛流形没有曲率型的局部几何，这是它区别于黎曼几何的根本，也决定了辛几何必须整体化。

## 6 小结

- **Darboux 定理**：辛流形每点附近存在正则坐标，使 $\omega = \sum dq_i \wedge dp_i$；**辛流形没有局部不变量**。
- **对比黎曼**：曲率是黎曼局部不变量，辛形式在局部永远可化成标准型——差异只在整体。
- **Moser 稳定性**：紧流形上同调类内的辛形式族可被同痕拉回；条件是闭性 + 非退化性 + 同调类固定。
- **Moser 技巧三步**：Cartan 公式化简李导数、目标方程化 $d\iota_X\omega = -d\alpha$