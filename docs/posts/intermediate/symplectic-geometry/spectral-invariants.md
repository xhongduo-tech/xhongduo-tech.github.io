---
title: 哈密顿流与谱不变量
date: 2026-08-07
---

# 哈密顿流与谱不变量

<div class="epigraph">
<p>谱不变量把 Floer 同调的滤波信息变成一族数值不变量——它们是辛世界的特征值。</p>
<footer>—— 尤里 · 谢廖金（Yuriy Eliashberg）与列昂尼德 · 波尔特洛维奇（Leonid Polterovich）教学传统</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第13章 ｜ 2026-08-07</p>
</div>

## 为什么从谱不变量开始

Floer 同调给出「周期轨道有多少」，但还藏着更细的信息：每条轨道有一个**作用值**（作用泛函的临界值）。对 Floer 同调施加**作用滤波**——只看作用值低于某个阈值的部分——就得到一族随阈值变化的中介同调。从这些中介同调可以提取**谱不变量（spectral invariants）**：对每个「同调类」$\sigma$ 与每个哈密顿量 $H$，一个数 $c_\sigma(H)$，度量「$H$ 的流需要多大作用才能实现 $\sigma$」。这一篇讲谱不变量的构造、性质（单调性、谱性、三角不等式），以及它们如何给出 Hofer 距离的下界、证明 Hofer 度量的非退化、并构造 Entov-Polterovich 拟态。<span class="marginnote">在课程地图上：谱不变量是前一篇 Floer 同调「滤波化」的产物，是 Hofer 几何（第3篇）的定量武器，也是通往 Calabi 拟态与镜面对称谱理论的门户。</span>

## 1 作用滤波与中介同调

回想 Floer 链复形 $CF_*^H$：生成元是周期轨道 $\gamma$，每条的**作用值**是

$$
\mathcal{A}_H(\gamma) = -\int_{D^2} \bar\gamma^*\omega + \int_0^1 H(t, \gamma(t)) dt
$$

对阈值 $\lambda$，考虑「作用值 $\lt  \lambda$」的生成元生成的子复形 $CF_*^H(\lambda)$（微分只连接作用递减的轨迹，故子复形封闭）。它的同调

$$
HF_*^H(\lambda) := H_*(CF_*^H(\lambda))
$$

叫**中介同调（filtered Floer homology）**。当 $\lambda \to \infty$ 时 $HF_*^H(\lambda) \to HF_*^H \cong QH_*$。<span class="marginnote">作用滤波的直觉：周期轨道按「能量」排队，中介同调只看「低能量部分」。随 $\lambda$ 升高，中介同调「长」出新的类——谱不变量记录「某个类在哪个 $\lambda$ 出现」。</span>

**关键性质**：对 $H \le K$（逐点 $H_t \le K_t$），有**单调性** $HF_*^H(\lambda) \to HF_*^K(\lambda)$（比较同态）。这使中介同调随 $H$ 单调变化——谱不变量就是这种单调性的数值化。

## 2 谱不变量

**谱不变量（spectral invariant）**：选一个「基本类」$\sigma \in QH_*(M)$（如单位类或点类），定义

$$
c_\sigma(H) = \inf\{ \lambda : \sigma \in \mathrm{im}\big( HF_*^H(\lambda) \to HF_*^H \big) \}
$$

即**「$\sigma$ 首次出现所需的最小作用阈值」**。<span class="marginnote">对单位类 $\sigma = [M]$（或点类），$c_\sigma(H)$ 是「把整个流形'制造'出来需要的最小作用」。对 $\sigma = [\text{pt}]$，它度量「最小能量轨道」。不同 $\sigma$ 给出不同谱不变量——一族特征值。</span>

**谱不变量的四大性质**（对单位类 $\sigma$）：

1. **单调性**：$H \le K \Rightarrow c(H) \le c(K)$；
2. **正规化**：$c(0) = 0$，$c(H + c) = c(H) + c$（加常数）；
3. **谱性**：$c(H) \in \mathrm{Spec}(H)$（是某条轨道的作用值——「谱」的来由）；
4. **三角不等式**：$c(H \# K) \le c(H) + c(K)$（对复合流），其中 $H\#K$ 是哈密顿量的合成。

**谱范数（spectral norm）**：

$$
\gamma(\phi) := c(\bar\phi H) + c(\phi H)
$$

其中 $H$ 生成 $\phi$，$\bar H$ 是反向流。$\gamma$ 满足**谱范数性质**：$\gamma(\phi) \ge 0$、$\gamma(\phi) = \gamma(\phi^{-1})$、$\gamma(\phi\psi) \le \gamma(\phi) + \gamma(\psi)$。

## 3 谱不变量与 Hofer 距离

谱不变量最强大的应用是约束 Hofer 几何：

**谱不等式**：对生成 $\phi$ 的哈密顿量 $H$，

$$
c(H) \le \|H\| \quad \text{（Hofer 范数）}, \qquad \gamma(\phi) \le d_H(\mathrm{id}, \phi)
$$

**谱范数是 Hofer 距离的下界**：$d_H(\mathrm{id}, \phi) \ge \gamma(\phi)$。

**Hofer 度量的非退化（谱证明）**：设 $\phi \neq \mathrm{id}$。存在小开集 $U$ 使 $\phi$ 把它移开。用「局部谱不变量」$c_U$（对 $U$ 内的单位类），有 $c_U(\phi) > 0$。由谱不等式，$d_H(\mathrm{id},\phi) \ge c_U(\phi) > 0$——**非平凡哈密顿同胚需要正能量**。这是上一篇「$d_H$ 是度量」中「非退化」部分的谱证明（比容量版本更直接）。<span class="marginnote">谱不变量把「能量-容量不等式」改进为「能量-谱不等式」：容量给出「移开小球至少花 $c(U)$」，谱给出「实现同调类至少花 $c_\sigma(H)$」。后者更细——因为谱依赖 $\sigma$，可以逐类度量。</span>

**能量-容量不等式（谱版本）**：对开集 $U$ 与「支撑在 $U$ 内」的哈密顿量 $H$，$c_U(H) \le c(U)$——**谱上界由容量给出**。这与上一不等式合起来：

$$
c(U) \le d_H(\mathrm{id}, \phi) \le \ldots
$$

给出「移开 $U$ 所需能量至少 $c(U)$」的精确化。

## 4 公式解析：谱不变量与三角不等式

**核心公式（谱范数与 Hofer 范数的下界）：**

$$
\gamma(\phi) = c(\phi) + c(\phi^{-1}) \le \|H\| = d_H(\mathrm{id}, \phi) \text{ 的生成元}
$$

拆解：

- **第一步，分解 $c(H)$**：$c(H) \le \|H\|$ 来自「$H \le \max H_t$ 与 $H \ge \min H_t$」夹逼：把 $H$ 写成「常数部分 + 振荡部分」，单调性给出 $c(H)$ 被振荡范数控制。
- **第二步，加反向**：$\gamma(\phi) = c(H\#\bar H) \le c(H) + c(\bar H)$（三角不等式）$\le \|H\| + \|\bar H\| = 2\|H\|$? 不对——$c(\bar H) = -c(H)$（正规化 + 反向），所以 $\gamma(\phi) = c(H) + c(\bar H) = c(H) - c(H)$? 这里需要小心约定。

让我理清：$\gamma(\phi) := c(\phi) + c(\phi^{-1})$，其中 $c(\phi) := c(H)$（$H$ 生成 $\phi$）。由于 $c(H) + c(\bar H) \ge c(H \# \bar H) = c(0) = 0$，得 $c(\bar H) \ge -c(H)$。所以 $\gamma(\phi) = c(H) + c(\bar H) \ge 0$。而 $c(\bar H) \le \|\bar H\| = \|H\|$、$c(H) \le \|H\|$，故 $\gamma(\phi) \le 2\|H\|$——**谱范数不超过 Hofer 范数的两倍（对任意生成元）**。取下确界：$\gamma(\phi) \le d_H(\mathrm{id},\phi)$。

- **第三步，下界方向**：$c_U(\phi) > 0$ 对非平凡 $\phi$（谱性 + 局部化），所以 $\gamma(\phi) > 0$——非退化。
- **第四步，结论**：谱范数夹在中间：$0 \lt  \gamma(\phi) \le d_H(\mathrm{id},\phi)$。**Hofer 距离 ≥ 谱范数 ≥ 0**，谱不变量是「从下方测量 Hofer 距离」的尺子。

**直觉总结：** 谱不变量把「能量」分配到「同调类」上——每个类有一个「谱值」。Hofer 距离于是被谱下界「钉住」：要想把 $\phi$ 造出来，至少要花 $\gamma(\phi)$ 的能量。

## 5 Entov-Polterovich 拟态与 Calabi

谱不变量还能**代数化**成拟态：

**定理（Entov-Polterovich）**：在满足「可微谱」条件的辛流形（如 $\mathbb{CP}^n$、$S^2$）上，对单位类 $\sigma$，谱不变量 $c_\sigma$ 诱导一个**Calabi 拟态**

$$
\mu: \mathrm{Ham}(M,\omega) \longrightarrow \mathbb{R}
$$

满足：$\mu(\phi\psi) = \mu(\phi) + \mu(\psi)$ 当 $\phi, \psi$ 对易（拟态性质）、$\mu(\phi^n) = n\mu(\phi)$、以及 **Lipschitz 性** $|\mu(\phi)| \le d_H(\mathrm{id}, \phi)$。**拟态把 Hofer 距离的「大尺度」变成数值**，并证明 $S^2$ 上「大旋转」需要大能量。<span class="marginnote">Calabi 拟态的构造：$\mu(\phi) = c_\sigma(H) + c_\sigma(\bar H)$ 型组合（或用「谱范数的无穷次根」）。它的存在说明 $\mathrm{Ham}(S^2)$ 的 Hofer 直径有限但「几何不平凡」——$S^2$ 与 $T^2$（直径无穷）的差异被拟态捕捉。</span>

**应用**：拟态给出「带符号的能量」——区分「顺时针 vs 逆时针」的旋转花费；也用于证明嵌入/填充障碍（与 ECH 谱联动）。

**辨析｜易错点：** 谱不变量**依赖 $\sigma$ 与系数环**——不同 $\sigma$ 给不同谱；Novikov 环下谱取「有理值/实值」需选「谱区间」。另外谱不变量对 $H$ 是**非光滑**的（只在谱值处有「台阶」），所以「谱函数」不是可微函数——它是「阶梯函数 + 单调」的奇异对象，分析时要小心。

## 6 小结

- **中介同调 $HF^H(\lambda)$**：作用滤波下的 Floer 同调，随 $\lambda$ 单调「生长」。
- **谱不变量 $c_\sigma(H)$**：$\sigma$ 首次出现的最小作用阈值；单调、正规、谱性、三角不等式。
- **谱范数 $\gamma(\phi)$**：$c(\phi) + c(\phi^{-1})$，是 Hofer 距离的下界。
- **非退化证明**：谱下界 $> 0$ 给出 $d_H$