---
title: 量子振幅放大与振幅估计
date: 2026-08-07
---

# 量子振幅放大与振幅估计

<div class="epigraph">
<p>振幅放大是比 Grover 搜索更基本、更通用的量子技巧。</p>
<footer>—— 布拉斯（Gilles Brassard）、海尔（Peter Høyer）、莫斯卡（Michele Mosca）与塔普（Alain Tapp）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§6.1–6.3（振幅放大视角）｜ 2026-08-07</p>
</div>

## 为什么从振幅放大开始

Grover 搜索（下一节）通常被当作「平方加速」的范例，但它的本质其实是**量子振幅放大（quantum amplitude amplification）**的一个特例。振幅放大是更通用的框架：给定一个「标记好解」的黑盒，它能把「解的分量振幅」从一个小的 $\sin\theta$ 放大到接近 1——把「采样成功率 $p$」提升到「$O(1/\sqrt{p})$ 次就能找到」。<span class="marginnote">振幅放大出自 Brassard, Høyer, Mosca, Tapp 2002 年的论文 "Quantum Amplitude Amplification and Estimation"（arXiv:quant-ph/0005055）。它把 Grover 从「搜索 $N$ 项」推广到「放大任意态里某个好子空间的振幅」。</span>本节先立起振幅放大的几何框架，再引出它的姊妹算法——**振幅估计（amplitude estimation）**，它用量子计数、蒙特卡洛加速提供了平方级加速。

## 1 从概率放大到振幅放大

经典的概率放大：若某事件发生概率为 $p$，重复尝试 $O(1/p)$ 次几乎必然成功——这是经典蒙特卡洛的朴素想法，代价与 $1/p$ 成正比。

量子振幅放大换了一个思路：**不放大「概率」，而是放大「振幅」。** 设状态 $\lvert\psi\rangle = \cos\theta\lvert g\rangle + \sin\theta\lvert b\rangle$，其中 $\lvert b\rangle$ 是「好」子空间（解），$\lvert g\rangle$ 是「坏」子空间。好消息子空间的概率是 $p = \sin^2\theta$。振幅放大通过反复施加一个「反射」算符 $G$，把 $\theta \to 3\theta \to 5\theta \to \cdots$，约 $O(1/\theta) = O(1/\sqrt{p})$ 次后，$\sin\theta$ 被放大到接近 1。<span class="marginnote">这就是「振幅放大」的名字由来：不是把概率 $p$ 每步线性加，而是把振幅 $\sin\theta$ 的「角度」$\theta$ 每步加一倍——角度线性增长 → 概率二次增长，总代价从 $O(1/p)$ 降到 $O(1/\sqrt p)$，平方加速。</span>关键是这个放大是**相干**的：它发生在叠加态内部，不引入测量，所以可以反复迭代。

## 2 Grover 算符 $G$：两个反射的复合

振幅放大靠**Grover 算符** $G$ 实现，它是两个反射的乘积：

$$
G = (2\lvert\psi\rangle\langle\psi\rvert - I)\,(2\lvert b\rangle\langle b\rvert - I)
$$

其中 $2\lvert b\rangle\langle b\rvert - I$ 是「对好子空间反射」（翻转好分量的相位，即 oracle），$2\lvert\psi\rangle\langle\psi\rvert - I$ 是「对初始态反射」（扩散算符）。<span class="marginnote">在二维平面（由 $\lvert g\rangle$ 与 $\lvert b\rangle$ 张成）里，$G$ 是「两次反射的复合」，而平面上的两次反射 = 一次旋转。旋转角 $2\theta$：每作用一次 $G$，态从角度 $\theta$ 转到 $3\theta$。这就是 Grover 迭代的几何本质。</span>「两个反射」的结构可推广：只要把「好子空间反射」换成任意标记函数，「对初始态反射」换成「对任意起始态反射」，就得到通用振幅放大。

## 3 公式解析：为什么两次反射是旋转

设当前态与好子空间夹角为 $\theta$，在 $\{\lvert g\rangle, \lvert b\rangle\}$ 张成的平面上：

- **第一步，好反射**：$\lvert\psi\rangle \xrightarrow{oracle} \cos\theta\lvert g\rangle - \sin\theta\lvert b\rangle$——好分量的振幅变号（绕 $\lvert g\rangle$ 轴反射）。
- **第二步，初始态反射**：再绕 $\lvert\psi\rangle$ 反射。两次反射的复合 = 绕交点旋转 $2\theta$，把 $\theta$ 变成 $3\theta$。
- **第三步，迭代**：$G^k\lvert\psi\rangle = \sin((2k+1)\theta)\lvert b\rangle + \cdots$。当 $(2k+1)\theta \approx \frac{\pi}{2}$ 时好分量取最大值。<span class="marginnote">几何要点：两次反射绕「初始态方向」与「好方向」的中间轴旋转 $2\theta$，每次旋转把好分量「转出来」更多。迭代次数 $k \approx \frac{\pi}{4\theta} = O(1/\sqrt p)$ 时到顶。</span>

## 4 振幅估计：用相位估计数「好」的次数

如果我们的目标不是「找到好态」，而是「知道 $p$ 有多大」——这就是**振幅估计**。把 Grover 算符 $G$ 当作「相位估计里的酉算符」，因为 $G$ 的本征相位恰与 $\theta$ 挂钩：$G$ 的本征值 $e^{\pm 2i\theta}$ 对应本征相位 $\pm 2\theta/\pi$。用相位估计测出 $\theta$，就得到

$$
\tilde{p} = \sin^2\theta, \qquad \lvert \tilde{p} - p\rvert \le \frac{2\pi\sqrt{p(1-p)}}{M} + \frac{\pi^2}{M^2}
$$

其中 $M = 2^m$ 是相位估计的控制比特数。<span class="marginnote">经典估计概率 $p$ 需要 $O(1/\epsilon^2)$ 次采样（中心极限定理），振幅估计只要 $O(1/\epsilon)$ 次「相位估计」——又是平方加速。它是量子蒙特卡洛（quantum Monte Carlo）加速的根基：把「多次随机采样」压缩成「一次带 QFT 的相位读出」。</span>

- **第一步，连接两套机器**：相位估计要求「酉算符的本征相位」，振幅估计把 $G$ 塞进去，本征相位 $\pm 2\theta$ 藏着的正是 $p = \sin^2\theta$。
- **第二步，读出角度**：相位估计输出 $\theta$ 的二进制近似，换算成 $\tilde p$。
- **第三步，误差控制**：误差随 $M$（控制比特数）线性下降，$M$ 越大越准；加 $O(1/\epsilon)$ 比特即可达到精度 $\epsilon$。<span class="marginnote">应用实例：量子计数（quantum counting）就是「Grover + 相位估计」——不直接找解，而是数「有几个解」。第八篇里 Grover 的计数应用、以及量子机器学习里求期望值的量子蒙特卡洛，都从这里发芽。</span>

**辨析｜易错点：** 振幅估计的输出 $\tilde p$ 可能落在 $[0,1]$ 之外或与真实值有微小偏差，需要做「投影 + 二次采样」的经典后处理。另外，振幅放大**假设起始态已知**（$\lvert\psi\rangle$ 可制备），若起始态本身是混合的，放大效率会打折——这是「振幅放大对相干性要求高」的另一种表述。

## 5 应用版图

振幅放大与估计是「万能放大器」，几乎无处不用：

- **Grover 搜索**：$p = M/N$（$M$ 个解），$\sqrt{N/M}$ 次找到解——下一节的主角。
- **量子计数**：用振幅估计数解的个数，$O(\sqrt N)$ 次查询替代经典 $O(N)$。
- **量子蒙特卡洛**：用振幅估计把期望值计算的采样代价从 $O(1/\epsilon^2)$ 压到 $O(1/\epsilon)$，是金融、物理模拟里量子加速的主要来源。<span class="marginnote">这条「振幅估计 → 量子蒙特卡洛 → 金融定价」的链路，是当下「有实用价值的量子加速」里最被看好的几条之一，很多 NISQ 公司的主打应用就在这条线上。</span>

## 6 小结

- **振幅放大**：把「好子空间振幅」的**角度**每步增 $2\theta$，$O(1/\sqrt p)$ 次即可接近确定成功——对成功概率的平方加速。
- **Grover 算符** $G$ = 好反射 × 初始态反射 = 二维平面上的旋转。
- **振幅估计**：用相位估计读出 $G$ 的本征相位 → 推出 $p$，误差 $O(1/M)$。
- 应用：Grover 搜索、量子计数、量子蒙特卡洛（$O(1/\epsilon)$ 替代 $O(1/\epsilon^2)$）。

在下一节，我们回到最朴素的问题设定——**无结构搜索问题与经典下界**，为 Grover 算法铺好「$\sqrt{N}$ 最优」的论证基础。
