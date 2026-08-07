---
title: 多次解与部分解的搜索
date: 2026-08-07
---

# 多次解与部分解的搜索

<div class="epigraph">
<p>当答案不止一个时，搜索的艺术在于知道何时停下。</p>
<footer>—— 洛弗 · 格罗弗（Lov Grover）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§6.1.4 ｜ 2026-08-07</p>
</div>

## 为什么从多次解开始

前几节的 Grover 都假设**恰有一个解**。但真实问题里解的个数 $M$ 往往未知：SAT 的可行赋值可能有几千个，碰撞检测的碰撞对可能有多个。<span class="marginnote">当解有 $M$ 个时，Grover 的旋转角变成 $\theta = \arcsin\sqrt{M/N}$——角度变大了，最优迭代次数 $k \approx \frac{\pi}{4}\sqrt{N/M}$ 变少了。但若 $M$ 未知，你就不知道该迭代多少次；迭代太少概率不够，太多则「转过了头」。这正是本节要解决的工程问题。</span>本节给出 $M$ 已知与未知两种情形的处理方案。

## 1 已知 $M$：直接调角度

若已知恰有 $M$ 个解，则均匀叠加态里解的分量占比 $\frac{M}{N}$。在「解子空间」与「非解子空间」张成的平面里，角度满足

$$
\sin^2\theta = \frac{M}{N}, \qquad \theta = \arcsin\sqrt{\frac{M}{N}}
$$

最优迭代次数

$$
k \approx \frac{\pi}{4}\sqrt{\frac{N}{M}}
$$

测量命中的概率 $\ge 1 - \frac{M}{N}$。<span class="marginnote">几何直觉：解越多，「解方向」越粗，$\theta$ 越大，旋转一步跨过的角度越大，需要转的步数越少。$M = N$（全是解）时 $\theta = \pi/2$，$k = 0$，一步不转直接成功——完全合理。</span>**核心结论**：$M$ 越大，Grover 越快，加速比从 $\sqrt N$ 变成 $\sqrt{N/M}$。

## 2 未知 $M$：量子计数先估个数

若 $M$ 未知，标准策略是**先用量子计数（quantum counting）估计 $M$，再定 $k$**。量子计数 = 相位估计 + Grover 迭代（见第五篇《振幅放大与振幅估计》）：把 Grover 迭代 $G$ 当酉算符，它的本征相位 $\pm 2\theta$ 与 $M$ 通过 $\sin^2\theta = M/N$ 相连。

流程：

1. 用量子计数估计 $\tilde{M}$（误差 $\le \sqrt{M}$ 量级）；
2. 令 $\tilde\theta = \arcsin\sqrt{\tilde M / N}$，取 $k = \lfloor \frac{\pi}{4\tilde\theta}\rfloor$；
3. 迭代 $k$ 次、测量、验证是否解；若不是，重复。

量子计数的代价约 $O(\sqrt N)$ 次查询，之后每次 Grover 迭代 $O(\sqrt{N/M})$ 次——总代价仍是 $O(\sqrt N)$ 量级，不失去平方加速。<span class="marginnote">这个「先数后搜」的两段式是工业级 Grover 应用的标准姿势：与其赌「$M$ 恰等于我猜的数」，不如花一点代价把 $M$ 估准。量子计数的细节见第五篇《振幅估计》与第七篇应用一节。</span>

## 3 未知 $M$：固定次数策略

一个更朴素的方案：**直接迭代固定次数 $\lfloor\frac{\pi}{4}\sqrt{N}\rfloor$（按 $M=1$ 的最优次数），测量，若不中再重复**。这个策略的成功概率是多少？<span class="marginnote">分析见 Nielsen–Chuang §6.1.4：若实际有 $M$ 个解，迭代 $t = \lfloor\pi/4\sqrt N\rfloor$ 次后命中概率近似 $\frac{M}{N}(1 + \Omega(\sqrt{N/M}))$ 的某种形式；对 $M \le N/2$ 都有不坏的保底。工程上常配合「重复 + 验证」，把失败摊到多轮。</span>

更精巧的「不知 $M$ 的最优策略」是 **固定次数 + 指数递增重试**（类似「Exponential Search」）：按 $k = 1, 2, 4, 8, \dots$ 迭代，每次测量；一旦命中即停。总代价的期望仍被控制，且对任意 $M$ 有界——这保证「即使完全不知道 $M$，代价也不超过 $O(\sqrt{N/M})$ 的常数倍」。

## 4 公式解析：$M$ 个解时的振幅几何

把单解的平面推广到多解：定义「解子空间」$\lvert S\rangle = \frac{1}{\sqrt M}\sum_{x \in \text{解}}\lvert x\rangle$ 与「非解子空间」$\lvert \bar S\rangle$。均匀叠加

$$
\lvert s\rangle = \sqrt{\frac{M}{N}}\lvert S\rangle + \sqrt{\frac{N-M}{N}}\lvert \bar S\rangle = \sin\theta\lvert S\rangle + \cos\theta\lvert \bar S\rangle
$$

- **第一步，系数读出**：解方向的系数 $\sqrt{M/N}$ 直接是 $\sin\theta$，于是 $\sin^2\theta = M/N$。
- **第二步，旋转角**：每次 Grover 迭代旋转 $2\theta$，迭代 $k$ 次后解分量 $\sin\big((2k+1)\theta\big)$。
- **第三步，求最优**：$(2k+1)\theta \approx \pi/2 \Rightarrow k \approx \frac{\pi}{4}\sqrt{N/M}$。<span class="marginnote">注意这里 $\lvert S\rangle$、$\lvert\bar S\rangle$ 只是两个方向，Grover 迭代依旧限制在它们张成的平面里——多解只是把「目标方向」从一条线加粗成一个子空间，几何框架原封不动。</span>

**辨析｜易错点：** 用 $M=1$ 的最优次数去搜 $M>1$ 的问题，不是「多迭代几次更保险」，而是**会过头**。例如 $M = N/2$ 时最优 $k\approx \frac{\pi}{4}\sqrt{2}$，若按 $k\approx\frac{\pi}{4}\sqrt N$ 迭代（$N$ 很大），早已转了好几圈，概率完全不可控。**先估 $M$ 再迭代，是未知个数时唯一可靠的路。**

## 5 部分解：不完美 oracle

另一种推广是**部分解（partial search）**：oracle 不必精确标记所有解，只需有较高概率标记「好」的方向（如近似 oracle、带噪声的 oracle）。此时 Grover 依然工作，但旋转角被「模糊化」：每次迭代的旋转不再是精确 $2\theta$，而是带噪声的 $2\tilde\theta$。<span class="marginnote">对 NISQ 硬件，门误差、退相干让 oracle 天然带噪。分析表明：只要「好方向」的保真度足够高，Grover 仍能在 $\Theta(\sqrt{N})$ 次内达到不错的成功概率；噪声太大则加速消失——这是「量子搜索对噪声的韧性有极限」的著名结论。</span>定量的处理：把部分解建模为「$\lvert s\rangle$ 与 $\lvert S\rangle$ 夹角 $\theta$ 的估计有误差」，误差会累积，需在迭代次数与噪声水平之间取平衡。

## 6 小结

- **已知 $M$**：$\theta = \arcsin\sqrt{M/N}$，最优 $k \approx \frac{\pi}{4}\sqrt{N/M}$，解越多越快。
- **未知 $M$**：标准方案是**先量子计数估 $\tilde M$，再定 $k$**；朴素方案是固定次数 + 重复验证。
- 几何框架从单解推广到多解：解子空间方向 + 非解方向张成平面，旋转逻辑不变。
- **部分解/噪声**：oracle 不完美时旋转角模糊化，需平衡迭代次数与噪声；噪声过大则加速消失。
- **易错点**：$M$ 未知时按 $M=1$ 迭代会「过头」，先估个数是关键。

在下一节，我们回答最深刻的问题——为什么 $\sqrt N$ 就是量子搜索的极限：**Grover 算法的复杂度分析：O(√N) 的最优性**。
