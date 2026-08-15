---
title: 相位估计（phase estimation）算法
date: 2026-08-07
---

# 相位估计（phase estimation）算法

<div class="epigraph">
<p>相位估计是量子算法工具箱里最重要的子程序之一。</p>
<footer>—— 基塔耶夫（Alexei Kitaev）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§5.2 ｜ 2026-08-07</p>
</div>

## 为什么从相位估计开始

上一节的 QFT 是一个「基变换」工具，但要让它发挥作用，得有一个「把相位翻出来读」的子程序。**相位估计（phase estimation, PE）**正是这样的万能子程序：给定一个酉算符 $U$ 和它的一个本征向量 $\lvert u\rangle$，求对应的本征相位 $\theta$（满足 $U\lvert u\rangle = e^{2\pi i\theta}\lvert u\rangle$）。<span class="marginnote">相位估计由 Kitaev 在 1995 年提出。它的地位怎么强调都不为过：Shor 分解、量子化学的能量计算（VQE 的相位估计版）、量子模拟、甚至一些量子机器学习算法，全都以它为子程序。学完 PE，几乎等于拿到了整个「QFT 家族」的钥匙。</span>本节给出它的完整线路与精度分析。

## 1 问题与基本思路

**相位估计问题**：已知酉算符 $U$ 与其本征向量 $\lvert u\rangle$（$U\lvert u\rangle = e^{2\pi i\theta}\lvert u\rangle$，$0 \le \theta < 1$），求 $\theta$。

思路分三步：

1. **第一寄存器** $t$ 个比特全部 $\lvert0\rangle$，作用 $H^{\otimes t}$ 得到均匀叠加；
2. 以第一寄存器为控制，施加**受控-$U^{2^j}$**（$j = 0,\dots,t-1$），把本征相位「拷贝」进第一寄存器的振幅；
3. 第一寄存器作用**逆 QFT**，测量读出 $\theta$ 的二进制近似。

关键机制：受控-$U^{2^j}$ 作用在叠加态上，给第 $j$ 个控制比特引入相位 $e^{2\pi i \theta 2^j}$，整个第一寄存器因此携带一个「相位编码的叠加」，逆 QFT 把它还原成 $\theta$ 的二进制表示。

## 2 线路的数学推导

经过第 1、2 步，整个态变成

$$
\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi i\theta k}\lvert k\rangle \otimes \lvert u\rangle
$$

因为受控-$U^{2^j}$ 把 $\lvert k\rangle$ 的每一位 $k_j$ 对应的相位 $e^{2\pi i\theta k_j 2^j}$ 乘上，累加得到 $e^{2\pi i\theta\sum_j k_j 2^j} = e^{2\pi i\theta k}$。<span class="marginnote">这一步是「相位复制」的精华：控制比特的叠加态吸收了本征相位，而 $\lvert u\rangle$ 本身不动（因为它是本征向量）。「用受控门把本征相位翻到控制寄存器上」——这个模式后面在 Shor、在 HHL 里反复出现。</span>若 $\theta = \frac{a}{2^t}$ 恰是二进制可表示的，则第一寄存器就是 $\frac{1}{\sqrt{2^t}}\sum_k e^{2\pi i ak/2^t}\lvert k\rangle$，它**正是** $QFT_{2^t}\lvert a\rangle$！逆 QFT 后直接得到 $\lvert a\rangle$，测量即得 $\theta = a/2^t$ 精确值。

## 3 公式解析：精确相位的情形

设 $\theta = a/2^t$ 恰好精确。逆 QFT 作用：

$$
QFT_{2^t}^{-1}\left( \frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi i ak/2^t}\lvert k\rangle \right) = \lvert a\rangle
$$

三步拆解：

- **第一步，识别正向 QFT**：由上一节定义，$QFT_N\lvert a\rangle = \frac{1}{\sqrt N}\sum_k e^{2\pi i ak/N}\lvert k\rangle$，其中 $N = 2^t$。左边的叠加正是这个式子。
- **第二步，逆变换**：QFT 是酉变换，$QFT^{-1} \cdot QFT = I$，于是第一寄存器坍缩到 $\lvert a\rangle$。
- **第三步，读出**：测量得到整数 $a$，相位 $\theta = a/2^t$ 被**精确**恢复。<span class="marginnote">记忆点：当相位能用 $t$ 个比特精确表示时，相位估计是一次成功、零误差的——QFT 的正交性保证了这一点。</span>

## 4 公式解析：非精确相位与误差界

若 $\theta$ 不是恰好 $a/2^t$（一般情形），则测得的 $\lvert a\rangle$ 满足：以高概率，$\frac{a}{2^t}$ 是 $\theta$ 的**最近 $t$ 比特近似**。精确的误差界由 Nielsen–Chuang 给出：

$$
P\left( \left\lvert \frac{a}{2^t} - \theta \right\rvert \le \frac{1}{2^t} \right) \ge \frac{8}{\pi^2} \approx 0.81, \qquad
P\left( \left\lvert \frac{a}{2^t} - \theta \right\rvert \le 2^{-t+m} \right) \ge 1 - \frac{1}{2(m-1)}
$$

- **第一步，几何求和**：非精确时振幅是等比级数 $\sum_k (e^{2\pi i(\theta - a/2^t)})^k$，其模在 $a$ 接近 $\theta 2^t$ 时最大。
- **第二步，干涉集中**：相位差 $\delta = \theta - a/2^t$ 越小，等比级数模越大；离 $a$ 越远，模按 $\lvert\sin(\pi 2^t\delta)\rvert^{-1}$ 衰减——概率集中在最近的整数。
- **第三步，失败控制**：对「误差超过 $2^{-t}$」的事件，总概率被压到 $\le \frac{\pi^2}{8}\cdot$(最近整数距离的比值)。给 $m$ 个额外比特就能把失败压到指数小。<span class="marginnote">工程读法：想要误差 $\le 2^{-t}$，就准备 $t$ 个控制比特；每次失败概率约 19%，重复 2–3 次取多数即可把整体失败压到可忽略。这个「用重复换精度」的套路是 PE 的标准配套。</span>

**辨析｜易错点：** 相位估计的前提是**已知 $\lvert u\rangle$ 并能制备**。许多应用（如 Shor）中 $\lvert u\rangle$ 未知，但可以把输入态准备成本征态的**叠加**——测量的结果按本征值加权混合。对 Shor 而言，巧妙选初始态让「正确本征态占主导」，这就是为什么 Shor 不需要先知道周期。

## 5 应用：从 Shor 到量子化学

相位估计是 Shor 算法的第二步（在模幂 oracle 之后）：把「模乘的周期」当成本征相位读出，从而求周期、分解大数。它也是**量子化学**里最精确的能量算法（如 phase-estimation-based quantum chemistry）的核心——直接读出哈密顿量的本征能量。<span class="marginnote">在 NISQ 时代，完整相位估计对相干时间要求太高，工程上多用变分方法（第十篇 VQE）先做近似，再用 PE 精修。PE 与 VQE 是「高精度贵方法」与「低精度廉价方法」的两端。</span>

## 6 常见误区与自查练习

| 误区 | 事实 |
| --- | --- |
| 「相位估计需要知道 $\lvert u\rangle$」 | 通常可制备本征态叠加；Shor 巧妙选初始态让正确本征态占主导 |
| 「相位估计总是有误差」 | $\theta=a/2^t$ 精确可表示时一次成功、零误差 |
| 「PE 只用于 Shor」 | 量子化学能量、量子模拟、部分 QML 都以它为子程序 |
| 「误差可以随便压」 | 需更多控制比特（$2^{-t}$），并用重复多数投票压失败 |

**自查问题**：

1. 受控-$U^{2^j}$ 怎么把相位「拷贝」进寄存器？——$\lvert k\rangle$ 每位引入相位 $e^{2\pi i\theta k_j 2^j}$。
2. 精确相位为什么一次成功？——第一寄存器恰是 $QFT\lvert a\rangle$，逆 QFT 精确还原。
3. 非精确时误差界是什么？——误差 $\le 2^{-t}$ 的概率 $\ge 8/\pi^2$。
4. NISQ 时代为什么多用变分法？——完整 PE 对相干时间要求太高，VQE 先近似、PE 精修。

## 7 术语速查表

| 术语 | 含义 |
| --- | --- |
| 相位估计 | 给定 $U$ 与本征向量，求本征相位 $\theta$ |
| 受控-$U^{2^j}$ | 把本征相位翻到控制寄存器上的门 |
| 逆 QFT | 把相位编码的叠加还原成二进制表示 |
| 本征相位 | $U\lvert u\rangle=e^{2\pi i\theta}\lvert u\rangle$ 中的 $\theta$ |

## 8 小结

- **相位估计**：给定 $U$ 与本征向量 $\lvert u\rangle$，求本征相位 $\theta$；线路 = **$H^{\otimes t}$ → 受控-$U^{2^j}$ → 逆 QFT → 测量**。
- 受控-$U^{2^j}$ 把相位「拷贝」进控制寄存器；逆 QFT 读出二进制近似。
- **精确情形**：$\theta = a/2^t$ 时一次成功、零误差。
- **非精确情形**：误差 $\le 2^{-t}$ 的概率 $\ge \frac{8}{\pi^2}$，加冗余比特可指数压失败。
- 应用：Shor、量子化学能量计算、量子模拟——QFT 家族的万能子程序。
- **记忆链**：$H^{\otimes t}$ 叠加 → 受控-$U^{2^j}$ 拷贝相位 → 逆 QFT 读出——「把相位翻出来读」的三段式。
- **数值锚点**：$t=3$、$\theta=1/8$ 时 $a=1$ 一次成功；误差 $\le 2^{-t}$ 的概率 $\ge 0.81$。
- **一句话**：PE = 把「特征相位」翻出来读——Shor、量子化学、量子模拟的公共子程序。
- **工程权衡**：NISQ 时代完整 PE 太贵，VQE 先近似、PE 精修——「廉价近似 + 精确修正」的组合拳。

**练习**：

1. 手算 $t=3$、$\theta=\tfrac18$ 的情形——$\theta=\tfrac{1}{8}=\tfrac{1}{2^3}$，$a=1$，一次成功读出 $\theta=1/8$。
2. 说明「相位拷贝」机制——受控-$U^{2^j}$ 给第 $j$ 控制位引入相位 $e^{2\pi i\theta 2^j}$。
3. 用误差界设计参数——想要误差 $\le 2^{-10}$，需 $t=10$ 个控制比特，失败概率约 19% 用重复压。
4. 对比 PE 与 VQE——PE 高精度贵，VQE 低精度廉，工程上 VQE 先近似、PE 精修。

在下一节，我们回到查询模型的另一条主线——**量子振幅放大与振幅估计**，它把 Grover 的思想推广成通用的概率放大工具。
