---
title: 量子熵与 coherent information
date: 2026-08-11
---

# 量子熵与 coherent information

<div class="epigraph">
<p>熵并不是混乱与无序的代名词，而是信息缺乏的度量。</p>
<footer>—— 冯 · 诺伊曼（John von Neumann，转引自 W. H. Zurek 的回忆）</footer>
</div>

<div class="article-byline">
<p>第九级 · 交叉与前沿 · 量子信息 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从量子熵开始

第二级的经典信息论告诉我们：一条消息携带多少信息，用**香农熵** $H(X) = -\sum_x p(x)\log p(x)$ 度量；一条信道能可靠传多少信息，用**信道容量** C 度量。量子信息论需要一个平行的体系：**量子态的熵是什么？量子信道最多可靠传多少？** 本节对应 Wilde §11 与 Preskill Part II，引入两个主角：**von Neumann 熵**（量子版的香农熵）与 **coherent information**（量子版的互信息）。

<span class="marginnote">香农的框架建立在「符号 + 概率」上，而量子态还携带相位与纠缠。von Neumann 熵把「不确定性」推广到密度算子上，coherent information 则回答一个更刁钻的问题：量子信道能保留多少「纠缠信息」——它不仅要求信息被读出，还要求量子关联被保持。</span>

## 1 von Neumann 熵的定义

对密度算子 $\rho$，定义 **von Neumann 熵**：

$$
S(\rho) = -\mathrm{Tr}(\rho \log_2 \rho)
$$

如果 $\rho$ 的本征分解为 $\rho = \sum_i \lambda_i |i\rangle\langle i|$（$\lambda_i \ge 0$，$\sum_i \lambda_i = 1$），则

$$
S(\rho) = -\sum_i \lambda_i \log_2 \lambda_i
$$

正好是概率分布 $\{\lambda_i\}$ 的香农熵。<span class="marginnote">取对数底为 2 时熵的单位是比特。本征值就是「把密度矩阵对角化后出现的概率」——von Neumann 熵 = 密度矩阵在正交基下的经典不确定性。底为 e 时单位是 nat，量子信息学默认底 2。</span>

**核心概念：von Neumann 熵 $S(\rho)$**：密度算子 $\rho$ 的量子熵，等于其本征值分布的香农熵。它度量态「有多不纯」：纯态 $S = 0$，最大混合态 $S = \log d$（$d$ 为维数）。

几个立刻要记住的性质：

$S(\rho) \ge 0$，等号当且仅当 $\rho$ 是纯态。
- $d$ 维空间里 $S(\rho) \le \log d$，等号当且仅当 $\rho = I/d$（最大混合）。
- **可加性缺陷**：$S(\rho_{AB})$ 与 $S(\rho_A)$、$S(\rho_B)$ 之间只有**次可加性** $S(\rho_{AB}) \le S(\rho_A) + S(\rho_B)$，而不是香农熵那种漂亮的等式或不等式——因为量子关联（纠缠）会把「负条件熵」带进这个世界。

## 2 纠缠让条件熵变负

定义量子条件熵 $S(A|B) = S(\rho_{AB}) - S(\rho_B)$，其中 $\rho_{AB}$ 是两体态，$\rho_B = \mathrm{Tr}_A \rho_{AB}$ 是约化密度算子。经典条件熵恒非负（知道 B 只会减少不确定性），但量子条件熵**可以为负**——这恰好是纠缠的签名。<span class="marginnote">回顾第五篇的 Bell 态例子：$|\Phi^+\rangle$ 的整体态是纯态所以 $S(AB)=0$，而每个约化态是 $I/2$ 所以 $S(A)=S(B)=1$，于是 $S(A|B) = 0 - 1 = -1$。负条件熵意味着「整体比部分更确定」——只有纠缠能带来这种反常。</span>

**辨析｜易错点：**不要用「量子熵 = 香农熵」的直觉硬套。经典互信息 $I(A:B) = H(A) + H(B) - H(AB)$ 永远非负；量子互信息 $I(A:B) = S(A) + S(B) - S(AB)$ 也非负，但它衡量的是「总关联」，把经典关联和纠缠捆在一起，分不开。要单独度量纠缠得用下一节《纠缠度量与量子资源理论》里的**纠缠熵** $E(|\psi\rangle) = S(\mathrm{Tr}_B |\psi\rangle\langle\psi|)$——对纯态，单侧约化态的熵正好度量纠缠量。

## 3 coherent information：量子信道能保留多少纠缠

经典信道容量的核心是互信息 $I(X;Y)$。量子侧的对应物不是量子互信息，而是 **coherent information**（相干信息）：

$$
I(A\rangle B)_{\rho} = S(\rho_B) - S(\rho_{AB})
$$

注意这就是条件熵的相反数。为什么「-条件熵」才是正确候选？直觉：信道若想**量子**地传信息，不仅要有经典相关，还要**保留纠缠**——而被保留的纠缠量恰好由负条件熵（即 $-S(A|B)$）刻画。<span class="marginnote">几何记忆法：coherent information 的记号 $I(A\rangle B)$ 故意写成「尖括号朝右」，强调它是「A 的信息流向 B」且要求 B 那边能相干地重建 A。</span>

对纯态 $\rho_{AB}$（编码成 A、信道出 B），$I(A\rangle B) = S(B) - S(AB) = S(A)$——输入端的熵被完整保留，这就是「信道没丢任何纠缠信息」的情形。有噪声时 $S(AB) > 0$，相干信息下降，降为零即信道不再能量子传态。

**量子容量（quantum capacity）**：信道 $\mathcal{N}$ 的量子容量定义为

$$
Q(\mathcal{N}) = \lim_{n\to\infty} \frac{1}{n} \max_{\rho} I(A\rangle B)_{\mathcal{N}^{\otimes n}(\rho)}
$$

也就是「用量子态编码、用量子态读出」的可靠传态速率上界。著名的退化信道 $Q(\mathcal{N})=0$ 例子说明：有些信道经典能传信息但量子完全不能——信息可以在经典意义上流过，纠缠却一滴都过不去。<span class="marginnote">这个「经典能传、量子不能」的反直觉结论（Smith–Smolin 例子、PPT 纠缠约束）是量子信道理论最精妙的主题之一，Wilde 书中 §11 有系统讨论。它提醒我们：量子信息是比经典信息更「挑剔」的货物。</span>

## 4 公式解析：Bell 态的熵谱

把熵的计算完整走一遍，用 $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$ 作为试金石：

$$
\rho_{AB} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}(|00\rangle\langle00| + |00\rangle\langle11| + |11\rangle\langle00| + |11\rangle\langle11|)
$$

- **第一步，求约化态**：对 B 做部分求迹，$\rho_A = \mathrm{Tr}_B \rho_{AB} = \frac{1}{2}(|0\rangle\langle0| + |1\rangle\langle1|) = \frac{I}{2}$。求迹时交叉项 $\mathrm{Tr}_B(|00\rangle\langle11|)$ 消失——纠缠的非对角元在约化时被抹平。
- **第二步，求熵谱**：$\rho_A$ 本征值 $\lambda_1 = \lambda_2 = \frac{1}{2}$，于是 $S(\rho_A) = -\left(\frac12\log_2\frac12 + \frac12\log_2\frac12\right) = 1$ 比特。
- **第三步，算相干信息**：整体是纯态，$S(\rho_{AB}) = 0$，所以 $I(A\rangle B) = S(B) - S(AB) = 1 - 0 = 1$——完整的一比特纠缠被保住；同时条件熵 $S(A|B) = -1$，符号翻负，缠结签名出现。

这套「部分求迹 → 谱 → 熵」的三步法，是计算量子信息论里一切熵量（纠缠熵、相干信息、信道容量）的标准动作，务必熟练。

## 5 从香农到量子香农

把两条谱线并列，量子香农理论的骨架就清晰了：

| 经典（香农） | 量子（von Neumann / 相干信息） |
| --- | --- |
| $H(X) = -\sum p\log p$ | $S(\rho) = -\mathrm{Tr}(\rho\log\rho)$ |
| 联合熵 $H(X,Y)$ | $S(\rho_{AB})$ |
| 条件熵 $H(X|Y) \ge 0$ | $S(A|B)$ 可以为负（纠缠） |
| 互信息 $I(X;Y)$ | coherent information $I(A\rangle B)$ |
| 信道容量 C | 量子容量 $Q(\mathcal{N})$ |

<span class="marginnote">注意表里那根竖线：经典框里的每个量都能映射到量子框，但量子框多出来的「负条件熵」「相干信息」没有经典对应物——它们正是纠缠留下的指纹。这是「量子信息 = 经典信息 + 纠缠信息」这句口号最精确的展开。</span>

**辨析｜易错点：**von Neumann 熵可加性成立仅当系统之间**无关联**（乘积态）：$S(\rho_A \otimes \rho_B) = S(\rho_A) + S(\rho_B)$。一旦有纠缠，整体熵小于各部分之和——「整体比部分更确定」的反直觉正是量子信息论与经典信息论最刺眼的分界。下一节我们把这套熵的语言接到信道上，回答「量子信道到底能传多少」。

## 6 Schumacher 压缩：量子版的香农无噪编码

熵不只是度量，它还有直接的操作意义——**量子压缩**。经典无噪信道编码定理说：n 个独立同分布消息可以被压缩到 $nH(X)$ 比特而无损。量子侧对应的是 **Schumacher 压缩**（1995）：n 个来自源 $\rho$ 的量子态可以被压缩到约 $nS(\rho)$ 个量子比特，且渐近保真度趋近 1。<span class="marginnote">这是量子香农理论的奠基结果之一（Nielsen & Chuang §12.4 有完整介绍）：$S(\rho)$ 是「每个量子态最少需要多少个 qubit 才能无损存储/传输」的答案——与香农的 $H(X)$ 扮演完全平行的角色。它的证明用到的正是「典型子空间」的量子版本：绝大多数拷贝落在 $\rho^{\otimes n}$ 的典型子空间里，维度约 $2^{nS(\rho)}$。</span>

这个定理把 von Neumann 熵从「度量」升格为「限额」：

源 $\rho$ 若接近纯态（$S \approx 0$），几乎不用带宽——信息早已确定；
- 源 $\rho = I/2$（$S = 1$），每个态至少 1 qubit——无法再压缩；
- 压缩比 $nS(\rho)$ 是**物理下限**：想比它更省，就必然损失保真度。

**辨析｜易错点：**Schumacher 压缩压缩的是「量子态本身」，输出仍是量子态；它不等于「先测量、再压缩经典结果」。后者会先破坏相干，损失量子信息——压缩必须在量子层面完成。<span class="marginnote">把 Schumacher 压缩与第七篇 Holevo 界放在一起，就构成量子香农理论的左右手：Schumacher 管「量子数据能压缩多省」，Holevo 管「量子载体能传多少经典」。两条线合起来，就是「量子信息论 = 量子数据 + 量子信道」的完整版图。</span>

## 7 小结

- **von Neumann 熵** $S(\rho) = -\mathrm{Tr}(\rho\log\rho)$ 是本征值分布的香农熵；纯态为 0，最大混合为 $\log d$。
- 量子熵只满足**次可加性**；纠缠使**条件熵变负**，负条件熵是纠缠的签名。
- **coherent information** $I(A\rangle B) = S(B) - S(AB)$ 度量信道保留的纠缠信息；**量子容量** $Q$ 由它的正则极限定义。
- 计算熵量的标准三步：**部分求迹 → 谱分解 → 熵**。
- 量子熵体系 = 经典香农体系的直接推广 + 纠缠特有的负条件熵。

在下一节，我们将正式走进信道——用量子信道与容量理论回答「信息经过噪声到底能传多少」，并认识那个以苏联科学家命名的惊人上界：**Holevo 界**。
