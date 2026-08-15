---
title: Eliashberg强耦合超导理论
date: 2026-08-07
---

# Eliashberg强耦合超导理论

<div class="epigraph">
<p>BCS 假设了一个简单的、常数能隙的世界；真实超导体里声子谱有丰富的结构、耦合可以很强。Eliashberg 理论把 BCS 的能隙方程「升级」成两条自洽的 Green 函数方程，让强耦合超导也能从第一性原理被计算。</p>
<footer>—— G. M. Eliashberg（*Soviet Physics JETP\*, 1960）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 10 ｜ 2026-08-07</p>
</div>

## 为什么需要 Eliashberg 理论

BCS 理论有两个「弱耦合」假设：能隙在动量空间取常数（$\Delta_{\mathbf{k}} \approx \Delta$），且用单个参数 $\lambda = N(0)V$ 概括声子耦合。对铅、汞、铝这样的**强耦合超导体**，这些假设不够用：

- 能隙在费米面上**各向异性**且频率依赖；
- 声子谱有真实结构（多个声子支、Van Hove 奇点）；
- 库仑排斥必须与声子吸引**竞争**（推迟效应）。

**Eliashberg 理论（Eliashberg theory）**在 1960 年把 BCS 推广为**自洽的 Green 函数方程**：正常自能 $\Sigma(\mathbf{k},\omega)$ 与反常自能 $\phi(\mathbf{k},\omega)$（配对自能）耦合在一起，由声子 Green 函数与电子-声子顶点决定。借助 Migdal 定理（声子慢、电子快，顶点修正可忽略），Eliashberg 方程是**精确到 $O(\omega_{ph}/\varepsilon_F)$** 的系统性理论——它把超导从「能隙方程」升级成「两条自洽的积分方程」，是目前第一性原理超导计算（结合 DFT 的 SCDFT）的标准框架。<span class="marginnote">Eliashberg 理论的意义：它第一次让「从声子谱 $\alpha^2F(\omega)$ 出发计算 $T_c$」成为可能，不需要 BCS 的常数能隙假设。1970 年代 McMillan 与 Allen-Dynes 据此给出半经验公式，让 $T_c$ 预测成为一门工程。直到今天，MgB₂、氢化物高温超导的 $T_c$ 计算仍以 Eliashberg 框架为基础。</span>

## 1 Eliashberg 函数与电子-声子耦合常数

Eliashberg 理论的核心输入是**Eliashberg 谱函数** $\alpha^2F(\omega)$，它把声子谱「加权」上电子-声子耦合强度：

$$\alpha^2F(\omega) = \frac{1}{N(0)}\sum_{\mathbf{k}\mathbf{k}'\nu}|g_{\mathbf{k}\mathbf{k}'}^{\nu}|^2\,\delta(\omega-\omega_{\nu})\,\delta(\xi_{\mathbf{k}})\,\delta(\xi_{\mathbf{k}'})$$

**重点：$\alpha^2F(\omega)$ 度量「费米面上的电子，以多大强度吸收频率为 $\omega$ 的声子」。** 它是实验可测的（隧穿谱的导数反演，McMillan-Rowell 方法），也是理论可算的（DFPT）。由它定义**电子-声子耦合常数**：

$$\lambda = 2\int_0^\infty\frac{d\omega}{\omega}\,\alpha^2F(\omega)$$

$\lambda$ 是无量纲耦合强度：$\lambda\ll1$ 弱耦合（回到 BCS 极限），$\lambda\sim1\text{-}2$ 强耦合（Pb: $\lambda\approx1.55$，Hg: $\lambda\approx1.6$），$\lambda\gg1$ 超强耦合（H₃S 等氢化物）。<span class="marginnote">$\alpha^2F(\omega)$ 与隧穿谱的关系：超导/正常隧穿电导在 $\Delta$ 以上有精细结构，其导数直接反演出 $\alpha^2F(\omega)$。这是「从实验测出理论的输入函数」的经典范例——理论预言与实验测量的闭环在超导领域做得最彻底。</span>

## 2 Eliashberg 方程：Nambu 形式

把正常自能 $\Sigma$ 与配对自能 $\phi$ 统一处理的最优雅方式是 **Nambu 旋量形式**：把 $(\psi_\uparrow, \psi^\dagger_\downarrow)$ 合并成二分量旋量，Green 函数变成 $2\times2$ 矩阵。在 Nambu 空间，Eliashberg 方程（用 Migdal 定理保留骨架图）为：

$$\hat{G}^{-1}(\mathbf{k},i\omega_n) = i\omega_n\hat{\tau}_0 - \xi_{\mathbf{k}}\hat{\tau}_3 - \hat{\Sigma}(\mathbf{k},i\omega_n)$$

自能矩阵含正常分量（$\propto\hat{\tau}_0,\hat{\tau}_3$）与反常分量（$\propto\hat{\tau}_1$，即配对）：

$$\hat{\Sigma} = i\omega_n(1-Z)\hat{\tau}_0 + \chi\hat{\tau}_3 + \phi\hat{\tau}_1$$

**Eliashberg 方程**（声子频率求和后）给出三个未知函数的自洽方程组：

$$Z(i\omega_n) = 1 + \frac{\pi k_BT}{\omega_n}\sum_{n'}\lambda(n-n')\,\frac{\omega_{n'}}{\sqrt{\omega_{n'}^2+\Delta^2}}$$

$$\Delta(i\omega_n)\,Z(i\omega_n) = \pi k_BT\sum_{n'}\big[\lambda(n-n') - \mu^*\big]\,\frac{\Delta_{n'}}{\sqrt{\omega_{n'}^2+\Delta_{n'}^2}}$$

其中 $\lambda(n-n')$ 由 $\alpha^2F(\omega)$ 的声子频率求和给出，$\mu^*$ 是 **Coulomb 赝势**。<span class="marginnote">$\mu^*$ 的引入是 Eliashberg 理论最精妙的一步：裸库仑排斥很强（$\mu\sim O(1)$），但它被<strong>推迟效应</strong>大幅压低到 $\mu^* = \mu/[1+\mu\ln(\varepsilon_F/\omega_{ph})]\sim0.1\text{-}0.15$——因为库仑排斥即时作用、声子吸引有延迟，电子更「信任」慢的声子。$\mu^*$ 的小值是声子超导能存在的前提。</span>

**重点：Eliashberg 方程把「配对」与「正常自能」耦合起来——配对 $\Delta$ 影响正常态（质量重正化 $Z$），正常自能又反馈进配对。** 这是比 BCS 更深的自洽：$Z$ 不是 1，而是频率依赖的重正化因子，它捕捉了强耦合下准粒子的推迟重正化。

## 3 强耦合的效应：从 BCS 到 Eliashberg

Eliashberg 理论相对 BCS 的修正可以归纳为三点：

**有效耦合常数增大**：配对通道的有效耦合从 $\lambda$ 变成 $\lambda/(1+\lambda)$（分母含 $Z$ 的重正化）。强耦合下 $T_c$ 不再由 $e^{-1/\lambda}$ 单调决定，而是被推迟效应「饱和」。

**普适比修正**：BCS 的 $2\Delta_0/k_BT_c = 3.52$ 在强耦合下增大：$2\Delta_0/k_BT_c = 3.53[1+12(T_c/\omega_{log})^2\ln(\omega_{log}/T_c)]$。铅的实验值约 4.3，与强耦合修正一致——这是 Eliashberg 理论胜过 BCS 的明确信号。

**$T_c$ 的 McMillan 公式**：把 Eliashberg 方程数值结果拟合成半经验公式（McMillan 1968，Allen-Dynes 修正）：

$$T_c = \frac{\omega_{log}}{1.2}\exp\Big[-\frac{1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\Big]$$

其中 $\omega_{log}$ 是声子的对数平均频率。**重点：$T_c$ 由两个因子决定——声子频率尺度 $\omega_{log}$（大频率 → 高 $T_c$）与有效耦合 $\lambda-\mu^*$（强耦合+弱库仑 → 高 $T_c$）。** 这直接指导了材料设计：氢化物超导（H₃S, $T_c\approx203\,\text{K}$）正是用轻的氢原子提供大 $\omega_{log}$。<span class="marginnote">McMillan 公式的适用性：$\lambda\lesssim1.5$ 时准确；对 $\lambda\gg1$（氢化物），Allen-Dynes 修正引入额外因子，但基本结构不变。2020 年代碳质硫氢化物在超高压下 $T_c$ 接近室温（约 $287\,\text{K}$），仍是 Eliashberg 框架的预言——「声子机制能否到室温」的现代答案基本是「能，但需要极端压力」。</span>

## 4 公式解析：从 BCS 到 Eliashberg 的一步之遥

比较 BCS 与 Eliashberg 的能隙方程，看清「强耦合」到底改了什么：

**第一步，写 BCS 能隙方程**（零温，常数 $\Delta$）：$1 = N(0)V\int_0^{\hbar\omega_D}\frac{d\xi}{\sqrt{\xi^2+\Delta^2}}$。这里相互作用被压缩成常数 $V$，截断是 $\hbar\omega_D$——BCS 是「单参数 + 硬截断」的模型。
**第二步，写 Eliashberg 配对方程**（松原频率形式）：$\Delta(i\omega_n) = \frac{\pi k_BT}{Z(i\omega_n)}\sum_{n'}\big[\lambda(n-n')-\mu^*\big]\frac{\Delta_{n'}}{\sqrt{\omega_{n'}^2+\Delta_{n'}^2}}$。
- **第三步，对比差异**：(a) $\lambda(n-n')$ 是**频率依赖**的（声子谱结构进来），BCS 的 $V$ 是常数；(b) 多了 $\mu^*$ 库仑项；(c) 分母有 $Z(i\omega_n)$ 重正化；(d) $\Delta$ 依赖频率（不再是常数）。
- **第四步，取弱耦合极限**：$\lambda\to$ 常数、$Z\to1$、$\mu^*\to0$，Eliashberg 方程**精确回到 BCS 能隙方程**。Eliashberg 是 BCS 的「带结构的推广」，BCS 是 Eliashberg 的「单模简化」。

**重点：Eliashberg 与 BCS 的差别不是「谁对谁错」，而是「分辨率」——Eliashberg 看到了声子谱的频率结构，BCS 只看平均值。** 当 $\alpha^2F(\omega)$ 在某一频率集中（如 Einstein 模），BCS 够用；当谱有多个峰、强耦合，必须 Eliashberg。

**辨析｜易错点：** 初学者常误以为「强耦合 = 大 $\lambda$ 就完事」。实际上**推迟效应**是强耦合理论的核心概念：声子吸引有延迟（$\sim1/\omega_{ph}$），库仑排斥即时，两者在**不同时间尺度**上作用，所以不能简单地「吸引减排斥」——这正是 $\mu^*$ 与 $\lambda(\omega)$ 的频率结构存在的意义。忽视推迟效应，就会高估库仑排斥、低估超导可能性。

## 5 Eliashberg 与「从极限到大模型」

Eliashberg 理论的方法论启示：**当「简单模型」（BCS 的常数能隙）在某个参数区失效时，不是推翻框架，而是给框架增加「结构」——把常数量升级成频率/动量依赖的函数。** 这种「从常数到函数」的升级路径，在机器学习里反复出现：从常数学习率到自适应学习率（Adam 的逐参数尺度）、从固定表征到上下文依赖表征（attention 的逐 token 加权）、从常数温度到退火调度。<span class="marginnote">更深的对应：Eliashberg 的「推迟效应」≈ 机器学习的「时序/记忆效应」——当前状态的效应要等一段时间才显现（类似 RNN 的延迟反馈）。而「$\mu^*$ 用能量尺度差把强排斥压低」与「用时间尺度分离做课程学习」也共享逻辑：<strong>先处理慢变量，再处理快变量</strong>。可参考第四级《大模型微调》与《生成模型》。</span>

对多体理论自身，Eliashberg 是超导计算的第一性原理标准；下一节，我们把超导「接通」外界：**超导隧穿与 Josephson 效应**——超导电流如何穿过绝缘势垒，以及宏观相位如何导致量子干涉。

## 6 小结

- **Eliashberg 理论**是强耦合超导的 Green 函数框架：两条自洽方程耦合正常自能与配对自能，靠 Migdal 定理支撑。
- 核心输入 **Eliashberg 函数** $\alpha^2F(\omega)$ 实验可测（隧穿谱）、理论可算（DFPT）；耦合常数 $\lambda=2\int\alpha^2F/\omega\,d\omega$。
- **Nambu 形式**把正常/反常自能统一进 $2\times2$ 矩阵；方程含频率依赖 $\lambda(n-n')$、$\mu^*$、重正化因子 $Z$。
- **$\mu^*$**（Coulomb 赝势）被推迟效应压低到 $0.1\text{-}0.15$，是声子超导存在的前提。
- 强耦合修正：普适比 $2\Delta_0/k_BT_c$ 从 3.52 增大；**McMillan 公式** $T_c = \frac{\omega_{log}}{1.2}e^{-1.04(1+\lambda)/(\lambda-\mu^*(1+0.62\lambda))}$。
- 弱耦合极限下 Eliashberg 精确回到 BCS；区别是「分辨率」而非「对错」。

在下一节，我们研究超导的宏观量子效应：**超导隧穿与 Josephson 效应**——电子对如何隧穿势垒，超导相位差如何驱动零电压电流，以及 SQUID 如何把相位变成可读的干涉信号。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| Eliashberg 函数 | $\alpha^2F(\omega)$ | 费米面电子吸收频率 $\omega$ 声子的强度 |
| 耦合常数 | $\lambda = 2\int_0^\infty\frac{d\omega}{\omega}\alpha^2F(\omega)$ | 无量纲耦合强度 |
| 能隙方程 | $\Delta_n = \frac{\pi k_BT}{Z_n}\sum_{n'}[\lambda(n-n')-\mu^*]\frac{\Delta_{n'}}{\sqrt{\omega_{n'}^2+\Delta_{n'}^2}}$ | 频率依赖配对，含库仑赝势 |
| McMillan 公式 | $T_c = \frac{\omega_{log}}{1.2}e^{-1.04(1+\lambda)/(\lambda-\mu^*(1+0.62\lambda))}$ | $T_c$ 由声子尺度与有效耦合决定 |
| 普适比修正 | $2\Delta_0/k_BT_c$ 从 3.52 增大 | 强耦合指纹，Pb 实测约 4.3 |

**易错复盘**：两点要盯住。其一，强耦合不是「大 $\lambda$ 就完事」——推迟效应是关键：声子吸引有延迟、库仑排斥即时，两者在不同时间尺度作用，所以 $\mu^*$ 被压低到 0.1-0.15；其二，Eliashberg 与 BCS 的区别是「分辨率」——前者看到声子谱频率结构，后者只看平均值，弱耦合极限下前者精确回到后者。

**知识连线**：本篇把第 3 篇 BCS 从弱耦合推广到强耦合，Migdal 定理（第 3 篇电子-声子）保证其合法性；$\alpha^2F(\omega)$ 由隧穿谱反演（第 3 篇 Josephson）直接测量。「从常数到频率依赖函数」的升级路径，是「从极限到大模型」里「简单模型失效时给模型加结构」的范例。

## 7 数值参照：强耦合参数的谱系

| 超导体 | $\lambda$ | $T_c$（K） | $2\Delta_0/k_BT_c$ |
| --- | --- | --- | --- |
| 铝 | 0.41 | 1.2 | 3.4 |
| 铌 | 1.0 | 9.2 | 3.8 |
| 铅 | 1.55 | 7.2 | 4.3 |
| 汞 | 1.6 | 4.2 | 4.6 |

**读数要领**：铝是弱耦合（普适比接近 BCS 的 3.52），铅与汞是典型强耦合（普适比显著增大）。普适比偏离 3.52 的幅度，就是「强耦合程度」的直接读数——Eliashberg 理论的价值正在于把这个偏离从经验事实变成可计算的预言。

**实践与辨析**：为什么氢化物超导（H₃S）的 $T_c$ 能高达 200 K？提示：轻的氢提供大 $\omega_{log}$，McMillan 公式 $T_c\propto\omega_{log}$。为什么 $\mu^*$ 必须存在？提示：裸库仑排斥 $\mu\sim O(1)$ 被推迟效应压低到 $\mu^*\approx0.1\text{–}0.15$，否则吸引被库仑淹没、声子超导不可能存在。