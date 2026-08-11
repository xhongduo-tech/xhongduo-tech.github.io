---
title: 热力学性质计算
date: 2026-08-11
---

# 热力学性质计算

<div class="epigraph">
<p>Nature does not jump, and its properties are connected like the links of a chain.</p>
<p>自然界不跳跃，它的性质像链条的环一样彼此相连。</p>
<footer>—— 吉布斯（Josiah Willard Gibbs）</footer>
</div>

<div class="article-byline">
<p>第六级 · 工程技术 · 化工热力学 ｜ 对标教材 Smith, Van Ness, Abbott, Swihart §6 ｜ 2026-08-11</p>
</div>

## 为什么从热力学性质计算开始

前面反复用到「焓」「熵」，但从未交代它们的数值从哪来。手册只给理想气体与极少数状态下的数据，而工程师需要的是高压、低温、汽液两相下的焓与熵。<span class="marginnote">从这一篇开始，热力学不再「只能定性」，而是变成一套可以手算的数值流程：有了状态方程，输入 $P$、$T$，就能输出焓、熵、逸度。这是从「物理」到「工程」的临门一脚。</span>核心思想一句话：**真实流体的性质 = 理想气体性质 + 剩余性质（departure function）**，而剩余性质全部可由状态方程算出来。这一步打通了第二篇（状态方程）到后续（相平衡、反应平衡）的整条链路。

## 1 热力学性质的数学骨架：全微分与麦克斯韦关系

焓、熵这些状态函数的微元变化可用全微分联系。对摩尔量，恒组成的单相流体有

$$
\mathrm{d}H = C_p\,\mathrm{d}T + \left[V - T\left(\frac{\partial V}{\partial T}\right)_P\right]\mathrm{d}P
$$

$$
\mathrm{d}S = \frac{C_p}{T}\,\mathrm{d}T - \left(\frac{\partial V}{\partial T}\right)_P \mathrm{d}P
$$

关键在哪？式中的偏导数 $\left(\partial V/\partial T\right)_P$ 由状态方程给出。<span class="marginnote">这两条式子意味着：只要知道热容 $C_p(T)$ 与状态方程，焓和熵从任意基准积分到任意状态都办得到——状态方程是第二篇埋下的伏笔，这里兑现。第二级《高等数学》里全微分、积分路径无关的概念，在此直接上岗。</span>

从这些关系可以导出**麦克斯韦关系（Maxwell relations）**，如

$$
\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V
$$

它把不可测的偏导换成可测的偏导，是性质计算的身份证明。<span class="marginnote">麦克斯韦关系的源头是「混合偏导相等」这条纯数学定理：因为 $U$、$H$、$A$、$G$ 都是恰当微分（exact differential），交叉二阶偏导必相等。热力学的全部推导魔力，几乎都来自这一个数学事实。</span>

## 2 剩余性质：把「理想」与「真实」的差包装起来

**剩余性质（residual property）**定义：同一温度、同一压力下，真实流体的摩尔性质减去理想气体的摩尔性质

$$
M^R = M - M^{ig}
$$

对焓与熵，$H^R$ 与 $S^R$ 都不是零——只有理想气体的焓与压力无关（吉布斯—亥姆霍兹关系保证熵的剩余不为零）。有了剩余性质，真实焓熵的算法就统一为

$$
H = H^{ig}(T) + H^R(T,P), \qquad S = S^{ig}(T,P) + S^R(T,P)
$$

**重点：理想气体部分查表或积分热容，剩余部分全由状态方程算——两条腿走路，把「数据需求」降到了最低。**

## 3 由状态方程计算剩余焓与剩余熵

剩余性质的通用公式由积分给出。对以 $P(T,V)$ 形式给的状态方程，剩余焓与剩余熵可写成对摩尔体积的积分：

$$
\frac{H^R}{RT} = Z - 1 + \frac{1}{RT}\int_{\infty}^{V}\left[T\left(\frac{\partial P}{\partial T}\right)_V - P\right]\mathrm{d}V
$$

$$
\frac{S^R}{R} = \ln Z + \frac{1}{R}\int_{\infty}^{V}\left[\left(\frac{\partial P}{\partial T}\right)_V - \frac{R}{V}\right]\mathrm{d}V
$$

<span class="marginnote">积分从无穷大体积（理想极限）出发，代表「从零压把分子『装进来』到当前摩尔体积」的过程中性质偏离的累积。把 $P(T,V)$ 的表达式代入、求偏导、积分，剩下的就是代数，任何立方型状态方程都能照此办理。</span>

对纯组分性质计算，用理想气体热容多项式求出 $H^{ig}$、$S^{ig}$，再加上 $H^R$、$S^R$，就得到任意 $P$、$T$ 下的焓与熵——**这是汽轮机、压缩机、闪蒸罐等一切设备核算的底层算法**。

## 4 公式解析：范德华方程下的剩余焓

拿最简的范德华方程 $P = RT/(V-b) - a/V^2$ 走一遍，把抽象公式落到可见的数字。

- **第一步，算偏导**：$(\partial P/\partial T)_V = R/(V-b)$（$a$ 项不含 $T$，对温度求导消失）。
- **第二步，代入剩余焓积分**：被积函数 $T(\partial P/\partial T)_V - P = TR/(V-b) - RT/(V-b) + a/V^2 = a/V^2$，于是 $H^R/RT = Z - 1 + (1/RT)\int_\infty^V (a/V^2)\,\mathrm{d}V = Z - 1 - a/(RTV)$。
- **第三步，读出结论**：$H^R = RT(Z-1) - a/V$。其中 $-a/V$ 项来自分子间引力（$a>0$），把焓压低——放大的正是范德华方程对吸引项的建模。$RT(Z-1)$ 项来自体积修正 $b$。两项的物理来源一目了然。

例：某真实气体在 $T = 400\,\text{K}$、$P = 20\,\text{bar}$ 下 $Z = 0.85$，$a = 3.6\,\text{bar·L}^2/\text{mol}^2$，$V = 1.41\,\text{L/mol}$，则 $H^R = 8.314\times400\times(0.85-1) - 3.6/1.41 \approx -499 - 2.55 \approx -501\,\text{J/mol}$。若理想气体焓为 4200 J/mol，则真实焓约 3699 J/mol。**过程比数字重要**：同一套「偏导 → 代入 → 积分」的打法，换 PR、SRK 方程照样成立，只是代数更繁、精度更高。

## 5 从剩余性质到逸度

最后一站：同一条积分路线还能算出**逸度（fugacity）**，它是真实气体「有效压力」的度量

$$
\ln \varphi = \frac{G^R}{RT} = Z - 1 - \ln Z + \frac{1}{RT}\int_{\infty}^{V}\left(P - \frac{RT}{V}\right)\mathrm{d}V
$$

其中 $\varphi = f/P$ 称为**逸度系数（fugacity coefficient）**。<span class="marginnote">理想气体 $\varphi = 1$、$f = P$；真实气体 $f$ 偏离 $P$，相当于「修正过的压力」参与一切相平衡公式。逸度是第七篇相平衡判据的核心道具——它在这里被状态方程算出，在那边被用来判相平衡。</span>

**重点：焓、熵、逸度三位一体**，都源自同一条「理想气体 + 剩余性质」路线，都由同一本状态方程账本出账。学会一条，就等于学会了全部。

## 6 小结

- 焓与熵的全微分里，一切 $P$、$T$ 偏导数都由**状态方程**提供——这是性质计算的输入。
- **麦克斯韦关系**把不可测偏导换成可测偏导，源头是恰当微分的混合偏导相等。
- **剩余性质** $M^R = M - M^{ig}$ 把「真实 − 理想」的差距集中起来，全部可由状态方程积分得到。
- 真实性质 = 理想气体性质（查表/热容积分）+ 剩余性质（状态方程积分）。
- **逸度系数**由同一条路线得出，是相平衡计算的钥匙。

在下一节，我们把这些性质用在动力机械上，看看蒸汽、燃气如何把热变成功、效率极限在哪——这是**动力循环（朗肯/布雷顿/联合循环）**。
