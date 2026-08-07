---
title: 自适应学习率：AdaGrad 与 RMSProp
date: 2026-08-07
---

# 自适应学习率：AdaGrad 与 RMSProp

<div class="epigraph">
<p>给每条路配一双合脚的鞋，才走得远。</p>
<footer>—— 依据查尔斯 · 狄更斯（Charles Dickens）的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§8.5.1、§8.5.2、李沐《动手学深度学习》§4.6 ｜ 2026-08-07</p>
</div>

## 为什么从 AdaGrad 与 RMSProp 开始

前几节的动量、学习率调度都在调整「**步长**」，但它们用的是**全局统一**的步长——所有参数维度共享同一个学习率。而真实损失曲面在各维度的尺度差异悬殊（**病态条件**）：某些方向陡、某些方向缓，统一步长「顾此失彼」。**自适应学习率（adaptive learning rate）**的思路是：**让每个参数维度拥有自己的步长**——陡峭方向步长小一点、平缓方向步长大一点，从而「从根上」缓解条件数问题。

**AdaGrad**（Duchi 等，2011）是第一个现代自适应方法：它用「历史梯度的平方和」为每维做缩放，频繁更新的维度步长自动变小。**RMSProp**（Tieleman & Hinton，2012）修正了 AdaGrad 的一个致命缺陷——平方和无限累积导致步长单调趋零——改用「指数加权平均」让缩放因子能「跟得上」梯度变化。**Adam** 则把 RMSProp 与动量结合，成为深度学习默认优化器。本节先把 AdaGrad 与 RMSProp 讲透，Adam 是它们的直接后继。<span class="marginnote">自适应方法的思想其实很古老：它等价于「对角近似的二阶方法」——用「梯度平方的估计」当「黑塞矩阵对角元的近似」，再用它逐维缩放步长。理解这条线索，就明白 AdaGrad 家族与《二阶优化近似》的内在联系：<strong>自适应学习率 = 廉价的逐维牛顿法</strong>。</span>

## 1 AdaGrad：按「历史梯度平方」缩放

**AdaGrad（Adaptive Gradient）**的更新规则：对每个维度 $i$，维护历史梯度平方的累积和 $\boldsymbol{r}$，用它缩放学习率：

$$
\boldsymbol{r} \leftarrow \boldsymbol{r} + \boldsymbol{g} \odot \boldsymbol{g}, \qquad
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{\delta + \sqrt{\boldsymbol{r}}} \odot \boldsymbol{g}
$$

其中 $\boldsymbol{g}$ 是梯度，$\delta$ 是防除零的小常数。三步拆解：

- **第一步，看累积**：$\boldsymbol{r}$ 累加每个维度的**梯度平方**——它在度量「这个维度最近有多活跃」。
- **第二步，看缩放**：每维的学习率是 $\frac{\eta}{\delta + \sqrt{r_i}}$。活跃维度（$r_i$ 大）→ 学习率小；不活跃维度（$r_i$ 小）→ 学习率大。
- **第三步，看效果**：**经常更新的维度被「踩刹车」，少见更新的维度被「加油门」**——这让每个维度以「自己的节奏」前进，直接缓解病态条件。

**AdaGrad 对稀疏特征特别友好**：推荐系统、NLP 里的特征（如「用户 ID」）出现频率悬殊，高频维度步长自动变小、低频维度步长自动变大——**AdaGrad 天然适配稀疏数据**，这是它在词嵌入与推荐场景里的成名之处。<span class="marginnote">AdaGrad 的「平方累积」是它「稀疏友好」的原因：稀疏特征梯度通常大而少，累积的 $r_i$ 不会太大，学习率得以保持；密集特征梯度小而多，$r_i$ 累积得快，学习率被压小。这个「按出现频率自动调步长」的性质，让它成为处理「幂律分布特征」的首选。</span>

**AdaGrad 的致命缺陷**：$\boldsymbol{r}$ 是**单调累积**（只加不减），随时间无限增长，学习率 $\frac{\eta}{\delta+\sqrt{r}}$ **单调递减到零**。训练后期，所有维度的步长都趋近于零，模型「学不动了」——这正是 RMSProp 要修的问题。

## 2 RMSProp：指数加权平均修复「死锁」

**RMSProp（Root Mean Square Propagation）**把「梯度平方的累积」换成「梯度平方的**指数加权平均**」：

$$
\boldsymbol{r} \leftarrow \rho\,\boldsymbol{r} + (1-\rho)\,\boldsymbol{g}\odot\boldsymbol{g}, \qquad
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{\delta+\sqrt{\boldsymbol{r}}}\odot\boldsymbol{g}
$$

与 AdaGrad 的唯一区别：$\rho$（常取 0.9 或 0.99）控制「近期梯度平方」的记忆长度。$\boldsymbol{r}$ 现在是一个**滑动平均**：

- **近期梯度大** → $\boldsymbol{r}$ 快速上升 → 步长变小（对「新出现的陡峭」快速反应）。
- **近期梯度小** → $\boldsymbol{r}$ 下降 → 步长恢复（不再被远古大梯度「永久锁死」）。

**修复了什么？** AdaGrad 的 $\boldsymbol{r}$ 只增不减，训练后期步长趋零「死锁」；RMSProp 的 $\boldsymbol{r}$ 能增能减，步长**动态调节**——这是 RMSProp 相比 AdaGrad 的决定性改进。<span class="marginnote">RMSProp 用 $\sqrt{\boldsymbol{r}}$（均方根）做缩放：$r_i$ 是「梯度平方的平均」，$\sqrt{r_i}$ 度量「梯度的典型大小」。所以 RMSProp 的缩放是「除以典型梯度大小」——直觉上，<strong>它把梯度「归一化」了：不管梯度量级多大，更新步长的量级大致一致</strong>。这正是它对新尺度变化鲁棒的原因。</span>

**易错点一：** AdaGrad 与 RMSProp 的 $\boldsymbol{r}$ 语义不同——AdaGrad 是「**累积**（单调增）」，RMSProp 是「**滑动平均**（可增可减）」。这导致 AdaGrad「训练不能太久」，RMSProp「可以持续训练」。混淆两者是调参新手常见的困惑来源。

## 3 自适应方法的统一视角：逐维缩放 = 对角预处理

把 AdaGrad/RMSProp 放进「**预处理矩阵**」的统一框架。梯度下降的标准形式是 $\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\boldsymbol{g}$；自适应方法把它推广为

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\boldsymbol{D}^{-1/2}\boldsymbol{g}
$$

其中 $\boldsymbol{D}$ 是对角矩阵，对角线是「梯度二阶矩的估计」（$\boldsymbol{D}_{ii}=r_i$）。对比**牛顿法** $\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\boldsymbol{H}^{-1}\boldsymbol{g}$：

- **牛顿法**：用完整的黑塞矩阵 $\boldsymbol{H}$ 预处理——**精确**但代价 $O(n^2)$。
- **自适应方法**：用「梯度平方的对角估计」当黑塞的**对角近似**——**粗糙**但代价 $O(n)$。

**「自适应 = 廉价的二阶方法」**这个视角解释了它的核心收益：它校正了「各维尺度不一致」（即病态条件），虽然不如牛顿法精确，但每步只要 $O(n)$，成本可控。<span class="marginnote">把 $\boldsymbol{D}_{ii} = \mathbb{E}[g_i^2]$ 与黑塞对角元 $\boldsymbol{H}_{ii} = \mathbb{E}[(\partial J/\partial \theta_i)^2]$ 对比（在高斯近似的 Fisher 信息框架下），两者惊人地相似——这就是「自适应学习率在逼近 Fisher 信息」的深层联系，也是 K-FAC 等「二阶自适应」方法的出发点。</span>

**易错点二：** 自适应方法的「逐维缩放」可能**扭曲方向**。牛顿法用 $\boldsymbol{H}^{-1}$ 精确旋转方向；对角近似只缩放坐标轴——如果真实最优方向是「坐标轴的 45° 混合」，对角缩放无法完美对齐。这在实践中通常无伤大雅，但解释了为什么「自适应方法在病态但不沿坐标轴」的问题上不如真正的二阶方法。

## 4 公式解析：RMSProp 的归一化效应

把 RMSProp 的更新拆开，看它如何「逐维归一化」。记当前维度的梯度为 $g$，其均方根估计为 $\sqrt{r}$，则有效更新步长为

$$
\Delta\theta = \frac{\eta}{\delta+\sqrt{r}}\,g = \eta \cdot \frac{g}{\sqrt{r}}
$$

- **第一步，看比值**：$\frac{g}{\sqrt{r}}$ 是一个「无量纲」的量——它度量「当前梯度相对于『典型梯度』有多突出」。若当前梯度与历史典型大小一致，比值约为 $\pm 1$；若异常大，比值大于 1（步长大）；若异常小，比值小于 1（步长小）。
- **第二步，看常数步长**：在「梯度大小稳定」的区域，$\frac{g}{\sqrt{r}} \approx \pm 1$，于是每步 $\Delta\theta \approx \pm\eta$——**不管原梯度量级多大，RMSProp 的步长都接近 $\eta$**。
- **第三步，看对尺度的鲁棒性**：若把整个损失函数放大 100 倍（所有梯度 ×100），$\sqrt{r}$ 也 ×100，比值 $\frac{g}{\sqrt{r}}$ 不变——**RMSProp 对「函数尺度」不敏感**，这是它相比朴素 SGD 的显著优势。<span class="marginnote">「尺度不变性」是自适应方法相对 SGD 的隐藏优势：SGD 的学习率要随损失尺度调整（损失大 100 倍，学习率要小 100 倍），RMSProp/Adam 则自动归一化。这在实践中意味着「<strong>换损失函数、换数据尺度时，Adam 的学习率不用怎么动，SGD 的要大调</strong>」——这是很多人「用了 Adam 就回不去 SGD」的原因之一。</span>

## 5 AdaGrad vs RMSProp vs SGD 的选型

| 方法 | 每维步长 | 稀疏数据 | 长期训练 | 典型场景 |
| --- | --- | --- | --- | --- |
| SGD | 全局固定 | 不友好 | 可持续 | 泛化优先、成熟 pipeline |
| AdaGrad | 单调递减 | 极友好 | 会「死锁」 | 稀疏特征、嵌入训练 |
| RMSProp | 动态调节 | 友好 | 可持续 | 需要自适应但不想用 Adam |

**易错点三：** 自适应方法**不等于「免调参」**。它们确实对学习率更鲁棒（量级容差大），但「选择哪个自适应方法、$\rho$ 取多少、$\eta$ 取多少」仍是需要验证的超参数。**「自适应」消解的是「尺度问题」，不是「一切调参问题」**。

**易错点四：** 自适应方法的「逐维归一化」在稀疏奖励（强化学习）、离群梯度上可能「过分自信」——大梯度被自动缩小，可能掩盖「该大步退出的信号」。在需要「方向精确」的场景（如 RNN 的语言建模），**SGD + 动量**有时反而优于 Adam——「自适应 vs 朴素」没有绝对胜者，要按任务裁决。<span class="marginnote">一个流传很广的经验：<strong>CNN 图像分类常用 SGD + 动量（+ 权重衰减），Transformer/NLP 常用 AdamW</strong>。原因之一是自适应方法自带隐式正则（梯度平方的平滑近似于 L2），与显式权重衰减叠加时需用 AdamW 解耦——下一节讲 Adam 时会把这条线索接上。</span>

## 6 小结

- **AdaGrad**：按「历史梯度平方累积」逐维缩放；**稀疏友好**但学习率**单调趋零**（死锁）。
- **RMSProp**：把累积换成**指数加权平均**，步长可增可减，修复死锁；缩放用「均方根」。
- 统一视角：自适应 = **对角近似的二阶方法**（$\boldsymbol{D}^{-1/2}\boldsymbol{g}$），代价 $O(n)$。
- RMSProp 的归一化效应：步长 $\approx \pm\eta$，对函数尺度不敏感。
- 选型：稀疏用 AdaGrad，长期训练用 RMSProp/Adam，泛化优先用 SGD+动量。
- 自适应 ≠ 免调参，且「方向扭曲」在特殊方向上有代价。

在下一节，我们把自适应学习率与动量**合二为一**，得到今天最常用的优化器——这就是 **Adam 及其变体（AdamW、AMSGrad）**。
