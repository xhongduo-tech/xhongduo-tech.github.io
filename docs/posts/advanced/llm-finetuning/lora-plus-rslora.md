---
title: LoRA+ 与 rsLoRA：非对称学习率与秩稳定缩放
date: 2026-08-07
---

# LoRA+ 与 rsLoRA：非对称学习率与秩稳定缩放

<div class="epigraph">
<p>更新步伐的大小，往往比更新方向更重要。</p>
<footer>—— 引意自优化理论共识</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第四章 ｜ 2026-08-07</p>
</div>

## 为什么从 LoRA+ 与 rsLoRA 开始

前几节我们不断调 LoRA 的「表达力」——秩、目标模块、初始化。但还有一个几乎每个 LoRA 用户都会碰到、却很少深究的暗坑：**更新步幅**。同一套 LoRA，学习率与缩放系数差一点，收敛速度和最终效果可能差几个点。

2023–2024 年的两篇工作盯上了这块。**LoRA+** 发现 $A$ 与 $B$ 需要**不同的学习率**；**rsLoRA** 发现缩放系数 $\alpha/r$ 在**秩增大时失稳**。两者都只改一个数字，却都带来稳定的收益——它们是「LoRA 动力学」里性价比最高的两处改良。<span class="marginnote">把这两篇放在一起讲，是因为它们共享同一个视角：<strong>LoRA 的更新不是「均匀」的——A 和 B 的角色不对称，缩放与秩的关系也不平凡</strong>。理解了更新动力学，你就能解释很多 LoRA 调参中的玄学。</span>

## 1 有效学习率：A 与 B 真的对称吗

LoRA 的前向是 $W_0 + \frac{\alpha}{r}BA$。训练时，$A$ 和 $B$ 都在被梯度更新——但**它们对输出的影响方式完全不同**：

- 更新 $A$（$A \leftarrow A + \delta A$）：输出的变化量是 $\frac{\alpha}{r} B \delta A x$——**与 $\|B\|$ 成正比**；
- 更新 $B$（$B \leftarrow B + \delta B$）：输出的变化量是 $\frac{\alpha}{r} \delta B A x$——**与 $\|A\|$ 成正比**。

而 $B$ 从 0 初始化、$A$ 从高斯初始化——**训练初期 $\|B\| \ll \|A\|$**。这意味着：同样的学习率下，更新 $A$ 对输出的实际影响远小于更新 $B$。**两个矩阵用了「看似相同、实则不对称」的有效学习率**。

这就是 LoRA+ 的核心观察：**为了平衡两边的有效更新，$B$ 应该用比 $A$ 更大的学习率**。这不是玄学，而是「用不同学习率补偿不同缩放」的必然结论。

## 2 公式解析：LoRA+ 的非对称学习率

LoRA+ 的做法极简：给 $A$ 和 $B$ 各配一个学习率。

$$
\eta_A = \eta, \qquad \eta_B = \lambda \cdot \eta, \qquad \lambda \ge 2
$$

逐项拆解：

- $\eta$：基准学习率（与普通 LoRA 的配置相同）；
- $\lambda$：**学习率比值**，论文建议至少 2，常用 4–16；
- 训练时：$A$ 用 $\eta_A$ 更新、$B$ 用 $\eta_B$ 更新，其余一切不变。

**为什么 $\lambda \ge 2$ 更稳？** 论文从理论上证明：为了让 $A$、$B$ 的「有效更新」量级相当，$\eta_B/\eta_A$ 应与 $\|A\|/\|B\|$ 的某个稳态比值匹配——而该比值在训练中通常 > 1，所以 $\lambda$ 应当取较大的值。实验上，$\lambda$ 从 1 提到 4 通常带来稳定提升，之后进入平台期。

一个直觉支撑：$B$ 从 0 出发、负责「放大」$A$ 学到的东西；给 $B$ 更大的学习率，等于让「放大器」更快跟上 $A$ 的表达——两边的节奏对上了，收敛自然更快。<span class="marginnote">LoRA+ 论文还指出一个反直觉结论：<strong>$\lambda=1$（即普通 LoRA）在理论上并非最优，甚至在某些设置下是「次优的」</strong>——不是因为参数不够，而是因为两个矩阵的更新节奏不匹配。把这个「节奏」问题单独解决，是 LoRA+ 的全部贡献。</span>

实现上，LoRA+ 只需要在优化器里把参数分成两组、给不同的学习率：

```python
# 按参数名区分 A 与 B：约定 lora_A 前缀的是 A、lora_B 前缀的是 B
param_groups = [
    {"params": [p for n, p in model.named_parameters() if "lora_A" in n],
     "lr": 1e-4},                              # η_A
    {"params": [p for n, p in model.named_parameters() if "lora_B" in n],
     "lr": 4e-4},                              # η_B = 4 × η_A
    {"params": [p for n, p in model.named_parameters() if "lora" not in n],
     "lr": 0.0},                               # 冻结部分，学习率 0
]
optimizer = AdamW(param_groups)
```

注意「冻结部分学习率置 0」这行：LoRA+ 要求只有 $A$、$B$ 在动，基座零学习率——这既是 LoRA 的原则，也是分组优化的天然写法。HF PEFT 也把这个封装成了 `loraplus_lr_ratio` 参数，传一个 `loraplus_lr_ratio=4` 即可。

## 3 rsLoRA：秩增大时，缩放系数要跟着改

第二个暗坑藏在缩放系数 $\alpha/r$ 里。回想：$\Delta W = \frac{\alpha}{r} BA$，其中 $r$ 是秩。**问题：$r$ 改变时，$\Delta W$ 的幅度会怎样变？**

直觉上你可能觉得「除以 $r$ 已经把 $r$ 的影响消掉了」——但 rsLoRA 论文指出这是错的。关键在 $BA$ 这个乘积的范数：

$$
\mathbb{E}\big[\|BA\|\big] \propto \sqrt{r} \qquad \Longrightarrow \qquad \Big\| \frac{\alpha}{r} BA \Big\| \propto \frac{\sqrt{r}}{r} = \frac{1}{\sqrt{r}}
$$

逐项拆解：

- $BA$ 的期望范数随 $\sqrt{r}$ 增长——$r$ 越大，$A$、$B$ 合成的矩阵「天生更大」；
- 除以 $r$ 之后，残余的是 $1/\sqrt{r}$——**秩越大，有效更新幅度反而越小**；
- 结果：**大秩 LoRA 被缩放系数「压住」了**——你加了秩想增强表达，更新步幅却悄悄缩水，高秩收益被抵消。

**rsLoRA 的修复**：把缩放从 $1/r$ 改成 $1/\sqrt{r}$：

$$
\Delta W = \frac{\alpha}{\sqrt{r}}\, B A
$$

这样 $\sqrt{r} \cdot \frac{1}{\sqrt{r}} = 1$，**有效更新幅度不再随秩变化**——这就是「秩稳定缩放（rank-stabilized scaling）」名字的由来。<span class="marginnote">一个直观比喻：$\alpha/r$ 是「按面积分蛋糕」，$r$ 翻倍时每块小一半；rsLoRA 的 $\alpha/\sqrt{r}$ 是「按边长分蛋糕」——$r$ 翻倍时每块只小 $1/\sqrt{2}$，更接近「稳定」。对高秩 LoRA 用户，这个修正能直接兑现「加秩的收益」。</span>

## 4 实践用法：两个数字，怎么调

LoRA+ 与 rsLoRA 都是「改一个数字」的微调，改动成本几乎为零：

| 方法 | 改动 | 何时受益 | 实现 |
| --- | --- | --- | --- |
| LoRA+ | $\eta_B = \lambda \eta_A$ | 各种规模，尤其大秩/小学习率 | `optimizer` 分组学习率 |
| rsLoRA | $\alpha/\sqrt{r}$ 替代 $\alpha/r$ | 高秩（$r \ge 64$）尤其明显 | 缩放系数直接改 |

实践建议：

1. **LoRA+**：先跑普通 LoRA 基线，再给 $B$ 的学习率乘 2–4——训练脚本只需把参数分成 $A$、$B$ 两组、设不同学习率，改动几行。收益是收敛更快、通常也更好；
2. **rsLoRA**：如果 $r$ 较大（64+），把缩放从 $\alpha/r$ 换成 $\alpha/\sqrt{r}$。$r$ 小（8–16）时两者差别不大，可以不改；
3. **两者叠加**：LoRA+ 改的是「学习率分组」，rsLoRA 改的是「缩放系数」，完全正交——高秩 + 分组学习率可以同时用。

还有一个与 $\alpha$ 的关系值得记：**改 rsLoRA 后，$\alpha$ 的语义从「与 $r$ 联动」变成「纯步幅旋钮」**——因为 $\alpha/\sqrt{r}$ 中 $r$ 的影响已被消掉，$\alpha$ 独立控制更新幅度，调参直觉更干净。<span class="marginnote">一个常见的误区：<strong>以为 $\alpha = r$ 是「固定搭配」</strong>。事实上 $\alpha$ 是一个独立超参，与 $r$ 的「搭配」只是为了补偿 $1/r$ 缩放对步幅的压扁。一旦理解「$\alpha$ 本质是有效学习率旋钮」（前一篇《LoRA 工程细节》），你就不会死守 $\alpha = r$，而是按评测调它。</span>

最后提醒一点：这两个方法解决的是「更新动力学」，**不会改变 LoRA 的参数量**——它们不是「表达力增强」，而是「让已有表达力更充分释放」。所以在「参数预算不变」的前提下想提分，LoRA+ 与 rsLoRA 是最便宜的两刀；想真正的表达力上限，还得靠秩与初始化（DoRA、PiSSA）。## 5 小结

- **有效学习率不对称**：更新 $A$ 的效果与 $\|B\|$ 成正比、更新 $B$ 与 $\|A\|$ 成正比；$B$ 从 0 出发导致两边节奏失衡。
- **LoRA+**：$\eta_B = \lambda \eta_A$，$\lambda \ge 2$（常用 4）——给 $B$ 更大的学习率，收敛更快、效果更好。
- **rsLoRA**：$\|BA\| \propto \sqrt{r}$，所以 $\alpha/r$ 让大秩更新「缩水」；改用 $\alpha/\sqrt{r}$ 让更新与秩无关。
- 两个方法是「一个数字」的改动，**正交可叠加**：LoRA+ 管学习率分组、rsLoRA 管缩放系数。
- $\alpha$ 本质是有效学习率旋钮，不是 $r$ 的固定搭配——理解这个，调参不再玄学。
- 一句话记忆：**LoRA+ 管「两个矩阵的节奏」，rsLoRA 管「缩放随秩的稳定」**——一个调学习率、一个调缩放，正交可叠加。
- 实操顺序：**先跑通 LoRA 基线，再逐个开关这两个「数字级」改良**——每个都是一行改动，用评测确认收益再保留。

在下一节，我们看一个「让秩自己会呼吸」的方法：**AdaLoRA——基于重要性评分的秩自适应分配**。
