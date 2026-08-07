---
title: DeepFM：用 FM 替代 Wide 侧的手工特征
date: 2026-08-07
---

# DeepFM：用 FM 替代 Wide 侧的手工特征

<div class="epigraph">
<p>最好的结构，是不需要你手动搭的那一种。</p>
<footer>—— 综合自 DeepFM 论文（Huawei / 2017）</footer>
</div>

<div class="article-byline">
<p>第四级 · 推荐系统 ｜《推荐系统实践》第 7 章 + DeepFM 论文（2017）｜ 2026-08-07</p>
</div>

## 为什么从 DeepFM 开始

Wide & Deep 解决了「记忆 + 泛化」双通道，但留下一个刺眼的手工活：**wide 侧的交叉特征还得人工构造**。DeepFM 的动机非常干脆——**用 FM 这一自动二阶交互模块，替换掉 wide 侧的全部人工特征**。从此，低阶交叉（FM）与高阶交叉（深度网络）都由模型自动完成，唯一的输入就是原始特征。<span class="marginnote">DeepFM 出自华为诺亚方舟实验室 2017 年论文《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》。它和 Wide & Deep 的最大区别就在 wide 侧：一个用人工交叉，一个用 FM。名字里的「F」即 Factorization Machine。</span>

这一节讲 DeepFM 的结构、它如何让「低阶 + 高阶」共享 embedding、以及它为何成为工业 CTR 模型的经典基线。

## 1 DeepFM 的整体结构

DeepFM 由两部分组成，**共享同一份特征 embedding**：

- **FM 组件**：负责建模一阶与二阶特征交互（见 [[fm-factorization-machines]]）。
- **Deep 组件**：多层 MLP，负责建模高阶特征交互。

两部分输出相加后过 sigmoid：

$$
\hat{y} = \sigma\left( \hat{y}_{\text{FM}} + \hat{y}_{\text{DNN}} \right)
$$

关键设计：**FM 组件与 DNN 组件共用底层的 embedding 表**。同一个特征 $i$ 的 embedding $\mathbf{e}_i$，既用于 FM 的二阶交互（内积），也送入 DNN。共享的意义不只是省参数，更是让**低阶与高阶信息在同一个表示空间里协同学习**——互相补充，而不是各学各的。

## 2 FM 组件的组成

FM 组件的预测分两部分：

$$
\hat{y}_{\text{FM}} = \underbrace{w_0 + \sum_i w_i x_i}_{\text{一阶项}} + \underbrace{\sum_{i<j} \langle \mathbf{e}_i, \mathbf{e}_j \rangle x_i x_j}_{\text{二阶项}}
$$

- **一阶项**：线性加权，类似 LR，刻画特征的独立影响。
- **二阶项**：用 embedding 内积刻画两两交互，且可用 $O(nk)$ 的「平方减自身平方」公式高效计算（见 [[fm-factorization-machines]] 的公式解析）。

这里没有人工交叉特征——**一切两两组合都由 FM 自动学**。这就是「用 FM 替代 wide 侧手工特征」的精确含义。

## 3 Deep 组件：高阶交叉交给网络

Deep 组件的输入是各特征 embedding 的拼接：

$$
a^{(0)} = [\mathbf{e}_{x_1}; \mathbf{e}_{x_2}; \dots; \mathbf{e}_{x_n}]
$$

然后过 $L$ 层全连接（带激活函数）：

$$
a^{(l+1)} = \sigma\left( W^{(l)} a^{(l)} + b^{(l)} \right)
$$

最终输出 $\hat{y}_{\text{DNN}}$。**DNN 的每一层都在做「特征之间的再组合」，层数越多，能表达的交互阶数越高**。FM 止步于二阶，DNN 可以学到三阶、四阶乃至全连接的交互——这是 DeepFM 高阶能力的来源。

## 4 公式解析：FM 与 DNN 为什么能互补

把「共享 embedding + 相加输出」的设计拆三步：

- **第一步，同一份 embedding，两种用法**：特征 $i$ 的 embedding $\mathbf{e}_i$，FM 组件拿它做二阶内积，DNN 组件拿它做深层网络的输入。**这要求 embedding 同时承载「适合两两比较」和「适合非线性组合」的信息**，二者在训练中相互约束、相互增强。
- **第二步，低阶项防止「网络只顾高阶」**：DNN 理论上也能表达二阶交互，但实际训练中高阶交互的噪声会让低阶规律学不牢。FM 组件显式承担低阶，把网络解放出来专攻高阶——**分工明确，各司其职**。
- **第三步，相加而非拼接**：$\hat{y} = \sigma(\hat{y}_{\text{FM}} + \hat{y}_{\text{DNN}})$。相加是最简单的融合，让模型自动权衡低阶与高阶哪个信号更重要。**可解释性也友好**：FM 部分可单独解读为「二阶交互得分」。

## 5 DeepFM 在模型谱系中的位置

| 模型 | 低阶交互 | 高阶交互 | 需人工特征 |
| --- | --- | --- | --- |
| LR | 人工 | 无 | 是 |
| FM | 自动二阶 | 无 | 否 |
| Wide & Deep | 人工交叉 | DNN | 是 |
| **DeepFM** | **FM 自动二阶** | **DNN** | **否** |

DeepFM 可以看作 **Wide & Deep 的「免人工」版本**：把 wide 侧换成了 FM。它在大量公开数据集上稳定优于单独用 FM 或单独用 DNN，且端到端训练、无两阶段割裂，因此成为工业 CTR 模型的**标配基线**——任何新模型都要先跟 DeepFM 比一比。

工程注意：DNN 部分对 embedding 维度敏感（常见 8~32 维）、层数不宜过深（2~3 层足矣），FM 部分的二阶项可复用 $O(nk)$ 高效实现。

## 辨析｜DeepFM 与 Wide & Deep 的选型：不止「省人工」

DeepFM 是 Wide & Deep 的「免人工」版，但两者选型不是简单「省不省事」的问题，有三点值得权衡：

**权衡一：人工交叉是否还有价值。** Wide & Deep 的 wide 侧如果放「业务验证过的强交叉」——如「新用户 × 首屏位」「促销季 × 高转化类目」——这些交叉是「业务知识」的显式注入，FM 自动学不一定能精准捕捉（自动学到的交叉是「统计驱动」的，不是「业务驱动」的）。**当团队有强业务交叉时，Wide & Deep 的显式 wide 侧仍是优势**；纯数据驱动的场景，DeepFM 更省心。

**权衡二：二阶交互的「粒度」。** FM 组件的二阶交互是「两两内积」，覆盖的是「所有特征对」的低阶交互；DNN 组件的高阶交互是隐式的。**DeepFM 假设「低阶靠 FM、高阶靠 DNN」的分工**——如果你的场景「低阶交互」特别重要（用户×类目这种强二阶），DeepFM 天然匹配；如果高阶关系更重要，可考虑显式高阶交叉的 DCN/xDeepFM（[[dcn-xdeepfm-explicit-feature-crossing]]）。

**权衡三：训练成本与调参面。** DeepFM 省了「人工构造交叉」，但多了一个「FM 与 DNN 如何分担」的调参面（FM 侧的隐向量维度、DNN 的深度宽度）。**没有一个模型是「零调参」的**——省下来的交叉工作量，转移到了结构超参上。

**一句话**：有强业务交叉 → 保留 Wide & Deep 的显式 wide 侧；纯数据驱动、想自动低阶高阶兼得 → DeepFM。两者都是「记忆/泛化双通道」家族，选型看「业务知识有多值钱」。

**补充｜DeepFM 的「变体」与「基线意义」**：

DeepFM 是工业 CTR 模型的「事实基线」——新模型几乎都要先跟它对比。围绕它有两条常见的变体方向：

- **共享 embedding 的深度与宽度**：DNN 部分加宽加深、或换成「序列建模」（把用户行为序列喂进去，见 [[din-deep-interest-network]]）——「FM + 更强的主干」。
- **高阶交叉的显式化**：把 DNN 换成「显式交叉网络」（DCN 的 Cross 或 xDeepFM 的 CIN）——「FM + 显式高阶」——这是 [[dcn-xdeepfm-explicit-feature-crossing]] 的直接动机。

**为什么 DeepFM 适合当基线**：它结构对称（低阶/高阶各司其职）、无人工特征、训练稳定——**作为「复杂度与效果」的参照点，比「更强的模型」和「更弱的模型」都更有信息量**。看论文时，先看「它比 DeepFM 好多少」，比看「绝对指标」更能判断真伪。

**收尾一句话**：DeepFM 是「低阶（FM）自动 + 高阶（DNN）自动」的范式——它的意义不是「最佳」，而是「一个让『特征交叉自动化』这件事变得可复制的基线」。

（理解 DeepFM 的关键，是把它看成「两个自动交叉模块的接力」——FM 把低阶交互「显式」做掉，DNN 把高阶交互「隐式」做掉，而「共享 embedding」让两者在同一表示上协同。）

## 6 小结

- **DeepFM = FM（低阶）+ DNN（高阶）**，共享 embedding、输出相加，全自动特征交叉。
- FM 组件含一阶与二阶项，**替代 Wide & Deep 的 wide 侧人工特征**。
- DNN 组件逐层组合特征，**负责高阶交互**，层数决定交互阶数上限。
- 共享 embedding 让低阶与高阶在同一个表示空间协同，**相加输出让模型自己权衡两者**。
- DeepFM 是工业 CTR 的标准基线，后续的 DCN、xDeepFM 等在此基础上增强高阶交叉。

在下一节，我们将看到把「显式高阶特征交叉」做到更彻底的——**DCN 与 xDeepFM**。
