---
title: 近似推断：MCMC 与变分推断
date: 2026-08-07
---

# 近似推断：MCMC 与变分推断

<div class="epigraph">
<p>你不可能总是得到你想要的，但若你尝试，有时你会发现得到的是你需要的。</p>
<footer>—— 滚石乐队（The Rolling Stones）</footer>
</div>

<div class="article-byline">
<p>第四级 · 机器学习 ｜ 周志华《机器学习》第14章 ｜ 2026-08-07</p>
</div>

## 为什么从近似推断开始

上一节说：图模型在链/树上可精确推断，但在高树宽的复杂图上，精确推断是 **NP 难**。可现实里的模型偏偏常是复杂的——稠密的贝叶斯网、大图上的 MRF、高维潜变量模型。面对「精确算不动」的敌人，机器学习有一条古老而通用的出路：**近似**。

**近似推断（approximate inference）**分两大流派：

- **采样派（MCMC）**：既然算不出后验的解析式，就从后验里**大量采样**，用样本均值近似期望——「抽得够多，均值就准」；
- **优化派（变分推断）**：既然真实后验太复杂，就找一个**简单的近似分布**，通过优化让它尽量贴近真实后验——「用一个好算的分布去逼近」。

两条路线是「用随机性凑」与「用优化凑」的两种哲学，也分别是当代贝叶斯深度学习（MCMC）与变分自编码器（变分）的数学根基。<span class="marginnote">「采样 vs 变分」的分野贯穿整个贝叶斯机器学习：MCMC 无偏但慢（要采很多样本才收敛）；变分快但可能有偏（近似分布族不一定够灵活）。大模型时代，VAE 用变分、贝叶斯推理用 MCMC、扩散模型用「去噪」——近似推断的思想渗进了最前沿。</span>

## 1 MCMC：用采样近似期望很多推断任务本质是「求后验下的期望」：$E_P[f] = \int f(\boldsymbol{x}) P(\boldsymbol{x}) d\boldsymbol{x}$。若能从 $P$ 中独立采样，大数定律给出
$$E_P[f] \approx \frac{1}{T}\sum_{t=1}^{T} f\left(\boldsymbol{x}^{(t)}\right), \qquad \boldsymbol{x}^{(t)} \sim P$$

但**直接从 $P$ 采样**通常也难——后验的归一化常数（配分函数）算不出，只知道未归一化的 $\tilde{P}(\boldsymbol{x})$。**MCMC（Markov Chain Monte Carlo）**解决「不能直接采样」：构造一条**马尔可夫链**，其**平稳分布**恰好是目标分布 $P$，让链游走足够久，游走轨迹的样本就「近似独立」地来自 $P$。<span class="marginnote">「让一条马尔可夫链的平稳分布等于目标分布」是 MCMC 的核心构造：即使不知道 $P$ 的归一化常数，只要知道未归一化形式，就能设计转移规则使链收敛到 $P$。这绕开了「算配分函数」这个 NP 难的坎——用「游走」替代「计算」。</span>

**Metropolis-Hastings（MH）算法**是 MCMC 的通用框架：

1. 从当前状态 $\boldsymbol{x}$，按提议分布 $q(\cdot \mid \boldsymbol{x})$ 提出新状态 $\boldsymbol{x}'$；
2. 按**接受概率** $\alpha = \min\left(1, \frac{\tilde{P}(\boldsymbol{x}')q(\boldsymbol{x}\mid\boldsymbol{x}')}{\tilde{P}(\boldsymbol{x})q(\boldsymbol{x}'\mid\boldsymbol{x})}\right)$ 接受或拒绝；
3. 接受则跳到 $\boldsymbol{x}'$，否则留在 $\boldsymbol{x}$；重复。

**吉布斯采样（Gibbs sampling）**是 MH 在「逐变量条件采样」下的特例：每次只更新一个变量，按「给定其余变量时的条件分布」采样——复杂图模型里常比通用 MH 高效。<span class="marginnote">MH 的接受率公式里有 $\tilde{P}$ 的<strong>比值</strong>——归一化常数被约掉了，这正是「只知道未归一化形式也能采样」的机制。吉布斯采样更进一步：每个变量的满条件分布通常可以从图结构直接读出（只依赖它的马尔可夫毯），实现极简单——它是 MRF、LDA 话题模型里的默认采样器。</span>

## 2 变分推断：用优化逼近后验**变分推断（variational inference）**不采样，而是**优化**。设真实后验 $P(\boldsymbol{z}\mid\boldsymbol{x})$ 难算，找一个简单分布族 $\mathcal{Q}$（如「各变量独立的分解分布」），在其中找 $q^*$ 使它最接近 $P$：
$$q^* = \arg\min_{q \in \mathcal{Q}} \; \text{KL}(q(\boldsymbol{z}) \| P(\boldsymbol{z}\mid\boldsymbol{x}))$$

直接最小化 KL 仍要算 $P(\boldsymbol{z}\mid\boldsymbol{x})$，但可以转而最大化**证据下界（ELBO）**——它与「最小化 KL」等价，却只含可算的项：

$$\mathcal{L}(q) = \mathbb{E}_{q}\left[\log P(\boldsymbol{x}, \boldsymbol{z})\right] - \mathbb{E}_{q}\left[\log q(\boldsymbol{z})\right]$$

最大化 ELBO 的第一项（鼓励 $q$ 落在「数据似然高」的区域）+ 第二项（熵项，鼓励 $q$ 分散）——平衡之下，$q$ 既贴近数据、又不过分集中。<span class="marginnote">「最大化 ELBO = 最小化 KL」的等价性，是变分推断的灵魂：ELBO 是「对数证据 $\log P(\boldsymbol{x})$」的下界（由琴生不等式，第7章 EM 的同款工具），把「算不了的对数证据」换成「能优化的下界」。VAE 的训练目标正是这个 ELBO——变分推断是生成模型的理论骨架。</span>

## 3 公式解析：ELBO 为什么是「下界」- **第一步，写对数证据**：$\log P(\boldsymbol{x}) = \log \int P(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z}$——这个积分一般不可算。- **第二步，引入 $q$**：$\log P(\boldsymbol{x}) = \log P(\boldsymbol{x}) \int q(\boldsymbol{z})d\boldsymbol{z}$，用琴生不等式把 $\log$ 放进积分：$\log P(\boldsymbol{x}) \geq \int q(\boldsymbol{z}) \log \frac{P(\boldsymbol{x},\boldsymbol{z})}{q(\boldsymbol{z})} d\boldsymbol{z}$。
- **第三步，认出 ELBO**：右边的积分正是 $\mathcal{L}(q) = \mathbb{E}_q[\log P(\boldsymbol{x},\boldsymbol{z})] - \mathbb{E}_q[\log q(\boldsymbol{z})]$——它是 $\log P(\boldsymbol{x})$ 的下界；
- **第四步，看差距**：$\log P(\boldsymbol{x}) - \mathcal{L}(q) = \text{KL}(q \| P(\boldsymbol{z}\mid\boldsymbol{x})) \geq 0$——**差距恰是 KL**，最大化 ELBO 即最小化 KL。

**直觉一句话**：变分推断把「算后验」变成「调一个简单分布的参数让 ELBO 最大」——「够不着的后验」被「够得着的下界」逼近，数学上等价。

## 4 核心对比表：MCMC vs 变分推断| 维度 | MCMC | 变分推断 || --- | --- | --- |
| 基本手段 | 采样（随机） | 优化（确定性） |
| 得到的后验 | 样本（无偏，渐进精确） | 近似分布（有偏，依赖分布族） |
| 计算特点 | 慢（要烧掉 burn-in、采很多） | 快（梯度优化） |
| 适用规模 | 小中规模、需精确 | 大规模、容忍近似 |
| 代表算法 | MH、吉布斯采样 | 平均场、VAE |
| 理论基础 | 马尔可夫链、大数定律 | 变分法、ELBO |
| 典型风险 | 收敛慢、混合差 | 分布族不够、欠拟合后验 |

**辨析｜易错点：** MCMC 的样本**不是独立**的（链上相邻样本相关），且需要**预烧（burn-in）**期丢弃收敛前的样本；变分的近似分布族若太简单（如平均场假设变量独立），会**低估后验的不确定性**——两者各有各的「不精确」。**「无偏但慢」vs「快但有偏」是选择的分水岭**。<span class="marginnote">深度时代两者都在进化：随机梯度 MCMC、变分自编码器（VAE）、以及「采样 + 优化的混合」。理解 MCMC 与变分的分工（随机 vs 优化），是读懂现代贝叶斯深度学习的钥匙——也呼应第16章强化学习里「探索（随机）vs 利用（优化）」的同一哲学。</span>

## 知识速查：近似推断

**本节关键词**
- MCMC
- MH
- 吉布斯采样
- 变分推断
- ELBO
- KL
- 采样
- 优化

**三条常见误区**
1. 把 MCMC 的样本当独立——链上样本相关需 burn-in；
2. 把变分当「精确」——近似分布族决定偏差；
3. 以为采样与变分互斥——现代方法常混合。

**核心结论**
1. MCMC 用马尔可夫链采样近似后验；
2. MH 接受率含未归一化概率的比值；
3. 变分推断最大化 ELBO 等价于最小化 KL；
4. MCMC 无偏但慢、变分快但有偏。

**与全书/后续的连接**
- 第14章 MRF 配分函数难算；
- 第7章 EM 与琴生不等式；
- VAE/扩散模型的数学根基。

**常见面试题**
1. 问：为什么 MCMC 能绕开配分函数？ 答：接受率只含概率比值，归一化常数被约掉。
2. 问：ELBO 与对数证据的差距是什么？ 答：恰是 KL(q||P)，最大化 ELBO 即最小化 KL。

**一句话记忆**
MCMC 无偏但慢、变分快但有偏——用随机性凑或优化凑，两条路都算近似。

## 5 小结- **近似推断**在精确推断 NP 难时出手，分**采样派（MCMC）**与**优化派（变分）**。- **MCMC** 构造平稳分布为目标分布的马尔可夫链，用样本均值近似期望；**MH 算法**与**吉布斯采样**是两大代表。
- **变分推断**在简单分布族里找 $q$ 逼近后验，最大化 **ELBO** 等价于最小化 KL。
- ELBO：$\mathcal{L}(q) = \mathbb{E}_q[\log P(\boldsymbol{x},\boldsymbol{z})] - \mathbb{E}_q[\log q(\boldsymbol{z})]$，是「对数证据」的可优化下界。
- 选择权衡：MCMC 无偏但慢、变分快但有偏；两者是大模型/贝叶斯深度学习的共同根基。

## 本节路线图

- **第1节**：MCMC：用采样近似期望
- **第2节**：变分推断：用优化逼近后验
- **第3节**：公式解析：ELBO 为什么是「下界」
- **第4节**：核心对比表：MCMC vs 变分推断
- **小结**：要点复盘与下一课衔接

## 复习自查清单

读完后，试着不翻书复述以下各点：

- [ ] **MCMC** 构造平稳分布为目标分布的马尔可夫链，用样本均值近似期望；**MH 算法**与**吉布斯采样**是两大代表。
- [ ] **变分推断**在简单分布族里找 $q$ 逼近后验，最大化 **ELBO** 等价于最小化 KL。
- [ ] ELBO：$\mathcal{L}(q) = \mathbb{E}_q[\log P(\boldsymbol{x},\boldsymbol{z})] - \mathbb{E}_q[\log q(\boldsymbol{z})]$，是「对数证据」的可优化下界。
- [ ] 选择权衡：MCMC 无偏但慢、变分快但有偏；两者是大模型/贝叶斯深度学习的共同根基。
- [ ] **第1节**：MCMC：用采样近似期望
- [ ] **第2节**：变分推断：用优化逼近后验
- [ ] **第3节**：公式解析：ELBO 为什么是「下界」
- [ ] **第4节**：核心对比表：MCMC vs 变分推断
- [ ] **小结**：要点复盘与下一课衔接

在下一节，我们看概率图模型的明星应用：**话题模型（隐狄利克雷分配 LDA）**。
