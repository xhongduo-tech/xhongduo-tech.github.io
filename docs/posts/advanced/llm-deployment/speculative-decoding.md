---
title: 投机解码的原理与接受率分析
date: 2026-08-07
---

# 投机解码的原理与接受率分析

<div class="epigraph">
<p>用便宜的猜测，换昂贵的确认。</p>
<footer>—— 投机解码思想的一句话总结（源自 Leviathan et al., 2023）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ 投机解码论文（Leviathan et al. / Chen et al., 2023） ｜ 2026-08-07</p>
</div>

## 为什么从投机解码开始

本专题反复强调：decode 是 Memory-Bound——每次只生成一个 token，但要把整个模型的权重从显存搬一遍。于是出现了一个反直觉的机会：**生成 token 很贵，但「验证 token」没那么贵**。如果有一个便宜的小模型能猜出大模型接下来会说什么，我们让大模型一次性「验证」一串猜测，猜对的部分就白赚了——一次权重搬运，换回多个 token。<span class="marginnote">这就是投机解码（speculative decoding）的直觉：<strong>用猜来换带宽</strong>。大模型一次 decode 的代价是固定的（权重访存），但能同时验证一批候选 token——把「串行生成」变成「批量验证」。</span>

本篇讲投机解码的核心算法（draft 与 verify）、接受率（acceptance rate）的数学，以及为什么它能保持**分布一致**（sample-consistent）——这是投机解码与普通加速技巧的本质区别。

## 1 投机解码的算法骨架

投机解码（Leviathan et al., 2023 与 Chen et al., 2023 同期提出）的流程分四步：

1. **草稿（draft）**：用小模型（或同一模型的自回归变体）逐 token 生成 $\gamma$ 个候选 token——小模型便宜，生成 $\gamma$ 个的代价远低于大模型生成 1 个。
2. **并行验证（verify）**：把「真实上下文 + $\gamma$ 个候选」一次性喂给大模型做一次 forward，得到每个候选位置的预测分布。
3. **逐个接受（accept）**：从位置 1 开始，按「与草稿一致则接受，不一致则拒绝并重采样」的规则判定。接受 $t$ 个就白赚 $t$ 个 token。
4. **回填**：被拒绝的位置，用大模型的真实采样替换；然后带着已接受的部分继续下一轮。

**关键性质：分布一致性（distribution-preserving）。** 接受/拒绝规则（典型是「从大模型分布里采样 $x$，若等于草稿则接受，否则重新采样」）被证明**保证输出分布与大模型直接自回归采样完全一致**。<span class="marginnote">这意味着投机解码不是近似加速，而是<strong>精确加速</strong>——输出分布零偏差，这对依赖概率性的生成（如采样温度）的应用极其重要。</span>

## 2 接受率：决定加速比的关键数字

投机解码的加速比由**接受率（acceptance rate）**驱动。定义：草稿模型猜的 token 与大模型分布一致的概率。对第 $t$ 个草稿 token，接受概率为：

$$\alpha_t = \sum_{x} \min(p(x), q(x))$$

其中 $p$ 是大模型分布、$q$ 是草稿模型分布（对已接受的前缀条件化）。$\alpha_t$ 衡量两个分布的重叠程度：**$\alpha_t = 1$ 当且仅当两分布完全相同**；$\alpha_t$ 越小，草稿越不靠谱。

期望每轮接受的 token 数（$\gamma$ 个候选）：

$$\mathbb{E}[\#\text{accepted}] = \sum_{t=1}^{\gamma} \prod_{i=1}^{t-1} \alpha_i \approx \frac{1 - \alpha^{\gamma}}{1 - \alpha} \quad (\text{假设平稳 } \alpha)$$

**接受率 $\alpha$ 越高，加速越接近 $\gamma$ 倍；$\alpha$ 越低，收益趋近于 1（没收益）。** 典型 LLM 上自回归草稿的 $\alpha \approx 0.7–0.8$，$\gamma = 4$ 时加速约 2–3 倍。<span class="marginnote">加速比上限的直觉：<strong>大模型 forward 一次的成本约等于小模型 $\gamma$ 次 + 一次验证</strong>，当接受率高时，平均每个大模型 forward 能产出 $\approx 1/(1-\alpha)$ 个 token。</span>

## 3 草稿模型的代价平衡

投机解码不是免费的：草稿模型的生成与验证都有开销。设大模型单次 forward 耗时 $T$，草稿模型单步耗时 $t \ll T$，草稿生成 $\gamma$ 个 + 验证一次的总耗时：

$$T_{\text{round}} = \gamma \cdot t + T$$

每轮产出约 $N_{\text{acc}}$ 个 token，因此**平均每 token 耗时**：

$$\frac{T_{\text{round}}}{N_{\text{acc}}} = \frac{\gamma t + T}{1 + \sum_{t=2}^{\gamma}\prod_{i<t}\alpha_i}$$

- **第一步，看分子**：$\gamma t$ 是草稿成本，$T$ 是大模型验证成本。**草稿模型越小 $t$ 越小，但 $\alpha$ 也越低**（猜得不准）——「小快灵」vs「大而准」需要平衡。
- **第二步，看分母**：$N_{\text{acc}}$ 随 $\alpha$ 增长。$\alpha$ 太低时 $N_{\text{acc}} \approx 1$，总耗时退化为 $\gamma t + T$，**比直接 decode 的 $T$ 还慢**——投机解码在不匹配的草稿模型下会负优化。
- **第三步，看收益条件**：加速条件 $\gamma t + T < N_{\text{acc}} T$，即 $\gamma t < (N_{\text{acc}} - 1) T$。**只有当草稿足够便宜、且接受率够高时，投机解码才划算**。

这就是为什么「草稿模型的选择与训练」（下一篇）那么关键：它直接决定 $\alpha$，进而决定投机解码是 2 倍加速还是负优化。

## 4 公式解析：期望接受数与加速比

把加速比写成接受率的显式函数。令 $\gamma$ 个候选、平稳接受率 $\alpha$，则：

- **第一步，算期望接受数**：每轮期望产出

$$N_{\text{acc}} = 1 + \alpha + \alpha^2 + \cdots + \alpha^{\gamma-1} = \frac{1 - \alpha^\gamma}{1 - \alpha}$$

（第 1 个 token 无条件接受，第 $t$ 个需前 $t-1$ 个都接受，概率 $\alpha^{t-1}$。）
- **第二步，算期望耗时**：$T_{\text{round}} = \gamma t + T$。
- **第三步，比加速**：假设草稿成本可忽略（$t \ll T$），加速比 $\approx N_{\text{acc}} = (1-\alpha^\gamma)/(1-\alpha)$。代入 $\alpha = 0.8$、$\gamma = 4$：$N_{\text{acc}} = 1 + 0.8 + 0.64 + 0.512 = 2.95$——**接近 3 倍加速**。代入 $\alpha = 0.5$：$N_{\text{acc}} = 1.875$，不足 2 倍。

**加速比对接受率极其敏感**：$\alpha$ 从 0.8 降到 0.5，加速比几乎腰斩。这解释了所有投机解码变体（Medusa、EAGLE）的核心战场：**提高 $\alpha$**。

## 5 小结

- **投机解码 = 草稿 + 批量验证**：小模型猜 $\gamma$ 个，大模型一次 forward 验证，猜对的白赚。
- **分布一致**：接受/拒绝规则保证输出分布与大模型直接采样**零偏差**，是精确加速而非近似。
- **接受率 $\alpha$ 决定加速**：期望每轮产出 $\approx (1-\alpha^\gamma)/(1-\alpha)$，$\alpha$ 越高加速越接近 $\gamma$ 倍。
- **有成本平衡**：草稿太贵或不匹配会负优化，收益条件 $\gamma t < (N_{\text{acc}}-1)T$。
- **所有变体的战场是提高 $\alpha$**：Medusa、EAGLE 都是「让草稿猜得更准」的工程方案。

在下一节，我们聚焦「草稿从哪来」——**草稿模型的选择与训练**，看自回归草稿、同一模型自推测与蒸馏式草稿的取舍。
