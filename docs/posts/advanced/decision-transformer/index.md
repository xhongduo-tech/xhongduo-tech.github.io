---
pageClass: plain-doc
---

# Decision Transformer（序列建模 RL）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Richard S. Sutton & Andrew G. Barto, "Reinforcement Learning: An Introduction" (2nd, 2018)
- Lili Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (NeurIPS 2021)
- Sergey Levine et al., "Offline Reinforcement Learning: Tutorial and Review" (2020)

## 主题规划

<ProgressGrid cat="advanced/decision-transformer" />

### 第1篇

- [x] [MDP 与序列建模对比 (Sutton & Barto §3)](./mdp-vs-sequence-modeling)
- [x] [Return-to-go 表示 (Chen et al., Decision Transformer §3)](./return-to-go-representation)
- [x] [因果自注意力 Transformer (Chen et al., Decision Transformer §3)](./causal-self-attention-transformer)
- [x] [离线 RL 与数据分布 (Levine et al., Offline RL Tutorial §3)](./offline-rl-data-distribution)
- [x] [轨迹采样策略 (Chen et al., Decision Transformer §4)](./trajectory-sampling-strategy)
- [x] [与 Q-learning 对比 (Chen et al., Decision Transformer §5)](./decision-transformer-vs-q-learning)
- [x] [传统离线 RL 基线 BCQ/CQL/IQL (Fujimoto et al., 2019; Kumar et al., 2020; Kostrikov et al., 2022)](./offline-rl-baselines-bcq-cql-iql)
- [x] [Return-conditioned 监督学习范式（行为克隆视角） (Emmons et al., RvS 2022)](./return-conditioned-supervised-learning)

### 第2篇

- [x] [Trajectory Transformer 与其他序列建模 RL (Janner et al., Trajectory Transformer 2021)](./trajectory-transformer)
- [x] [Online Decision Transformer 扩展 (Zheng et al., 2022)](./online-decision-transformer)
- [x] [序列建模 RL 局限性 (Chen et al., Decision Transformer §6)](./limitations-of-sequence-modeling-rl)
