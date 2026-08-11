---
pageClass: plain-doc
---

# MoE 混合专家架构

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Dmitry Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2020)
- William Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (JMLR 2022)
- Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017)

## 主题规划

<ProgressGrid cat="advanced/mixture-of-experts" />

### 第1篇

- [ ] 稀疏门控 MoE 原理 (Shazeer et al., Sparsely-Gated MoE 2017)
- [ ] Top-k 路由机制 (Fedus et al., Switch Transformer 2022)
- [ ] 负载均衡损失 (Lepikhin et al., GShard 2020)
- [ ] 负载均衡与专家退化（dead experts）机制 (Dai et al., DeepSeek-MoE 2024)
- [ ] 专家容量因子 (Fedus et al., Switch Transformer 2022)
- [ ] 专家并行 Expert Parallelism (Lepikhin et al., GShard 2020)
- [ ] 路由策略对比 (Shazeer et al., 2017)
- [ ] DeepSeek MoE 细粒度专家 (Dai et al., DeepSeek-MoE 2024)

### 第2篇

- [ ] MoE 训练稳定性 (Fedus et al., Switch Transformer 2022)
- [ ] MoE 推理部署与专家卸载 (Rajbhandari et al., DeepSpeed-MoE 2022)
