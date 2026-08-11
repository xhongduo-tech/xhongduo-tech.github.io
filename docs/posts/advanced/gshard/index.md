---
pageClass: plain-doc
---

# GShard（论文解析）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Dmitry Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2020)
- William Fedus et al., "Switch Transformers" (JMLR 2022)
- Noam Shazeer et al., "Outrageously Large Neural Networks" (ICLR 2017)

## 主题规划

<ProgressGrid cat="advanced/gshard" />

### 第1篇

- [x] [条件计算动机 (Lepikhin §2)](./conditional-computation-motivation)
- [x] [MoE 层结构 (Lepikhin §3)](./moe-layer-structure)
- [x] [Top-2 路由机制 (Lepikhin §3.2)](./top-2-routing)
- [x] [辅助负载均衡损失 (Lepikhin §3.3)](./load-balancing-loss)
- [x] [专家并行 Expert Parallelism (Lepikhin §4)](./expert-parallelism)
- [x] [XLA SPMD 编译 (Lepikhin §5)](./xla-spmd-compilation)
- [x] [万亿参数模型训练 (Lepikhin §6)](./trillion-parameter-training)
- [x] [多语言翻译应用 (Lepikhin §7)](./multilingual-translation)
