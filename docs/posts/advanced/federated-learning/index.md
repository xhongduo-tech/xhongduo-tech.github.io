---
pageClass: plain-doc
---

# 联邦学习

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Qiang Yang, Yang Liu & Tianjian Chen, "Federated Learning" (Synthesis Lectures on AI and ML 2021)
- Brendan McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
- Peter Kairouz et al., "Advances and Open Problems in Federated Learning" (Foundations and Trends in ML 2021)

## 主题规划

<ProgressGrid cat="advanced/federated-learning" />

### 第1篇

- [x] [联邦学习框架与分类 (Yang et al., Ch.1)](./federated-learning-framework)
- [x] [联邦平均 FedAvg (McMahan et al., 2017 §3)](./client-selection-incentives)
- [x] [异构数据与非 IID (Kairouz et al., 2021 §2)](./communication-efficiency-quantization)
- [x] [联邦优化算法 FedProx/SCAFFOLD (Kairouz et al., 2021 §3)](./differential-privacy)
- [x] [通信效率与梯度压缩/量化 (Kairouz et al., 2021 §3)](./federated-averaging-fedavg)
- [x] [差分隐私保护 (Kairouz et al., 2021 §4)](./federated-learning-fairness)
- [x] [安全聚合协议 (Kairouz et al., 2021 §5)](./federated-learning-framework)
- [x] [客户端选择与激励 (Yang et al., Ch.4)](./federated-learning-systems)

### 第2篇

- [x] [纵向联邦与联邦迁移 (Yang et al., Ch.5)](./fedprox-scaffold)
- [x] [联邦学习的公平性 (Kairouz et al., 2021 §6)](./heterogeneous-data-non-iid)
- [x] [联邦学习系统部署 (Kairouz et al., 2021 §8)](./secure-aggregation)
