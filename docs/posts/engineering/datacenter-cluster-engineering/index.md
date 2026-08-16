---
pageClass: plain-doc
---

# 算力集群与数据中心工程（SuperPOD/InfiniBand/万卡组网）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Barroso, Hölzle & Ranganathan, "The Datacenter as a Computer" (3rd ed., 2018)
- NVIDIA, "DGX SuperPOD Reference Architecture" 官方文档 (2024)
- Faisal et al., "The Datacenter as a Networked Computer: RDMA 与 RoCE 实践" (IEEE HotI, 2015)

## 主题规划

<ProgressGrid cat="engineering/datacenter-cluster-engineering" />

### 第1篇

- [x] [集群架构层次（整机柜→SuperPOD→多园区算力中心）](./cluster-architecture-hierarchy)
- [x] [GPU 互联网络（NVLink 域内 + InfiniBand/RoCE 域外的两层设计）](./gpu-interconnect-nvlink-fabric)
- [x] [无损以太网（RoCEv2、PFC/ECN 拥塞控制、DCQCN）](./lossless-ethernet-rocev2-dcqcn)
- [x] [网络拓扑（Fat-Tree/Dragonfly+/轨式优化 Rail-Optimized）](./network-topology-fat-tree-dragonfly)
- [x] [万卡集群的集合通信（NCCL/拓扑感知 AllReduce、网络拥塞实测）](./collective-communication-nccl-allreduce)
- [x] [作业调度（Slurm/Kubernetes 拓扑感知调度、Gang Scheduling）](./job-scheduling-slurm-kubernetes)
- [x] [训练容错（Checkpoint 策略、故障预测、弹性训练）](./training-fault-tolerance-checkpoint)
- [x] [存储系统（并行文件系统 Lustre/GPFS、检查点带宽墙）](./parallel-filesystem-lustre-gpfs)

### 第2篇

- [x] [供电基础设施（市电→UPS→母线→机柜的配电链、柴发与储能）](./power-infrastructure-ups-distribution)
- [x] [制冷基础设施（风冷/液冷混合、冷却塔、PUE/WUE 指标）](./cooling-infrastructure-liquid-cooling-pue)
- [x] [数据中心等级与可靠性（Tier I-IV、2N 冗余、可用性数学）](./datacenter-tier-reliability)
- [x] [绿色算力（余热回收、碳足迹、东数西算与算力网络）](./green-computing-carbon-footprint)
