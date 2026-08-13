---
pageClass: plain-doc
---

# AI 硬件：NVIDIA GB200/GB300-NVL72

对标 NVIDIA 官方白皮书与机架系统文档，按章节逐节写成博文，系统重建从 Grace Blackwell 超级芯片、NVLink 域到 NVL72 机架系统、液冷供电与大规模集群扩展的完整知识。学完这个专题，就写完了 NVIDIA 当前旗舰级 AI 基础设施的软硬件全貌。

## 对标教材

- NVIDIA, "NVIDIA Blackwell Platform for the Era of AI" (White Paper, 2024)
- NVIDIA, "NVIDIA GB200 NVL72 Server Reference Design" (官方文档, 2024)
- NVIDIA, "NVIDIA GB300 NVL72 White Paper" (官方文档, 2025)

## 主题规划

<ProgressGrid cat="advanced/nvidia-gb200-gb300-nvl72" />

### 第1篇

- [x] [Grace Blackwell 超级芯片与 B200/GB300 GPU（GB200 NVL72 白皮书 第1章）](./grace-blackwell-superchip)
- [x] [Blackwell GPU 架构与 SM/张量核心（Blackwell 架构白皮书 第2章）](./blackwell-gpu-architecture)
- [x] [第二代 Transformer Engine 与 FP4/FP6 精度（Blackwell 架构白皮书 第3章）](./transformer-engine-fp4)
- [x] [Grace CPU 与 NVLink-C2C 异构互连（GB200 NVL72 白皮书 第2章）](./grace-cpu-nvlink-c2c)
- [x] [HBM3e 显存与显存带宽（Blackwell 架构白皮书 第4章）](./hbm3e-memory-bandwidth)

### 第2篇

- [x] [NVLink5 与 NVSwitch 无阻塞全互联（GB200 NVL72 白皮书 第3章）](./nvlink5-nvswitch)
- [x] [NVLink 域与 72-GPU 单域互连（GB200 NVL72 白皮书 第3章）](./nvlink-domain-72-gpu)
- [x] [GB200 NVL72 机架物理结构与机柜布局（GB200 NVL72 官方文档 第2章）](./nvl72-rack-layout)
- [x] [机架级 Scale-up 与 GPU 对架构（GB200 NVL72 白皮书 第4章）](./scale-up-gpu-pair)

### 第3篇

- [x] [液冷散热系统（冷板与 CDU）（GB200 NVL72 官方文档 第4章）](./liquid-cooling-cdu)
- [x] [电源架构与母线配电（GB200 NVL72 官方文档 第5章）](./power-architecture)
- [x] [机架管理、固件与 NVIDIA 软件栈（GB200 NVL72 官方文档 第6章）](./management-software-stack)
- [x] [系统可靠性、冗余与 OOB 管理（GB200 NVL72 官方文档 第7章）](./reliability-oob-management)

### 第4篇

- [x] [训练与推理性能基准对比（NVIDIA 白皮书 性能章）](./training-inference-performance)
- [x] [从 H100/H200 到 GB200 的代际提升（Blackwell 架构白皮书 第5章）](./hopper-to-blackwell)
- [x] [GB300 NVL72 升级：增强算力与网络（GB300 NVL72 白皮书 第2章）](./gb300-nvl72-upgrade)
- [x] [NVL576 机架群与万卡集群扩展（GB300 NVL72 白皮书 第3章）](./nvl576-cluster-scaling)
