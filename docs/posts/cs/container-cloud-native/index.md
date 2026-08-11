---
pageClass: plain-doc
---

# 容器与云原生

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Brendan Burns, Joe Beda & Kelsey Hightower, "Kubernetes: Up & Running" (3rd, 2022)
- Brendan Burns, "Designing Distributed Systems" (2018)
- Lee Atchison, "Architecting for the Cloud" (2020)

## 主题规划

<ProgressGrid cat="cs/container-cloud-native" />

### 第1篇

- [ ] 容器原理与 namespaces/cgroups (Burns Ch.2)
- [ ] Docker 镜像与运行时 (Burns Ch.2)
- [ ] Kubernetes 集群架构与控制平面（etcd/API Server/Scheduler） (Burns Ch.3)
- [ ] Kubernetes 对象模型 (Burns Ch.6)
- [ ] Pod 与服务发现 (Burns Ch.7)
- [ ] 控制器与声明式 API (Burns Ch.9)
- [ ] 弹性伸缩 HPA/VPA (Burns Ch.10)
- [ ] 服务网格 Istio (Atchison §8)

### 第2篇

- [ ] 云原生 12 因素应用 (Atchison §3)
- [ ] 分布式系统设计模式 (Burns "Designing" Ch.2)
- [ ] 持久化存储与 CSI (Burns Ch.16)
- [ ] 网络模型（CNI/Service Mesh） (Burns Ch.15)
- [ ] Ingress 与流量管理/负载均衡 (Burns Ch.8)
- [ ] 安全（RBAC/NetworkPolicy/Secrets） (Burns Ch.14,19)
- [ ] Helm 与配置管理（ConfigMap/Secret） (Burns Ch.13)
- [ ] 可观测性（Prometheus/OpenTelemetry） (书目外)
