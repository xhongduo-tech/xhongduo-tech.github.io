---
pageClass: plain-doc
---

# 电源技术（开关电源/VRM/供电网络）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Pressman, Billings, Morey, "Switching Power Supply Design" (3rd ed., 2009)
- Erickson, Maksimović, "Fundamentals of Power Electronics" (3rd ed., 2020)
- Maniktala, "Switching Power Supplies A to Z" (2nd ed., 2012)

## 主题规划

<ProgressGrid cat="engineering/power-supply-technology" />

### 第1篇

- [ ] 电源架构总览（AC-DC 整流→PFC→DC-DC 的能量链）
- [ ] 开关变换拓扑（Buck/Boost/Buck-Boost 的工作模态）
- [ ] 隔离拓扑（反激/正激/半桥全桥/LLC 谐振）
- [ ] 磁性元件设计（变压器/电感、磁芯损耗与绕组损耗）
- [ ] 控制环路（电压/电流模式、补偿网络设计）
- [ ] 功率器件选型（MOSFET/GaN 在电源中的权衡，与功率半导体专题互链）
- [ ] 同步整流与多相 VRM（CPU/GPU 供电的瞬态响应挑战）
- [ ] 功率因数校正 PFC（升压 PFC、图腾柱无桥）

### 第2篇

- [ ] EMI 与安规（传导/辐射抑制、绝缘耐压认证）
- [ ] 热设计与效率优化（损耗分解、80 PLUS 体系）
- [ ] 电池充电管理（CC/CV、快充协议、电量计）
- [ ] 数字电源与智能供电（PMBus、服务器 48V 架构）
