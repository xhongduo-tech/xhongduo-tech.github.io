---
pageClass: plain-doc
---

# AI 服务器整机柜工程（NVL72/NVLink/液冷/供电母排）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- NVIDIA, "GB200 NVL72 System Architecture" 官方技术文档 (2024)
- Barroso, Hölzle & Ranganathan, "The Datacenter as a Computer" (3rd ed., 2018)
- ASHRAE, "Liquid Cooling Guidelines for Datacom Equipment Centers" (2nd ed., 2021)
- OCP, "Open Rack V3 (ORV3) 供电与机柜规范" 官方规范

## 主题规划

<ProgressGrid cat="engineering/ai-server-rack-engineering" />

### 第1篇

- [ ] 从单机到整机柜（Scale-Up 与 Scale-Out 的架构分界）
- [ ] NVLink 域设计（72 GPU 全互连、NVLink Switch 托盘拓扑）
- [ ] 铜互连背板（ACC/AEC 有源电缆、背板布线的信号完整性）
- [ ] 计算托盘结构（Bianca 板：Grace CPU + Blackwell GPU 的 1U 形态）
- [ ] 供电母排（Busbar 大电流传输、48V/±400V 高压直流演进）
- [ ] 机柜级液冷（冷板/manifold 分液器/CDU 冷量分配单元）
- [ ] 120kW+ 机柜的热设计（热密度、进出水温、漏液检测）
- [ ] 机柜管理（RMC 机柜管理控制器、遥测与固件带外管理）

### 第2篇

- [ ] 可靠性工程（RAS 特性、故障域隔离、GPU 热插拔与降频降级）
- [ ] OCP 开放计算（ORV3/DC-MHS 规范、供应链开放生态）
- [ ] 交付形态（L10→L11→L12 集成级别、数据中心部署约束）
- [ ] 演进路线（GB200→GB300→Rubin Ultra、600kW 机柜与 Kyber 架构）
