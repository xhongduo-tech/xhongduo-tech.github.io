---
pageClass: plain-doc
---

# 高速互连与信号完整性（SerDes/PCIe/DDR）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Bogatin, "Signal and Power Integrity—Simplified" (3rd ed., 2018)
- Johnson, Graham, "High-Speed Digital Design: A Handbook of Black Magic" (1993)
- Hall, Heck, "Advanced Signal Integrity for High-Speed Digital Designs" (2009)

## 主题规划

<ProgressGrid cat="engineering/high-speed-interconnect" />

### 第1篇

- [ ] 从并行到串行的历史转折（时钟偏移为何逼出 SerDes）
- [ ] 传输线理论（特性阻抗、反射、端接策略）
- [ ] S 参数与信道表征（插损/回损/串扰、TDR 测量）
- [ ] 编码与均衡（8b/10b→PAM4、CTLE/DFE/FFE 均衡链）
- [ ] 时钟与抖动（PLL/CDR、抖动分解 RJ/DJ）
- [ ] PCIe 协议栈（物理层→数据链路→事务层、代际翻倍史）
- [ ] DDR 存储接口（拓扑/端接/读写训练、信号时序余量）
- [ ] 封装与板级协同（Die-封装-PCB 三级互连的信号接力）

### 第2篇

- [ ] 光互连（AOC/光模块、共封装光学趋势）
- [ ] 电源完整性（SSN 同步开关噪声、PDN 设计）
- [ ] 仿真工作流（IBIS-AMI、信道仿真、眼图合规）
- [ ] 标准生态（PCI-SIG/JEDEC/OIF 的规范工程）
