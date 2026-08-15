---
pageClass: plain-doc
---

# EDA 算法与芯片设计流程

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Wang, Chang, Cheng, "Electronic Design Automation: Synthesis, Verification, and Test" (Morgan Kaufmann, 2009)
- Kahng, Lienig, Markov, Hu, "VLSI Physical Design: From Graph Partitioning to Timing Closure" (2nd ed., 2022)
- Lavagno, Martin, Scheffer (eds.), "Electronic Design Automation for IC System Design, Verification, and Testing" (2006)

## 主题规划

<ProgressGrid cat="cs/eda-algorithms" />

### 第1篇

- [x] [设计流程总览（规格→RTL→综合→布局布线→签核→流片）](./design-flow-overview)
- [x] [逻辑综合（两级/多级逻辑优化、工艺映射）](./logic-synthesis)
- [x] [高层次综合 HLS（调度、分配、绑定）](./high-level-synthesis)
- [x] [布图规划与布局（划分、模拟退火、解析式布局器）](./floorplanning-placement)
- [x] [时钟树综合（CTS、偏斜控制）](./clock-tree-synthesis)
- [x] [布线（Steiner 树、全局/详细布线、轨道分配）](./routing)
- [x] [静态时序分析 STA（时序图、RC 提取、OCV/AOCV）](./static-timing-analysis)
- [x] [物理验证（DRC/LVS/ERC、天线效应）](./physical-verification)

### 第2篇

- [x] [仿真与验证（事件驱动仿真、覆盖率、UVM 方法学）](./simulation-verification)
- [x] [形式验证（等价性检查、模型检验）](./formal-verification)
- [x] [可制造性设计 DFM（OPC 交互、良率感知设计）](./design-for-manufacturability)
- [x] [ML for EDA（布局/布线/良率预测的机器学习方法）](./ml-for-eda)
