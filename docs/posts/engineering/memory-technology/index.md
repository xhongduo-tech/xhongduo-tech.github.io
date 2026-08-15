---
pageClass: plain-doc
---

# 存储器技术（DRAM/Flash/HBM/新型存储）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Sharma, "Semiconductor Memories: Technology, Testing, and Reliability" (IEEE Press, 1997)
- Keeth, Baker, Johnson, Lin, "DRAM Circuit Design: Fundamental and High-Speed Topics" (2nd ed., 2008)
- Cappelletti, Gola (eds.), "Flash Memories" (Springer, 1999)

## 主题规划

<ProgressGrid cat="engineering/memory-technology" />

### 第1篇

- [x] [存储层次与存储器指标（容量/带宽/延迟/耐久/成本）](./memory-hierarchy-and-metrics)
- [x] [SRAM（6T 单元读写分析、稳定性 SNM、外围电路）](./sram-6t-cell)
- [x] [DRAM 单元与阵列（1T1C、刷新、读出放大器）](./dram-cell-and-array)
- [x] [DRAM 接口演进（SDR→DDR5、LPDDR、GDDR）](./dram-interface-evolution)
- [x] [HBM（TSV 堆叠、宽接口、与 GPU/AI 芯片的协同）](./hbm-tsv-stacking)
- [x] [NAND Flash 单元（浮栅/电荷俘获、编程/擦除机理）](./nand-flash-cell)
- [x] [多值存储（MLC/TLC/QLC）与读扰动/保持力](./multilevel-storage-mlc-tlc-qlc)
- [x] [3D NAND（沟道孔刻蚀、层数演进、串堆叠）](./3d-nand)

### 第2篇

- [x] [NOR Flash 与嵌入式存储（eFlash 的 scaling 困境）](./nor-flash-and-eflash)
- [x] [新型非易失存储（PCM、RRAM、MRAM、FeRAM 原理与现状）](./emerging-nvm-pcm-rram-mram-feram)
- [x] [存储可靠性（磨损均衡、ECC、LDPC 纠错）与主控](./storage-reliability-ssd-controller)
- [x] [存内计算与近存计算（CIM/PIM、CXL 内存池化）](./compute-in-memory-cxl)
