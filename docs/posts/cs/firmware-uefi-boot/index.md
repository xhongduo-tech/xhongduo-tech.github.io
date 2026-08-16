---
pageClass: plain-doc
---

# 固件与启动链（BIOS/UEFI/嵌入式引导）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- UEFI Forum, "UEFI Specification" 与 "Platform Initialization (PI) Specification"（公开规范）
- Vincent Zimmer, Jiming Sun, Marc Jones, Stefan Reinauer, "Embedded Firmware Solutions: Development Best Practices for the Internet of Things" (Apress, 2015)
- Intel 开源固件文档（coreboot/EDK II 官方文档）

## 主题规划

<ProgressGrid cat="cs/firmware-uefi-boot" />

### 第1篇

- [x] [固件的位置（硬件与 OS 之间的隐形层、从上电复位开始）](./firmware-role-and-position)
- [x] [x86 启动链（RESET→SEC→PEI→DXE→BDS→OS 的接力）](./x86-boot-chain)
- [x] [UEFI 体系（驱动模型、Protocol、UEFI Shell 与变量服务）](./uefi-architecture)
- [x] [传统 BIOS 与 legacy 兼容（CSM、实模式遗产）](./legacy-bios-and-csm)
- [x] [安全启动（Secure Boot 信任链、TPM 度量启动）](./secure-boot-and-tpm)
- [x] [内存初始化（MRC 内存参考代码、SPD 读取与训练）](./memory-initialization-mrc)
- [x] [外设枚举（PCIe 枚举、ACPI 表的生成）](./device-enumeration-acpi)
- [x] [嵌入式引导（ARM 的 BootROM→TF-A→U-Boot→内核链）](./embedded-arm-boot)

### 第2篇

- [x] [开源固件（coreboot/LinuxBoot、固件供应链透明化）](./open-source-firmware)
- [x] [固件更新机制（ capsules 更新、防回滚、A/B 分区）](./firmware-update-capsules)
- [x] [固件安全（BIOS rootkit、Intel ME/PSP 的争议与边界）](./firmware-security)
- [x] [调试手段（串口日志、POST code、JTAG/SWD）](./firmware-debugging)
