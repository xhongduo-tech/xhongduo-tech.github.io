---
pageClass: plain-doc
---

# 虚拟化技术

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Jim Smith & Ravi Nair, "Virtual Machines: Versatile Platforms for Systems and Processes" (2005)
- Andrew S. Tanenbaum & Herbert Bos, "Modern Operating Systems" (4th, 2014)
- Bryant & O'Hallaron, "Computer Systems: A Programmer's Perspective" (3rd, 2015)

## 主题规划

<ProgressGrid cat="cs/virtualization" />

### 第1篇

- [x] [虚拟化导论与分类 (Smith §1)](./virtualization-intro-taxonomy)
- [x] [进程虚拟机与系统虚拟机 (Smith §2-3)](./process-vs-system-vms)
- [x] [二进制翻译与动态翻译 (Smith Ch.2)](./binary-translation)
- [x] [高级语言虚拟机（JVM/.NET CLR） (Smith Ch.5-6)](./high-level-language-vm)
- [x] [Hypervisor 类型与架构 (Tanenbaum §7.7)](./hypervisor-types-architecture)
- [x] [CPU 虚拟化与陷入模拟 (Smith §3.3)](./cpu-virtualization-trap-and-emulate)
- [x] [内存虚拟化与影子页表 (Smith §3.4)](./memory-virtualization-shadow-page-tables)
- [x] [半虚拟化与硬件内存虚拟化 EPT/NPT (Smith Ch.8)](./paravirt-and-hardware-memory-virtualization)

### 第2篇

- [x] [I/O 虚拟化与设备模拟 (Smith §3.5)](./io-virtualization-device-emulation)
- [x] [I/O 虚拟化 SR-IOV / VT-d 直通 (Tanenbaum §7.7)](./io-virtualization-sriov-vtd)
- [x] [硬件辅助虚拟化 Intel VT-x (Tanenbaum §7.7.2)](./hardware-assisted-virtualization-intel-vtx)
- [x] [容器与操作系统级虚拟化 (Tanenbaum §7.8)](./containers-os-level-virtualization)
