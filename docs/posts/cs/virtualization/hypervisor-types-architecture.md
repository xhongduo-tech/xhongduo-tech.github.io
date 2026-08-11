---
title: Hypervisor 类型与架构
date: 2026-08-11
---

# Hypervisor 类型与架构

<div class="epigraph">
<p>构建软件设计有两种方式：一种是把它做得简单到没有明显的缺陷，另一种是把它做得复杂到看不出缺陷在哪里。</p>
<footer>—— 东尼 · 霍尔（C. A. R. Hoare）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 虚拟化技术 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么要把 Hypervisor 单独拆出来

系统虚拟机的灵魂是 **VMM（虚拟机监视器）**，今天更常叫 **Hypervisor**。它是整个虚拟化的「指挥中枢」：客户 
OS 以为自己是机器的主人，而真实硬件其实只听 Hypervisor 的。这一节解决三个问题：Hypervisor 有哪几类、
各自架构长什么样、以及它们各自的性能与适用场景。讲清楚了，后面 CPU、内存、I/O 三块虚拟化的细节，就都是在给某一类 
Hypervisor 的某个模块补课。

## 1 二分法：Type 1 与 Type 2

1974 年，戈德伯格（Robert P. Goldberg）在他的博士论文工作中给出了至今通用的分类：

**Type 1（裸机型 / bare-metal，也称原生型 native）**：Hypervisor 直接运行在硬件之上，
没有宿主 OS。它是「硬件之上的第一层软件」，拥有机器全部特权。代表：VMware ESXi、Xen、Microsoft 
Hyper-V、KVM（Linux 内核模块形态）。<span class="marginnote">Type 1 的名字直译「第一类」——直接贴着硬件；Type 2 是「第二类」——隔了一层宿主 OS。</span>

Type 1 的「没有宿主 OS」是它的灵魂：没有宿主 OS，就少一层被攻击、被调度、被抢占的中间软件；虚拟机的 CPU、内存、
中断直接面对硬件，性能损失压到最低。代价是它必须自带一套精简的设备驱动——所以 ESXi 只支持有限的网卡与存储卡，
换来的是「一个精简内核 + 虚拟化」的极致形态。

**Type 2（宿主型 / hosted）**：Hypervisor 作为宿主操作系统上的**普通程序**运行，
像任何应用一样受宿主 OS 调度。代表：VMware Workstation、Oracle VirtualBox、
QEMU（宿主形态）。

| 维度 | Type 1（裸机型） | Type 2（宿主型） |
| --- | --- | --- |
| 与硬件的关系 | 直接占有硬件 | 通过宿主 OS 间接访问 |
| 额外开销 | 一层（仅 VMM） | 两层（宿主 OS 调度 + VMM） |
| 性能 | 高，接近原生 | 低，受宿主 OS 调度影响 |
| 驱动依赖 | 自带精简驱动，设备支持面窄 | 复用宿主 OS 全部驱动 |
| 典型场景 | 数据中心、云 | 开发测试、个人桌面 |
| 例子 | ESXi、Xen、Hyper-V、KVM | Workstation、VirtualBox |

**核心结论：Type 1 用「放弃设备驱动复用」换「独占硬件的高性能」；Type 2 用「性能」换「即插即用的便利」**。

## 2 边界正在模糊：KVM 与 QEMU 的组合

教科书式的二分法在现实中并不干净，最典型的就是 Linux 上的 KVM 组合拳：

**KVM**（Kernel-based Virtual Machine，2007）是一个 Linux 内核模块，把「虚拟机」变成一个受内核管理的实体（进程）。它复用内核的调度器、内存管理、设备驱动——KVM 身兼「内核的一部分」，性质上更像 Type 1。
- **QEMU** 在用户态提供设备模拟与客户机 BIOS 等，本质是 Type 2 式的设备服务。

KVM + QEMU 因此被称为「Type 1.5」：**CPU/内存虚拟化的核心在特权模式的内核态（Type 1 精神），设备模拟在普通进程（Type 2 做法）**。
今天 AWS、阿里云、腾讯云的几乎所有云主机都是这个架构，它证明了二分法只是理解工具，现实产品是混血。

**辨析｜易错点：** 有人说「KVM 是 Type 2，因为它在 Linux 内核里」。判据不该是「有没有宿主 OS」，
而是**虚拟化核心路径是否绕过宿主 OS 的调度直接面对硬件**。KVM 的 vCPU 由内核直接调度，几乎没有宿主 OS 参与，
所以工程上公认把它归入 Type 1 家族。

## 3 特权层级：Hypervisor 站多高

无论 Type 1 还是 Type 2，Hypervisor 与客户 OS 之间必须分出一个高低。
经典模型是**特权环（privilege ring）**：

- 传统 OS 自己占据最内环（x86 的 ring 0，特权指令只允许在 ring 0 执行），应用在 ring 3。
- 虚拟化引入后：**Hypervisor 占据比 ring 0 更内的一环**（有人画作 ring −1），客户 OS 被压到 ring 0 或更外。客户 OS 执行特权指令时，因权限不足触发陷阱，落入 Hypervisor 处理——这就是「陷入模拟」的物理基础，也是下一节的主题。

架构设计上还有一个流派之争：

- **宏内核式 Hypervisor**：虚拟化的全部功能（调度、内存、设备）都塞进一个高特权内核，代表是 ESXi 与 Hyper-V。功能强、性能好，但攻击面大——一个设备模拟漏洞可能拖垮整台宿主机。<span class="marginnote">2015–2018 年陆续公开的 CVE-2015-3456（Venom）、CVE-2018-3646（L1TF）等虚拟化安全漏洞，都把矛头指向了 Hypervisor 高特权层的攻击面——这是「少即是多」的工程哲学在安全领域的又一次验证。</span>
- **微内核式 Hypervisor**：只保留「特权级不可少」的极简内核（调度、隔离），设备模拟与驱动放进取特权进程或宿主分区。代表是 Xen（dom0 承担管理）与各类微内核验证系统（seL4、OKL4）。**代码越少，越可验证，越安全**——这正是 Hoare 那句「简单到没有明显缺陷」在系统软件里的极致追求。

## 4 公式解析：Type 1 与 Type 2 的性能差在哪

设一次虚拟化操作的总开销为 $T$，原生执行为 $T_0$，则虚拟化开销为 $T - T_0$。Type 1 与 Type 2 的差异在于这笔开销里多不多宿主 OS 那一层：

$$
T_{\text{Type1}} = T_0 + T_{\text{vmm}}
$$

$$
T_{\text{Type2}} = T_0 + T_{\text{vmm}} + T_{\text{host}}
$$

拆三步：

- **第一步，$T_{\text{vmm}}$ 是两者都付的**：陷入处理、地址翻译、设备访问的拦截，Type 1 与 Type 2 都要做。
- **第二步，$T_{\text{host}}$ 是 Type 2 独有**：VMM 作为一个进程，它发起 I/O 要经过宿主 OS 的系统调用、进程调度、驱动路径；客户 OS 的中断要先被宿主 OS 接收再转发给 VMM 进程。每一步都多一次模式切换与上下文切换。
- **第三步，为什么差距没那么大**：现代 CPU 虚拟化把 $T_{\text{vmm}}$ 压得很低后，$T_{\text{host}}$ 的相对占比反而变大。但 Type 2 受益于宿主 OS 成熟的调度与驱动，很多工作负载下（尤其 I/O 少、CPU 密集的任务）差距被压缩到百分之几。**选 Type 1 还是 Type 2，先看瓶颈在哪。**

## 5 Hypervisor 架构的历史一瞥

- **1980 年代，VM/370 的 CP**：Type 1 的鼻祖，直接驱动 S/370 硬件。
- **1998 年，VMware**：先在 Windows/Linux 上做 Type 2（Workstation），2001 年推出 ESX（Type 1 服务器产品）。
- **2003 年，Xen**：开源 Type 1 半虚拟化，剑桥大学出身，性能惊艳，一度是云的默认选择。
- **2006–2007 年**：微软 Hyper-V（Type 1，依赖 Windows 的 hypervisor 层）、KVM 进入 Linux 内核（把 Type 1 的能力带进通用内核）。
- **2008 年起，硬件辅助成熟**：VT-x/EPT 成为标配，Type 1 与 Type 2 的性能差距进一步缩小，Type 2 在个人桌面依旧流行（VirtualBox、VMware Fusion）。

架构上的一个常见变体值得单独提一下：**微内核式 Hypervisor 的商业化**。除了学术派的 seL4、OKL4，实际部署里 
Xen 的 dom0 分区、以及容器安全的 Kata Containers（把容器装进一个轻量虚机）都体现了「把高特权代码尽量做小」
的原则。安全不是虚拟化的副业，而是 Hypervisor 架构的第一性约束——**特权代码越少，越值得被信任**。

把 Hypervisor 的架构抉择放进「从极限到大模型」的坐标系里，还有一层当代意义：**大型机留下的「分区」思想，正在 GPU 时代重新登场**。
NVIDIA 的 MIG（把一块 A100 切成多个独立 GPU 实例）、AMD 的 MxGPU、云上的 GPU 直通与 
vGPU——它们都是「把一块硬件按隔离边界切成多份」的 Hypervisor 式哲学，只是对象从 CPU 换成了 GPU。理解 
Type 1/Type 2 与宏微内核之争，等于拿到了读懂这些硬件级「分区器」的钥匙。

## 6 小结

- Hypervisor 分 **Type 1（裸机）** 与 **Type 2（宿主）**：前者直接面对硬件，后者作为宿主 OS 的普通进程运行。
- Type 1 用设备驱动便利换高性能，是数据中心的标配；Type 2 图省事，是开发桌面首选。
- **KVM + QEMU** 是混血「Type 1.5」：内核态做 CPU/内存虚拟化，用户态做设备模拟，是今日云的绝对主流。
- Hypervisor 位于特权环最内层（ring −1），客户 OS 被降权运行，特权指令触发陷阱落入 VMM。
- 性能模型：$T_{\text{Type2}} = T_{\text{Type1}} + T_{\text{host}}$，Type 2 多付一层宿主 OS 的调度与转发开销。
- 宏内核 Hypervisor 功能全、攻击面大；微内核 Hypervisor 极简、可验证、更安全。

在下一节，我们钻入 Hypervisor 最核心的活计——**CPU 虚拟化**：特权环如何运行、敏感指令如何被捕获、
陷入模拟为何是教科书正解，又为何在 x86 上失灵。
