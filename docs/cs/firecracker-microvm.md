---
title: Firecracker
date: 2026-09-08
section: cs
---

# Firecracker

<div class="epigraph">
<p>Firecracker 是用 Rust 写的瘦 VMM：KVM + 极少 virtio 设备，为每函数或每容器起一台微 VM，启动以毫秒计。</p>
<footer>—— 据 Firecracker 设计文档；[KVM/QEMU](/cs/kvm-qemu) 为完整 VMM 对照</footer>
</div>

[gVisor](/cs/gvisor) 截 syscall。[QEMU](/cs/kvm-qemu) 设备太多。缺口是 **microVM**：Firecracker 砍设备模型。

## 问题

只留 virtio-net/blk、串口、非常少的 PCI。缺口：与 [容器](/cs/container-runtime) 的 jailer（chroot、cgroup、seccomp 锁 VMM）；快照恢复。本课不把 Lambda 产品写成云广告。

<span class="marginnote">无 USB、无 VGA。对象是攻击面与冷启动，不是桌面虚拟化。</span>

<span class="marginnote">术语翻译：microVM 就是「砍到只剩必需设备的虚拟机」：没有显卡、USB 和大部分 PCI 设备，内核一醒就能跑，攻击者能碰的入口因此变少。传统 QEMU 为兼容几十年老硬件背了一大堆设备模型，Firecracker 干脆不实现。</span>

## 方法

jailer 起进程 → KVM 建 VM → 客内核 + 极简 rootfs。对照 QEMU：少 exit 路径种类。对照 runc：多一层客户内核。对照 [SR-IOV](/cs/sriov-passthrough)：微 VM 常用 virtio 而非直通。

<span class="marginnote">数字实例：传统云主机冷启动常以秒到几十秒计（固件引导、设备枚举、挂盘）；Firecracker 的启动在百毫秒量级、内存占用个位数 MB 起步——数量级的差距，才让「每个函数起一台 VM」在成本上成立。</span>

```mermaid
flowchart TD
  JAIL["jailer 锁 VMM"] --> FC["Firecracker"]
  FC --> KVM["KVM vCPU"]
  KVM --> GUEST["极简客户机"]
```

## 机制

Firecracker 把「每租户一内核」的隔离做到能水平铺开，用砍设备换启动与 CVE 面。客户仍是真内核，syscall 兼容好于 gVisor。不要写成 serverless 计费。与 [热迁移](/cs/live-migration)：有快照，完整 live 不是主路径。

```mermaid
flowchart TD
  APP["应用 / 函数"] --> R["容器 runc"]
  APP --> GV["gVisor"]
  APP --> FC["Firecracker 微 VM"]
  R --> HK["共享宿主内核: 隔离最薄, 启动最快"]
  GV --> UL["用户态内核过滤 syscall: 兼容性折损"]
  FC --> GK["独立裁剪客户内核 + KVM: 隔离强, 启动毫秒级"]
```

内核配置要为微 VM 裁剪——[kbuild](/cs/kernel-config-build)。

<span class="marginnote">常见误区：初学者把微 VM 当「更小的容器」。容器共享宿主内核，内核漏洞可能直接越过边界；微 VM 里跑的是另一个内核，逃逸得先攻破 KVM 与 VMM 这一层。代价是多一份内核内存与启动开销。</span>


实现上：API 套接字在 jailer 之后才听，攻击面从 VMM 进程开始锁。无 PCI 扫描减少客户攻击。快照把内存+设备状态当微函数冷启动加速。 读法上只引用[上一课](/cs/gvisor)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「虚拟化与隔离进阶 / 容器与内核形态」课序里，对象是 **Firecracker**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 API 套接字。不保证嵌套。下一课 Kata：容器接口 + VM。


版本字段会变，课序钉的是机制对象「Firecracker」，不是某一主线内核的结构体名。
后课默认：微 VMM 用 KVM 跑裁剪客户。Kata 把 OCI 接到 VM，下一课。

## 小结

- Firecracker：瘦设备模型的 KVM VMM。
- jailer 锁住 VMM 进程本身。
- Kata 是下一课。
- 出处：Firecracker 文档；KVM；OCI。
