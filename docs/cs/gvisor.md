---
title: gVisor
date: 2026-09-08
section: cs
---

# gVisor

<div class="epigraph">
<p>gVisor 的 Sentry 在用户态实现 Linux 系统调用：应用 ptrace 或 systrap 进入 Sentry，宿主内核看见的是 Sentry 的调用，而不是应用的。</p>
<footer>—— 据 gVisor 文档；[seccomp]/ptrace 与 [FUSE](/cs/fuse) 为用户态代内核对照</footer>
</div>

[上一课](/cs/oci-image-layers)留下的缺口接到本课。 [runc](/cs/container-runtime) 把 syscall 直接给宿主。[KVM](/cs/kvm-qemu) 是另一极端。缺口是 **应用内核在用户态**：gVisor。

## 问题

Sentry 实现 VFS、net、mm 子集。Gofer 代文件。缺口：兼容性（不是每个 ioctl）；性能税；与 [LSM](/cs/lsm-selinux) 宿主仍包一层。本课不把 Sentry 内部写成 Go 课。

<span class="marginnote">术语翻译：Sentry 就是一个跑在普通用户进程里的「迷你 Linux 内核」——用 Go 把 VFS、网络栈、内存管理这些内核子系统重写了一遍，专门接应用发来的系统调用。</span>

<span class="marginnote">直觉类比：Sentry 像一位全职翻译中介。应用只会说「Linux 方言」（300 多个 syscall），中介把它翻译成宿主内核听的少数几句「基础用语」；宿主从头到尾只见中介，见不到应用本人——应用带进来的恶意请求也就够不着宿主内核。</span>

<span class="marginnote">平台：ptrace 慢，KVM 平台用硬件隔离跑 Sentry。对象是缩减内核攻击面。</span>

## 方法

应用 syscall → 陷入 Sentry → Sentry 用有限宿主 syscall 完成。对照 [FUSE](/cs/fuse)：文件可走 gofer。对照 [seccomp]：过滤 vs 重新实现。对照 [unikernel](/cs/unikernel) 后课：一个截 syscall，一个无 syscall。

```mermaid
flowchart TD
  APP["应用 syscall"] --> SEN["Sentry 用户态内核"]
  SEN --> HOST["少量宿主调用"]
  SEN --> GOF["Gofer 文件"]
```

## 机制

gVisor 把 Linux ABI 的实现从宿主内核挪到用户进程，使内核漏洞对应用更远。代价是慢与不全。不要写成「比 VM 安全」的绝对句。与 [eBPF](/cs/ebpf-observability)：观测要看 Sentry 不是宿主进程树直觉。

不支持的 syscall 会 ENOSYS，应用崩。

```mermaid
flowchart TD
  subgraph COMPARE["syscall 路径对比"]
    R["runc：应用"] -->|syscall 直接进入| HK["宿主内核（攻击面全暴露）"]
    G["gVisor：应用"] --> SEN2["Sentry（用户态实现）"] -->|少量调用| HK
    V["VM：应用"] --> GUEST["客户内核"] -->|虚拟化出口| HV["VMM / 硬件"]
  end
```

<span class="marginnote">常见误区：gVisor 不是让容器跑得更快的加速器，而是一张「性能税换安全距离」的价目表——syscall 越密集（高频小文件、高频网络）税越重；syscall 稀疏的作业付得很少。</span>


实现上：Sentry 的网络栈自己实现，和宿主 conntrack 不是一张表。ptrace 平台每个 syscall 两次陷入，KVM 平台用硬件隔离跑 Sentry。不支持的 ioctl 直接失败。 读法上只引用[上一课](/cs/oci-image-layers)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「虚拟化与隔离进阶 / 容器与内核形态」课序里，对象是 **gVisor**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有平台后端。不保证 GPU。下一课微型 VM：Firecracker。


版本字段会变，课序钉的是机制对象「gVisor」，不是某一主线内核的结构体名。
后课默认：可用用户态内核截 syscall。专用微型 VMM，下一课 Firecracker。

## 小结

- gVisor 用 Sentry 在用户态实现 Linux ABI。
- 减宿主内核暴露，付性能与兼容。
- Firecracker 是下一课。
- 出处：gVisor 文档；OCI；KVM 平台说明。
