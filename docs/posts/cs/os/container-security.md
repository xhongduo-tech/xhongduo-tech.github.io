---
title: 容器安全：Capability、Seccomp 与 AppArmor
date: 2026-08-07
---

# 容器安全：Capability、Seccomp 与 AppArmor

<div class="epigraph">
<p>Namespace 是「看不见」，cgroup 是「用不多」——而容器安全要回答更硬的问题：「就算被攻破，还能干什么？」</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Linux 内核设计与实现》与容器安全 ｜ 2026-08-07</p>
</div>

## 为什么从容器安全开始

容器共享宿主内核——**这是它的优势（轻）也是它的软肋（一旦突破，共享内核 = 直达宿主）**。Namespace 只是「视图隔离」，**不是安全边界**。容器安全的三件套——**Capability（权限收缩）、Seccomp（系统调用过滤）、AppArmor（强制访问控制）**——把「攻破容器后的破坏力」压到最小。这是容器专题的收尾，也是最小特权原则（回顾《保护域》）的容器落地。<span class="marginnote">回顾《Namespace》：视图隔离 ≠ 安全边界。容器里的进程<strong>共享宿主内核</strong>——攻击者若拿到容器进程的控制权，它的系统调用、权限都直达宿主内核。<strong>容器安全 = 把「被攻破的容器进程」的权限与能力压到最小。</strong></span>

## 1 Capability：不给 root，给「最小能力」

**Linux Capability**：把 root 的**全部特权拆成一组小能力**——进程可以只持有部分能力，而不是「root = 一切」。

- **全部特权**（传统 root）：CAP_SYS_ADMIN、CAP_NET_ADMIN、CAP_SYS_PTRACE、CAP_KILL…
- **capability 模型**：**给进程最小必需的能力**（回顾最小特权）。

**容器默认 capability（Docker）**：容器内进程**不是全 root**，只保留少数必要能力（如 `CAP_NET_BIND_SERVICE` 绑定低端口、`CAP_SETUID` 设 UID）。

**capability 的意义**：

- 容器内即使「我是 root（UID 0）」，**也只是 UID 0，不是 full root**——关键特权（`CAP_SYS_ADMIN` 等）默认被丢弃。
- `--cap-drop ALL --cap-add NET_BIND_SERVICE`：丢弃全部、只加必要——**最小特权的显式表达**。

**公式解析：capability 收缩攻击面**

设 root 全部能力为 $C_{all}$，容器保留 $C_{container} \subseteq C_{all}$：

$$\text{攻击者可用的特权} \propto |C_{container}|$$

- 无 capability：容器 root = 全部能力——攻击者拿到一切。
- 最小 capability：容器只保留必要能力——**攻击者能做的（如改宿主内核参数、挂载）被剥夺**。
- **capability 把「root 的一票否决权」拆成「可细分的权限」**——这是最小特权的直接实现。

**直觉**：capability 回答「**容器里的 root 是不是真 root**」——**不是**，它是「被阉割的 root」。`CAP_SYS_ADMIN`（挂载、命名空间操作）被丢弃，攻击者就不能靠「容器内 root」动宿主。

## 2 Seccomp：过滤系统调用

**Seccomp（secure computing mode）**：**限制进程能调用的系统调用**——白名单/黑名单过滤。

- **容器默认 Seccomp 配置**：**阻止危险系统调用**——如 `mount`、`kexec_load`、`open_by_handle_at`（绕过路径检查）、`reboot`、`ptrace`（调试/注入）。
- **为什么需要**：Namespace 让容器「看不到」宿主，但**系统调用是内核接口**——若容器能调 `mount` 或 `ptrace`，就可能操纵宿主。**Seccomp 从「系统调用」层面设卡。**

**Seccomp 的机制**：

- 进程设置 seccomp 过滤器（BPF 程序）——每次系统调用被过滤。
- 默认动作：`ALLOW`（允许）、`ERRNO`（返回错误）、`KILL`（杀死进程）。

**Seccomp 与 Capability 的分工**：

- **Capability**：管「特权操作」（有没有权限做）。
- **Seccomp**：管「能不能调这个系统调用」（连调用都不行）。

**Seccomp 的收益**：**即使容器进程被攻破，攻击者想调的危险系统调用（mount、ptrace）直接被杀/报错**——系统调用层面封死。

**辨析｜易错点：** 「Namespace 已经隔离了，不需要 Seccomp」是危险认知。**Namespace 隔离「视图」，不隔离「系统调用」**——容器里的进程可以调任何没被限制的系统调用。**`CAP_SYS_ADMIN` + `mount` 的组合可以让容器「挂载宿主文件系统」突破隔离**——这正是 Seccomp 要封的。**「视图隔离 + 系统调用过滤」缺一不可。**

## 3 AppArmor：强制访问控制

**AppArmor（Application Armor）**：Linux 的 **MAC（强制访问控制）** 实现（类似 SELinux，但基于路径）——**限制进程能访问的文件与资源**。

- **AppArmor 配置文件**：定义「进程能读/写/执行哪些路径」。
- Docker 默认 AppArmor 配置：限制容器**访问宿主敏感路径**（如 `/proc` 的宿主信息、`/sys` 的设备）。

**AppArmor vs SELinux**（回顾《SELinux 与 MAC》）：

- **SELinux**：基于标签（label），粒度细，配置复杂。
- **AppArmor**：基于路径（path），更易用，现代 Ubuntu 默认。

**AppArmor 的收益**：**即使 Seccomp 与 Capability 都没拦住，AppArmor 还限制「碰哪些文件」**——多层防御的最后一层。

**纵深防御（Defense in Depth）**：容器安全是**多层叠加**：

```
Namespace（视图隔离）
  → Capability（权限收缩）
    → Seccomp（系统调用过滤）
      → AppArmor（文件访问控制）
        → 只读文件系统 / 非 root 用户（最后防线）
```

**每一层被攻破，下一层仍在**——攻击者要穿透所有层才能抵达宿主（回顾《SELinux》的纵深防御与《保护域》的最小特权）。

**公式解析：容器安全的「洋葱模型」**

设各层防御被攻破的概率为 $p_1, p_2, \ldots, p_n$：

$$\text{攻击到达宿主} \approx p_1 \times p_2 \times \cdots \times p_n$$

- 每层独立设防，攻击要**全部穿透**。
- 各层概率相乘——**层数越多，整体攻破概率指数下降**（对照双因素认证的概率相乘）。
- 没有单层是完美的，但**叠加让整体足够安全**。

**直觉**：容器安全不是「一道铁门」，而是「**洋葱的很多层皮**」——攻击者剥一层还有一层。**「每一层都不完美，但每一层都让攻击更难」是纵深防御的哲学。**

## 4 核心对比表：三种安全机制

| 机制 | 管什么 | 粒度 | 典型动作 |
| --- | --- | --- | --- |
| Capability | 特权操作 | 能力位 | 丢弃 CAP_SYS_ADMIN |
| Seccomp | 系统调用 | 调用级 | 阻止 mount/ptrace |
| AppArmor | 文件访问 | 路径级 | 限制敏感路径 |

**设计启示**：三种机制对应三层的「最小特权」——**Capability 管权限、Seccomp 管调用、AppArmor 管访问**。它们共同回答容器安全的核心问题：「**即使被攻破，还能干什么？**」——答案是「几乎什么都干不了」。

## 5 小结

- 容器共享宿主内核——**视图隔离 ≠ 安全边界**，安全靠多层机制。
- **Capability**：把 root 特权拆成最小能力，容器 root 不是真 root。
- **Seccomp**：过滤危险系统调用（mount、ptrace），系统调用层面封死。
- **AppArmor**：路径级 MAC，限制敏感文件访问。
- **纵深防御（洋葱模型）**：Namespace → Capability → Seccomp → AppArmor → 只读/非 root——层数越多越安全。

至此，第十八篇「Linux 专题：虚拟化与容器」收官，**操作系统专题全部 140 篇博文写作完成**。
