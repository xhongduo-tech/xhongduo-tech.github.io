---
title: oops 与 panic
date: 2026-09-08
section: cs
---

# oops 与 panic

<div class="epigraph">
<p>oops 记录一次内核非法访问或 BUG，进程可能被杀而系统继续；panic 认为无法继续，停机或转 dump。</p>
<footer>—— 据 Linux 对 oops/panic 的文档；[kdump](/cs/kdump-crash) 为转储先修</footer>
</div>

[kdump](/cs/kdump-crash) 在 panic 路径上触发。缺口是 **oops vs panic 的策略**：`panic_on_oops`、RCU stall、软锁。

## 问题

空指针在进程上下文：oops，杀该任务。中断上下文或 idle：常直接 panic。缺口：tainted 标志；连续 oops；与 [hardening](/cs/kernel-hardening) 的 BUG_ON。本课不分析具体漏洞。

<span class="marginnote">sysctl `kernel.panic` 超时重启。对象是失败模式，不是调试符号安装。</span>

<span class="marginnote">直觉类比：oops 像航班上一名乘客晕倒——处理该任务的进程被「抬走」，航班照飞；panic 像机长宣布紧急迫降——宁可整机停摆，也要保住黑匣子（dump）里的现场。选哪档，就是可用性与可诊断性的交换。</span>

## 方法

fault → `oops_begin` 打栈 → 若策略则 `panic`。对照 [ASan](/cs/asan-mechanism)：用户非法 vs 内核非法。对照 [NMI](/cs/nmi)：NMI 里 printk 受限。对照 SELinux：拒绝不是 oops。

```mermaid
flowchart TD
  BUG["非法访问/BUG"] --> OOPS["oops 记录"]
  OOPS -->|"可恢复"| CONT["杀任务继续"]
  OOPS -->|"panic_on_oops 或致命上下文"| PAN["panic"]
  PAN --> DUMP["kdump 或重启"]
```

<span class="marginnote">数字实例：生产服务器的常见组合是 `kernel.panic_on_oops=1`（任何 oops 都升级为 panic）加 `kernel.panic=10`（panic 后 10 秒自动重启），再配 kdump 先抓内存转储——用一次确定的停机换一份可信的现场。</span>

## 机制

内核把「局部腐败」和「全局不可信」分开：oops 允许服务器继续卖服务，也允许静默损坏——故生产常 panic_on_oops。不要写成恐吓。与 [memcg](/cs/memcg) OOM：那是杀用户，不是内核 oops。

栈损坏的 oops 本身不可信。

oops 后「继续跑」为什么可能更糟：

```mermaid
flowchart TD
  OOPS["一次 oops 后继续运行"] --> GOOD["多数情况: 只死一个任务"]
  OOPS --> BAD["发生在持锁 / 半途写路径"]
  BAD --> STALL["锁不释放, 慢慢拖成软锁"]
  BAD --> SILENT["数据改了一半, 静默损坏"]
  SILENT --> LATER["数天后爆发, 无从查起"]
```

<span class="marginnote">常见误区：初学者以为「系统没崩就没事」。oops 若发生在持锁路径，锁可能永远不释放，其他任务逐个卡死；表面「还活着」的机器实际已经残废——这正是生产宁选 panic_on_oops 的原因。</span>


实现上：tainted 标志告诉你是否加载了专有模块或发生过严重警告。中断上下文 oops 几乎必然 panic。连续 oops 可能已损坏到栈不可信。 读法上只引用[上一课](/cs/kdump-crash)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「安全、启动与调试 / 观测与调试」课序里，对象是 **oops 与 panic**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 lockdep 警告。不保证嵌入式无 console 时的策略。下一课不停机补内核：livepatch。


版本字段会变，课序钉的是机制对象「oops 与 panic」，不是某一主线内核的结构体名。
后课默认：oops 可转 panic 以便 dump。运行时替换函数，下一课 livepatch。

## 小结

- oops 记录并可能继续；panic 停机。
- 策略决定安全性与可用性。
- livepatch 是下一课。
- 出处：Linux oops；panic sysctl；kdump。
