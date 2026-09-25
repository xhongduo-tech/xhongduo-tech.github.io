---
title: cpuidle 与 C-state
date: 2026-09-08
section: cs
---

# cpuidle 与 C-state

<div class="epigraph">
<p>C-state 越深越省电，退出延迟越大；cpuidle 根据预计空闲时间选档，避免刚睡就被包打断。</p>
<footer>—— 据 Linux cpuidle 文档；ACPI C-state；[NAPI](/cs/napi) 与 [轮询 I/O](/cs/io-polling) 为忙碌对照</footer>
</div>

[cpufreq](/cs/cpufreq-governor) 管忙时频率。核 idle 时管 **睡多深**。缺口是 cpuidle：菜单 governor、与中断的博弈。

## 问题

poll 路径（DPDK、IOPOLL）故意不进 C-state。普通服务器：idle → 选 C1/C6…。预计空闲来自调度器（下一个 tick 或 hrtimer）。缺口：错估则 exit 延迟伤害 [cyclictest](/cs/latency-measurement)；nohz 后课改「下一个 tick」。本课不把每个 CPU 的 residencies 表抄来。

<span class="marginnote">tick 会阻止最深 C-state。irq 亲和把打断集中到部分核，让别的核能睡。</span>

<span class="marginnote">术语翻译：C-state 是 CPU 的「睡觉档位」——C0 在干活，C1 打个盹（时钟停走），C6 睡得更沉（更多内部状态断电）。越深越省电，但叫醒它要付的退出延迟也越长。</span>

## 方法

`cpuidle_enter`：查预测 → 选 state → `mwait`/WFI。唤醒：中断，记 residency。对照 EAS：忙时选核选频，闲时选 C。对照 [zswap](/cs/zswap)：一个用 CPU 换内存，一个用延迟换焦耳。

```mermaid
flowchart TD
  IDLE["无任务"] --> PRED["预计空闲时长"]
  PRED --> C["选 C-state"]
  IRQ["中断"] --> EXIT["付退出延迟"]
```

<span class="marginnote">数字实例：浅档 C1 的退出延迟通常在 1-2 微秒量级，深档 C6 可到几十微秒。若一段空闲只持续 5 微秒却选了 C6，光叫醒就比空闲本身还久——省下的电抵不过白付的延迟，倒贴一笔。</span>

## 机制

cpuidle 把空闲变成分级睡眠，使平均功耗可降，代价是最坏唤醒。RT 系统常限制最深 C。不要写成电池广告。与 [RSS](/cs/rss-multiqueue)：包若打到睡眠核，要付 exit。

错误预测在高 IRQ 率下会振荡，菜单 governor 有修正。

```mermaid
flowchart TD
  IDLE2["核空闲 5 ms"] --> Q{"预计空闲长还是短?"}
  Q -->|"短"| C1["选浅档 C1: 省得少, 唤醒快"]
  Q -->|"长"| C6["选深档 C6: 省得多, 退出慢"]
  C1 --> W1["中断一来马上响应"]
  C6 --> Q2{"预测准吗?"}
  Q2 -->|"准"| WIN["整段空闲都在省电"]
  Q2 -->|"不准"| PAY["很快来中断, 白付长退出延迟"]
```

<span class="marginnote">常见误区：初学者容易以为省电只看 cpufreq 降频。空闲核若进不了深档，功耗同样降不下来；反过来，中断没集中好、老打在睡熟的核上，省的电又被退出延迟和唤醒开销吃回去——两件事是一对。</span>


实现上：预测过深则 wakeup 付长 exit 延迟，cyclictest max 爆炸。poll 忙等的核 residency 为 0。irq 亲和把打断集中，让旁核睡得着。 读法上只引用[上一课](/cs/cpufreq-governor)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「调度进阶 / 公平、实时与能耗」课序里，对象是 **cpuidle 与 C-state**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入 S-state 睡眠整机——[挂起](/cs/suspend-resume) 后课。不保证虚拟机 halt 是真 C-state。下一课关掉周期性 tick：tickless。


版本字段会变，课序钉的是机制对象「cpuidle 与 C-state」，不是某一主线内核的结构体名。
后课默认：空闲可选深 C-state。无任务时可停时钟中断，下一课 tickless。

## 小结

- C-state 用退出延迟换功耗。
- cpuidle 按预计空闲选档。
- tickless 是下一课。
- 出处：Linux cpuidle；ACPI；Intel idle。
