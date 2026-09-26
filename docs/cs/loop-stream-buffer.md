---
title: 循环流缓冲
date: 2026-09-08
section: cs
---

# 循环流缓冲

<div class="epigraph">
<p>一旦认出循环体已经全部译码进一小块缓冲，取指单元可以休眠：uop 从缓冲循环供给，直到退出分支被预测为离开。</p>
<footer>—— 据 Intel 对 Loop Stream Detector 的公开描述；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/uop-cache)用按 IP 索引的 uop cache 绕过译码器，每拍仍要做标签比较、仍可能跨行。[循环预测](/cs/loop-predictor)已经能猜何时离开。本课不重讲 DSB 填入。缺口是把**足够小的循环锁定在更浅的流缓冲**，连 uop cache 的查找都关掉。

## 问题

科学计算与多媒体里，十几到几十 uop 的循环会跑百万次。uop cache 命中已经便宜，但取指流水、BTB 读口、分支预测查表仍在跳。缺口不是更大的 uop cache，而是**检测「循环体已完整驻留」后切换到从 LSB 顺序弹出 uop**，退出才回到普通前端。

<span class="marginnote">Intel Core 起的 LSD（Loop Stream Detector）有 uop 数上限；体太大、含过多分支或调用则不锁定。这是功耗与前端端口的交易，不是新的 ISA 循环指令。</span>

<span class="marginnote">可以类比把常做的五步体操动作背进肌肉记忆：背熟之后，不用每遍都翻教材（取指、译码）照着比划，直接凭记忆连做一百遍；直到口令说「换动作」（预测退出）才重新翻书。省的不只是翻书时间，更是「翻书」这件事本身的力气。</span>

## 方法

前端统计连续向后分支且目标落在近期已译码窗口内。若循环体 uop 数 ≤ 缓冲容量、且满足「无调用、间接可预测」等实现限制，把该体复制进 LSB，置锁定。供给：每拍从 LSB 取 $D$ 个 uop，IP 在体内绕回。循环预测器说「这一次退出」则解锁，从退出目标重新走 BTB/uop cache。

```mermaid
flowchart TD
  DET["检出短循环体"] --> LOCK["填入 LSB 并锁定"]
  LOCK --> POP["从 LSB 弹 uop"]
  POP --> REN["重命名"]
  EXIT["预测退出"] --> UNL["解锁，按目标重取"]
```

## 机制

锁定期间前端各部件谁在干活、谁在休息，值得拆开看：LSB 直接供 uop，一整条取指译码链路被门控休眠。

```mermaid
flowchart TD
  LK["循环体已锁定进 LSB"] --> P["每拍从 LSB 顺序弹出 uop"]
  P --> RN["重命名 / 分配"]
  OFF["被门控休眠的部件"] --> S1["I-cache 取指"]
  OFF --> S2["译码器"]
  OFF --> S3["BTB 与部分预测器读口"]
  P --> E["前端功耗大降，供给不再受行对齐打断"]
```

<span class="marginnote">初学者容易以为 LSB 是架构状态、程序能感觉到它。实际它纯属微结构优化：自修改代码、异常、误预测都会解锁冲刷，架构层面什么都没变——这也是为什么同一份程序在不同核上是否用 LSD 都完全合法。</span>

功耗：I-cache、uop cache、相当一部分预测器读口可以门控。带宽：LSB 的端口按重命名宽度设计，不再被 cache 行对齐打断。与[循环器](/cs/loop-predictor) 的配合：行程 $N$ 决定何时解锁；LSB 本身不学 $N$。

<span class="marginnote">数字实例：一个 20 uop 的小循环跑 100 万次。普通前端每遍都要查 I-cache、uop cache、BTB，加起来约 2000 万次查表；LSB 锁定后这些查找几乎归零——对服务长热循环的代码，省下的前端功耗相当可观，这也是 LSD 的首要动机。</span>

自修改或 SMC 仍要打破锁定。异常与误预测同样解锁并冲刷——LSB 不是架构状态。

## 边界

本课不把软件展开后的巨循环硬塞进 LSB。也不把 GPU 的循环缓冲、DSP 的零开销循环指令当成同一结构：那些是 ISA 可见的。宏融合会改变「几条宏指令对应几个 uop」，影响能否装进容量，下一课才讲融合。

后课默认：短热循环可以在 uop cache 之下再短一截供给路径。把相邻宏指令合成更少 uop，是另一条减前端压力的路。

## 小结

- LSB 锁定短循环的 uop 流，让取指与部分预测器休眠。
- 容量与分支限制决定能否锁定；退出靠循环预测。
- 减少 uop 条数的融合是下一课。
- 出处：Intel 优化参考手册中的 LSD；Hennessy and Patterson, *CA:AQA*。
