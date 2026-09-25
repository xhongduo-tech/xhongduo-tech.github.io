---
title: BGP 属性与决策
date: 2026-09-08
section: cs
---

# BGP 属性与决策

<div class="epigraph">
<p>同一前缀多条路径时，决策过程按 LOCAL_PREF、AS_PATH、ORIGIN、MED、eBGP vs iBGP、IGP 度量、Router ID 逐步打破平局。</p>
<footer>—— 据 RFC 4271 第 9.1 节决策过程；Kurose and Ross BGP 属性整理</footer>
</div>

主干[AS 路径与策略](/cs/bgp-policy) 已给进出口直觉。[上一课](/cs/rip) 对照了无政策的跳数。[BGP 直觉](/cs/bgp-intuition) 不背阶梯。缺口是**属性怎么排序**：谁先谁后、哪些过 iBGP、哪些不。本课不把路由反射器写完。

## 问题

IGP 选最小代价。BGP 选「政策最佳」。典型顺序：拒无效 → 最高 LOCAL_PREF → 最短 AS_PATH → 最低 ORIGIN → 最低 MED（同 AS）→ 偏 eBGP → 最近 IGP 下一跳 → 其它平局。LOCAL_PREF 不发给 eBGP 邻居，故是 AS 内部的客户优先旋钮。MED 可对外暗示入口，邻居可无视。

<span class="marginnote">术语翻译：LOCAL_PREF 是「我们 AS 内部更偏爱谁」的打分牌，分越高越优先，只贴在自家墙上（不随 eBGP 外发）；MED 则是贴给邻居看的「请优先从我这进来」的提示牌，邻居可以根本不看。</span>

不要把 AS_PATH 长度当延迟：绕远可以是故意的 LOCAL_PREF。

<span class="marginnote">RFC 4271。厂商在阶梯里插入权重等私有步，课上以标准为主。Communities 是政策标签，本身不是阶梯里的一步，除非你的政策把它映到 LOCAL_PREF。</span>

### 不是最短跳

LOCAL_PREF 内部优先于 AS_PATH。MED 可被邻居无视。下一跳必须 IGP 可达才有效。Communities 要映到阶梯才起作用。

## 方法

走一条例子：客户路由 vs 对等路由，LOCAL_PREF 已决定。再走 MED 选入口。画决策漏斗。对照 RIP：没有这些属性，无法表达合同。

```mermaid
flowchart TD
  IN["候选路径"] --> LP["LOCAL_PREF"]
  LP --> ASP["AS_PATH 长"]
  ASP --> MED["MED"]
  MED --> NH["IGP 到下一跳"]
  NH --> BEST["安装 FIB"]
```

<span class="marginnote">数字实例：同一前缀，客户路由 AS_PATH 是「65001」，对等路由是「65002 65003」。论 AS_PATH 客户短、本该赢；但若给客户路由打 LOCAL_PREF 200、对等 100，阶梯第一步就定了客户，根本轮不到数 AS_PATH——这就是「绕远可以是故意的」。</span>

## 机制

选出的下一跳还要 IGP 可达（递归），否则无效——这把域内 SPF 与域间政策接起来。ECMP 在 BGP 里默认常关，多路径要额外开，避免环与不对称。主干 LPM 课：FIB 仍按最长前缀，BGP 只是填充某前缀的下一跳。

哪些属性跨 AS、哪些留在 AS 内：

```mermaid
flowchart LR
  NE["eBGP 进入：AS_PATH、MED 可见"] --> R1["入口路由器"]
  R1 -->|"iBGP 扩散：附 LOCAL_PREF"| R2["iBGP 同伴"]
  R2 --> R3["出口路由器"]
  R3 -->|"eBGP 发出：LOCAL_PREF 不发送"| NX["邻居 AS"]
```

<span class="marginnote">为什么重要：BGP 的下一跳常是邻居 AS 的地址，域内路由表若查不到它，再「最佳」的路径也装不进转发表、整条作废。这道递归检查把域内 IGP 和域间政策拴在一起：域内断了，域间选得再好也白选。</span>

收敛慢：属性抖动会反复决策，后课 dampening。

## 边界

本课不引入 ADD-PATH 的全部。iBGP 与路由反射器是下一课。后课默认：eBGP 政策阶梯以 RFC 4271 为准；私有步要显式声明。

不要用 MED 当全球负载均衡：它只在同意比较的 AS 之间有意义。

下一课[iBGP 与路由反射器](/cs/ibgp-route-reflector)。

## 小结

- 决策是属性漏斗，不是最短跳。
- LOCAL_PREF 内部，MED 对外可被忽略。
- 下一跳必须 IGP 可达。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 4271；Kurose and Ross。
