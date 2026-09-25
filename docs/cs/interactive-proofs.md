---
title: 交互证明与 IP = PSPACE
date: 2026-09-08
section: cs
---

# 交互证明与 IP = PSPACE

<div class="epigraph">
<p>验证者多项式、可掷硬币，与全能证明者回合交谈；IP 恰好等于 PSPACE。图不同构因此有交互证明。</p>
<footer>—— 据 Goldwasser, Micali and Rackoff；Shamir, IP = PSPACE, 1992；Arora and Barak 整理</footer>
</div>

上一课[BPP](/cs/bpp-randomized-class) 的硬币只在一台机器里。缺口是**交互**：证明者 $P$ 无限能，验证者 $V$ 多项式随机，多轮消息。完备：是实例 $V$ 几乎总接受；可靠：否实例任意 $P^*$ 几乎总被拒。$\mathrm{IP}=\mathrm{PSPACE}$（ Shamir）。GNI（图不同构）在 IP 里，却未知是否在 NP——短证书与交互不是一类。

## 问题

NP 是一轮、确定性验证的证书。交互允许：隐藏随机挑战、自适应问题。求和检查、算术化把布尔改成多项式，验证者抽随机点比较。#SAT 的计数证明可放进 IP，从而 PSPACE（TQBF）也可。本课要这层归约直觉，不写 Shamir 的全部多项式。

<span class="marginnote">数字实例：单轮 GNI 里，恶意证明者闭眼猜也有 1/2 答对率；独立重复 20 轮后，靠猜骗过验证者的概率只剩 $2^{-20}$，约百万分之一——可靠性就是这样按轮数指数级压下去的。</span>

公开随机（Arthur–Merlin）与私硬币在足够轮数下能力接近，点名 AM。一轮公开随机大致在 PH 的第二层附近。

### 交互不是「口头 NP」

证明者不能被相信：可靠对恶意 $P^*$ 成立。零知识是额外性质（模拟），本课不定义 ZK 协议，只声明 IP 的类不等于「有 ZK 证明」。

<span class="marginnote">术语翻译：「完备性」是说真话时验证者会被说服；「可靠性」是说假话时任何狡猾的证明者都几乎骗不过。前者防「证明太弱说不服人」，后者防「证明太容易伪造」。</span>

<span class="marginnote">GMR 交互证明。Shamir 1992；Lund–Fortnow–Karloff–Nisan 的 #SAT。Sipser 有简写。本课不引入 MIP = NEXP。</span>

## 方法

用图不同构：验证者随机藏一张图的随机重标，问证明者来自 $G$ 还是 $H$。不同构则证明者能分；同构则无法优于猜测。说明为何这不是 NP 证书（验证者必须藏随机性）。再点名：算术化把 PSPACE 送进 IP。

```mermaid
flowchart TD
  V["多项式随机验证者"] o--o P["全能证明者"]
  V --> IP["IP"]
  IP --> EQ["= PSPACE"]
```

## 机制

$\mathrm{NP}\subseteq\mathrm{IP}$ 显然（证书当消息）。$\mathrm{IP}\subseteq\mathrm{PSPACE}$：空间枚举消息树、递归算接受概率。另一边 Shamir。于是交互恰好填满上一课的 PSPACE，而不是「NP 的随机版」。

轮数、私硬币是精细结构；类的等式在多项式轮下成立。

$\mathrm{IP}\subseteq PSPACE$ 的递归：验证者消息树深度多项式、每层多项式分支，空间可复用。另一边 Shamir 把 TQBF 算术化到有限域多项式，证明者逐步打开求和。GNI 展示交互可以藏随机挑战；NP 证书没有「验证者私硬币」这一资源。

GNI 协议的单轮长这样：关键在重标后的图把「来自哪张」的信息藏进了验证者的私硬币。

```mermaid
flowchart TD
  S["验证者掷硬币选 G 或 H"] --> R["随机重标顶点得 H′，隐藏选择"]
  R --> Q["把 H′ 发给证明者：它来自哪张原图?"]
  Q --> A{"证明者回答"}
  A --> OK["答对一轮不够，独立重复 k 轮"]
  OK --> ACC["全部答对才接受"]
  S --> ISO{"若 G ≅ H"}
  ISO --> L["重标后分布相同，猜中率至多 1/2"]
```

<span class="marginnote">常见误区：初学者容易把交互证明想成「递一张短证书、验证者看完签字」；关键差别恰恰是验证者的随机挑战要保密——NP 证书没有「验证者私硬币」这个资源，GNI 才因此有交互证明、却未知有 NP 证书。</span>

## 边界

本课不写零知识证明实现，不引入密码学假设下的简洁证明。不把 MIP、PCP 混成一轮（PCP 下一课）。后课默认：交互证明的能力是 PSPACE，不是 NP。下一课 PCP 与不可近似。

IP 填满 PSPACE，不是「随机的 NP」。轮数、私硬币是精细结构；类的等式在多项式轮下成立。零知识是额外模拟性质，本课不把 ZK 协议当 IP 的定义。

## 小结

- IP：多项式随机验证者 + 交互；等于 PSPACE。
- GNI 展示「有交互证明 ≠ 已知有 NP 证书」。
- 可靠必须对恶意证明者成立。
- 出处：Goldwasser, Micali and Rackoff；Shamir, 1992。
