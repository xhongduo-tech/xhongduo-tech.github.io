---
title: 通用散列
date: 2026-09-08
section: cs
---

# 通用散列

<div class="epigraph">
<p>函数从族里随机抽：任意一对不同键碰撞的概率 $\le 1/m$；最坏输入对期望仍然成立。</p>
<footer>—— 据 Carter and Wegman, Universal Classes of Hash Functions, JCSS 1979；Cormen, Leiserson, Rivest and Stein 第 11 章整理</footer>
</div>

[上一课](/cs/art-adaptive-radix) 结束字符串结构。主干[散列碰撞](/cs/hash-collision) 已说明固定 $h$ 可被对抗。[字符串哈希](/cs/string-hashing) 用了随机模，未给族的定义。本课不画 ART 节点。缺口是 Carter–Wegman 通用散列：期望分析的合法前提。

## 问题

链地址、布隆、草图都要「碰撞不多」。对手若知道 $h$，可构造全撞一槽。通用族 $\mathcal{H}$：对任意 $x\neq y$，

$$
\Pr_{h\in\mathcal{H}}[h(x)=h(y)] \le 1/m
$$

（$m$ 为槽数）。强通用（pairwise independent）还要求任意两槽对均匀。缺口是**把随机性放在选 $h$，而不是假设输入随机**。

<span class="marginnote">Carter and Wegman, *JCSS*, 1979。多项式、乘法散列 $h_a(x)=\lfloor m(ax\bmod 2^w)/2^{w}\rfloor$ 等是常见构造。CLRS 第 11 章。</span>

## 方法

实践：程序启动抽 $a,b$，线性 $((ax+b)\bmod p)\bmod m$。证明两键碰撞化为一次均匀。多重独立（$k$-wise）给更好尾界，Count-Min 后课用。本课只钉 pairwise 级。

```mermaid
flowchart TD
  ADV["任意固定 x ≠ y"] --> H["随机 h ∈ H"]
  H --> COL["P(碰撞) ≤ 1/m"]
```

与密码学哈希：通用散列不要求原像困难，只要碰撞概率；更快、键更短。不要混用 SHA 当布隆的 $k$ 个索引——可以，但过重。

## 机制

一次完整的线性通用散列计算，以及碰撞过多时的换族流程：

```mermaid
flowchart TD
  X["输入键 x"] --> M1["y1 = a·x + b"]
  M1 --> M2["y2 = y1 mod p (p 为大素数)"]
  M2 --> M3["slot = y2 mod m"]
  M3 --> INS["插入槽 slot"]
  INS --> CHK{"链长超出阈值?"}
  CHK -->|否| OK["照常运行"]
  CHK -->|是| RE["换族: 重抽新 (a, b)"]
  RE --> REB["全表按新 h 重散列"]
```

期望链长：对固定键集，期望 $\alpha$ 与简单均匀假设相同，但现在对对抗输入也成立（期望对 $h$）。再散列换 $h$ 即换族中另一员。实现须真随机种子，不能写死 $a=31$。

<span class="marginnote">通用族可以翻译成「保险单」：这一篮子函数里没有哪个单独靠谱，但任意一对键撞进同一槽的概率都被保险到 $\le 1/m$。$m=1000$ 个槽时，无论对手挑什么键对，碰撞概率都不超过千分之一——概率论从「假设输入随机」换成「假设我抽函数随机」。</span>

<span class="marginnote">常见误区：以为写死 $a=31$、$b=0$ 的经典字符串哈希也算通用散列。族是集合，抽签才给界；写死等于把唯一的骰子钉在桌面上，对手照样能构造全撞一槽的键集。素数 $p$ 也别随手取，常用 $2^{61}-1$ 这类梅森素数。</span>

<span class="marginnote">「重抽 $(a,b)$」是最后手段：换一个 $h$ 等于换族里另一名成员，此前的坏运气不继承。代价是全表重建一次，所以工程上只在负载因子失控或链长统计显著超标时触发。</span>

完美散列下一课在通用之上再消除碰撞（静态）。

## 边界

本课不把有限域全部构造写完。一致性哈希解决的是槽增减，不是碰撞界。

后课默认：字典分析用通用族。静态零碰撞用完美散列。

## 小结

- 通用族：任意一对碰撞概率 $\le 1/m$。
- 随机在函数，不在数据。
- 静态无碰撞字典是完美散列。
- 出处：Carter and Wegman, *JCSS*, 1979；Cormen et al.。
