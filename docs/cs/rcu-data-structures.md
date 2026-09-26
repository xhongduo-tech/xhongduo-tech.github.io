---
title: RCU 友好的结构
date: 2026-09-08
section: cs
---

# RCU 友好的结构

<div class="epigraph">
<p>读者不取锁：只记已进入读侧临界区。写者复制或拼接新结构，宽限期过后再释放旧节点。</p>
<footer>—— 据 McKenney and Slingwine, Read-Copy Update: Using Execution History to Solve Concurrency Problems, PDCS 1998；McKenney, RCU 系列文献；[RCU 宽限期](/cs/rcu-gp) 整理</footer>
</div>

[上一课](/cs/concurrent-hashmap) 的 get 仍常碰原子或桶锁。[RCU 宽限期](/cs/rcu-gp) 已定义 GP。[路径复制](/cs/persistent-path-copy) 提供写复制图像。本课不 Treiber。缺口是哪些结构适合 RCU：单向链表、基数树、哈希链——读者遍历，写者 `rcu_assign_pointer`。

## 问题

读远多于写（路由表、配置、目录）。锁读者会互撞。RCU：读侧 `rcu_read_lock` 几乎是关抢占或记状态；写者做 copy-update，发布新指针，`synchronize_rcu` 等所有读者离开旧视图。缺口是**结构必须允许「旧新并存」直到 GP**：不能原地改读者正扫的字段除非小心（或只用追加）。

<span class="marginnote">直觉类比：宽限期像酒店退房——写者想拆掉一间房改造，不能把还住着的客人赶出来，只能先盖好新房、把门牌改过去，等最后一位老客人退房（GP 过）再拆旧房。读者自始至终没排过队，这正是「读多写少」场景付得起的价格。</span>

<span class="marginnote">McKenney–Slingwine 1998。内核哈希桶、`list_replace_rcu` 是实例。用户态 urcu 同构。</span>

## 方法

链表删除：摘指针发表，GP 后 free。更新：新节点填好，CAS 或锁下改前驱 next。树：写路径复制或 COW 根。哈希表：桶链 RCU，扩容仍难，常读锁写或分层。

```mermaid
flowchart TD
  R["读者: 无锁遍历"] --> OLD["可能看见旧节点"]
  W["写者: 复制/拼接"] --> PUB["发表新指针"]
  PUB --> GP["synchronize_rcu"]
  GP --> FREE["再释放旧节点"]
```

与 HP：RCU 读侧更轻，写侧延迟更大、要能停或异步回调。与 HAMT 持久：语义像无限版本；RCU 通常只保到 GP 的读者。

## 机制

内存屏障：发表前写节点字段。读者遍历要用 `rcu_dereference`。本课不重开调度器。分配器下一课：RCU 延迟释放会给分配器压力，与空闲链结构有关。

```mermaid
flowchart TD
  OLD["旧链：A 的 next 指向 B，B 指向 C"] --> R["老读者正站在 B 上扫描"]
  W["写者删 B：把 A 的 next 改指 C"] --> NEW["新读者从 A 直达 C，绕过 B"]
  R --> KEEP["B 暂时不能释放：老读者还在用"]
  NEW --> GP["synchronize_rcu 等宽限期"]
  KEEP --> GP
  GP --> FREE["确认老读者全离开后，才释放 B"]
```

<span class="marginnote">为什么重要：这张图的顺序不能颠倒——写者必须先把 B 的字段填好、再改 A 的 next「发表」新视图。反过来的话，某读者可能顺着新指针看到半成品节点（字段还是垃圾值），而且没有任何锁能抓住这个错误，只能靠内存屏障保证「先写后发布」的顺序可见。</span>

## 边界

本课不把内核 API 清单当正文。写多读多仍用锁或无锁表。不要用 RCU 保护任意图的随便原地改。

<span class="marginnote">常见误区：初学者容易把 `rcu_read_lock` 当普通自旋锁——以为可以拿住不放、甚至在临界区里睡眠。读侧临界区必须短小、不可阻塞；长期持有会让宽限期永远过不去，写侧的延迟释放无限堆积，最终内存被旧节点撑爆。</span>

后课默认：读多链表/树可用 RCU 发表。分配器自身的空闲结构下一课。

## 小结

- RCU 结构：读者无锁，写者发表，GP 后回收。
- 适合链表、某些树与哈希链；原地改受限。
- 下一课分配器内部空闲结构。
- 出处：McKenney and Slingwine, 1998；McKenney RCU 文献。
