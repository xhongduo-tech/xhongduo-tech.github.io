---
title: Grace hash join
date: 2026-09-08
section: cs
---

# Grace hash join

<div class="epigraph">
<p>内存装不下 build 侧时，两边按连接键划分成对等分区，分区对分别做内存哈希；递归直到能放下。</p>
<footer>—— 据 Kitsuregawa et al. GRACE；DeWitt et al. 混合哈希；Blasgen and Eswaran；主干哈希连接的溢盘补全</footer>
</div>

[上一课](/cs/hash-aggregation)对单输入分区。本课不更新组状态。缺口是等值连接在内存不够时：主干 [NLJ 与哈希](/cs/nlj-hash-join) 点名 Grace 式分区。进阶把算法钉死，并行 shuffle 后课再加网络。

## 问题

内存哈希：小侧 build，大侧 probe，一次扫描。build 超过工作内存：简单做法失败。Grace：用同一哈希函数的高位把 R 与 S 划成 $P$ 个桶文件，保证 $R_i$ 只可能连 $S_i$。<span class="marginnote">「同键同分区」可以类比分拣快递：两堆包裹用同一个邮编规则分拣，同一邮编的件保证落进编号相同的筐——之后只需把 1 号筐对 1 号筐核对，绝不用跨筐配对。正确性全押在「两边用同一个哈希函数」这一条上。</span>然后对每个 $i$，把较小的一侧读入内存哈希，扫另一侧。若某分区仍太大，递归再划分。

混合哈希：第一分区常驻内存，其余落盘，减少 I/O。

```mermaid
flowchart LR
  SCAN["顺序扫 R 与 S 划分"] -->|"分区 0"| RES["build 侧直接常驻内存"]
  SCAN -->|"分区 1 .. P-1"| DISK["写入盘上分区对"]
  RES --> J0["分区 0 连接：免落盘与重读"]
  DISK --> JI["分区 i 逐对读入内存连接"]
  J0 --> OUT["合并输出"]
  JI --> OUT
```

缺口是倾斜：一键对应极多行，分区无法切开该键——需要特殊处理（独立该键、或改 NLJ）。<span class="marginnote">常见误区：以为把分区数 $P$ 调大就能消化倾斜。倾斜键的所有行被同一个哈希值钉进同一个分区，$P$ 再大它也整块跟着走——切开的是分区、不是键。正确动作是把该键摘出来单独连接，或对那一小块退回嵌套循环。</span>

<span class="marginnote">Kitsuregawa, Tanaka, Moto-oka 的 GRACE 数据库机。DeWitt 等混合哈希。选择 build 侧仍靠估计；估错会把大侧当 build，分区文件爆。</span>

## 方法

划分遍：顺序扫两输入，写 $P$ 对文件。处理遍：对 $i=1..P$ 做内存连接。I/O：约两输入各读两次、写一次（划分写、处理读），常数优于朴素反复扫内表。$P$ 选使期望分区适合内存。<span class="marginnote">数字实例：build 侧 10 GB、工作内存 1 GB，取 $P=32$——每分区期望约 313 MB，留足余量。代价是约「读两遍写一遍」：10 GB 的表多付约 20 GB 的 I/O；混合哈希让分区 0 免掉落盘加重读，一个分区就省约 $2\times313$ MB。</span>

位图过滤、布隆：probe 前丢弃必不命中行，后课 LSM 读放大也会用布隆，思想同源，结构不同。

```mermaid
flowchart TD
  R["R"] --> P["按键划分成 R_i"]
  S["S"] --> Q["按键划分成 S_i"]
  P --> JOIN["R_i 与 S_i 内存哈希"]
  Q --> JOIN
  JOIN --> REC["分区仍大则递归"]
```

## 机制

迭代器：划分是阻塞的（先写完文件）。向量化按批算哈希、写页。编译内联哈希函数。WAL 不记录临时划分文件；崩溃则查询失败重来，不是恢复连接中途。

外连接：分区仍按键，未匹配行在处理遍按外连接规则吐 NULL——逻辑课已钉，执行器实现分支。

## 边界

本课不讲跨节点的 exchange 与网络 shuffle——下一课并行。单机多核可用同一划分思想在内存队列上做，仍叫分区哈希。

后课默认：等值连接内存不够走 Grace/hybrid；倾斜键单独处理。并行查询把划分换成跨线程/跨节点的 exchange。

Grace 的正确性来自「同键同分区」；性能来自各分区独立且期望能放下。

## 小结

- Grace 把两输入划成对等分区再分别哈希。
- 混合哈希保留一个内存分区；倾斜键破划分假设。
- 并行查询与 exchange 下一课：把分区送到多个工人。
- 出处：Kitsuregawa et al.；DeWitt 等；Ramakrishnan and Gehrke。
