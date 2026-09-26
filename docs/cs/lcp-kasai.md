---
title: LCP 与 Kasai
date: 2026-09-08
section: cs
---

# LCP 与 Kasai

<div class="epigraph">
<p>$H[i]$ 是 $SA$ 上相邻两后缀的最长公共前缀；Kasai 用 $ISA$ 按文本顺序扫，摊还线性算出整张 $H$。</p>
<footer>—— 据 Kasai et al., Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications, CPM 2001；Manber and Myers 整理</footer>
</div>

[上一课](/cs/suffix-array) 二分时可能对 $P$ 反复比已经相等的前缀。RMQ 若建在 $LCP$ 上，任意两后缀的 LCP 是 $SA$ 区间最小值——接回[稀疏表](/cs/sparse-table)。本课不重排后缀。缺口是定义 $LCP$ 数组，以及 Kasai 线性算法。

## 问题

$LCP[i]=\mathrm{lcp}(T[SA[i-1]..], T[SA[i]..])$（下标约定课内固定一种）。已知 $SA$ 与 $ISA$，Kasai：按 $k=ISA$ 的文本位置 $i=0,1,\ldots$ 考虑后缀 $T[i..]$ 与在 $SA$ 中的前一名。若上一步 LCP 为 $h\gt 0$，则这一步至少 $h-1$（去掉一个字符）。缺口是**用这个 $h-1$ 下界避免从零比**，合计比较 $O(n)$。

<span class="marginnote">直觉类比：$h-1$ 下界像一摞上下对齐的卡片抽掉最上面一张——两张卡片剩余的重合部分只少一层。后缀 $T[i..]$ 删掉首字符就是抽掉公共前缀的最上层，所以「与某后缀共享 $h$」不会凭空变成「共享 0」。</span>

<span class="marginnote">术语翻译：$ISA$ 就是 $SA$ 的反查表——$SA[i]$ 回答「排名 $i$ 的后缀从哪开始」，$ISA[i]$ 回答「从位置 $i$ 开始的后缀排第几」。一个按名次查人，一个按人查名次，互为逆函数。</span>

<span class="marginnote">Kasai et al., CPM 2001。有了 LCP，后缀数组 + RMQ 可模拟后缀树许多查询。</span>

## 方法

实现：`h=0`，对 `i in 0..n-1`，若 `ISA[i]>0`，令 `j=SA[ISA[i]-1]`，从 `h` 起比较 $T[i+h]$ 与 $T[j+h]$，写入 `LCP[ISA[i]]`，然后 `h=max(0,h-1)`。任意两后缀 $p,q$ 的 lcp 等于 $LCP$ 在 $SA$ 上两者秩之间的最小值。

```mermaid
flowchart TD
  SA["SA 与 ISA"] --> KASAI["按文本序扫, h-1 下界"]
  KASAI --> LCP["LCP 数组"]
  LCP --> RMQ["稀疏表 RMQ"]
  RMQ --> PAIR["任意两后缀 lcp"]
```

应用：本质不同子串个数 $n(n+1)/2-\sum LCP$；重复串；与 KMP 前缀函数不同——这里对全体后缀。

<span class="marginnote">数字实例：文本 banana 的 $SA$ 顺序是 a、ana、anana、banana、na、nana，$LCP=[1,3,0,0,2]$。全部子串 $\frac{6\times 7}{2}=21$ 个，减去 $\sum LCP=6$ 个重复，得 15 个本质不同子串——一行公式就把枚举全部子串的 $O(n^2)$ 活干完了。</span>

## 机制

正确性：文本位置 $i$ 与 $i+1$ 的后缀关系给出 $h$ 的传递。不要在无哨兵时让比较越界。LCP 最小值用静态 RMQ，不需 Fenwick。

<span class="marginnote">常见误区：初学者容易把 $ISA$ 当可有可无的辅助数组。实际上 Kasai 的线性复杂度全靠它：算法按文本顺序扫 $i$，必须 $O(1)$ 知道「后缀 $T[i..]$ 在 $SA$ 里排第几」才能定位它的前一名；没有 $ISA$ 就得每轮扫一遍 $SA$，线性立刻退化成平方。</span>

```mermaid
flowchart TD
  A["后缀 i 与其 SA 前一名: 公共前缀长 h"] --> B["删去首字符得到后缀 i+1"]
  B --> C["公共前缀最多被削掉 1: 至少剩 h-1"]
  C --> D["后缀 i+1 的 SA 前一名也共享这 h-1"]
  D --> E["本轮从 h-1 起比, 不从 0 起"]
  E --> F["真实 LCP 算出后更新 h, 传给下一轮"]
  Z["若无下界: 每轮从 0 比, 最坏 O(n^2)"] -.-> E
```

这张图回答的问题：Kasai 的 $h-1$ 下界从哪来、省了什么。删首字符只会把已知的公共前缀削一层、不会清零，所以每轮的比较都从上一轮的余量接着数；所有「从 0 开始」的比较加起来被摊平，总工作量才是 $O(n)$。

后缀树把这些 lcp 变成显式边，下一课收指针树；本课数组已够多数统计。

## 边界

本课不写稀疏后缀数组。动态 LCP 维护复杂。后缀树与 SAM 仍有「所有出现位置」的显式 coprime 结构需求。

后课默认：静态 $SA$ 必配 $LCP$。要显式后缀链与边，用后缀树。

## 小结

- $LCP[i]$：SA 相邻后缀公共前缀长。
- Kasai：$O(n)$ 借助 $h-1$。
- 任意对 lcp = 区间最小 LCP。
- 出处：Kasai et al., CPM 2001；Manber and Myers。
