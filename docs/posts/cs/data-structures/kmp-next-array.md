---
title: KMP 算法与 next 数组
date: 2026-08-07
---

# KMP 算法与 next 数组

<div class="epigraph">
<p>匹配失败时，不要从头再来——记住模式串自己有多「自相似」。</p>
<footer>—— KMP 格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·Trie 与字符串匹配 ｜ 2026-08-07</p>
</div>

## 为什么朴素匹配「白费了力气」

§4.3 的朴素匹配失配时，主串指针 $i$ 回退、模式串指针 $j$ 复位——**已比过的信息被丢弃**。**KMP 算法**（Knuth-Morris-Pratt）的突破：**失配时，$i$ 不回退，只把模式串指针 $j$ 移动到「模式串自己的最长相等前后缀」处**。这个「跳转目标」预先算好，存在 **next 数组**里。于是匹配过程 $i$ 一路向前、永不回溯——总复杂度 $O(n+m)$。KMP 的核心不是匹配本身，而是**预先计算模式串的「自相似性」**。本节讲透 next 数组的定义与两种计算法（经典教材版与工程简化版）。

## 1 next 数组的定义

设模式串 $T = t_1 t_2 \cdots t_m$。next 数组的定义（严蔚敏教材版）：

$$
next[j] = \begin{cases} 0 & j = 1 \\ \max\{k \mid 1 \le k < j, \ t_1..t_{k} = t_{j-k+1}..t_j\} & \text{存在这样的 } k \\ 1 & \text{否则} \end{cases}
$$

直观说：**next[j] = 失配时，模式串「最长真前缀 = 真后缀」的长度 + 1**——即失配后 $j$ 该回退到哪。

**重点：next[j] 回答「如果第 $j$ 位失配，模式串前 $j-1$ 位已经匹配，下一步从哪继续」**。它利用的是「$t_{j-1}$ 结尾的后缀，与 $t_1$ 开头的前缀，最长能重合多长」。<span class="marginnote">「<strong>next 存的是失配后的跳转位</strong>」：<strong>失配于 $j$，说明 $T[1..j-1]$ 都匹配过——这些字符的后缀里，最长等于某个前缀的那一段，可以「复用」而不必重比</strong>。<strong>「自相似性」是 KMP 的一切</strong>。</span>

## 2 匹配过程：i 不回退

用 next 数组匹配主串 $S$ 与模式串 $T$：

```c
int KMPIndex(SString S, SString T, int next[]) {
    int i = 1, j = 1;                       /* 教材版下标从 1 开始 */
    while (i <= S.length && j <= T.length) {
        if (j == 0 || S[i] == T[j]) { i++; j++; }  /* 匹配成功，或 j 已回退到 0 */
        else j = next[j];                   /* 失配：i 不动，j 跳 next[j] */
    }
    if (j > T.length) return i - T.length;  /* 匹配成功，返回起始位置 */
    return 0;                               /* 匹配失败 */
}
```

**重点：失配时 $i$ 纹丝不动，$j$ 跳到 `next[j]`——这是 KMP 与朴素匹配最根本的区别。** $j$ 表示「连第一个字符都不匹配」，此时 $i$ 前进一位、$j$ 回到 1。<span class="marginnote">「<strong>$i$ 只进不退</strong>」是 KMP 的全部承诺：<strong>主串指针一路向前，绝不回头——所以主串只扫一遍，$O(n)$</strong>。<strong>朴素匹配的 $i$ 回退是它 $O(nm)$ 的根源，KMP 砍掉回退，就砍掉了重复比较</strong>。</span>

## 3 next 数组的递推计算

next 数组本身可以递推求解——**求 next 的过程就是一个「小的 KMP」**：

```c
void GetNext(SString T, int next[]) {
    int j = 1, k = 0;
    next[1] = 0;                        /* 教材版：next[1] 恒为 0 */
    while (j < T.length) {
        if (k == 0 || T[j] == T[k]) { j++; k++; next[j] = k; }
        else k = next[k];               /* 失配：k 沿 next[k] 回退 */
    }
}
```

**重点：计算 next 的循环与 KMP 匹配长得一模一样**——只是「主串」换成了「模式串自己」。「求 next 是自己在和自己做 KMP」这句话，是理解它的钥匙。<span class="marginnote">「<strong>求 next = 模式串自己跟自己匹配</strong>」：<strong>next 的递推用的是「前缀的 next」——$k$ 失配就沿 `next[k]` 回退</strong>。<strong>这个「自匹配」的递归结构，正是「前缀 = 后缀」的自我寻找</strong>——理解了它，KMP 不再神秘。</span>

## 4 公式解析：next 数组示例

求模式串 `ababaa` 的 next 数组：

$$
\begin{array}{c|cccccc}
j & 1 & 2 & 3 & 4 & 5 & 6 \\
\hline
T[j] & a & b & a & b & a & a \\
next[j] & 0 & 1 & 1 & 2 & 3 & 4
\end{array}
$$

- **第一步，读 next[3]**：$j=3$ 失配时，前两位 `ab`——「真前缀 = 真后缀」无重合（`a` vs `b`），回退到 1。
- **第二步，读 next[4]**：前三位 `aba`——最长真前后缀重合是 `a`（长 1），回退到 2。
- **第三步，读 next[5]**：前四位 `abab`——最长重合 `ab`（长 2），回退到 3。<span class="marginnote">「<strong>next 反映的是模式串的「周期性」</strong>」：`<strong>`ababaa` 有周期 `ab`（长 2），所以失配能回退 2 步复用。<strong>周期越强的模式串，next 越大、匹配越省</strong>——这也是 KMP 在「重复模式」上远胜朴素匹配的原因。</span>

## 5 辨析｜易错点：教材版 next vs 工程版

**教材版（严蔚敏）**：下标从 1 计数、`next[1] = 0`，`next[j]` = 「最长真前后缀长 + 1」，匹配时 `j = 0` 特判；
**工程版（-1 版）**：`next[0] = -1`（或教材版 `next` 的整体右移写法），数组下标从 0 起——逻辑相同，边界不同；
**常见陷阱**：两版混用、边界初始化错、`j = 0`（或 `j = -1`）忘记特判。

**重点：教材版与工程版「内核相同、边界不同」——先吃透一种，另一种是「换个下标起点」的翻译。** 死记两套会混乱，理解一套能推导另一套。<span class="marginnote">「<strong>next 数组有「+1 版」和「-1 版」，别混用</strong>」：<strong>教材从 1 计数、next[1]=0；工程从 0 计数、next[0]=-1</strong>。<strong>两种的「跳转逻辑」完全一致，只是边界常量不同</strong>——写代码前先选定一套，别在两种之间「翻译出错」。</span>

## 6 小结

- KMP 核心：失配时 $i$ 不回退、$j$ 跳 `next[j]`——复用已匹配信息。
- `next[j]`：模式串前 $j-1$ 位的最长真前后缀长度 + 1。
- 匹配过程 $O(n+m)$：主串只扫一遍。
- 求 next = 模式串自己跟自己匹配，递推求解。
- 周期越强，next 越大、越省。
- 教材版（0/1 计数）与工程版（-1）内核相同、边界不同。

在下一节，我们严格证明 KMP 的正确性与复杂度——**KMP 的正确性证明与复杂度分析**。
