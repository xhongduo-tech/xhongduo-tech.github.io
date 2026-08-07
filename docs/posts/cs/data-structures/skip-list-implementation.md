---
title: 跳表的实现与复杂度分析
date: 2026-08-07
---

# 跳表的实现与复杂度分析

<div class="epigraph">
<p>三十行代码，逼近平衡树的全部能力——概率的恩赐。</p>
<footer>—— 跳表格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·跳表 ｜ 2026-08-07</p>
</div>

## 为什么跳表的复杂度是「期望」的

上一节讲了跳表的原理与随机层数。本节做两件事：**给出可运行的完整实现**（查找、插入、删除），并**严格说明那套期望复杂度分析**——为什么查找是期望 $O(\log n)$、为什么「运气差」的概率指数级小。跳表的复杂度分析是「概率分析」的入门范本：**它不保证最坏 $O(\log n)$，但保证「不是 $O(\log n)$ 的概率小到可以忽略」**。理解这套「期望 + 高概率」的语言，是进入随机化算法世界的门票。

## 1 跳表的完整实现

```c
#define MAX_LVL 32

typedef struct SkipNode {
    int key;
    struct SkipNode *forward[MAX_LVL];   /* 每一层的前向指针 */
} SkipNode;

typedef struct {
    SkipNode *header;                    /* 头结点：每层链表的哨兵头 */
    int level;                           /* 当前最高层数 */
} SkipList;

SkipNode *NewNode(int key, int lvl) {
    SkipNode *p = (SkipNode *)malloc(sizeof(SkipNode));
    p->key = key;
    for (i = 0; i < lvl; i++) p->forward[i] = NULL;
    return p;
}

SkipList *CreateList() {
    SkipList *sl = malloc(sizeof(SkipList));
    sl->header = NewNode(-1, MAX_LVL);   /* 哨兵头，负无穷键 */
    sl->level = 0;
    return sl;
}
```

**重点：跳表的实现三件套——结点带「层数组」指针、头结点是每层的哨兵、`level` 记当前最高层。** 这与「链表 + 哨兵」一脉相承，只是哨兵有多个（每层一个）。<span class="marginnote">「<strong>forward 数组 = 一个结点在多个层里的『分身指针』</strong>」：<strong>第 $k$ 层只有层数 $\ge k$ 的结点才有这个指针</strong>。<strong>哨兵头 + 层数组</strong>让「从最高层出发」的查找可以从 `header->forward[level]` 一步进入——<strong>实现虽短，结构却完整</strong>。</span>

## 2 插入与删除的完整代码

```c
void Insert(SkipList *sl, int key) {
    SkipNode *update[MAX_LVL];           /* 记录每层「要插入的位置」 */
    SkipNode *p = sl->header;
    for (int lvl = sl->level; lvl >= 0; lvl--) {   /* 查找：记录每层前驱 */
        while (p->forward[lvl] && p->forward[lvl]->key < key)
            p = p->forward[lvl];
        update[lvl] = p;
    }
    int newLvl = RandomLevel();
    if (newLvl > sl->level) {            /* 新层：补哨兵层 */
        for (lvl = sl->level + 1; lvl <= newLvl; lvl++) update[lvl] = sl->header;
        sl->level = newLvl;
    }
    SkipNode *node = NewNode(key, newLvl);
    for (lvl = 0; lvl <= newLvl; lvl++) {        /* 逐层插入 */
        node->forward[lvl] = update[lvl]->forward[lvl];
        update[lvl]->forward[lvl] = node;
    }
}

void Delete(SkipList *sl, int key) {
    SkipNode *update[MAX_LVL];
    SkipNode *p = sl->header;
    for (int lvl = sl->level; lvl >= 0; lvl--) {
        while (p->forward[lvl] && p->forward[lvl]->key < key)
            p = p->forward[lvl];
        update[lvl] = p;
    }
    SkipNode *node = p->forward[0];
    if (node && node->key == key) {              /* 逐层摘除 */
        for (lvl = 0; lvl <= sl->level; lvl++)
            if (update[lvl]->forward[lvl] == node)
                update[lvl]->forward[lvl] = node->forward[lvl];
        free(node);
    }
}
```

**重点：插入/删除都用 `update[]` 数组记录「每层的前驱」**——一次查找，把每一层「该接/该摘的位置」全部记下，再逐层操作。这个「一趟查找 + 记录每层位置」的技巧是跳表插入/删除的统一骨架。<span class="marginnote">「<strong>update 数组 = 一次查找、全部层的位置快照</strong>」：<strong>查找时在每一层都停在一个位置，存进 update[lvl]</strong>——插入/删除时按层取用即可。<strong>「先查后改、一次到位」</strong>让跳表的增删代码极其对称，也是它比平衡树「边查边旋」清爽的原因。</span>

## 3 公式解析：期望查找长度

设 $n$ 个结点、层高 $L \approx \log_2 n$。查找一个 key 的期望比较次数：

$$
\mathbb{E}[\text{步数}] \le L + \frac{n}{2^L} \approx \log_2 n + O(1)
$$

- **第一步，读「每层最多走几步」**：每层从「上一层的降落点」向右走，直到遇到「高于本层的结点」——由于层数随机独立，**每层期望走 $O(1)$ 步**（几何分布的无记忆性）。
- **第二步，读「层数」**：最高层 $L \approx \log_2 n$——每层 $O(1)$ 步、共 $L$ 层，总期望 $O(\log n)$。
- **第三步，读「高概率」**：层数比 $\log_2 n$ 大很多的概率指数级小——「运气极差」几乎不可能。<span class="marginnote">「<strong>每层期望常数步</strong>」是跳表复杂度分析的核心：<strong>几何分布的「无记忆性」让每一层的步数期望都是常数，不累积</strong>。<strong>这是跳表「期望 $O(\log n)$」的数学根基</strong>——它比 BST 的「平均深度」分析更干净，因为它把「平衡」写进了随机层数本身。</span>

## 4 辨析｜易错点：跳表的复杂度是「期望」不是「最坏」

- **期望 $O(\log n)$**：平均意义，硬币公平；
- **最坏 $O(n)$**：理论上可能全部结点都抛到第 0 层（退化成链表）——但概率 $2^{-n}$，指数级小；
- **实际保证**：对任何固定输入，「不是 $O(\log n)$」的概率可以忽略——**工程上视为 $O(\log n)$**。

**重点：跳表的「最坏退化」概率指数级小，所以工程完全可用**——这与哈希表的「最坏全冲突」同理（随机化哈希后概率可忽略）。「高概率」是随机化结构的通用保证语言。<span class="marginnote">「<strong>概率指数级小 = 工程可忽略</strong>」是随机化算法的通用信念：<strong>$2^{-n}$ 的概率比硬件故障概率还低，工程上不值得担心</strong>。<strong>「期望 + 高概率」取代「最坏」</strong>，是跳表、随机哈希、Treap 共同的分析语言。</span>

## 5 跳表的工程优化

- **层数上限**：`MAX_LVL` 设为 $O(\log n)$ 上界，防止极端层数；
- **指针压缩**：低层指针更常被访问，可单独优化缓存；
- **并发版本**：跳表天然适合「无锁并发」——不同层、不同区间的操作互不干扰，比平衡树的「重平衡锁」并发友好得多（这也是 ConcurrentSkipListMap 选它的原因）。

**重点：跳表的「层」结构让它并发友好**——查找/插入只影响局部指针，无需锁整棵树。这是它胜过红黑树的隐藏优势，也是 Java `ConcurrentSkipListMap` 选跳表而不用 TreeMap 的理由。<span class="marginnote">「<strong>结构简单 → 并发友好</strong>」：<strong>跳表没有全局重平衡，每个操作只碰自己的指针段，可以细粒度加锁甚至无锁</strong>。<strong>平衡树的旋转/染色需要「锁住局部子树」，并发代价高</strong>——<strong>「简单结构」在并发时代成了隐藏优势</strong>。</span>

## 6 小结

- 实现三件套：结点层指针数组 + 哨兵头 + `level`。
- 插入/删除：一趟查找记录 `update[]`，逐层操作。
- 期望查找 $O(\log n)$：每层期望常数步 + 层数 $\log_2 n$。
- 最坏 $O(n)$ 但概率 $2^{-n}$——「期望 + 高概率」保证。
- 层数上限、指针压缩、并发无锁是工程优化方向。
- 简单结构在并发场景成为隐藏优势（ConcurrentSkipListMap）。

在下一节，我们对比跳表与平衡树的工程取舍——**跳表 vs 平衡树（Redis 为什么选跳表）**。
