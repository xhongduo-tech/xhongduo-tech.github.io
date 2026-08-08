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

```cpp
#define MAX_LEVEL 16                        // 层数上限

struct Node {
    int key;
    vector<Node*> forward;                  // forward[k] = 第 k 层的下一个结点
    Node(int k, int lvl) : key(k), forward(lvl + 1, nullptr) {}
};

Node* head = new Node(-1, MAX_LEVEL);       // 哨兵头：每层都是入口
int level = 0;                              // 当前最高层（从第 0 层起）

// 查找 key：从最高层开始，能跳就跳、不能跳就降层
Node* find(int key) {
    Node* p = head;
    for (int k = level; k >= 0; --k) {      // 逐层下降
        while (p->forward[k] && p->forward[k]->key < key)
            p = p->forward[k];              // 同层向右跳到「不大于 key」的最大结点
    }
    return p->forward[0];                   // 第 0 层再走一步，目标若存在必在附近
}
```

**重点：跳表的实现三件套——结点带「层数组」指针、头结点是每层的哨兵、level 字段记当前最高层。** 这与「链表 + 哨兵」一脉相承，只是哨兵有多个（每层一个）。<span class="marginnote">「<strong>forward 数组 = 一个结点在多个层里的『分身指针』</strong>」：<strong>第 $k$ 层只有层数 $\ge k$ 的结点才有这个指针</strong>。<strong>哨兵头 + 层数组</strong>让「从最高层出发」的查找可以从 head 一步进入——<strong>实现虽短，结构却完整</strong>。</span>

## 2 插入与删除的完整代码

```cpp
// 随机层数：连续抛「正面」就升一层
int randomLevel() {
    int lvl = 0;
    while (rand() % 2 == 1 && lvl < MAX_LEVEL) lvl++;
    return lvl;
}

// 插入 key
void insert(int key) {
    Node* update[MAX_LEVEL + 1];            // 记录每一层的前驱
    Node* p = head;
    for (int k = level; k >= 0; --k) {      // 一趟查找，逐层记下前驱
        while (p->forward[k] && p->forward[k]->key < key)
            p = p->forward[k];
        update[k] = p;
    }
    int lvl = randomLevel();                // 抛硬币定层数
    if (lvl > level) {                      // 层数超过当前最高层
        for (int k = level + 1; k <= lvl; ++k) update[k] = head;
        level = lvl;
    }
    Node* cur = new Node(key, lvl);
    for (int k = 0; k <= lvl; ++k) {        // 逐层插入
        cur->forward[k] = update[k]->forward[k];
        update[k]->forward[k] = cur;
    }
}

// 删除 key
void erase(int key) {
    Node* update[MAX_LEVEL + 1];
    Node* p = head;
    for (int k = level; k >= 0; --k) {      // 同样一趟查找记录每层前驱
        while (p->forward[k] && p->forward[k]->key < key)
            p = p->forward[k];
        update[k] = p;
    }
    Node* cur = p->forward[0];
    if (cur && cur->key == key) {           // 找到目标才删除
        for (int k = 0; k <= level; ++k)
            if (update[k]->forward[k] == cur)
                update[k]->forward[k] = cur->forward[k];   // 逐层摘除
        delete cur;
        while (level > 0 && head->forward[level] == nullptr) level--;  // 修正最高层
    }
}
```

**重点：插入/删除都用 update 数组记录「每层的前驱」**——一次查找，把每一层「该接/该摘的位置」全部记下，再逐层操作。这个「一趟查找 + 记录每层位置」的技巧是跳表插入/删除的统一骨架。<span class="marginnote">「<strong>update 数组 = 一次查找、全部层的位置快照</strong>」：<strong>查找时在每一层都停在一个位置，存进 update[lvl]</strong>——插入/删除时按层取用即可。<strong>「先查后改、一次到位」</strong>让跳表的增删代码极其对称，也是它比平衡树「边查边旋」清爽的原因。</span>

## 3 公式解析：期望查找长度

设 $n$ 个结点、层高 $L \approx \log_2 n$。查找一个 key 的期望比较次数：

$$
\mathbb{E}[\text{步数}] \le L + \frac{n}{2^L} \approx \log_2 n + O(1)
$$

- **第一步，读「每层最多走几步」**：每层从「上一层的降落点」向右走，直到遇到「高于本层的结点」——由于层数随机独立，**每层期望走 $O(1)$ 步**（几何分布的无记忆性）。
- **第二步，读「层数」**：最高层 $L \approx \log_2 n$——每层 $O(1)$ 步、共 $L$ 层，总期望 $O(\log n)$。
- **第三步，读「高概率」**：层数比 $\log_2 n$ 大很多的概率指数级小——「运气极差」几乎不可能。<span class="marginnote">「<strong>每层期望常数步</strong>」是跳表复杂度分析的核心：<strong>几何分布的「无记忆性」让每一层的步数期望都是常数，不累积</strong>。<strong>这是跳表「期望 $O(\log n)$」的数学根基</strong>——它比 BST 的「平均深度」分析更干净，因为它把「平衡」写进了随机层数本身。</span>

## 4 辨析｜易错点：跳表的复杂度是「期望」不是「最坏」

**期望 $O(\log n)$**：平均意义，硬币公平；
**最坏 $O(n)$**：理论上可能全部结点都抛到第 0 层（退化成链表）——但概率 $2^{-n}$，指数级小；
**实际保证**：对任何固定输入，「不是 $O(\log n)$」的概率可以忽略——**工程上视为 $O(\log n)$**。

**重点：跳表的「最坏退化」概率指数级小，所以工程完全可用**——这与哈希表的「最坏全冲突」同理（随机化哈希后概率可忽略）。「高概率」是随机化结构的通用保证语言。<span class="marginnote">「<strong>概率指数级小 = 工程可忽略</strong>」是随机化算法的通用信念：<strong>$2^{-n}$ 的概率比硬件故障概率还低，工程上不值得担心</strong>。<strong>「期望 + 高概率」取代「最坏」</strong>，是跳表、随机哈希、Treap 共同的分析语言。</span>

## 5 跳表的工程优化

**层数上限**：MAX_LEVEL 设为 $O(\log n)$ 上界，防止极端层数；
**指针压缩**：低层指针更常被访问，可单独优化缓存；
**并发版本**：跳表天然适合「无锁并发」——不同层、不同区间的操作互不干扰，比平衡树的「重平衡锁」并发友好得多（这也是 ConcurrentSkipListMap 选它的原因）。

**重点：跳表的「层」结构让它并发友好**——查找/插入只影响局部指针，无需锁整棵树。这是它胜过红黑树的隐藏优势，也是 Java ConcurrentSkipListMap 选跳表而不用 TreeMap 的理由。<span class="marginnote">「<strong>结构简单 → 并发友好</strong>」：<strong>跳表没有全局重平衡，每个操作只碰自己的指针段，可以细粒度加锁甚至无锁</strong>。<strong>平衡树的旋转/染色需要「锁住局部子树」，并发代价高</strong>——<strong>「简单结构」在并发时代成了隐藏优势</strong>。</span>

## 6 小结

- 实现三件套：结点层指针数组 + 哨兵头 + level 字段。
- 插入/删除：一趟查找记录 update 数组，逐层操作。
- 期望查找 $O(\log n)$：每层期望常数步 + 层数 $\log_2 n$。
- 最坏 $O(n)$ 但概率 $2^{-n}$——「期望 + 高概率」保证。
- 层数上限、指针压缩、并发无锁是工程优化方向。
- 简单结构在并发场景成为隐藏优势（ConcurrentSkipListMap）。

在下一节，我们对比跳表与平衡树的工程取舍——**跳表 vs 平衡树（Redis 为什么选跳表）**。
