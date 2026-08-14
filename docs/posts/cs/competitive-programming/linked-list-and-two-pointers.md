---
title: 链表与双指针
date: 2026-08-07
---

# 链表与双指针

<div class="epigraph">
<p>指针即数据，数据即指针——改一条 `next`，就改写了一整个世界。</p>
<footer>—— 数据结构课堂箴言</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法竞赛与编程实践 ｜ LeetCode 链表专题 ｜ 2026-08-07</p>
</div>

## 为什么从链表与双指针开始

数组把元素挨个排好，靠下标直达——但它插入、删除中间元素要移动后面全部数据。**链表（linked list）** 反其道而行：每个节点只存「数据 + 指向下一个的指针」，插入、删除只改指针，$O(1)$ 完成；代价是不能随机访问，必须从头逐个找。链表是「用指针组织数据」的第一课，也是面试最爱考的数据结构。而**双指针**则是在链表、数组上「一个循环内用两个指针互相配合」的万能技巧——快慢指针、左右指针，一套组合拳解决一大类问题。

## 1 链表的节点与基本操作

链表的节点是一个结构体，含数据域与指针域：

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};
```

头节点是链表的入口。遍历、反转、插入删除，全部围绕 `next` 指针展开。**重点：改指针前先保存原值。** 反转链表是链表操作的第一道坎：

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) {
        ListNode *nxt = cur->next;  // 先保存下一个
        cur->next = prev;           // 反转
        prev = cur; cur = nxt;      // 前进
    }
    return prev;
}
```

<span class="marginnote">反转链表的关键是<strong>三步走</strong>：记下一个 → 反向指向前 → 双双前进。丢了 `nxt` 就丢了下半段链表，这是新手反转链表的头号崩溃现场。</span>

**辨析｜易错点：** 链表操作最常见的 bug 是**空指针解引用**——`cur->next` 在 `cur` 为空时崩溃。循环条件 `while (cur)` 与 `while (cur->next)` 语义完全不同，动手前先想清楚「我能不能保证 cur 非空」。

## 2 哑节点：统一边界情况的技巧

**哑节点（dummy node）** 是在真正头节点前加的一个哨兵节点，让「头节点」也有前驱，从而把「处理头节点」和「处理中间节点」统一起来。

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0); dummy.next = head;
    ListNode *fast = &dummy, *slow = &dummy;
    for (int i = 0; i <= n; i++) fast = fast->next;  // fast 先走 n+1 步
    while (fast) { fast = fast->next; slow = slow->next; }
    slow->next = slow->next->next;  // 删掉倒数第 n 个
    return dummy.next;
}
```

**重点：哑节点让删除头节点变得安全。** 若没有哑节点，「删除头节点」需要特判；有了它，统一走「前驱的 next 跳过去」一条路，代码更简洁、更难出错。<span class="marginnote">返回时写 `return dummy.next;` 而不是 `return head;`——因为头节点可能被删掉了，`head` 可能已失效。哑节点是链表的「防空洞」，凡是可能动头的操作都值得加上。</span>

## 3 快慢指针：一步与两步的节奏

**快慢指针（fast-slow pointers）** 让两个指针以不同速度前进，靠「速度差」解决问题。

**找链表中间节点**：快指针每次走两步，慢指针每次走一步，快指针到末尾时，慢指针恰好在中点——这同时是「链表判断回文」「链表排序」的前置步骤。

**判断环并找环入口（Floyd 判圈算法）**：

```cpp
ListNode* detectCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {          // 相遇，说明有环
            ListNode *p = head;
            while (p != slow) { p = p->next; slow = slow->next; }
            return p;                // 环入口
        }
    }
    return nullptr;
}
```

**重点：相遇后如何找入口。** 这是 Floyd 判圈算法最精妙的一步——两个指针在环内相遇后，让一个指针从**头节点**重新出发，两个都每次走一步，再次相遇的位置就是**环的入口**。背后的数学，正是下一节的公式解析。

## 4 公式解析：Floyd 判圈算法的距离分析

为什么「快指针从头部重走，与慢指针再次相遇处即环入口」？设头节点到环入口的距离为 $a$，环长为 $L$，两指针第一次相遇时，慢指针在环内走了 $b$：

$$
\text{慢指针总路程} = a + b, \qquad \text{快指针总路程} = a + b + kL
$$

- **第一步，列方程**：快指针速度是慢指针的 2 倍，同时刻路程也是 2 倍：$2(a + b) = a + b + kL$。
- **第二步，化简**：移项得 $a + b = kL$，即 $a = kL - b$。
- **第三步，翻译**：$a$（头到入口）与 $kL - b$（从相遇点继续走到入口再绕整圈）在环上**是同一条弧**。所以让一个指针从头出发、一个从相遇点出发，各自每次一步，必然在环入口相遇。

**重点：距离方程决定算法正确性。** 这段推导说明 Floyd 判圈不只是「碰运气相遇」，而是有明确距离关系支撑的确定性算法。理解它，你就掌握了「用方程证明算法」的方法——这在竞赛里是把「我觉得对」升级成「我知道对」的关键一步。

## 5 双指针在数组上的延伸：左右指针

同样的「双指针」思想搬到有序数组上，就演化出**左右指针（two-pointer sweep）**。经典问题是「有序数组两数之和」：左指针在最左、右指针在最右，和太大右指针左移，和太小左指针右移：

```cpp
vector<int> twoSumSorted(vector<int>& a, int target) {
    int l = 0, r = a.size() - 1;
    while (l < r) {
        int s = a[l] + a[r];
        if (s == target) return {l, r};
        else if (s < target) l++;
        else r--;
    }
    return {};
}
```

**核心概念：为什么能一次扫描完成？** 因为数组有序——左指针右移使和变大，右指针左移使和变小，每一步都**排除了不可能的区域**，总步数不超过 $n$，于是 $O(n)$。这比两数之和的哈希表解法更省空间，是「有序性带来算法红利」的典范。<span class="marginnote">左右指针的本质是「剪枝」：每移动一次指针，就有一整行或一整列的解被排除。这个思想在二维矩阵的搜索、滑动窗口里会反复出现，是 LeetCode 双指针专题的主旋律。</span>

## 6 小结

- 链表节点 = 数据 + `next` 指针；改指针前先保存原值，小心空指针解引用。
- 反转链表是基础操作：记下一个 → 反向 → 双双前进。
- 哑节点统一头节点处理，动头的操作一律加它。
- 快慢指针解决中点、判环；Floyd 判圈用距离方程 $a + b = kL$ 证明环入口的求法。
- 左右指针利用有序性做一次扫描剪枝，$O(n)$