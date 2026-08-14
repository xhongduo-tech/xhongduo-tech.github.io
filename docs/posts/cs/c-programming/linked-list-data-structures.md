---
title: 链表与基本数据结构实现
date: 2026-08-07
---

# 链表与基本数据结构实现

<div class="epigraph">
<p>数组是连续的长队，链表是手拉手的环——各有各的插队方式。</p>
<footer>—— 对数组与链表的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ C Primer Plus 第17章 ｜ 2026-08-07</p>
</div>

## 为什么从链表讲起

数组的大小在声明时定死，插入/删除元素要搬动整段数据。真实世界的数据是**动态增长**的：用户列表不断添加、任务队列不断进出。**链表（linked list）** 用「指针把节点串起来」的方式解决了这两个痛点——节点在堆上按需分配，插入删除只需改指针。<span class="marginnote">链表是 C 里第一个「指针驱动的数据结构」，也是后续树、图、哈希表的基础：<strong>节点 + 指针串联 + 动态分配</strong>三件套。学链表，本质是学「如何用指针组织动态数据」。理解了链表，数据结构的大门就开了。</span>这一节实现一个完整单链表：节点定义、头插/尾插、删除、遍历与销毁。

## 1 节点：数据 + 指针

链表的基本单元是**节点（node）**：一块装着数据、又牵着下一个节点的动态内存。

```c
typedef struct Node {
    int data;             /* 数据域 */
    struct Node *next;    /* 指针域：指向下一个节点 */
} Node;
```

自引用结构体（`struct Node *next`）我们在第 3 篇见过。一个链表由若干节点**首尾相连**构成，最后一个节点的 `next` 是 `NULL`（链表结束的哨兵）。链表有一个**头指针（head）** 指向第一个节点，`head == NULL` 表示空链表。

![单向链表：头指针依次串联各节点，末节点指向 NULL](/images/c-programming/linked-list-data-structures-1.svg)

**为什么链表能动态增长？** 节点用 `malloc` 在堆上分配，需要多少就分配多少——不像数组需要预先定大小。每新来一个节点，`malloc` 一个再接到链上。

## 2 创建节点与插入

**创建节点**（分配 + 赋值 + 返回指针）：

```c
Node *make_node(int value)
{
    Node *n = (Node *)malloc(sizeof(Node));
    if (n == NULL)
        return NULL;        /* 分配失败 */
    n->data = value;
    n->next = NULL;
    return n;
}
```

**头插法**（新节点插到头部，最常用，O(1)）：

```c
Node *push_front(Node *head, int value)
{
    Node *n = make_node(value);
    if (n == NULL)
        return head;
    n->next = head;       /* 新节点指向原头 */
    return n;             /* 新节点成为新头 */
}
```

注意 `push_front` **返回新的头指针**——因为头变了，调用者必须更新：`head = push_front(head, 10);`。这体现了「函数改指针必须传指针的地址或返回新指针」的纪律。

**尾插法**（新节点接到末尾，需遍历找尾，O(n)）：

```c
void push_back(Node **head, int value)
{
    Node *n = make_node(value);
    if (n == NULL)
        return;
    if (*head == NULL) {          /* 空链表：新节点即头 */
        *head = n;
        return;
    }
    Node *p = *head;
    while (p->next != NULL)       /* 走到最后一个节点 */
        p = p->next;
    p->next = n;                  /* 接上 */
}
```

`push_back` 用**二级指针 `Node **head`**：因为要修改调用者的头指针（空表时），直接传 `Node *` 改不到外部。<span class="marginnote">「函数要修改指针变量本身，就得传指针的指针」——这是 C 里「改指针」的标准手法。头插返回新头、尾插用二级指针，两种方案殊途同归：都是为了让外部看见头的改变。</span>

## 3 删除节点

**删除第一个匹配节点**：要维护「前一个节点」指针，让前一个跳过被删节点：

```c
void delete_value(Node **head, int value)
{
    Node *p = *head, *prev = NULL;
    while (p != NULL && p->data != value) {
        prev = p;
        p = p->next;
    }
    if (p == NULL)
        return;               /* 没找到 */
    if (prev == NULL)
        *head = p->next;      /* 删的是头：头后移 */
    else
        prev->next = p->next; /* 跳过被删节点 */
    free(p);                  /* 释放节点内存 */
}
```

三步：**找到 → 让前一个节点绕过它 → `free`**。删头与删中间节点分开处理，因为删头要动 `*head`。

**销毁整条链表**（释放每个节点，注意先保存 `next` 再 `free`）：

```c
void free_list(Node *head)
{
    while (head != NULL) {
        Node *tmp = head->next;   /* 先记下下一个 */
        free(head);               /* 再释放当前 */
        head = tmp;
    }
}
```

**先存 `next` 再 `free`** 是铁律——`free` 之后访问 `head->next` 是 use-after-free 未定义行为。

## 4 遍历与查找

遍历链表是从头走到尾的经典循环：

```c
void print_list(Node *head)
{
    for (Node *p = head; p != NULL; p = p->next)
        printf("%d ", p->data);
    printf("\n");
}
```

查找同理：

```c
Node *find(Node *head, int value)
{
    for (Node *p = head; p != NULL; p = p->next)
        if (p->data == value)
            return p;
    return NULL;
}
```

**遍历的时间复杂度是 O(n)**：找第 k 个节点必须从头走 k 步——没有随机访问。这是链表相对数组的固有代价：数组 `arr[k]` 是 O(1)，链表只能顺序找。

## 5 核心对比表：数组 vs 链表

| 维度 | 数组 | 链表 |
| --- | --- | --- |
| 内存布局 | 连续 | 分散（节点在堆上） |
| 大小 | 声明时固定 | 动态增长 |
| 随机访问 `a[k]` | O(1) | O(n) 需遍历 |
| 头部插入 | O(n) 搬数据 | O(1) 改指针 |
| 中部删除 | O(n) 搬数据 | O(n) 查找 + O(1) 改指针 |
| 额外内存 | 无 | 每节点一个指针 |
| 缓存友好 | 是（连续） | 否（跳转访问） |

一句话选型：**需要频繁随机访问、大小固定 → 数组；需要频繁插入删除、大小动态 → 链表**。现代 CPU 缓存对连续内存极度友好，所以实际工程里数组往往比链表快得多——链表的教学价值在于「指针动态组织数据」的思维，而不一定是性能首选。

## 6 公式解析：头插法的指针操作

头插的两条赋值把指针操作的精髓浓缩成一行图：

$$
\text{new} \to \text{next} = \text{head}，\quad \text{head} = \text{new}
$$