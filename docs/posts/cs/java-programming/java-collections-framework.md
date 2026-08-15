---
title: Java 集合框架
date: 2026-08-07
---

# Java 集合框架

<div class="epigraph">
<p>算法 + 数据结构 = 程序；集合框架就是 Java 送你的那一整套数据结构。</p>
<footer>—— 改编自 Niklaus Wirth</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第9章 ｜ 2026-08-07</p>
</div>

## 为什么从集合框架开始

数组定长、下标访问、同质——它把「数据容器」做对了，却不够灵活：不知道数据有多少条时怎么办？要按值快速查找时怎么办？要去重、要排队、要按 key 存取时怎么办？**集合框架（Collections Framework）**是 JDK 提供的整套数据结构库：`List` 管有序序列，`Set` 管不重复集合，`Map` 管键值对，`Queue`/`Deque` 管队列与双端队列。它把第三级《数据结构》课程里学的链表、哈希表、树，都做成了「开箱即用、类型安全、有复杂度承诺」的标准库——你选实现，JDK 保证性能。这一篇不求穷举每个方法，而是把**「选哪个实现」**这个每天都会遇到的决定讲清楚：同一套接口之下，不同实现的复杂度与语义差别，正是数据结构理论在工程里的直接投影。

## 1 集合的接口骨架

集合框架的核心是四组接口，都在 `java.util` 包下：

$$

\text{Collection} \supset \{\text{List},\ \text{Set},\ \text{Queue}\} \qquad \text{Map 独立成系}

$$

- **`Collection`**：所有单元素集合的根接口，声明 `add`、`remove`、`contains`、`size`、`iterator` 等通用操作。
- **`List`**：**有序、可重复**的序列，按下标访问。实现：`ArrayList`、`LinkedList`。
- **`Set`**：**无序（迭代序不定）、不可重复**的集合。实现：`HashSet`、`LinkedHashSet`、`TreeSet`。
- **`Queue` / `Deque`**：先进先出（FIFO）队列与双端队列。实现：`ArrayDeque`、`LinkedList`、`PriorityQueue`。
- **`Map`**：**键值对**表，键不可重复。实现：`HashMap`、`LinkedHashMap`、`TreeMap`。<span class="marginnote">`Map` 不继承 `Collection`——它存的是「键→值」的映射而非元素。但 `map.keySet()`、`map.values()`、`map.entrySet()` 会返回集合视图，因此 `Map` 能参与到集合的遍历中。</span>

**统一性**：四组接口都要求元素是**对象**，不能是基本类型（自动装箱兜底）；都支持泛型，`List<String>` 比裸 `List` 安全；都支持 for-each 遍历，因为都实现了 `Iterable`。

## 2 List：ArrayList 与 LinkedList

`List` 的两个实现是「连续内存」与「链式节点」两种经典数据结构在 Java 里的化身：

| 维度 | ArrayList | LinkedList |
| --- | --- | --- |
| 底层 | 动态数组 | 双向链表 |
| 按下标访问 `get(i)` | O(1) | O(n) |
| 尾部添加 `add(e)` | 均摊 O(1) | O(1) |
| 中间插入/删除 | O(n)（元素搬移） | O(1)（只改指针） |
| 内存 | 紧凑连续 | 每个节点带前后指针，开销大 |
| 常用度 | **几乎总是首选** | 很少 |

**重点结论：默认选 `ArrayList`。** 它缓存友好、随机访问快、内存紧凑；`LinkedList` 的「中间插入 O(1)」听起来美，但**找到插入位置本身要 O(n)**，实际场景几乎总是 `ArrayList` 更快。只有「频繁在头部操作」或「需要实现队列语义」时才考虑它——而那通常用 `ArrayDeque` 更合适。

**辨析｜易错点：`remove(i)` 与 `remove(Object)` 是两个重载。** `list.remove(2)` 按下标删，`list.remove(Integer.valueOf(2))` 按值删。对 `List<Integer>` 调用 `remove(2)` 想删「值为 2 的元素」，实际删的是「下标 2」——经典翻车现场。想要按值删，务必装箱。

## 3 Set：HashSet、LinkedHashSet 与 TreeSet

`Set` 的语义是「不重复」，三个实现用三种机制保证去重与排序：

**`HashSet`**：基于 `HashMap`，用 `hashCode()` 定位桶、`equals()` 判重复。**查找 O(1)**，迭代顺序**不保证**（对新手：别依赖 HashSet 的打印顺序）。
**`LinkedHashSet`**：在 HashSet 上加了**插入顺序**的链表——「去重 + 保持插入序」都要时选它。
- **`TreeSet`**：基于红黑树，元素**按自然顺序或自定义 `Comparator` 排序**，查找 O(log n)，支持「取第一个/最后一个」等有序操作。

**Set 去重的判据**：先比 `hashCode`，相等再比 `equals`。因此**放进 HashSet 的对象必须同时正确重写 `hashCode()` 与 `equals()`**——只写 `equals` 不写 `hashCode`，HashSet 会漏判重复。<span class="marginnote">「hashCode 相等但 equals 不等」是允许的（哈希冲突）；「equals 相等但 hashCode 不等」则是致命违约——违反 hashCode 契约的对象放进 HashSet 会「同一个对象出现两遍」。正确实现见《覆盖 equals、hashCode 与 toString》。</span>

**辨析｜易错点：可变对象放进 HashSet 后再改字段**是危险的——它的桶位由改动前的 hashCode 决定，改完 hash 变了，`contains` 就找不到了。**放进集合的对象应视为不可变**，至少改动要极其克制。

## 4 Map：HashMap、LinkedHashMap 与 TreeMap

`Map` 管「键→值」，三个实现的差别与 Set 的姊妹一一对应：

| 维度 | HashMap | LinkedHashMap | TreeMap |
| --- | --- | --- | --- |
| 底层 | 哈希表（数组+链表/红黑树） | 哈希表 + 插入序链表 | 红黑树 |
| 按键查找 | O(1) | O(1) | O(log n) |
| 迭代顺序 | 不保证 | 插入序（默认） | 键的排序序 |
| 典型场景 | 绝大多数查找场景 | LRU 缓存、保持插入序 | 需要有序遍历、范围查询 |

**`HashMap` 的关键机制**：通过 `key.hashCode()` 计算桶位，桶内用 `equals` 定位。键为 null 是允许的（hashCode 为 0）。`put` 返回旧值、`get` 找不到返回 null、`getOrDefault(key, 默认值)` 更安全。<span class="marginnote">HashMap 的<strong>负载因子</strong>与<strong>扩容</strong>：默认负载因子 0.75，元素数超过容量×0.75 就扩容为两倍并<strong>重新散列所有元素</strong>——所以「预先知道容量」时用 `new HashMap<>(预期大小)` 能避免多次扩容的停顿。</span>

**`LinkedHashMap` 的一个著名用途**：重写 `removeEldestEntry` 就能实现 **LRU（最近最少使用）缓存**——访问过的条目被移到尾部，满了就淘汰最久未用的。

**`TreeMap` 的键必须可比较**（自然顺序或传入 `Comparator`），否则 `put` 时抛 `ClassCastException`。

**Map 的遍历**：`entrySet()` 是最高效的——一次拿键值对，避免按 key 再查一次：

```java
for (Map.Entry<String, Integer> e : map.entrySet()) {
    System.out.println(e.getKey() + " → " + e.getValue());
}
```

## 5 公式解析：HashMap 的查找为什么是 O(1)

HashMap 的常数时间查找，本质是「把元素直接算到它该待的位置」：

$$

\text{桶下标} = \text{hash}(key) \bmod \text{容量}

$$

对这条公式做三步拆解：

- **第一步，算 hash**：对 key 调 `hashCode()`，得到一个 32 位整数——**同一个 key 每次 hash 都相同**，这是确定性。
- **第二步，取模定位桶**：`hash % 容量` 把 key 映射到数组的一个**桶（bucket）**下标——不用遍历比较，直接算到「它该待的位置」。
- **第三步，桶内找**：如果多个 key 撞进同一桶（**哈希碰撞**），桶内用 `equals` 线性比较；Java 8 起，**桶内元素超过 8 个时把链表转成红黑树**，把最坏情况从 $O(n)$ 压到 $O(\log n)$。

**为什么整体是 O(1)**：碰撞不频繁时，绝大多数 key 落进「空桶或单元素桶」，一次 hash + 一次 equals 就找到——与集合大小无关的常数时间。**代价是空间**：数组要预留容量，且 `loadFactor 0.75` 触发扩容（翻倍重散列）。理解「用空间换时间」，是理解哈希表的关键。

**核心对比：List / Set / Map 的选型逻辑**

| 需求 | 用 | 理由 |
| --- | --- | --- |
| 有序、可重复、按下标访问 | `ArrayList` | 随机访问 O(1)、缓存友好 |
| 去重、快速查找 | `HashSet` | 查找 O(1) |
| 去重 + 保持插入序 | `LinkedHashSet` | 哈希 + 链表 |
| 去重 + 排序 | `TreeSet` | 红黑树，有序遍历 |
| 键值对、快速按键查 | `HashMap` | 查找 O(1) |
| 键值对 + 插入序/LRU | `LinkedHashMap` | 可作 LRU 缓存 |
| 键值对 + 有序遍历 | `TreeMap` | 红黑树，范围查询 |
| FIFO 队列 | `ArrayDeque` | 双端、数组实现 |

**重点结论：选实现 = 先定「接口语义」（List/Set/Map/Queue），再定「要不要排序、要不要保持顺序」。** 90% 的日常代码 `ArrayList` + `HashMap` 就够；需要有序/去重/缓存时才换 `Linked*`/`Tree*`。这套「接口 + 实现」的分层，正是《数据结构》课程「抽象数据类型 + 具体实现」在标准库里的投影。

## 6 小结

- 集合框架四接口：**`List`**（有序可重复）、**`Set`**（去重）、**`Queue`/`Deque`**（队列）、**`Map`**（键值对）。
- `List` 默认 **`ArrayList`**；`remove(i)` 与 `remove(值)` 是两个重载，`List<Integer>` 别踩坑。
- `Set` 去重靠 `hashCode` + `equals`；**放进集合的对象务必正确实现两者**，且视为不可变。
- `Map` 三兄弟：`HashMap`（O(1)）、`LinkedHashMap`（插入序/LRU）、`TreeMap`（有序）。
- HashMap 的 O(1) 来自「hash 直接算桶位」；负载因子 0.75、扩容重散列是它的幕后机制。

在下一节，我们把「对象怎么造、怎么销毁」的纪律提上日程——**对象创建与销毁的最佳实践**。