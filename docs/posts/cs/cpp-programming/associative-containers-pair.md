---
title: 关联容器与 pair
date: 2026-08-07
---

# 关联容器与 pair

<div class="epigraph">
<p>任何查找问题都可以归结为：如何把键快速映射到值。</p>
<footer>—— Donald Knuth（唐纳德 · 克努特）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从关联容器开始

顺序容器按「位置」存取：`v[i]` 问的是「第几个」。但现实中更多问题按「键」存取：查字典要「查单词」，记电话簿要「找名字」，统计词频要「数每个词出现几次」。**关联容器（associative container）**把「键（key）→ 值（value）」的映射变成一等公民：`map`（字典）、`set`（集合）、以及它们的多重版与无序版。这一章还介绍 **`pair`**——把两个值捆在一起的「最小编程积木」。理解关联容器，等于理解「查找」在数据结构层面的两大流派：**有序树的 $O(\log n)$** 与 **哈希表的均摊 $O(1)$**。<span class="marginnote">Python 的 `dict`、Java 的 `HashMap`、C++ 的 `std::unordered_map` 都是「哈希表」这一思路的产物；而 C++ 的 `std::map` 走红黑树路线，牺牲一点速度换来自动排序与稳定迭代。选哪个，取决于你要不要「按键序遍历」。</span>

## 1 map 与 set：字典与集合

**`std::map`（map）**：**有序**的「键 → 值」容器，键唯一、按键的大小自动排序。**`std::set`（set）**：**有序**的「只含键」容器，每个键只出现一次。

```cpp
#include <map>
#include <set>

std::map<std::string, int> word_count;
word_count["hello"] = 1;          // 用下标访问：键不存在就插入并值初始化
++word_count["hello"];            // 经典词频写法

std::set<std::string> unique;
unique.insert("apple");
```

**重点：** `map` 的下标运算符 `m[key]` 与顺序容器截然不同——**键不存在时，它会自动插入一个「默认初始化的值」**再返回引用。所以 `++word_count["hello"]` 无需判存在：第一次访问建出 0、`++` 变 1。这是便利，也是陷阱：**用 `[]` 查「是否存在」会无意中插入空条目**。<span class="marginnote">只想查、不想插入，用 `m.find(key)`（返回迭代器，找不到返回 `end()`）或 `m.count(key)`（返回 0/1）。C++ Primer 第11章特意对比：<strong>`[]` 是「读改写」语义，`find`/`count` 是「只读」语义</strong>——查存在性永远用后者。</span>

## 2 pair：把两个值捆在一起

**`std::pair`（pair）**：保存**两个**（可不同类型的）值的类模板，定义在 `<utility>`：

```cpp
#include <utility>
std::pair<std::string, int> p{"apple", 3};
p.first;                 // "apple"
p.second;                // 3
```

**重点：** `map` 的每个元素其实就是一个 `pair<const Key, Value>`——遍历 map 时 `it->first` 是键、`it->second` 是值。`insert` 返回 `pair<iterator, bool>`：`bool` 表示「是否真的插入了」（键已存在则为 false）。C++11 的 `make_pair` 与 `auto` 让 pair 用起来几乎零负担。<span class="marginnote">`pair` 还有结构体绑定（structured binding，C++17）：`auto [k, v] = *it;` 一行解出键值。`pair` 是最小的「异构二元组」，它和 tuple（第17章）的关系是「二元组 vs 任意元组」。</span>

## 3 multimap 与 multiset：允许重复键

**`std::multimap` / `std::multiset`**：允许**多个相同键**并存，于是不能用下标运算符（`m[k]` 不知道该返回哪个），插入改用 `insert`、查找用 `count` + `lower_bound`/`upper_bound` 取整个区间：

```cpp
std::multimap<std::string, int> m;
m.insert({"apple", 1});
m.insert({"apple", 5});          // 允许同键多值
auto n = m.count("apple");       // 2
auto lo = m.lower_bound("apple");
auto hi = m.upper_bound("apple");
for (auto it = lo; it != hi; ++it)
    std::cout << it->second << ' ';   // 输出 1 5
```

## 4 无序容器：哈希表的均摊 O(1)

**`std::unordered_map` / `std::unordered_set`**（C++11）：用**哈希表**实现，不再按键排序，换来**均摊 $O(1)$** 的查找、插入、删除。它们与有序版本的接口几乎相同，只是多了**桶（bucket）**的概念：`bucket_count()`、`load_factor()`（负载因子 = 元素数/桶数）、`rehash(n)`。

**核心对比表：** 有序 vs 无序关联容器

| 维度 | `map`/`set`（红黑树） | `unordered_map`/`unordered_set`（哈希表） |
| --- | --- | --- |
| 查找复杂度 | $O(\log n)$ | 均摊 $O(1)$，最坏 $O(n)$ |
| 插入/删除 | $O(\log n)$ | 均摊 $O(1)$ |
| 按键遍历 | **有序**（升序） | **无序** |
| 需要 | 键支持 `<` | 键支持**哈希函数** + `==` |
| 典型用途 | 需要排序结果、区间查询 | 只需快速查、不在乎顺序 |

**易错点：** 想让自定义类型当 `unordered_map` 的键，必须自己提供**哈希函数**；而自定义类型当 `map` 的键只需提供 `operator<`。此外哈希表在元素极多、负载因子过高时会**重哈希（rehash）**——一次性搬桶，此时所有迭代器失效。<span class="marginnote">哈希表之所以均摊 O(1)，是因为「负载因子到阈值就扩容搬桶」——和 vector 扩容的均摊分析一模一样，只是搬的是「桶」而不是「元素」。而红黑树是「自平衡二叉搜索树」：插入删除后做旋转保持高度 $O(\log n)$，保证最坏情况稳定。</span>

## 5 公式解析：单词计数的复杂度

把「统计一段文本里每个词出现次数」的三种容器复杂度算清楚：

- **map 方案**：每次 `++word_count[word]` 先查找 $O(\log n)$，`n` 是不同单词数；总代价 $O(m \log n)$，`m` 是总词数。
- **unordered_map 方案**：每次查找均摊 $O(1)$，总代价 $O(m)$——但常数更大（哈希计算 + 冲突处理）。
- **multiset 方案**：只数词、不关心顺序时够用，但浪费了「值」这个维度。

$$T_{\text{map}}(m,n) = O(m\log n), \qquad T_{\text{unordered}}(m) = O(m)$$

**直觉**：文本越长、单词种类越多，哈希的优势越明显；但若「还要按出现次数排序输出」，map 已经有序、哈希还得再排——**没有银弹，按需求选容器**。<span class="marginnote">这正是第11章全部内容的浓缩：<strong>选容器 = 先问「要不要有序、要不要重复键、数据多大」</strong>。三个问题各对应一类容器，答完即选定。</span>

**自定义类型的哈希与键**：想让自己的类型当 `unordered_map` 的键，要么提供 `std::hash<T>` 的特化、要么给容器传自定义哈希对象：

```cpp
struct Point { int x, y; bool operator==(const Point &o) const { return x==o.x && y==o.y; } };
struct PointHash {
    std::size_t operator()(const Point &p) const {
        return (std::hash<int>{}(p.x) << 16) ^ std::hash<int>{}(p.y);   // 组合两个哈希
    }
};
std::unordered_map<Point, int, PointHash> m;
```

哈希函数的要点是**确定性**（同一键恒同值）与**均匀性**（不同键尽量散开）——把「满足 `==` 的对象映射到同一哈希值」是正确性底线，把「不同对象分散」是性能追求。C++20 的 `std::hash` 与结构体 `==` 让「值语义类型」的键定义更省心。<span class="marginnote">红黑树（map）与哈希表（unordered_map）对键的要求完全不同：map 要 `<`（可排序），unordered_map 要 `==` + 可哈希。<strong>同一类型往往两种都能当键</strong>——`std::string` 两者皆可；自定义类型则看你想用哪张表就提供哪套接口。这是「容器选型」落到「类型接口」的具体体现。</span>

**map 的遍历输出天然有序**：因为 `map`/`set` 底层是红黑树，`begin()` 到 `end()` 的迭代顺序就是**键的升序**——统计词频后想「按键字母序打印」，`map` 一行搞定；用 `unordered_map` 则顺序随机，还要先拷贝进 vector 再 sort。这个「要不要有序结果」的差异，直接决定容器选型。

**emplace 与 pair 的组合**：`m.emplace("apple", 3)` 用「完美转发」直接构造 pair 元素，比 `insert(make_pair(...))` 少一次构造。C++17 的结构化绑定 `for (const auto &[word, cnt] : word_count)` 让遍历 map 的键值解包成为一行——pair 的价值在与现代语法的配合中进一步放大。

## 6 小结

- **`map`** 是有序「键→值」字典，键唯一、`[]` 有「自动插入」副作用；查存在用 `find`/`count`。
- **`set`** 是有序唯一键集合；**`multimap`/`multiset`** 允许重复键，无 `[]`。
- **`pair`** 把两个值捆在一起；map 的元素就是 `pair<const Key, Value>`。
- **`unordered_map`/`unordered_set`** 用哈希表，均摊 $O(1)$ 查找、不排序。
- 有序容器要求键有 `<`；无序容器要求键有哈希函数 + `==`。
- 单词计数这类「按键聚合」是关联容器最典型的应用场景。

在下一节，我们把「内存管理」的权力从裸指针交还给类型——**动态内存与智能指针**：new/delete 的原始形态、shared_ptr/unique_ptr/weak_ptr 三种智能指针与引用计数。