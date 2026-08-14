---
title: C++ 与 STL 容器入门
date: 2026-08-07
---

# C++ 与 STL 容器入门

<div class="epigraph">
<p>先让它跑起来，再让它快起来，然后让它正确。</p>
<footer>—— 斯蒂芬 · 约翰逊（Stephen C. Johnson）</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法竞赛与编程实践 ｜ 刘汝佳《算法竞赛入门经典》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从 C++ 与 STL 开始

C 语言教我们「内存是怎么工作的」，但竞赛追求的是「把算法最快地写对」。C++ 在保持高性能的同时，带来了一整套标准模板库 **STL（Standard Template Library）**——vector 替你管动态数组、stack 替你管栈、sort 替你管排序。这意味着你可以**把精力从实现细节转移到算法本身**。今天的竞赛几乎清一色使用 C++，核心原因就是 STL。这一章我们把 C++ 的语法甜点和最常用的容器一次性上手。

## 1 C++ 的基本语法：cin/cout 与引用

C++ 兼容 C 的几乎全部写法，在此之上加了面向对象与泛型。入门最直观的差异是输入输出：`cin` 读入、`cout` 输出，不需要 `%d` 那种格式占位符：

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; double x; string s;
    cin >> n >> x >> s;          // 自动按类型读入
    cout << n << " " << x << "\n"; // 自动按类型输出
    return 0;
}
```

**核心概念：引用（reference）** 是 C++ 新增的「变量的别名」。函数参数用 `&` 声明，就能直接修改调用者的变量，绕开了 C 的按值传递：

```cpp
void swap2(int &a, int &b) { int t = a; a = b; b = t; }
```

<span class="marginnote">`cin/cout` 默认与 C 的 `stdio` 同步，导致偏慢。竞赛模板常在 `main` 开头写 `ios::sync_with_stdio(false); cin.tie(0);` 关闭同步，能让输入输出快一个数量级。若再追求极限，可以直接用 `scanf/printf`。</span>

**辨析｜易错点：** `#include <bits/stdc++.h>` 是竞赛圈的事实标准头文件，一次引入全部 STL；但它不是标准库的一部分，工程代码里不要用。`using namespace std;` 省去 `std::` 前缀，但大项目里易引发命名冲突——竞赛中用它是共识，工程中慎用。

## 2 动态数组 vector 与字符串 string

**`vector`** 是能自动扩容的数组，是 STL 里最常用的容器：

```cpp
vector<int> v;            // 空数组
v.push_back(5);           // 尾部追加
v.pop_back();             // 尾部弹出
int x = v[2];             // 下标访问，O(1)
v.size();                 // 元素个数
```

**重点：`vector` 的访问是 $O(1)$，尾部插入是**均摊** $O(1)$**——空间不够时它会把容量翻倍地搬去新内存，所以偶尔一次 $O(n)$，摊还下来仍是常数。

**`std::string`** 则是 C++ 的字符串，比 `char` 数组安全好用得多：

```cpp
string a = "hello", b = "world";
string c = a + " " + b;      // 直接拼接
int len = c.size();
c.substr(1, 3);              // 从下标 1 起取 3 个字符
c.find("wor");               // 查找子串，找不到返回 string::npos
```

<span class="marginnote">`string` 的比较、拼接都做了运算符重载，写起来像原生类型。注意 `find` 失败返回的是 `string::npos`（一个巨大的无符号值），判断要写 `if (c.find("xx") != string::npos)`，不能直接 `== -1`。</span>

## 3 线性容器：stack、queue 与 priority_queue

竞赛里三种「受限访问」的容器，各有各的出场场景：

**栈 `stack`**：后进先出（LIFO），`push`/`pop`/`top`。
**队列 `queue`**：先进先出（FIFO），`push`/`pop`/`front`。
**优先队列 `priority_queue`**：每次取出的都是最大（或最小）元素，底层是堆（heap）。

```cpp
priority_queue<int> pq;          // 大顶堆：每次取最大
pq.push(3); pq.push(9); pq.push(1);
pq.top();                        // 9
```

| 容器 | 取出顺序 | 插入 | 取出 | 经典用途 |
| --- | --- | --- | --- | --- |
| `stack` | 后进先出 | $O(1)$ | $O(1)$ | 括号匹配、DFS、单调栈 |
| `queue` | 先进先出 | $O(1)$ | $O(1)$ | BFS、层序遍历 |
| `priority_queue` | 按优先级 | $O(\log n)$ | $O(\log n)$ | Dijkstra、哈夫曼 |

**辨析｜易错点：** 优先队列默认是**大顶堆**，想取最小元素，要么存负数，要么自定义比较器：

```cpp
priority_queue<int, vector<int>, greater<int>> pq;  // 小顶堆
```

`greater<int>` 这种「模板里塞比较器」的写法是 C++ 泛型的典型形态，看熟了就不怕。

## 4 sort 与 algorithm 库

`<algorithm>` 头文件是算法仓库，其中 `sort` 是使用率之王——它用的是**内省排序（introsort）**，最坏也是 $O(n\log n)$，且自带常数优化：

```cpp
sort(a, a + n);              // 对数组排序，默认升序
sort(v.begin(), v.end());    // 对 vector 排序
sort(v.begin(), v.end(), greater<int>());  // 降序
```

自定义排序规则时，传入一个**比较函数**或 **lambda**：

```cpp
sort(people.begin(), people.end(), [](const Person &x, const Person &y) {
    return x.age < y.age;    // 按年龄升序
});
```

<span class="marginnote">比较函数必须满足<strong>严格弱序（strict weak ordering）</strong>：对任何两个元素都要给出确定的大小关系，相等时返回假。写出「既说 A 小于 B 又说 B 小于 A」的怪比较器，会让 `sort` 的行为未定义，甚至数组越界崩溃。</span>

## 5 关联容器：set、map 与 unordered 系列

**`set`** 是「有序、无重复」的集合，**`map`** 是「键值对、键有序」的字典，两者底层都是**红黑树**，插入、删除、查找都是 $O(\log n)$：

```cpp
set<int> s;
s.insert(3); s.insert(3);     // 重复插入无效
s.count(3);                    // 1 或 0
s.erase(3);

map<string, int> score;
score["alice"] = 95;           // 不存在就自动创建
score["alice"] += 5;
```

当只关心「在不在」而不需要有序时，`unordered_set` / `unordered_map` 用哈希表实现，查找均摊 $O(1)$，是竞赛里「查重、计数」的首选。<span class="marginnote">`map` 用 `[]` 访问不存在的键会<strong>自动插入一个默认值</strong>，这在「只想查询」时会造成多余元素。只想查不增，用 `find` 或 `count`：`if (score.count(k))`。</span>

## 6 公式解析：迭代器与 [first, last) 半开区间

STL 的一切容器都围绕**迭代器（iterator）** 运转，而迭代器遵循一条铁律——**半开区间 $[first, last)$**：`first` 指向第一个元素，`last` 指向最后一个元素的**下一个位置**。

$$
\text{有效元素} = \{a_{\text{first}}, a_{\text{first}+1}, \ldots, a_{\text{last}-1}\}
$$

- **第一步，为什么含头不含尾**：区间 `[first, last)` 的元素个数恰好是 `last - first`，空区间就是 `first == last`——循环终止条件与区间表示完全统一。
- **第二步，遍历**：`for (auto it = v.begin(); it != v.end(); ++it)`，`*it` 是当前元素，`++it` 移到下一个。
- **第三步，配合算法**：`sort(v.begin(), v.end())` 正是对 `[begin, end)` 排序——三个参数天然契合，也解释了为什么 `end()` 指向的是「尾后」。

**重点：半开区间是 STL 设计的地基**。几乎所有算法函数都以 `[first, last)` 为参数范围，所有循环都以 `!= end()` 为终止条件。理解了它，STL 的阅读与写作都会豁然开朗——这也是 C++ 体系里最值得记住的一个约定。

## 7 小结

- `cin/cout` + `ios::sync_with_stdio(false)` 是竞赛标配；引用 `&` 绕开按值传递。
- `vector` 动态数组均摊 $O(1)$ 尾插，`string` 封装了字符串的常用操作。
- `stack`/`queue`/`priority_queue` 三兄弟：后进先出、先进先出、按优先级。
- `sort` 用比较器自定义规则，比较器必须满足严格弱序。
- `set`/`map` 底层红黑树 $O(\log n)$，`unordered_*` 哈希表均摊 $O(1)$