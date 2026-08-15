---
title: 泛型算法与 lambda 表达式
date: 2026-08-07
---

# 泛型算法与 lambda 表达式

<div class="epigraph">
<p>算法 + 数据结构 = 程序。</p>
<footer>—— Niklaus Wirth（尼克劳斯 · 沃斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从泛型算法开始

第9章我们把容器家族都认全了，但如果每个容器都要「自己写查找、自己写排序」，那将是灾难性的重复劳动。C++ 的解法叫**泛型算法（generic algorithm）**：把「find、count、sort、accumulate」这类操作写成**只依赖迭代器、不依赖具体容器**的独立函数。`std::find` 既能作用于 vector 也能作用于 list——因为它根本不认识容器，只认识「一段迭代器区间」。这一章同时介绍 **lambda 表达式**：一种「就地写的小函数对象」，专门用来把「排序规则、查找条件」这类行为参数喂给算法。<span class="marginnote">这是 C++ 泛型编程（generic programming）的第一次系统亮相：<strong>算法与数据分离，靠迭代器耦合</strong>。Stepanov 设计 STL 的初衷正是如此——同一套 `find`、`sort` 服务所有容器，避免「为每个容器写一遍算法」。第16章的模板就是把这种「泛化」推向极致的机制。</span>

## 1 只读算法：find 与 count

算法通常接受**一对迭代器** `[beg, end)` 表示「作用范围」，返回值可能是迭代器或计数值：

```cpp
#include <algorithm>
#include <numeric>

std::vector<int> v{1, 2, 3, 4, 5, 3};

auto it = std::find(v.begin(), v.end(), 3);   // 找到第一个 3
if (it != v.end())
    std::cout << "找到：" << *it << '\n';

long long n = std::count(v.begin(), v.end(), 3);  // 3 出现 2 次
int sum = std::accumulate(v.begin(), v.end(), 0); // 求和，初值 0
```

**重点：** `find` 找不到时返回 `end()`——这与「尾后迭代器是哨兵」的约定天然契合：**算法用 `end()` 表达「没找到」**。`accumulate` 的第三个参数是**初值**，也是累加的类型起点（`0` 是 int，`0.0` 才是 double）。<span class="marginnote">`[beg, end)` 是 C++ 的<strong>半开区间</strong>惯例：左闭右开，`end` 指向最后一个元素之后。数学里这叫「区间表示法」，STL 全库贯彻——好处是「空区间」也能表示（beg == end），且遍历不重不漏。</span>

## 2 写算法与 back_inserter

有些算法要往目标写结果，目标通常是「空容器」。此时直接传 `v.begin()` 是错的（越界），要用**插入迭代器（insert iterator）**：

```cpp
std::vector<int> src{1, 2, 3};
std::vector<int> dst;
std::copy(src.begin(), src.end(), std::back_inserter(dst));  // 尾插式拷贝
```

`std::back_inserter(dst)` 生成一个「每次写入都 `push_back`」的迭代器——写多少就长多少。**易错点：** 忘记用 `back_inserter`、直接 `copy(..., dst.begin())` 会让程序往未分配的连续内存里写，是经典的越界 bug。<span class="marginnote">`front_inserter`（前插，配合 list）与 `inserter(container, pos)`（指定位置插入）是同一家族。插入迭代器把「写迭代器」翻译成「调用容器的插入」，让「写目标为空容器」的算法调用成为可能。</span>

## 3 排序与二分查找

`sort` 默认升序，可传**比较谓词**定制规则：

```cpp
std::vector<std::string> words{"banana", "apple", "cherry"};
std::sort(words.begin(), words.end());                // 字典序
std::sort(words.begin(), words.end(),
          [](const std::string &a, const std::string &b) {
              return a.size() < b.size();             // 按长度排序
          });
```

`sorted` 之后可用 `binary_search`、`lower_bound` 做 $O(\log n)$ 查找——这是「排序后享高效查找」的经典组合。<span class="marginnote">`lower_bound` 返回「第一个 ≥ 目标值的位置」，`upper_bound` 返回「第一个 > 目标值的位置」——两者的差就是「等于目标值的元素个数」，这个区间叫 equal_range。这套二分工具在有序数据上替代了线性 find。</span>

## 4 lambda 表达式：就地写的小函数

**lambda 表达式（lambda expression）**：一段**可调用的匿名函数**，C++11 引入，语法是 `[捕获列表](形参列表) -> 返回类型 { 函数体 }`：

```cpp
auto is_short = [](const std::string &s) { return s.size() < 5; };
auto cnt = std::count_if(words.begin(), words.end(), is_short);
```

**捕获列表（capture list）**决定 lambda 如何访问外层变量：

```cpp
int threshold = 3;
auto by_len = [threshold](const std::string &a, const std::string &b) {
    return a.size() < b.size();      // 用了外层 threshold（按值捕获）
};
auto bigger = [&threshold]() { ++threshold; };   // 按引用捕获：可改外层变量
auto all = [=]() { /* 按值捕获所有用到的外层变量 */ };
auto allr = [&]() { /* 按引用捕获所有用到的外层变量 */ };
```

**重点：** lambda 本质是一个**匿名函数对象**（functor，第14章会讲函数调用运算符），编译器给它生成一个匿名类、把捕获的变量存成成员。**按值捕获**的是拷贝（只读）、**按引用捕获**的是别名（可改、也可能悬垂——被捕获的局部变量销毁后再调 lambda 就是未定义行为）。<span class="marginnote">lambda 与函数指针的区别：无捕获的 lambda 可以隐式转成函数指针；有捕获的 lambda 不行，因为捕获的数据需要存储。`auto` 是唯一能「装下」lambda 的变量类型——你写不出它的类型名，它只有编译器知道。</span>

## 5 公式解析：sort 的复杂度

`std::sort` 的复杂度是标准保证的 $O(n \log n)$：

$$T(n) = O(n \log n), \qquad \text{比较次数 } \le C \cdot n \log n$$

- **第一步，读复杂度**：`n` 是区间元素个数，`log` 底数无关紧要（换底只是常数因子）。
- **第二步，理解来源**：`sort` 实现为**内省排序（introsort）**——多数时候走快速排序（平均 $O(n \log n)$），但**检测到递归过深（可能退化 $O(n^2)$）时切换到堆排序**，保住最坏情况仍是 $O(n \log n)$。
- **第三步，看常数**：对「基本有序」的数据，introsort 的常数比简单的快速排序更稳；而 `std::stable_sort` 走归并排序，额外 $O(n)$ 空间，换来「相等元素保持原相对顺序」。

**对照表**：为什么不用 $O(n^2)$ 的简单排序？因为 `n` 一上去差距就是数量级：$n=10^5$ 时 $n \log_2 n \approx 1.7\times10^6$，而 $n^2 = 10^{10}$——差近 6000 倍。<span class="marginnote">这套「平均快、最坏也保底」的设计正是工程化的体现：教科书教你快排会退化，STL 的 `sort` 直接内建了防退化机制。同理 `nth_element`（部分排序）只需 $O(n)$，找中位数用它比 `sort` 快一个数量级。</span>

**lambda 的 mutable 与返回类型推导**：按值捕获的 lambda 默认把捕获变量当 `const`（只读）；想修改拷贝，在形参表后加 `mutable`：

```cpp
int cnt = 0;
auto counter = [cnt]() mutable { return ++cnt; };   // 改的是拷贝，外层 cnt 不变
```

返回类型**自动推导**；当函数体不只是一条 `return` 时，可用**尾置返回类型**显式声明：`[](double x) -> double { return x * x; }`。这些细节在「把 lambda 当一等函数」的泛型代码里会反复用到。

## 6 小结

- **泛型算法**只依赖迭代器区间 `[beg, end)`，不认识具体容器——`find`、`count`、`accumulate`、`sort` 全家通用。
- 找不到的约定：算法返回 `end()`；**半开区间** `[beg, end)` 是 STL 全库的表示法。
- **写算法**要用插入迭代器 `back_inserter` 等，别往空容器的 `begin()` 上写。
- **lambda** = 匿名函数对象，`[捕获](形参){体}`；按值捕获只读、按引用捕获可改也可悬垂。
- `std::sort` 是 **introsort**，最坏情况仍 $O(n \log n)$；排序后配 `lower_bound` 二分查找。
- 算法与容器分离的哲学，是第16章模板泛型编程的思想源头。

在下一节，我们转向「按键组织数据」——**关联容器与 pair**：map 与 set、有序与无序、红黑树与哈希，以及 pair 这个「把两个值捆在一起」的最小容器。