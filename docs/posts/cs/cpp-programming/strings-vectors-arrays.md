---
title: 字符串、向量与数组
date: 2026-08-07
---

# 字符串、向量与数组

<div class="epigraph">
<p>优秀的程序员是那些不满足于表层工具、渴望理解底层机理的人。</p>
<footer>—— Bjarne Stroustrup（比雅尼 · 斯特劳斯特鲁普）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从库类型开始

前两章我们用的一直是 `int`、`double` 这类**内置类型**——它们由语言直接提供，不含方法、不做内存管理。但真实程序处理的是名字、句子、名单、数据表，这些「批量数据」需要一个更聪明的容器。C++ 的标准库为此提供了三类主力：**`std::string`**（可变字符串）、**`std::vector`**（可变数组）与**内置数组**。这一章是「库类型」的第一次正式登场，也是后续容器、算法、迭代器一切工作的起点。<span class="marginnote">标准库类型都定义在 `std` 命名空间里，用前要写 `#include <string>`、`#include <vector>`，再用 `using` 声明或 `std::` 前缀引用。Python 里容器是语言内置的，C++ 里容器是库提供的——这种「语言提供语法、库提供容器」的分工正是 C++ 的哲学。</span>理解 `string` 与 `vector` 的差别，是理解整个标准库的第一把钥匙。

## 1 string：会自我管理的字符串

**`std::string`（string）**：标准库定义的、**长度可变**的字符序列类型。它不是一个「数组加长度」的裸结构，而是一个会管理自己内存的类——你在 `s = "a very long text"` 之后继续 `s += "more"`，内存由它自己申请与释放，不需要你碰指针。

初始化一个 string 有多种写法，各有语义：

```cpp
#include <string>
using std::string;

string s1;                // 默认初始化：空串 ""
string s2 = "hello";      // 拷贝初始化：s2 是 "hello"
string s3("hello");       // 直接初始化：等价于 s2
string s4(10, 'c');       // "cccccccccc"：10 个字符 c
string s5(s2);            // 拷贝 s2
```

**重点：** `string` 的输入输出用 `>>` 与 `getline` 有本质区别。`cin >> s` **只读入第一个非空白字符直到再遇空白**，一次读一个「词」；`getline(cin, s)` 则读一整行（含空格），直到换行符为止（换行符本身被丢弃）。<span class="marginnote">想让「含有空格的整行」进字符串，必须用 `getline`——这是处理人名、地址、整段文本时的头号需求。`cin >> s` 与 `getline` 混用时，注意 `cin >>` 留下的换行符会被下一次 `getline` 直接吞掉，常见的修法是在中间调用一次 `cin.ignore()`。</span>

string 的常用操作：`s.size()` 返回字符个数（类型是 `size_type`，无符号）、`s.empty()` 判空、`s1 + s2` 拼接、`s1 == s2` 比较、`s[i]` 取下标、`s.substr(pos, len)` 取子串。<span class="marginnote">`s.size()` 返回的是无符号的 `size_t`，与上章「不要混用有符号/无符号」的教训直接相关：`s.size() > -1` 恒为真——`-1` 会被转成巨大的无符号数。写循环判断时请把下标变量也声明成 `size_t` 或直接交给范围 for。</span>

## 2 遍历与下标

**范围 for（range for）** 是 C++11 为遍历容器提供的语法糖，把「迭代器 + 取元素」的细节全部藏起来：

```cpp
string s = "C++ Primer";
for (char c : s) {          // 逐个读出字符（副本）
    std::cout << c;
}
for (char &c : s) {         // 引用版：可以修改
    c = std::toupper(c);    // 把每个字符转大写
}
```

**易错点：** 如果只想读、不想改，用 `const char &` 或按值 `char c`；如果想改，必须用引用 `char &`——否则改的只是临时副本，原字符串纹丝不动。范围 for 背后其实在走**迭代器（iterator）**，这一章先看现象，第9章再拆机制。

## 3 vector：可生长的动态数组

**`std::vector`（vector）**：标准库的**容器（container）**，保存同一类型元素的**可变长度**序列。vector 是一个**类模板**：尖括号里写元素类型，`vector<int>`、`vector<string>` 各自成类。第16章会讲「模板怎么来的」，这里先把 `vector<T>` 当「装着 T 的能生长的数组」用。

```cpp
#include <vector>
using std::vector;

vector<int> v1;               // 空 vector
vector<int> v2(10);           // 10 个默认初始化的 0
vector<int> v3(10, 42);       // 10 个 42
vector<string> sv{"a", "bb"}; // 列表初始化（C++11）
```

**重点：** 向 vector 尾部添加元素用 `push_back`，它**没有** `push_front`——因为「在头部插入」对 vector 的底层连续存储代价太高，那是 list 的强项。<span class="marginnote">vector 的底层是一块<strong>连续内存</strong>，`v[i]` 因此是 $O(1)$ 的随机访问；但一旦内存不够，它要整体搬到一块更大的空间——第9章我们会专门算这个「均摊 O(1)」的账，并引入 `capacity` 与 `reserve`。想在已知规模时省去反复搬迁，就先 `reserve`。</span>创建后不断 `push_back` 是「动态收集数据」的标准姿势：先建空 vector，边读边往里加。

## 4 内置数组与 vector 的对立

**内置数组（built-in array）** 是最原始的定长序列：

```cpp
int a[3] = {1, 2, 3};        // 定长 3，编译期确定
int b[] = {4, 5, 6};         // 由初始化列表自动推断长度为 3
```

数组的「先天缺陷」一目了然：**大小固定**、**没有 size 成员**、**数组之间不能直接赋值**、**不知道自己的长度**（`sizeof(a) / sizeof(a[0])` 这个惯用写法在某些语境下还会失效）。C++ 的设计倾向非常明确：**优先用 `std::vector`，内置数组只在极少数场合（定长、追求零开销、与 C 接口交互）才值得用**。

## 5 核心对比表：string / vector / 内置数组

| 维度 | `std::string` | `std::vector<T>` | 内置数组 |
| --- | --- | --- | --- |
| 元素类型 | `char` | `T`（任意） | `T`（任意） |
| 长度 | 可变，`+=` 或 `push_back` | 可变，`push_back` | 定长，编译期确定 |
| 大小 | `s.size()` | `v.size()` | 无成员，需自行跟踪 |
| 赋值 | 支持 `s1 = s2` | 支持 `v1 = v2` | **不支持** `a = b` |
| 内存 | 自动管理 | 自动管理 | 栈或静态区，不自动管理 |
| 随机访问 | `s[i]` $O(1)$ | `v[i]` $O(1)$ | `a[i]` $O(1)$ |
| 遍历 | 范围 for / 下标 | 范围 for / 迭代器 | 范围 for / 下标 |
| 推荐度 | 首选 | 首选 | 仅特殊场合 |

**辨析｜易错点：** 很多人分不清 `string` 与 C 风格字符串。`string` 是库类，自带长度与内存管理；而 `"hello"` 是**字符串字面量**，类型是 `const char[6]`（末尾有看不见的 `'\0'`）。把 string 传给 C 函数时用 `s.c_str()` 拿到 `const char*`——它指向以 `'\0'` 结尾的缓冲区。<span class="marginnote">`'\0'` 是 ASCII 码 0 的转义写法，作为字符串结束标记。C 风格的 `strlen`、`strcpy` 全都依赖这个「哨兵」，所以 `char buf[5] = "hello"` 会编译期报错：5 个字符加上结尾 `'\0'` 一共需要 6 个位置。这也是 C++ 力推 `std::string` 的深层原因——字符串的边界与长度由类替你管住。</span>

**C++17 的 string_view**：`std::string_view`（字符串视图）是「只读地看待一段字符，不拥有它们」——它存储「指针 + 长度」，构造零拷贝。函数签名用 `std::string_view` 接收「字符串参数」时，`"literal"`、`std::string`、`char[]` 都能零拷贝传入，比 `const std::string&` 更通用（`string_view` 不会因为「临时 string 要构造」而额外分配）。**注意**：string_view **不拥有**数据，被它引用的字符串必须活得比它久——这点与「悬垂引用」的纪律一致。<span class="marginnote">`string_view` 是「性能 + 通用性」双赢的现代写法：<strong>读字符串参数优先 `std::string_view`</strong>。但要守住两条：① 别把它当「能改内容」的引用用（它是只读）；② 别返回「指向局部字符串」的 string_view。它和 `std::span`（C++20，数组视图）是同一家族。</span>

**string 的查找与子串**：处理文本时最常用的三个成员——`s.find(sub)` 返回子串第一次出现的下标（找不到返回 `std::string::npos`）、`s.substr(pos, len)` 截取子串、`s.replace(pos, len, str)` 替换一段。它们与「C 风格字符串」的 `strstr`、`strncpy` 相比，最本质的进步是**边界由 string 自己管**——不需要你数长度、不需要担心溢出。判「没找到」用 `if (s.find("x") != std::string::npos)`，注意 `npos` 是 `size_t` 最大值，别与 `-1` 混淆。<span class="marginnote">`npos` 是 `std::string::npos`，类型为 `size_type`、值为 `(size_t)-1`——<strong>它等于无符号的最大值，不是 -1</strong>。因为 `size()` 返回无符号，`find` 用 `npos` 表达「找不到」与「下标从 0 计」天然兼容，这也是「永远把 string 下标当无符号数处理」的又一佐证。</span>

**三种「批量数据」的最终定位**：`std::string` 管「文本」、`std::vector<T>` 管「任意类型的动态序列」、内置数组退居「C 接口/定长极简场景」；而 `std::array`（C++11）补上「定长但现代」的空位。日常写代码，凡「不知道有多少个」一律 vector、凡「文本」一律 string、凡「要跟 C 库打交道」才碰数组。这一层的容器直觉，是第9章全部容器家族的第一块多米诺骨牌。

**一个补充练习视角**：把第3章当作「容器思维的养成」——`string` 是字符的容器、`vector` 是任意类型的容器、数组是最原始的容器，它们共享「下标、size、范围 for」的语言，也各自划定「变长/定长、自动/手动内存」的边界。下一章（表达式）会用到这一章的容器去写实际计算；而第9章会把「容器家族」完整展开。

**动手练习三题**（检验本章掌握度）：① 用 `getline` 读入一整行并统计其中单词个数；② 用 `vector<int>` 收集 10 个输入数，逆序输出；③ 写一个 `const vector<double>&` 参数函数求和并返回。三题分别覆盖「string 按行输入」「vector 动态收集」「const 引用传参」，是本章语法进入实战的敲门砖——建议直接打开编译器把三题各写一遍，比读十遍都有效。

**承上启下**：这一章的容器直觉（下标、size、范围 for、push_back）是第9章「顺序容器家族」的第一印象，也是第10章「泛型算法」的操作对象。带着「容器即数据、迭代器即指针」的视角继续，标准库的图景会越来越清晰。

## 6 小结

- **`std::string`** 是可变长字符串：`>>` 按词读，`getline` 按行读；`size()`、`empty()`、`+`、`substr` 是常用成员。
- **范围 for** 让遍历容器变成一行代码；要修改元素必须用引用 `char &c`。
- **`std::vector<T>`** 是可变长、连续内存、随机访问的容器；尾部添加用 `push_back`，没有 `push_front`。
- **内置数组**大小固定、无 `size`、不能整体赋值——首选 `vector`，数组只留给特殊场合。
- **字符串字面量**是 `const char[]`，与 `std::string` 不同；调 C 接口用 `s.c_str()`。
- 这一章的容器直觉（连续存储、下标、范围 for）是第9章所有容器设计的第一印象。

在下一节，我们把「表达式」单独拎出来——**表达式与运算符**：运算优先级与结合性、左值与右值、短路求值、位运算与类型转换。