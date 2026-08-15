---
title: IO 库：流、文件与字符串流
date: 2026-08-07
---

# IO 库：流、文件与字符串流

<div class="epigraph">
<p>一切输入输出皆是文本的流动。</p>
<footer>—— Ken Thompson（肯 · 汤普森）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从 IO 库开始

从第一个 `std::cout << "Hello"` 起，我们就在用流，但一直没有把它当作一个**类型系统**来看。C++ 的输入输出统称**IO 流（stream）**：程序与外部世界的每次对话——读键盘、写终端、读写文件、在内存字符串里做格式化——都抽象成「字符流」这一种形态。第8章的洞察是：**`cin`/`cout`、文件流、字符串流是同一套接口的三个实例**，学会了其中一个，其余两个自然迁移。这章还引入了「流状态」——判定输入是否失败、如何处理坏状态，是写健壮程序的必备技能。<span class="marginnote">C++ 没有把 IO 做进语言关键字（不像 Java 的 `System.out` 或 Python 的内置 `print`），而是做成标准库类型并交给运算符重载（`<<`、`>>`）。这是 C++ 一贯的哲学：<strong>语法最小化，能力交给库</strong>。</span>

## 1 流的类型族谱

C++ 的 IO 类型分三层，全部定义在三个头文件里：

| 头文件 | 类型 | 作用 |
| --- | --- | --- |
| `<iostream>` | `istream` / `ostream` / `iostream` | 终端/控制台的标准 IO |
| `<fstream>` | `ifstream` / `ofstream` / `fstream` | 读写**文件** |
| `<sstream>` | `istringstream` / `ostringstream` / `stringstream` | 读写内存中的**字符串** |

三族共用同一套运算符：`>>` 从流读入、`<<` 向流写出。**流是「字符序列的通道」，至于字符来自终端、文件还是内存字符串，使用者不关心**——这就是多态在 IO 库上的体现（第15章会揭示其实现）。

**cin/cout/cerr/clog** 四个预定义对象：`cin` 标准输入、`cout` 标准输出、`cerr` 标准错误（无缓冲）、`clog` 标准日志（缓冲）。<span class="marginnote">`cerr` 是无缓冲的——出错信息能立刻刷到屏幕上；`cout` 默认有缓冲，程序崩溃时缓冲可能没来得及刷出。所以「排错信息走 cerr、正常输出走 cout」是长期沉淀下来的惯例。</span>

## 2 读写的语法与流状态

`>>` 默认**跳过空白**、按类型解析：`int x; cin >> x;` 从输入里抠出一个整数；`string s; cin >> s;` 抠出一个「词」。`<<` 则把值转成字符序列写出。

**流状态（stream state）**：每个流维护一个状态位，标识流是否可用：

- `good()`：无错误，可正常读写。
- `fail()`：格式错误（如想读整数却读到字母）——流仍可继续，但需 `clear()` 复位。
- `eof()`：读到文件末尾。
- `bad()`：系统级损坏（磁盘错误等），流基本不可恢复。

**重点：** 判断「读入是否成功」的惯用法是把流当作布尔：

```cpp
int sum = 0, v;
while (std::cin >> v) {      // 读到非法输入或 EOF 时，流进入失败态，循环退出
    sum += v;
}
```

`std::cin >> v` 的返回值是 `std::cin` 本身，而流的**布尔转换**在 `fail()` 或 `bad()` 时为假——于是这个 while 在「读到尽头」时干净地停下来。<span class="marginnote">流还能直接用作条件：`if (cin)` 等价于 `if (!cin.fail())`。把「操作后检查流」写成一体，是 C++ 里少见的「表达优雅」——但要注意 <strong>eof 与 fail 的分工</strong>：`eof()` 只在「尝试读、而读到末尾」后为真，不是「还没读就知道到头」。</span>

## 3 文件流：ifstream 与 ofstream

**文件流（file stream）**让程序读写磁盘文件，基本步骤：**创建对象 → 绑定文件 → 读写 → 关闭**。

```cpp
#include <fstream>
std::ifstream in("input.txt");       // 创建并打开（只读）
if (in) {                            // 打开失败则流为 fail 态
    std::string line;
    while (std::getline(in, line))   // 按行读
        std::cout << line << '\n';
} // 离开作用域时自动 close

std::ofstream out("output.txt");     // 打开（默认覆盖写）
out << "hello, file\n";
```

打开文件**失败**（文件不存在、无权限）时，流进入失败态——所以**打开后一定要先检查 `if (in)` 再使用**。`ofstream` 默认**截断（trunc）**旧内容；想追加用 `out.open("log.txt", std::ios::app)`。<span class="marginnote">现代 C++ 倾向用 <strong>RAII</strong>（资源获取即初始化）：文件流对象析构时自动 `close()`，不必手动调用。把「资源生命周期」绑定到「对象生命周期」，是第4篇 Effective C++ 第13条的核心，也是 C++ 相比 C 的关键优势——C 语言里 `fclose` 忘了调就会泄漏文件句柄。</span>

## 4 字符串流：istringstream 与 ostringstream

**字符串流（string stream）**让「字符串」扮演流的角色：把一串文本**当输入**解析、或把格式化结果**写进字符串**。它的用武之地是「先组合、后输出」或「把一整行按空白拆词」：

```cpp
#include <sstream>
std::ostringstream os;
os << "合计：" << 42 << " 元";          // 拼装成字符串
std::string msg = os.str();              // "合计：42 元"

// 反向：解析一行数据
std::istringstream is("2026 8 15");
int y, m, d;
is >> y >> m >> d;                       // y=2026, m=8, d=15
```

**核心对比表：** 三种流的定位

| 维度 | iostream | fstream | sstream |
| --- | --- | --- | --- |
| 数据源 | 标准输入/输出设备 | 磁盘文件 | 内存字符串 |
| 关闭 | 无需 | 自动或显式 `close()` | 无需 |
| 典型场景 | 交互、日志 | 持久化、批量处理 | 拼字符串、解析文本、格式化 |
| 头文件 | `<iostream>` | `<fstream>` | `<sstream>` |

**易错点：** 字符串流与「字符串拼接」不是一回事。`os << 42` 会做**类型到文本的转换**，而 `"..." + 42` 是非法或灾难性的。所有「把数字/浮点/日期格式化成人类可读文本」的操作，统一交给 `ostringstream`，安全且可读。<span class="marginnote">Python 的 `str.format`、Java 的 `String.format`、C 的 `sprintf` 在 C++ 里的对应物就是 `ostringstream`（C++20 又加了更快的 `std::format`）。把格式化统一走流，是「类型安全 + 自动管理缓冲」的双重保障——sprintf 那种「目标缓冲区不够大就溢出」的历史性灾难在这里不会发生。</span>

**格式化操控符**：流的输出格式用「操控符（manipulator）」控制——`std::setw(10)` 设字段宽度、`std::setprecision(6)` 设有效位数、`std::boolalpha` 输出 `true/false`、`std::hex` 切十六进制、`std::fixed` 固定小数位。它们与 `<<` 一起使用，返回流本身，可以串联：

```cpp
#include <iomanip>
std::cout << std::setw(8) << std::setprecision(3) << 3.14159265 << '\n';  // 输出 "    3.14"
```

操控符是「函数式」的——`<<` 接受一个「接受流、返回流」的函数，这正是函数指针/函数对象在 IO 库里的应用。<span class="marginnote">`<iomanip>` 里的操控符（setw、setprecision）与 `<ios>` 里的（boolalpha、hex）分工不同：前者带参数、后者无参数。格式状态会「粘住」——`setprecision` 一旦设置对后续输出持续生效，而 `setw` 只对下一个输出生效。记不住就「用之前重新设置」。</span>

## 5 小结

- IO 类型分**iostream / fstream / sstream** 三族，共用 `<<`、`>>` 与同一套状态机制。
- **流状态**：`good/fail/eof/bad`；`while (cin >> x)` 是「读到尽头自动停」的标准惯用法。
- **文件流**：创建时绑定文件、先检查 `if (in)` 再读写；离开作用域自动 `close()`（RAII）。
- **字符串流**：把字符串当输入解析、或把格式化结果写进字符串；类型安全地替代 `sprintf`。
- `cerr` 无缓冲、排错走它；`cout` 有缓冲，崩溃前可能丢输出。
- IO 的「同一接口、三种来源」是多态的第一次实际亮相——第15章将揭示它背后的继承结构。

在下一节，我们系统化「批量数据的存储与遍历」——**顺序容器与迭代器**：vector 的连续存储与扩容、list 与 deque、forward_list，以及迭代器如何统一「怎么访问容器」。