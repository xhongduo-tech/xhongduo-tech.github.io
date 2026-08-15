---
title: 定制 new/delete 与杂项
date: 2026-08-07
---

# 定制 new/delete 与杂项

<div class="epigraph">
<p>内存分配器是最后一道需要你亲自把关的底层。</p>
<footer>—— Scott Meyers（斯科特 · 迈耶斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ Effective C++ 第8–9章 条款49–55 ｜ 2026-08-07</p>
</div>

## 为什么从这里收尾

Effective C++ 的最后 7 个条款分成两块：第8章「定制 new 和 delete」（49–52）深入**内存分配的最底层**——`new-handler` 的行为、何时值得替换 `operator new`/`delete`、写 `new`/`delete` 要守的惯例、`placement new` 与 `placement delete` 的配对；第9章「杂项」（53–55）则把眼光拉回工程习惯——**编译器警告、熟悉标准库、熟悉 Boost**。这两块放在最后是刻意的：先懂机制、再懂定制、最后懂「站在巨人的肩上」。读完这篇，本专题的 30 篇博文也就收束了。<span class="marginnote">为什么「定制 new/delete」值得学？因为<strong>默认分配器在「大量小对象、高并发、追求极低延迟」的场景下不是最优</strong>——游戏引擎、网络服务器、高性能库经常用自己的分配器。而第9章的三条更像「修行」，提醒你：<strong>编译器警告是你的朋友，标准库与 Boost 是你的武器库</strong>。</span>

## 1 条款49：了解 new-handler 的行为

**易错点：** 当 `operator new` 无法分配足够内存时，默认行为是**抛 `std::bad_alloc`**。而在此之前，C++ 会给程序一个**自救机会**——`new-handler` 回调函数：

```cpp
void outOfMem() {        // new-handler 回调
    std::cerr << "无法分配内存\n";
    std::abort();
}
int main() {
    std::set_new_handler(outOfMem);   // 注册
    int *p = new int[1'000'000'000];  // 失败时先调 outOfMem
}
```

**new-handler 必须做的事**（不然死循环）：**让更多内存可用**（释放缓存、缩小池）、**安装另一个 handler**、**解除 handler**（设 nullptr → 恢复抛异常）、**抛异常/终止**。`std::set_new_handler` 返回旧 handler，便于恢复；C++11 的 `std::nothrow` 版 `new` 则返回空指针而非抛异常。<span class="marginnote">设计一个「内存不足时优雅降级」的类（如缓存池），正是靠 new-handler：<strong>分配失败 → handler 清出缓存 → 重试分配</strong>。它把「内存告急」从「立刻崩溃」变成「一次可控的挽救机会」。但注意 handler 是<strong>全局</strong>的——多线程下要考虑它的线程安全性。</span>

## 2 条款50、51、52：定制 new/delete 的纪律

**条款50——了解何时替换 new 和 delete 才有意义**：默认分配器「正确但未必高效」。值得替换的场合：**调试**（记录每次分配的地址、大小、调用栈）、**统计**（分析分配模式）、**提高性能**（定制大小类、内存池）、**减少碎片**。用 `new`/`delete` 的类专用版本（在类内 `static operator new/delete`）做对象池，是高频小对象场景的标准手法。

**条款51——编写 new 和 delete 时要固守常规**：自定义 `operator new` 必须**遵守惯例**：

- 循环调用 new-handler，直到内存足够或 handler 不在了。
- 申请 **0 字节也要返回合法指针**（惯例是「视为 1 字节申请」）。
- 派生类会继承基类的 `operator new`——**按 `sizeof(Base)` 假设分配，对更大的派生对象可能分配不足**，需检查 `size != sizeof(Base)` 时回退默认 `::operator new`。

**条款52——写了 placement new 也要写 placement delete**：**placement new** 是「带额外参数的 new」（常见是 `new (buf) T(...)` 在缓冲区上构造，即**定位构造**）。它不能自己释放内存，因此——**若 placement new 构造中途抛异常，C++ 会去寻找「参数与之匹配的 placement delete」**来清理；找不到匹配的 delete 就不清理，资源泄漏。<span class="marginnote">placement new 的经典用途：内存池、栈缓冲、`std::vector` 的原地构造。它的规则是「<strong>在已分配的内存上构造对象</strong>」——所以「释放」要靠调用方（`buf` 的拥有者）显式 `ptr->~T()`，而不是 delete。<strong>三条纪律</strong>：① 用 placement new 就要配套 placement delete；② 分配/释放配对；③ 别在栈或全局对象上用 placement delete。</span>

**核心对比表：new/delete 家族一览**——它们常被误用，先列清单再讲惯例：

| 形式 | 作用 | 配套的释放 |
| --- | --- | --- |
| `new T` | 分配内存 + 构造一个 T | `delete` |
| `new T[n]` | 分配内存 + 构造 n 个 T | `delete[]` |
| `new (buf) T` | **定位构造**：在已有 buf 上构造（不分配） | `ptr->~T()`（手动析构） |
| `::operator new(size)` | 只分配内存、不构造 | `::operator delete(ptr)` |
| `new (std::nothrow) T` | 分配失败返回 nullptr（不抛） | `delete` |
| `make_unique/make_shared` | 一步「分配 + 构造 + 接管」 | 智能指针自动释放 |

**为什么惯例如此重要**：自定义 `operator new` 会被所有 `new T` 走一遍，`new T[n]` 也是——所以「0 字节也返回合法指针、失败时循环调用 new-handler、检查 `size` 再决定是否回退全局 `::operator new`」这三条是任何自定义分配器的**最低底线**；漏一条，就可能在某个角落制造未定义行为。而 placement new 的「手动析构」纪律，则是「内存归 buf 所有、对象归构造者所有」的所有权划分——谁的资源谁释放，正是整个专题反复强调的 RAII 精神在内存最底层的回响。

## 3 条款53：不要轻忽编译器的警告

**重点：** 编译器警告是「免费送你的静态分析」。认真对待每一条，尤其是「严肃警告」——`-Wall`、`-Wextra`（GCC/Clang）全开，把「想当然」变成「被证明」。警告的常见教训：未初始化的变量、`switch` 漏 case、有符号/无符号比较、忽略返回值、`delete` 与 `delete[]` 混用。<span class="marginnote">「忽略警告」与「依赖警告」只有一线之隔：<strong>高级别警告全开、把每个警告当成潜在 bug</strong>——一次 `-Wuninitialized` 就可能救下一个「运行时才崩」的线上事故。但也要知道：不同编译器的警告集不同，<strong>跨平台代码不能假设「我这个编译器不报就是没问题」</strong>。</span>

**对照表：三条常见警告的信号与对策**

| 警告 | 含义 | 对策 |
| --- | --- | --- |
| `uninitialized` | 用了未初始化变量 | 使用前初始化（条款4） |
| `sign-compare` | 有符号/无符号混比 | 统一类型、用 `size_t` |
| `return-type` | 非 void 函数缺返回值 | 补上返回值 |

## 4 条款54、55：熟悉标准库，熟悉 Boost

**条款54——让自己熟悉标准库，包括 TR1**：标准库是「免费、正确、经过亿万人考验」的组件库——`string`、容器、算法、`numeric`、`memory`、`iostream`、`locale`、regex、chrono、random、type_traits……**写代码前先问「标准库有没有现成的」**。TR1（Technical Report 1）是 2005 年那次大规模扩充（`shared_ptr`、`regex`、`tuple`、`function`、`random`、`type_traits`）的中间形态——C++11 已把它们全部并入 `std`，如今「熟悉 TR1」的实际含义是「熟悉 C++11/14/17 的 `<memory>`、`<regex>`、`<tuple>`、`<functional>` 等头文件」。

**条款55——让自己熟悉 Boost**：**Boost**（boost.org）是「标准库的孵化器」——`shared_ptr`、`regex`、`tuple`、`thread`、`filesystem`、`variant` 都是从 Boost 走进标准库的。当标准库缺件时，先看 Boost 再自己造轮子。<span class="marginnote">条款54+55 的底层哲学是「<strong>复用 > 重写</strong>」：标准库与 Boost 的组件被千万项目锤炼过，正确性与性能都经过验证。<strong>写代码的第一动作不是敲键盘，而是翻库目录</strong>——这是从「写得出」到「写得好」的分水岭，也是 Effective C++ 全书的收官之笔。</span>

**一个收尾清单：从标准库到 Boost 的「查库顺序」**——写任何功能前，按这个顺序问自己「现成的有没有」：

1. **C++ 标准库**：`<string>`、`<vector>`、`<algorithm>`、`<map>`、`<memory>`、`<regex>`、`<chrono>`、`<random>`、`<thread>`、`<filesystem>`、`<functional>`、`<type_traits>`。
2. **Boost**（标准库的预备役）：`Boost.Asio`（网络）、`Boost.Filesystem`（文件系统）、`Boost.Variant`、`Boost.Smart_Ptr`、`Boost.Multiprecision`（任意精度）、`Boost.Program_Options`（命令行解析）。
3. **语言设施**：`auto`、`constexpr`、`lambda`、结构化绑定、概念（C++20）。
4. 最后才轮到自己造。

**实践建议**：把「优先复用」变成肌肉记忆——遇到问题先假设「标准库/Boost 必有解」，翻过再写。这不仅省时间，更让代码的正确性站在千万人验证过的基础上。

**警告与错误的再区分**：编译器的输出分「错误」与「警告」——错误让编译停止、警告不阻止生成。真正危险的**不是警告多**，而是「编译器明确告诉你有问题、你却选择忽略」。`-Wall -Wextra` 全开之后，把「每个警告都当作错误来处理」是纪律：要么修，要么写注释说明「为什么这里的行为符合预期」。很多静态分析工具（clang-tidy、cppcheck）能在编译之外再抓一类「编译器不警告、但明显是 bug」的模式——它们是「编译器警告」的延伸，值得纳入日常工具链。<span class="marginnote">为什么「忽略警告」是反模式？因为<strong>警告是编译器在你编译的瞬间免费送你的静态分析结果</strong>——同一份分析你要在运行时抓 bug 可能要花几个小时。把「警报到错」的开关（`-Werror`）开在 CI 上，让「有警告就失败」成为团队约定，是把这条纪律固化到流程里的标准做法。</span>

**一条贯穿全章的判断**：从条款49到55，Effective C++ 的最后七条其实在回答同一个问题——「**当默认机制不够用时，你是否有能力优雅地接管，且接管得符合规矩？**」`new-handler` 接管「内存告急」、定制 `new`/`delete` 接管「分配策略」、`placement new/delete` 接管「已分配缓冲上的构造」、警告与库接管「正确的默认习惯」。接管不是炫技，而是「理解默认行为 → 确认边界 → 按惯例定制」的成熟路径。

**一个自查口诀**：把本章七条浓缩成三句——「内存不足，先救再抛」（49）；「替换 new/delete 要守惯例，placement 必须成对」（50–52）；「警告当错、库优先、Boost 兜底」（53–55）。这三句能让你在「要不要接管内存管理」「要不要自己造工具」的两个岔路口少走弯路。

**三个「new/delete 配对」的小练习**（判断对错，答案在下一段）：

```cpp
int *p = new int(5);          delete p;     // ① 对吗？
int *q = new int[5];          delete[] q;   // ② 对吗？
int *r = new int(5);          delete[] r;   // ③ 对吗？
```

① 正确：单个 new 配单个 delete；② 正确：数组 new 配数组 delete；③ **错误**：单个 new 配数组 delete，块布局对不上，未定义行为。**规则只有一条：new 与 delete 的形式必须一一对应，而用 `make_unique`/`make_shared`/`vector` 可以完全绕开裸配对。**

## 5 小结

- **条款49**：内存不足先触发 **new-handler** 回调（让内存可用 / 换 handler / 解除 / 终止）；`nothrow` 版 `new` 返回空指针。
- **条款50–52**：值得替换 new/delete 的场合（调试、统计、性能）；自定义 `operator new` 要守「0 字节也返回合法指针、处理派生类更大尺寸」；**placement new 必须配 placement delete**。
- **条款53**：**重视编译器警告**——全开 `-Wall -Wextra`，把警告当 bug。
- **条款54、55**：**先翻标准库，再翻 Boost，最后才自己写**——复用经过验证的组件。
- 至此 C++ Primer 与 Effective C++ 两套权威教材都走完了。

## 6 全专题收束

回顾整个 C++ 编程专题，我们铺开了一条完整的进阶曲线：从**程序结构与基本类型**起步（第1篇），掌握 **string/vector 与表达式、语句、函数**（第1篇）；进入**类与封装、IO、容器、算法、关联容器、智能指针**（第2篇）；攻克**拷贝控制、移动语义、运算符重载、继承与多态**（第3篇）；推进到**模板、特化、异常、命名空间与 RTTI**（第4篇）；最后用 Effective C++ 的 55 条实践经验，把「怎么写正确」升级为「怎么设计才对」——从**让自己习惯 C++** 一路走到**定制 new/delete**。

这一路的收获可以收拢成三条主线：

- **机制**：类型系统、内存模型、对象生命周期、模板实例化——C++ 的「是什么」。
- **纪律**：RAII、const、初始化、异常安全、接口设计——C++ 的「怎么用」。
- **品味**：复用标准库、尊重编译器警告、审慎使用继承与转型——C++ 的「怎么选」。

C++ 是一门「既奖励理解、又惩罚轻率」的语言。掌握了这三条主线，你不仅学会了 C++，也拥有了阅读操作系统、游戏引擎、高性能计算与大型基础设施代码的能力。从「从极限到大模型」的课程坐标看，这里正是你从语言层通向系统层、从单机程序通向高性能并发世界的重要一站。下一站，我们继续在知识树里向上攀登——去接触更多真正运行在「贴近硬件」位置的系统级技术。