---
title: 动态内存与智能指针
date: 2026-08-07
---

# 动态内存与智能指针

<div class="epigraph">
<p>内存泄漏不是一次事故，而是一次又一次忘了释放的累积。</p>
<footer>—— 谚语（C++ 社区）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第12章 ｜ 2026-08-07</p>
</div>

## 为什么从动态内存开始

栈上的局部变量随函数返回自动销毁，但程序里总有「必须活得比创建它的函数更久、或数量运行时才知道」的数据——这正是**动态内存（dynamic memory）**的舞台。C 时代用裸的 `malloc`/`free`，C++ 用 `new`/`delete`，但「谁负责释放」这个所有权问题始终悬而未决：忘释放 = **内存泄漏**，释放两次 = **双重释放崩溃**，释放后仍用 = **悬垂指针**。C++11 用**智能指针**——`shared_ptr`、`unique_ptr`、`weak_ptr`——把「释放」绑定到「对象销毁」这一必然事件上。这一章是理解 RAII（资源获取即初始化）的第一现场，也是 C++ 从「容易出错的手工内存管理」走向「自动、安全、零开销抽象」的分水岭。<span class="marginnote">C++ Primer 第12章的主线：裸 `new`/`delete` 只有在万不得已时才直接用，<strong>智能指针是动态内存的默认选择</strong>。这与第4篇 Effective C++ 第13条「以对象管理资源」完全一致——「资源」不止内存，还有文件句柄、锁、网络连接。</span>

## 1 new 与 delete：原始的动态内存

**`new` 表达式**在堆上分配对象并返回指针；**`delete` 表达式**销毁对象并归还内存：

```cpp
int *p = new int(42);       // 分配一个 int，初始化为 42
delete p;                   // 归还这块内存
p = nullptr;                // 习惯：delete 后置空，防悬垂误用

int *arr = new int[10];     // 动态数组
delete[] arr;               // 注意：数组用 delete[]，别写成 delete
```

三条铁律：**`new` 与 `delete` 必须配对**；**数组用 `new[]`/`delete[]`**；**delete 之后指针变悬垂**，任何继续使用都是未定义行为。<span class="marginnote">`delete[]` 与 `delete` 必须严格对应——`new int[10]` 返回的块头记录了「元素个数」，`delete[]` 才会逐个析构。写错（`new[]` 配 `delete`）在多数编译器上直接崩溃或内存泄漏。</span>正是「手动配对」这个动作太容易被打破，才有了智能指针。

## 2 shared_ptr：共享所有权与引用计数

**`std::shared_ptr`（shared_ptr）**：允许多个智能指针**共享**同一对象的所有权。它内部维护一个**引用计数（reference count）**：每当一个 shared_ptr 拷贝到另一个、计数 +1；某个 shared_ptr 销毁或被重置、计数 -1；计数归零时，对象自动被销毁。

```cpp
#include <memory>
std::shared_ptr<int> sp = std::make_shared<int>(42);  // 推荐创建方式
std::shared_ptr<int> sp2 = sp;    // 拷贝：引用计数 1 → 2
sp.reset();                       // 释放本指针的引用：计数 2 → 1
sp2.reset();                      // 计数 1 → 0，对象被销毁
```

**重点：** 优先用 **`make_shared`** 而非 `new`——它一次分配「对象 + 计数块」两块内存、只做一次内存分配，且在异常安全上更稳。<span class="marginnote">为什么不推荐 `shared_ptr<int> sp(new int(42))`？因为「`new int(42)` 成功、shared_ptr 构造抛异常」之间有一小段对象没人管——C++ Primer 第12章明确建议：<strong>能用 make_shared 就不用 new</strong>。同样的道理在第4篇 Effective C++ 第17条「以独立语句将 newed 对象置入智能指针」还会再强调一次。</span>

## 3 unique_ptr：独占所有权

**`std::unique_ptr`（unique_ptr）**：同一时刻**只能有一个** unique_ptr 指向某对象——它不可拷贝、只能**移动**。所有权可以在 unique_ptr 之间**移交**（转移给新主人，旧指针变空）：

```cpp
std::unique_ptr<int> up = std::make_unique<int>(10);  // C++14
std::unique_ptr<int> up2 = std::move(up);  // 所有权转移：up 变为空
if (!up) { /* up 现在是空指针 */ }
```

`unique_ptr` 是**移动唯一类型**——它没有拷贝构造函数。正因如此它近乎零开销（不像 shared_ptr 要维护计数），是「独占资源」的默认智能指针。<span class="marginnote">「不可拷贝、只能移动」是 C++ 类型设计的一个里程碑模式：<strong>拷贝意味着复制对象，移动意味着转移所有权</strong>。`unique_ptr`、`fstream`、`thread` 都是这种「移动唯一」类型——第13章右值引用与移动语义会把「移动」这个动作本身讲透。</span>

## 4 weak_ptr：打破循环引用的观察者

**`std::weak_ptr`（weak_ptr）**：一种**不增加引用计数**的「观察者」——它指向 shared_ptr 管理的对象，却不阻止对象被销毁。用 `lock()` 临时提升为 shared_ptr：

```cpp
std::shared_ptr<int> sp = std::make_shared<int>(7);
std::weak_ptr<int> wp = sp;        // 观察者，计数不变
sp.reset();                        // 对象被销毁
auto sp2 = wp.lock();              // lock() 返回空的 shared_ptr
if (sp2) { /* 对象还在 */ }
```

**weak_ptr 存在的意义是打破循环引用**：两个 shared_ptr 互相指向对方时，引用计数永远到不了 0——形成内存泄漏。把其中一环换成 weak_ptr，环就断了。<span class="marginnote">循环引用是智能指针的经典陷阱：A 持有 B 的 shared_ptr、B 持有 A 的 shared_ptr，两者互相保命，谁也不会销毁。解法是「环的一侧改用 weak_ptr」——数据拥有关系是树状的，观察关系用 weak_ptr 表达。这也解释了为什么「父持有子 shared_ptr、子持有父 weak_ptr」是树形结构的标准搭配。</span>

## 5 公式解析：引用计数的生命周期

shared_ptr 的引用计数可以精确追踪。设对象 O 被 $n$ 个 shared_ptr 持有：

$$\text{count}(O) = \#\{\text{shared\_ptr 指向 O}\}, \qquad O \text{ 销毁} \iff \text{count}(O) = 0$$

- **第一步，初始**：`make_shared` 创建时 $\text{count}=1$。
- **第二步，拷贝**：每次拷贝 `sp2 = sp`，$\text{count} \gets \text{count} + 1$；每次 `reset()` 或离开作用域，$\text{count} \gets \text{count} - 1$。
- **第三步，归零**：$\text{count}$ 减到 0 的那一刻，析构函数立刻执行，内存归还——**释放时机由「最后持有者消亡」决定**，程序员不再手工指定。
- **代价**：每个 shared_ptr 操作计数都涉及**原子操作**（线程安全），比裸指针和 unique_ptr 有常数开销——这正是「安全」的代价，用它换取「不可能泄漏」。

**对照**：unique_ptr 无计数（$\text{count}$ 恒为 1，仅移动所有权），weak_ptr 不计数（纯观察）。

## 6 动态数组与工厂模式

需要「运行期才知道长度」的数组，优先 `std::vector`；万不得已要动态数组，`make_unique<T[]>(n)` 或 `new T[n]`。智能指针也能管理**其他资源**——只要传入自定义删除器（deleter）：

```cpp
std::shared_ptr<FILE> f(fopen("a.txt", "r"), fclose);  // 管理文件句柄
```

这是 RAII 思想的外推：**任何「用完要释放」的资源都能交给智能指针**。工厂函数「返回 new 出来的对象」时，一律用 unique_ptr 承载：

```cpp
std::unique_ptr<Widget> make_widget();   // 调用方自动获得所有权
```

**`std::enable_shared_from_this`** 解决「成员函数里把自己转成 shared_ptr」的难题：一个被 shared_ptr 管理的类若想 `this` 共享同一所有权（比如把 `this` 塞进容器），直接 `shared_ptr<this>(this)` 会产生第二个控制块、双重释放。正确写法是继承 `std::enable_shared_from_this<T>` 并用 `shared_from_this()` 拿到「与自己共享控制块的 shared_ptr」——它内部维护一个指向控制块的 weak_ptr。<span class="marginnote">一句话讲清：<strong>shared_ptr 的所有权属于「控制块」，不随 `this` 指针移动</strong>。`shared_from_this()` 是从控制块反查 shared_ptr，而 `shared_ptr(this)` 是「再造一个新控制块」——两者天差地别。遇到「类内部要共享所有权」时先想 enable_shared_from_this。</span>

**make_unique 与 new 的取舍**：C++14 补上 `make_unique` 后，三种智能指针都有了「make 一步到位」的创建方式。为什么尽量用 make 系列？除了消除「new 与接管之间的异常窗口」（条款17），还有**性能**：`make_shared` 一次分配「对象 + 控制块」两块内存，而 `shared_ptr<T>(new T)` 要两次分配。唯一不适合 make 的场合是「需要自定义删除器」或「构造函数是 private（工厂会返回裸指针）」——那时才退回显式 `new` + 智能指针接管。<span class="marginnote">`weak_ptr` 的常见误区：它<strong>不能</strong>单独创建（必须从一个 shared_ptr 拷贝而来）；`lock()` 返回的 shared_ptr 可以短暂持有对象，保证「检查到使用」之间对象不被销毁——这是「安全地借对象」的标准姿势，比「先 `expired()` 再 `lock()`」更稳（后者有竞态窗口）。</span>

**一句话收束**：动态内存的意义不在「手动 new/delete」的快感，而在「把所有权交给 RAII、把释放交给析构」之后，程序再也没有「谁负责释放」的悬案。智能指针让「内存安全」从纪律变成类型——这正是 C++ 从 C 走向现代的分水岭，也是第13章拷贝控制、第13.6节移动语义的承接点。

**收尾前再看一眼「裸 new 的合法去处」**：极少数场景仍需要裸 `new`——与 C API 交互需要裸指针、实现「非拥有」的观察者、性能敏感的极简路径。判断标准一句话：**能用智能指针就绝不裸 new**；必须裸时，让「new 与 delete」相距最近、且处于同一函数同一路径。

**练习自查**：① 为什么 `make_shared` 比 `shared_ptr<T>(new T)` 少一次分配？② `unique_ptr` 为什么没有拷贝构造？③ 循环引用怎么用 `weak_ptr` 拆？④ 自定义删除器怎么让 `shared_ptr` 管理文件句柄？四问若能独立作答，本章核心已内化。

## 7 小结

- **`new`/`delete`** 是动态内存的原始形态：必须配对、数组用 `new[]`/`delete[]`、delete 后置空。
- **`shared_ptr`** 用**引用计数**共享所有权，计数归零自动销毁；优先 `make_shared`。
- **`unique_ptr`** 独占所有权、只能移动，近乎零开销，是默认智能指针。
- **`weak_ptr`** 是观察者、不增计数，用 `lock()` 提升；**打破循环引用**是它的本职。
- **引用计数公式**：对象销毁当且仅当 count = 0；shared_ptr 有原子计数开销。
- 智能指针不限于内存——自定义删除器让它管理文件、锁等一切「用完要释放」的资源。

在下一节，我们进入「对象生命周期」的完整控制——**拷贝控制：拷贝、移动、赋值与析构**：五大特殊成员函数、Rule of Three/Five、=default 与 =delete。