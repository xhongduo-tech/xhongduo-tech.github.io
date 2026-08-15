---
title: 资源管理
date: 2026-08-07
---

# 资源管理

<div class="epigraph">
<p>释放资源是所有人的事，最后就变成没有人负责的事。</p>
<footer>—— 谚语（资源所有权问题）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ Effective C++ 第3章 条款13–17 ｜ 2026-08-07</p>
</div>

## 为什么从资源管理开始

第12章讲了智能指针，那是「资源管理」的一种实现；Effective C++ 第3章（条款13–17）把「资源」从**内存**推广到一切**用完要归还**的东西——文件句柄、互斥锁、数据库连接、GUI 句柄，并把「以对象管理资源」上升到 C++ 最核心的设计思想 **RAII（Resource Acquisition Is Initialization）**。五个条款层层递进：**用对象持有资源**（13）、**资源管理类的拷贝行为要想清楚**（14）、**提供对原始资源的访问**（15）、**new/delete 必须同一形式**（16）、**以独立语句把 newed 对象放进智能指针**（17）。这章是「C++ 为什么会是 C++」的答案之一——它把「安全」变成类型系统的纪律，而不是程序员的自觉。<span class="marginnote">RAII 的核心理念一句话：<strong>资源的生命周期 = 持有它的对象的生命周期</strong>——对象构造时拿到资源（获取即初始化），对象析构时自动释放（析构即归还）。因为析构在栈展开、异常传播时必然执行，资源释放就「永远会发生」，无论代码走到哪条路径。</span>

## 1 条款13：以对象管理资源

**易错点：** 手写「`new` → 使用 → `delete`」的正确性取决于「使用」那一段**不提前退出**——一旦中间有 `return`、`throw`、`goto`，`delete` 就被跳过，资源泄漏。

```cpp
void f() {
    Investment *p = createInvestment();   // 裸指针，手工管理
    ...                                   // 中途 return 或抛异常 → p 泄漏！
    delete p;
}
```

**对策：** 把资源立刻交给智能指针，让「释放」跟随「对象的析构」自动发生：

```cpp
void f() {
    std::unique_ptr<Investment> p(createInvestment());  // 立刻接管
    ...                            // 无论怎么退出，p 析构时自动 delete
}
```

**重点：** 两条关键纪律——**资源获得后立即放进管理对象**（「获取即初始化」）；**管理对象用其析构函数确保资源被释放**。`shared_ptr` 用引用计数（多个持有者）、`unique_ptr` 独占（单个持有者），都能自动释放，区别只在「能不能复制」。<span class="marginnote">为什么「一获得就放进对象」重要？因为中间隔着一次函数调用，就可能插入一个 `throw`——<strong>new 出的对象还没进智能指针就抛异常，谁也来不及释放它</strong>。这正是条款17要专门处理的时序问题，也是 `make_unique`/`make_shared` 存在的理由：一步到位、无中间态。</span>

## 2 条款14：资源管理类中注意拷贝行为

**重点：** 当资源本身不是内存、无法用 `shared_ptr` 直接表达时，你得自己写 RAII 类。此时「拷贝这个类」意味着什么，必须想清楚。比如「不允许复制」的 `MutexLock`：

```cpp
class MutexLock {
public:
    explicit MutexLock(Mutex *m) : mutex(m) { lock(mutex); }
    ~MutexLock() { unlock(mutex); }
private:
    Mutex *mutex;
    MutexLock(const MutexLock &) = delete;            // 禁止复制
    MutexLock &operator=(const MutexLock &) = delete;
};
```

资源管理类的拷贝行为有四种选择：

- **禁止拷贝**（如上面的锁）。
- **引用计数共享**：底层引用计数归零才释放（`shared_ptr` 的做法）。
- **深拷贝**：复制底层资源（`string`、`vector` 的做法）。
- **转移所有权**：唯一持有权移交给新对象（`unique_ptr`、`auto_ptr` 的做法）。

**辨析：** 用 `shared_ptr` 的**自定义删除器**可以少写很多自己的 RAII 类——`shared_ptr<Mutex>` 加一个 `unlock` 删除器，就把锁的释放交给了智能指针。<span class="marginnote">条款14的陷阱常出在「复制资源管理类但没定义拷贝行为」——编译器默认的逐成员拷贝会把同一个资源句柄复制成两份，析构时<strong>双重释放</strong>（第13章的浅拷贝问题在资源管理类里同样致命）。所以自定义 RAII 类时，拷贝行为要么显式定义、要么 `= delete`，绝不能交给默认。</span>

## 3 条款15：在资源管理类中提供对原始资源的访问

**易错点：** RAII 类把资源藏起来了，但很多 C 风格 API 要的是**原始资源**（`Mutex*`、`FILE*`、`GLuint`）。于是 RAII 类必须提供「拿到原始资源」的出口，两种设计：

- **显式转换**：`.get()` 返回裸资源——明确但啰嗦。
- **隐式转换**：`operator Mutex*()` ——方便但容易意外触发（把锁对象当指针用）。

```cpp
class MutexLock {
public:
    Mutex *get() const { return mutex; }          // 显式访问
    // operator Mutex*() const { return mutex; }  // 隐式转换（可选）
};
```

**重点：** 显式 `.get()` 是 C++11 智能指针的标准接口（`shared_ptr::get`、`unique_ptr::get`），推荐沿用；隐式转换虽然写起来顺，却会让「该显式表达的地方被悄悄隐式化」——Meyers 的态度是「显式优于隐式」，除非能明确证明隐式带来明显收益。<span class="marginnote">「提供原始资源访问」与「保持封装」的张力是 RAII 的永恒议题：<strong>没有 get，C API 用不了；有了 get，外人又能绕过管理直接碰资源</strong>。工业界的平衡点是「get 返回 const 资源 + 文档声明『别长期持有』」——临时借走可以用，长期持有请拿 `shared_ptr` 的副本。</span>

## 4 条款16：成对使用 new 和 delete 时要采取相同形式

**易错点：** `new` 有「单个对象」和「对象数组」两种形式，`delete` 必须一一对应：

```cpp
std::string *p1 = new std::string;        // 单个对象
std::string *p2 = new std::string[10];    // 数组
delete p1;          // 单个：delete
delete[] p2;        // 数组：delete[]
// delete p2;       // 错误：单个 delete 数组 → 未定义行为
```

**为什么必须对应**：`new[]` 分配的内存块头记录着「元素个数」（为逐个调用析构用），`delete[]` 才知道析构几次；用单个 `delete` 去删数组，析构次数与块布局都对不上，行为未定义。**对策**：不写裸 `new[]`/`delete[]`——用 `std::vector` 或 `std::array` 代替动态/静态数组。<span class="marginnote">这条其实是「用库替代手工」的又一例证：<strong>数组的 new/delete 配对是 C 系语言的高频事故，而 vector 把长度、析构、释放全管住了</strong>。同理，typedef 一个「数组类型」再 delete 它，也是隐蔽事故源——`typedef std::string Addr[4]; delete p;` 实际是 `delete[]` 语义。</span>

## 5 条款17：以独立语句将 newed 对象置入智能指针

**易错点：** 一个微妙的时序问题——把 new 与智能指针放进**同一句**函数调用，可能被「调用顺序」害到：

```cpp
processWidget(std::shared_ptr<Widget>(new Widget), priority());
```

`processWidget` 的两个实参求值顺序未定义，若编译器先求 `new Widget`、再调 `priority()`、而 `priority()` **抛异常**——new 出的裸指针在构造 shared_ptr 之前就丢了，泄漏发生。

**对策：** 拆成独立语句，先让智能指针接管、再调用其他函数：

```cpp
std::shared_ptr<Widget> pw(new Widget);   // 独立语句：先接管
processWidget(pw, priority());            // 再调用
```

**重点：** 用 `std::make_shared`/`std::make_unique` 从根上消除这个窗口——它们把「分配 + 构造智能指针」合为一步，无中间裸指针态。<span class="marginnote">这条与条款13的「立即放入」是同一枚硬币的两面：<strong>「new 与智能指针之间不能有任何可失败的可观察步骤」</strong>。现代 C++ 的答案是 `make_shared`/`make_unique`——它们返回的就是智能指针，中间态根本不存在。写「`shared_ptr<T> p = make_shared<T>(...)`」是这条条款的正规最终形态。</span>

**核心对比表：资源管理类的四种拷贝策略**（条款14）——

| 策略 | 复制语义 | 代表 |
| --- | --- | --- |
| 禁止拷贝 | 无法复制 | MutexLock、unique_ptr |
| 引用计数 | 共享，归零才释放 | shared_ptr |
| 深拷贝 | 复制底层资源 | string、vector |
| 转移所有权 | 唯一持有，移动转移 | auto_ptr、unique_ptr |

## 6 小结

- **条款13**：**以对象管理资源**（RAII）——资源进对象，析构必释放；获取即初始化。
- **条款14**：资源管理类的拷贝行为要显式决定：禁止、引用计数、深拷贝、转移所有权，四选一。
- **条款15**：RAII 类提供 **`.get()`** 显式访问原始资源；隐式转换方便但慎用。
- **条款16**：**`new`/`delete`、`new[]`/`delete[]` 严格对应**；数组优先 `std::vector`。
- **条款17**：**以独立语句把 newed 对象放进智能指针**，或直接用 `make_shared`/`make_unique` 一步到位。
- 资源管理的总纲：把「释放」绑到「对象析构」，把「时序窗口」从语言层面抹掉——这就是 RAII 为何是 C++ 的立身之本。

在下一节，我们转向「接口设计」——**设计与声明**：让接口易于正确使用、类设计即类型设计、pass-by-reference-to-const、数据成员私有化、swap 与函数选择。