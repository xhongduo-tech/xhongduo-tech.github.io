---
title: 拷贝控制：拷贝、移动、赋值与析构
date: 2026-08-07
---

# 拷贝控制：拷贝、移动、赋值与析构

<div class="epigraph">
<p>明白你的对象何时被构造、何时被拷贝、何时被销毁，才算真正懂这门语言。</p>
<footer>—— Scott Meyers（斯科特 · 迈耶斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第13章 ｜ 2026-08-07</p>
</div>

## 为什么从拷贝控制开始

第12章我们知道了 `shared_ptr` 会自动管理内存，但它的「自动」背后依赖一套更深层的机制——**当对象被拷贝、被赋值、被销毁时，到底发生了什么？** 每个类都有五个「特殊成员函数」在暗中左右对象的生命周期：**拷贝构造、拷贝赋值、移动构造、移动赋值、析构**。不写它们，编译器就隐式生成「逐成员版本」；而一旦类内部持有指针、资源、连接，逐成员拷贝就会变成**浅拷贝**——两个对象共享同一块内存，析构时双重释放。这一章就是**拷贝控制（copy control）**：理解五个函数的触发时机，用 **Rule of Three/Five** 决定何时该自己写，用 `=default`/`=delete` 精确控制编译器行为。<span class="marginnote">C++ Primer 第13章有一句反复出现的话：<strong>「如果一个类需要自定义析构函数，几乎可以肯定它也需要自定义拷贝构造函数与拷贝赋值运算符」</strong>——这就是 Rule of Three 的来历。反向推演：需要深拷贝的类型，必然有指针成员，必然要析构清理。</span>

## 1 五大特殊成员函数

**拷贝控制成员（copy control members）**：五个函数，前四个都是**重载**（编译器按语境选择），析构函数是唯一的「只此一份」。下表是五者的核心对比表：

| 成员函数 | 什么时候被调用 | 默认行为 |
| --- | --- | --- |
| 拷贝构造 `T(const T&)` | 用已存在对象初始化新对象、按值传参、按值返回 | 逐成员拷贝 |
| 拷贝赋值 `T& operator=(const T&)` | 对已存在对象 `a = b` | 逐成员拷贝 |
| 移动构造 `T(T&&)` | 用右值（临时对象）初始化新对象 | 逐成员移动 |
| 移动赋值 `T& operator=(T&&)` | 对已存在对象 `a = std::move(b)` | 逐成员移动 |
| 析构 `~T()` | 对象生命周期结束时 | 逐成员析构 |

**重点：** 五者之间有微妙的「联动」——编译器会**隐式声明**你没写的那几个；但你一旦自定义了**析构函数**，旧标准里拷贝操作就可能被隐式删除（C++11 规则复杂），C++ 社区因此强烈建议：**要么全按 Rule 写齐，要么全用 `=default`**。<span class="marginnote">五函数联动是 C++11 之后最重要的规则之一：你声明了析构，编译器就不再隐式生成移动操作；移动一旦定义，拷贝也可能被拒绝。与其记忆这些「自动行为矩阵」，不如遵循纪律：<strong>显式 `=default` 你想要编译器代劳的，显式写你需要自定义的</strong>。</span>

## 2 深拷贝：为什么默认拷贝不够

默认拷贝是**逐成员拷贝（memberwise copy）**——对内置类型就是复制值，对指针就是**复制指针本身（浅拷贝）**：

```cpp
class HasPtr {
public:
    HasPtr(const std::string &s) : ps(new std::string(s)), i(0) {}
    ~HasPtr() { delete ps; }                 // 析构清理堆内存
private:
    std::string *ps;                         // 指向堆上的 string
    int i;
};

HasPtr a("hello");
HasPtr b = a;          // 默认拷贝构造：b.ps 与 a.ps 指向同一块内存！
```

**辨析｜易错点：** 默认拷贝后，`a.ps` 与 `b.ps` **指向同一个堆对象**。`a` 析构时 `delete ps`，随后 `b` 析构再次 `delete ps`——**双重释放（double free）**，直接崩溃。要修复，就必须自己写**深拷贝**：让 `b.ps` 指向**新的**、内容相同的 string：

```cpp
HasPtr(const HasPtr &rhs) : ps(new std::string(*rhs.ps)), i(rhs.i) {}
HasPtr &operator=(const HasPtr &rhs) {
    auto newp = new std::string(*rhs.ps);    // 先建新内存
    delete ps;                               // 再释放旧内存（自赋值安全）
    ps = newp;
    i = rhs.i;
    return *this;
}
```

拷贝赋值的**自赋值安全**是这里的经典考点：必须先分配新内存、再释放旧内存——若先 `delete ps` 再拷贝，遇到 `a = a` 时 `rhs.ps` 已被自己删掉，读到的就是悬垂数据。<span class="marginnote">为什么「先新建、后删除」能同时解决自赋值与异常安全？因为「新建失败」时旧状态还完好，异常抛出后对象仍处于合法状态——这是第4篇 Effective C++ 第29条「为异常安全而努力」的雏形，也是 copy-and-swap 惯用法的动机。</span>

## 3 Rule of Three 与 Rule of Five

**Rule of Three**：如果一个类自定义了**析构、拷贝构造、拷贝赋值**三者中的任意一个，就应该把**三个**都写齐。**Rule of Five**（C++11）：加上移动构造与移动赋值，五者齐备。<span class="marginnote">Rule 的本质是「资源管理的一致协议」：<strong>有堆资源 → 需要析构 → 需要深拷贝 → 需要自定义拷贝操作</strong>。不遵循 Rule，浅拷贝 + 双重释放就会悄悄回来。现代 `unique_ptr`、`string`、`vector` 这类「值语义」成员让 Rule 在很多类里自动满足——所谓「零规则」。</span>

**Rule of Five 的完整版本**，以 `HasPtr` 为例：

```cpp
HasPtr(HasPtr &&rhs) noexcept : ps(rhs.ps), i(rhs.i) { rhs.ps = nullptr; }
HasPtr &operator=(HasPtr rhs) {   // copy-and-swap：一次搞定拷贝+移动
    swap(*this, rhs);             // 用 swap 交换状态
    return *this;
}
```

**辨析：** 移动构造是「**偷**」而不是「拷贝」——直接把 `rhs.ps` 的指针**接管**过来，再把 `rhs.ps` 置空，避免一次深拷贝、也避免两个指针指向同一对象。移动操作通常标 `noexcept`，这样 `std::vector` 扩容时才敢用移动而非拷贝。

## 4 =default 与 =delete：声明你的意图

**`=default`**：明确要求编译器生成**默认实现**——「我知道你在，你就按逐成员来」；**`=delete`**：明确**禁止**某个操作——「这个函数不许存在」：

```cpp
class NoCopy {
public:
    NoCopy() = default;
    NoCopy(const NoCopy &) = delete;            // 禁止拷贝构造
    NoCopy &operator=(const NoCopy &) = delete; // 禁止拷贝赋值
    ~NoCopy() = default;
};
```

**重点：** `=delete` 让「禁止拷贝的类型」写起来一目了然——`unique_ptr` 正是用这招禁止拷贝的（只留移动）。`=delete` 只能用于**成员函数**声明处，且函数若被调用会**编译期报错**而非运行期崩溃。<span class="marginnote">Effective C++ 第6条「显式禁止不想自动生成的函数」在 C++11 之前要用「把拷贝操作声明为私有且不实现」的 hack——`=delete` 是它的正统替代。看到 `= delete`，读者立刻明白作者意图，而私有化 hack 只能靠注释。</span>

## 5 拷贝省略（copy elision）与返回值优化

编译器有权**省略**某些拷贝/移动——即使省略会改变可观察行为（但省略掉的拷贝必须仍「可选」，即拷贝/移动操作存在）。最典型的是**返回值优化（RVO）**：

```cpp
HasPtr make() {
    HasPtr p("tmp");
    return p;      // 直接构造到调用者处，无拷贝、无移动
}
```

在 C++17 之前这是「编译器可选的优化」；C++17 起，「返回纯右值」（如 `return HasPtr("x")`）的省略是**标准强制**的。<span class="marginnote">正因为省略拷贝，C++ 里「按值返回大对象」不仅不是反模式，反而是最清晰的写法——RVO 让它零开销。这也是「value semantics + 优化器」组合拳的体现，与 Python 的引用语义、Java 的引用语义都不同。</span>

## 6 小结

- **五大特殊成员函数**：拷贝/移动构造、拷贝/移动赋值、析构——编译器按语境调用、按需隐式生成。
- 默认拷贝是**逐成员（浅）拷贝**：指针成员被「复制指针」而非「复制所指」，导致**双重释放**。
- **深拷贝**要先新建、后删除旧内存，天然兼顾自赋值与异常安全。
- **Rule of Three/Five**：自定义析构者必自定义拷贝/移动操作；有资源就要有协议。
- **`=default`** 显式要编译器代劳，**`=delete`** 显式禁止（如禁止拷贝）。
- **拷贝省略 / RVO** 让按值返回零开销，C++17 起对纯右值强制。

在下一节，我们把「移动」这个动作从拷贝里彻底独立出来——**右值引用与移动语义**：什么是左值/右值、`std::move` 与 `std::forward`、完美转发与移动唯一类型。