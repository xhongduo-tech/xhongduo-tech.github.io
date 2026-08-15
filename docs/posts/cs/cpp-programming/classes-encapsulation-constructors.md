---
title: 类：封装、构造函数与访问控制
date: 2026-08-07
---

# 类：封装、构造函数与访问控制

<div class="epigraph">
<p>类是数据的组织方式，也是行为与数据之间契约的化身。</p>
<footer>—— Bjarne Stroustrup（比雅尼 · 斯特劳斯特鲁普）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从类开始

前面几章我们把数据装进容器、把逻辑写进函数，但数据与操作它的人之间还是松散的。**类（class）**把「数据成员」与「操作这些数据的函数」打包成一个整体，同时用**访问控制**划定外部世界的权限边界——这正是「面向对象」的第一块基石。C++ 里 `struct` 与 `class` 几乎同义，唯一的差别是默认访问级别，这一点让无数新手困惑。这一章以 C++ Primer 第7章的视角，把类的骨架拆开：成员、`this`、构造函数、初始化列表、访问级别与友元、`static` 成员。<span class="marginnote">「封装」这个词来自 1972 年 Dijkstra 提倡的信息隐藏（information hiding）：对象把自己的内部状态藏起来，只通过公开接口与人交往。第15章的继承、第13章的拷贝控制，全都要在「类」这个容器里才能成立。</span>

## 1 struct 与 class：同一件事的两种拼写

**类（class）**：把**数据成员（data member）**与**成员函数（member function）**封装在一起的用户自定义类型。定义类的两种关键字，几乎完全相同：

```cpp
struct Point {          // struct：默认 public
    double x = 0.0;
    double y = 0.0;
};

class Point {           // class：默认 private
    double x = 0.0;     // 外部不可直接访问
public:
    double y = 0.0;
};
```

**重点：** `struct` 与 `class` **唯一的差别是默认访问级别**——`struct` 默认 `public`，`class` 默认 `private`。惯例是：只装数据的 POD 结构用 `struct`，带行为与不变量（invariant）的用 `class`。<span class="marginnote">C++ Primer 第7章开篇就说：`struct` 是为了与 C 兼容而保留的，`class` 才是面向对象的主角。写代码时把「纯数据袋」与「有行为约束的类型」分开，是第4篇 Effective C++ 第18条「让接口易于正确使用」的前奏。</span>

## 2 成员函数与 this

**成员函数（member function）**在类内声明、通常类外定义：

```cpp
struct Sales {
    double revenue = 0;
    double avg_price() const {   // const 成员函数：不修改对象
        return units_sold == 0 ? 0.0 : revenue / units_sold;
    }
};
```

每个成员函数都隐含一个 **`this` 指针**——它指向「调用该函数的那个对象」。`obj.avg_price()` 被编译成 `Sales::avg_price(&obj)`，函数体内访问成员，实际都是通过 `this->` 完成的。<span class="marginnote">`this` 的类型是 `Sales* const`：指针本身不能改（不能换指向对象），但它指向的对象可以改。想表达「这个函数不会改动对象」，就在形参表后加 `const`——此时 `this` 变成 `const Sales*`，这被称为 <strong>const 成员函数</strong>。</span>

**const 成员函数**的价值在重载里体现：`const` 版本与非 `const` 版本可以并存，编译器按「对象是否 const」选择——这是第15章动态绑定之外，另一处「按调用对象性质分发」的机制。

## 3 构造函数与初始化列表

**构造函数（constructor）**：与类同名的特殊成员函数，在对象**创建时**被调用，负责把对象初始化到合法状态。没有构造函数时，编译器生成一个**默认构造函数（default constructor）**，但它对内置类型成员是「默认初始化」（可能有未定义值）——所以**只要类有不变量，就自己写构造函数**：

```cpp
struct Sales {
    Sales() = default;                 // 要求默认构造函数
    Sales(const std::string &isbn, unsigned n)
        : isbn_(isbn), units_sold_(n) {}   // 初始化列表
private:
    std::string isbn_;
    unsigned units_sold_ = 0;
};
```

**初始化列表（constructor initializer list）**：冒号后的 `成员(初值)` 才是**真正的初始化**——在进入函数体之前就已把成员建好；函数体里 `isbn_ = isbn;` 则是「先默认构造、再赋值」，多一次默认构造。<span class="marginnote">对 `const` 成员、引用成员这类「一旦创建不可改绑」的成员，<strong>必须用初始化列表</strong>——它们没有「先构造再赋值」这条路可走。C++ Primer 第7章强调：初始化列表的顺序应与成员声明顺序一致，否则会有微妙的初始化顺序告警。</span>

**委托构造函数（delegating constructor）**（C++11）：一个构造函数可以调用同类的另一个构造函数，复用初始化逻辑：

```cpp
Sales() : Sales("", 0) {}     // 委托给三参版本
```

## 4 访问控制与友元

**访问说明符**控制类成员的对外可见性：`public`（公开）、`private`（私有，仅类内成员可访问）、`protected`（受保护，类内与派生类可访问，第15章见）。这是**封装**的落地手段——数据成员默认私有，外部只能经由公开成员函数读写，从而守住不变量。

**友元（friend）**：类可以指定某个**外部函数或另一个类**为友元，让它们访问私有成员：

```cpp
class Sales {
    friend std::istream &read(std::istream &, Sales &);
    // ...
};
```

**辨析：** 友元**不是**成员——它只是「被特许访问私有的外部函数」。友元声明放在类内、却不参与访问级别；友元关系不能传递（A 是 B 的友元、B 是 C 的友元，不代表 A 是 C 的友元）。友元要慎用：它每用一次，就撕开一次封装的口子。<span class="marginnote">「私有 + 友元」的组合常见于「运算符重载函数」需要读取类内部状态的情形——`operator<<` 打印一个类时通常需要是它的友元。Effective C++ 第23条会告诉我们：非成员非友元函数是比友元更干净的设计，能用友元之外的接口就别用友元。</span>

## 5 static 成员：属于类，而非属于对象

**static 成员（static member）**：用 `static` 声明的数据成员或成员函数，**属于类本身**，所有对象共享一份，不随对象拷贝而复制。访问用 `类名::成员`：

```cpp
class Account {
public:
    static double rate() { return interestRate; }
private:
    static double interestRate;   // 声明
};
double Account::interestRate = 0.05;   // 定义（需在类外给出）
```

**核心对比表：** 普通成员 vs static 成员

| 维度 | 普通成员 | static 成员 |
| --- | --- | --- |
| 归属 | 每个对象一份 | 所有对象共享一份 |
| 访问 | `obj.member` | `ClassName::member` 或 `obj.member` |
| 有无 this | 有 | **没有 this**（不针对某个对象） |
| 能否调用非 static 函数 | 能 | **不能**（没有对象上下文） |
| 典型用途 | 对象状态 | 计数、共享配置、工具函数 |

**易错点：** static 数据成员在类内是**声明**，必须在**类外**定义一次并初始化（`constexpr` 整型等少数例外可类内初始化）。忘了类外定义，链接阶段会报「未定义引用」。

## 6 小结

- **`struct` 默认 public、`class` 默认 private**——同一机制的两种拼写，按「数据袋 vs 有行为类型」选用。
- 成员函数通过**`this` 指针**访问所属对象；`const` 成员函数承诺不改对象。
- **构造函数**在创建时把对象初始化到位；**初始化列表**是真正的初始化，const/引用成员必须走它。
- **访问控制**（public/private/protected）实现封装；**友元**是「特许访问私有的外部函数」，慎用。
- **static 成员**属于类而非对象，用 `类名::` 访问；static 数据成员要在类外定义。
- 类是后续一切高级特性（拷贝控制、继承、多态、模板）的容器。

在下一节，我们把输入输出系统化——**IO 库：流、文件与字符串流**：cin/cout 的流状态、fstream 读写文件、stringstream 在内存里做格式化。