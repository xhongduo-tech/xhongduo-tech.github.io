---
title: 设计与声明
date: 2026-08-07
---

# 设计与声明

<div class="epigraph">
<p>接口是类与使用者之间的合同，合同的措辞决定它的执行效果。</p>
<footer>—— Scott Meyers（斯科特 · 迈耶斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ Effective C++ 第4章 条款18–25 ｜ 2026-08-07</p>
</div>

## 为什么从设计声明开始

前两章解决「对象怎么管理」，第4章（条款18–25）往前一步问：**你暴露给使用者的接口，好不好用？** 好接口与坏接口的差别不在「功能全不全」，而在「错误是不是从一开始就不可能犯」。八个条款构成一套接口设计哲学：**让接口易于正确使用**（18）、**把类当类型设计**（19）、**尽量以 const 引用传参**（20）、**别返回局部对象的引用**（21）、**数据成员私有化**（22）、**非成员非友元函数优于成员**（23）、**需要类型转换时声明非成员函数**（24）、**支持不抛异常的 swap**（25）。<span class="marginnote">这一章可以浓缩成 Meyers 的一句话：<strong>「接口应该让正确用法更容易、让错误用法几乎不可能」</strong>（make interfaces easy to use correctly and hard to use incorrectly）。后面的每条都是这句话在某个具体面向的展开——类型、参数、封装、函数归属、交换操作。</span>

## 1 条款18：让接口容易被正确使用，不易被误用

**易错点：** 裸类型参数的接口几乎必然被误用——顺序反了、单位混了、边界错了，编译器浑然不觉。

```cpp
class Date {
public:
    Date(int month, int day, int year);   // 三个 int：month/day/year 顺序谁记得？
    Date(int day, int month, int year);   // 重载？照样分不清
};
```

**对策：** 用**强类型**把「无意义的位置参数」变成「有意义的类型」：

```cpp
struct Day { explicit Day(int d) : val(d) {} int val; };
struct Month { explicit Month(int m) : val(m) {} int val; };
struct Year { explicit Year(int y) : val(y) {} int val; };

class Date {
public:
    Date(const Month &m, const Day &d, const Year &y);   // 顺序由类型锁定
};
Date date(Month(3), Day(15), Year(2026));   // 想写错都难
```

**辅助**：`Month` 还可以把「合法取值」约束进接口——`static Month monthOf(int)` 拒绝非法月份；限制操作——`operator=` 设私有防赋值。**好接口让错误在编译期就露头**。<span class="marginnote">强类型（strong typing）的思路延伸到很多地方：`std::chrono` 用 `minutes`、`seconds` 区分时间单位；`std::filesystem::path` 用专门类型防止「字符串即路径」的误用；智能指针用「所有权语义」杜绝「谁 delete」的歧义。Meyers 的忠告：<strong>预见到用户会犯的错，并在接口层面让他们犯不成</strong>。</span>

## 2 条款19：设计 class 犹如设计 type

**重点：** 设计一个类之前，先问自己「我在设计一个**新的类型**」——这个类型要满足哪些语义？Meyers 给出一张清单，从「怎么创建/销毁」到「拷贝与移动」再到「谁允许访问」，逐条自查：

- 对象如何创建与销毁？（构造、析构、new/delete）
- 初始化和赋值有何区别？（拷贝构造 vs 拷贝赋值）
- 按值传递意味着什么？（深拷贝还是禁止拷贝）
- 合法值的边界？（不变量约束）
- 是否需要继承层次？（虚函数、虚析构）
- 需要哪些类型转换？（显式构造还是隐式转换）
- 哪些运算符重载有意义？（`==`、`<<`、`[]`……）
- 什么样的接口可以不让别人碰？（private/protected 的划分）
- 这个类型「真的需要吗」？（复用旧类型 vs 自造新类型）

**辨析：** 「类设计 = 类型设计」把「API 好不好用」从风格问题提升为**类型系统问题**——你定义的构造、拷贝、转换规则，决定了使用者的每一次赋值、传参、比较是否合法。这是「零开销抽象」的另一面：类型的语义由你定，错了也由你承担。<span class="marginnote">这条清单对应到 C++ Primer 就是第7章（类）到第14章（重载）的全部内容。设计者最常见的错误是「只想着数据 + 方法，忘了拷贝、转换、销毁的语义」——<strong>类型不是数据袋子，是行为契约</strong>。</span>

## 3 条款20：尽量以 pass-by-reference-to-const 替换 pass-by-value

**易错点：** 按值传参默认**拷贝**——对小类型无所谓，对类类型是「一次构造 + 一次析构 + 切片风险」的昂贵操作：

```cpp
bool validate(Student s);        // 按值：Student 的拷贝构造 + 析构
bool validate(const Student &s); // const 引用：零拷贝、不改实参
```

**重点：** 派生类对象按值传给「基类形参」会**切片**——派生类部分被切掉（第15章），虚函数调用退化为基类版本。而 `const Student&` 绑定派生对象时保持动态类型，多态完好。**按值传递唯一合理的场景是「内置小类型」（`int`、`double`、指针、迭代器）**——它们拷贝便宜、也没有切片问题。<span class="marginnote">这条与前一篇第2章「按 const 引用读大对象」同源：<strong>读参数用 `const &`、改参数用 `&`、只有小内置类型按值</strong>。函数对象与 lambda 按值传是另一回事（它们通常小、且按值捕获语义需要）——规则有例外，先记住「类对象默认别按值」这条主线。</span>

## 4 条款21 与 22：别返回局部引用、数据成员私有

**条款21——不要返回指向局部对象的引用或指针**：

```cpp
const Rational &operator*(const Rational &lhs, const Rational &rhs) {
    Rational result(lhs.n * rhs.n, lhs.d * rhs.d);
    return result;    // 局部对象已销毁！悬垂引用
}
```

局部对象在函数返回时销毁，返回它的引用/指针就是悬垂引用。**必须返回对象时，就按值返回**——第13章的 RVO 会让「按值返回」零开销。

**条款22——将成员变量声明为 private**：数据成员藏在 private 里，才能保证「对外接口不变、内部实现可换」。public 数据成员等于把「实现细节」焊死进接口——改内部结构就破坏所有用户代码。private 之外的另一个好处是**可读写分别控制**：`getX()`/`setX()` 可以加校验、可以延迟计算、可以上锁；裸 public 成员统统做不到。<span class="marginnote">「数据私有 + 接口公开」就是封装（第7章）在实践中的完整姿态：<strong>public 成员是「永不改变的契约」，private 成员是「随时可改的实现」</strong>。protected 成员（只对派生类开放）也值得警惕——它同样属于接口的一部分，改动会波及所有派生类，所以「数据成员一律 private」是更干净的铁律。</span>

## 5 条款23 与 24：非成员函数的归属与转换

**条款23——宁以 non-member、non-friend 替换 member 函数**：一个类的**公共接口越多，封装越差**——每个成员函数都能碰私有数据。能作为非成员非友元函数实现的「操作」，放在类外更利于封装（private 字段暴露面更小）、也便于扩展（不需要改类定义）。

```cpp
class Window { public: ... private: ... };
void clearAppWindow(Window &w);   // 非成员非友元：不增加类的封装暴露面
```

**辨析：** 「get/set 一堆访问器，然后所有逻辑都是外部函数」是常见的矫枉过正——**真正属于对象内部状态的操作**（读成员、改成员）当然是成员函数。判断标准：这个操作**必须**访问私有成员吗？不必，就放外面。

**条款24——若所有参数皆需类型转换，请为此采用 non-member 函数**：

```cpp
class Rational {
public:
    Rational(int n = 0, int d = 1) : n_(n), d_(d) {}   // 非 explicit：允许隐式转换
    int n_, d_;
};
// 成员版本：只能 lhs 是 Rational，2 * r 不合法
const Rational operator*(const Rational &lhs, const Rational &rhs) {
    return Rational(lhs.n_ * rhs.n_, lhs.d_ * rhs.d_);
}
Rational r = 2 * 3.14;   // 非成员：两个操作数都能被隐式转换，合法
```

**重点：** 成员 `operator*` 只有左侧操作数能隐式转换（左侧必须是对象才能调成员函数）；非成员让**左右两侧**都参与转换——这是第14章「对称二元运算符用非成员」的缘由。<span class="marginnote">条款23与24共同划定了「成员 vs 非成员」的边界：<strong>必须碰私有成员 → 成员（或友元）；需要对称转换 → 非成员；两可 → 倾向非成员</strong>。第14章的 `operator<<`、`operator+` 选择非成员，正是这套规则的直接应用。</span>

## 6 条款25：考虑写出一个不抛异常的 swap

**`std::swap`** 的默认实现是「三次移动/拷贝」——对「指针指向堆数据」的类，这是白费功夫。**定制 swap**：在类内提供 `swap` 成员、再在类外提供非成员的 `std::swap` 特化（或 ADL 重载），让交换变成「换指针」而非「深拷贝」：

```cpp
class Widget {
public:
    void swap(Widget &other) {   // 成员：交换指针成员即可
        std::swap(pImpl, other.pImpl);
    }
};
namespace std {
    template<> void swap<Widget>(Widget &a, Widget &b) { a.swap(b); }
}
```

**要点：** swap 之所以重要，是因为它也是 **copy-and-swap 赋值**（第2章条款11）、异常安全（条款29）、以及容器/算法内部操作的基础设施——一个不抛异常的 swap 让这些下游全部受益。<span class="marginnote">「不抛异常」是 swap 的黄金标准：swap 只是「交换两个对象的状态」，理论上不该失败。给标准库特化 `swap` 要守规矩：<strong>只能在 `std` 命名空间特化模板</strong>，或者在自己的命名空间提供 `swap` 让 ADL 找到——前者改标准库模板、后者更常见也更干净。</span>

**核心对比表：按值 vs 按 const 引用**（条款20）——

| 维度 | 按值 `T` | 按 `const T&` |
| --- | --- | --- |
| 拷贝 | 一次拷贝构造 + 一次析构 | 零拷贝 |
| 派生类 | **切片**，多态失效 | 保持动态类型 |
| 大对象开销 | 随大小增长 | 恒定 |
| 适用 | 内置小类型、函数对象 | 类类型（读参数） |

## 7 小结

- **条款18**：接口要「易正确用、难错误用」——强类型锁定参数语义、约束合法值。
- **条款19**：**设计类 = 设计类型**：构造/拷贝/转换/销毁的语义都要想清楚，不是数据袋子。
- **条款20**：读参数用 **`const&`**，按值只留给小内置类型；按值传派生类会切片。
- **条款21、22**：别返回局部对象引用；**数据成员一律 private**，接口与实现分离。
- **条款23、24**：不必碰私有的操作放非成员；需对称转换的二元运算符用非成员。
- **条款25**：提供**不抛异常的 swap**（换指针而非深拷贝），让赋值与异常安全受益。

在下一节，我们进入「实现细节」——**实现细节**：延迟变量定义、最小化转型、避免返回内部句柄、异常安全的三种保证、inline 与编译依赖。