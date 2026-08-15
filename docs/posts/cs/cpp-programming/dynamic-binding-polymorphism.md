---
title: 动态绑定、抽象基类与多态设计
date: 2026-08-07
---

# 动态绑定、抽象基类与多态设计

<div class="epigraph">
<p>多态不是「继承」的同义词，而是「对一致接口的多样实现」。</p>
<footer>—— Bjarne Stroustrup（比雅尼 · 斯特劳斯特鲁普）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第15章 ｜ 2026-08-07</p>
</div>

## 为什么从动态绑定开始

上一节建立了继承关系、声明了虚函数，但「多态到底怎么发生」还没有完全展开。这节把最后一环补上：**动态绑定（dynamic binding）**的机制（vtable 查找）、**抽象基类（abstract base class）**的设计（纯虚函数），以及「面向接口编程」如何让一段代码在**未来**继续适用于**未知的新类型**。多态的价值不在「继承」，而在**可扩展性**：写完的代码不用改，只要新增一个派生类、塞进同一个容器，行为就自动变。这就是「开闭原则」在 C++ 里的实现载体。<span class="marginnote">面向对象 vs 泛型编程的分工值得一提：多态是<strong>运行时</strong>的「一个接口、多种实现」（继承 + 虚函数）；模板是<strong>编译期</strong>的「一个实现、多种类型」。两者都能写出通用代码，但时机与代价完全不同——性能敏感的路径常选模板，需要运行时插拔的架构选多态。</span>

## 1 虚函数如何被调用：vtable

编译器为「含虚函数的类」生成一张 **vtable（虚函数表）**——一张**函数指针数组**，每个虚函数对应一个表项。每个对象里藏着一个 **vptr（虚表指针）**指向自己类的 vtable：

```cpp
Quote *p = getPtr();         // 可能是 Quote，也可能是 Bulk_quote
p->net_price(5);             // 运行期：p->vptr->[net_price 的表项](5)
```

**动态绑定（dynamic binding）**：编译器把 `p->net_price(5)` 编译成「先取 `p->vptr`、再按偏移取函数指针、再间接调用」——**到底调哪一版，运行期由对象的 vptr 决定**，编译期只知道「调用会发生」。<span class="marginnote">vtable 是「零开销抽象」的经典：多态只多一次指针解引用 + 一次间接跳转，没有运行时类型字符串匹配的开销。代价是：<strong>虚函数不能内联</strong>（目标要到运行时才确定）、对象体积多一个 vptr（通常 8 字节）。</span>

**析构函数进 vtable 的意义**：`delete p` 时先查 vtable 调到「实际类型」的析构函数，从而正确销毁派生类成员——这正是上节「基类析构必须 virtual」的机制根源。

## 2 纯虚函数与抽象基类

**纯虚函数（pure virtual function）**：在虚函数声明末尾加 `= 0`，表示「这个类不提供实现，交给派生类」：

```cpp
class Quote {                       // 抽象基类
public:
    virtual double net_price(std::size_t n) const = 0;  // 纯虚
    virtual ~Quote() = default;
};

class Bulk_quote : public Quote {
public:
    double net_price(std::size_t n) const override { /* 必须实现 */ }
};
```

**抽象基类（abstract base class）**：含至少一个纯虚函数的类。它有两大铁律：

- **不能实例化**——`Quote q;` 编译报错（「无法为抽象类创建对象」）。
- **派生类必须实现所有纯虚函数**，否则它自己仍是抽象类。

**重点：** 抽象基类的意义不是「能造对象」，而是**定义接口契约**——它把所有具体实现推迟到派生类，让调用方只依赖「抽象」，不依赖「具体」。这正是设计模式里「面向接口编程、而非面向实现编程」的第一原则。<span class="marginnote">C++ 里「抽象基类 + 虚函数」近似 Java 的 interface + 默认方法，但更灵活（可带数据成员、非虚函数、构造函数）。实践中常把抽象基类的析构函数写成虚的、把接口方法写成纯虚的——这就是「接口类」惯用法，是大型 C++ 架构的骨架。</span>

## 3 面向接口的多态设计

多态设计的完整画面：**调用方持有基类指针/引用，运行时塞入各种派生类对象**。

```cpp
double total(const std::vector<std::shared_ptr<Quote>> &items, std::size_t n) {
    double sum = 0;
    for (const auto &item : items)
        sum += item->net_price(n);   // 每个 item 各走自己的版本
    return sum;
}

std::vector<std::shared_ptr<Quote>> cart;
cart.push_back(std::make_shared<Bulk_quote>("book", 50.0, 10));
cart.push_back(std::make_shared<Quote>("other", 20.0));
double t = total(cart, 5);           // 混着算：两种策略自动各算各的
```

**要点：** 容器里装的是 `shared_ptr<Quote>`（多态必须指针/引用）；`net_price` 的调用在运行期分发；**新增一个 `Discount_quote` 类、`push_back` 进去，`total` 函数一行不改**——这就是「对扩展开放、对修改封闭」的开闭原则。<span class="marginnote">「把算法绑定到具体类型」与「把算法绑定到接口」的区别是面向对象与传统过程式设计的根本分歧：<strong>多态让你在编译期不认识具体类型的前提下，写出可复用的行为</strong>。这与第10章泛型算法「只认迭代器、不认容器」的哲学遥相呼应。</span>

## 4 核心对比表：静态绑定 vs 动态绑定

| 维度 | 静态绑定（非虚） | 动态绑定（虚） |
| --- | --- | --- |
| 决定时机 | **编译期** | **运行期**（vtable 查找） |
| 依据类型 | 静态类型 | 动态类型 |
| 调用开销 | 直接调用（可内联） | 间接跳转（不可内联） |
| 依赖接口 | 不依赖 | 依赖抽象基类 |
| 扩展方式 | 改代码重新编译 | 新增派生类、无需改调用方 |
| 典型场景 | 性能关键路径 | 插件、回调、框架扩展点 |

**辨析：** 静态绑定不是「劣质版」——`std::sort` 对 `vector<int>` 的比较函数在编译期就定死，快得多；而动态绑定用在「运行时才知道具体是哪种对象」的场合。选型标准是：**类型在编译期可穷举 → 静态；类型要开放给未来 → 动态**。<span class="marginnote">还有一个折中方案值得一提：C++17 的 <strong>`std::variant`</strong> 用「编译期封死的类型集合」模拟多态（`std::visit`），既保留静态绑定速度、又避免继承——适合「类型集合固定且已知」的场合。它是「继承多态 vs 模板多态」之外的第三条路。</span>

**覆盖的完整形态与协变返回**：派生类覆盖虚函数时，返回类型可以「收窄」到派生类自身——这称为**协变返回类型（covariant return type）**：`Bulk_quote* Bulk_quote::clone() const override { return new Bulk_quote(*this); }` 可以覆盖 `Quote* Quote::clone() const`。协变返回让「多态工厂」更顺——`clone` 返回的 `Bulk_quote*` 在 `auto` 里保持具体类型，调用方无需再 cast。**补充规则**：覆盖函数必须与基类签名一致（除了协变返回），`virtual` 关键字在派生类里可写可不写（写更清楚）；`override` 负责让编译器检查「真的覆盖上了」。

**一个对照表：多态三大「运行期入口」**——`virtual` 函数调用（vtable 分发）、`dynamic_cast`（安全下行转型）、`typeid`（动态类型查询）。三者都依赖 RTTI，都只在「含虚函数的类」上工作。设计上「优先虚函数、少用 cast、typeid 兜底」，是避免多态代码腐烂的常识。<span class="marginnote">协变返回的实现代价值得一提：编译器对「返回类型不同」的覆盖在底层要插入<strong>类型调整（adjustor thunk）</strong>——所以协变返回并不是「免费的语法糖」，但它是安全的：编译器保证返回的指针确实指向派生类子对象。日常能用就用，多态工厂里尤其顺手。</span>

**多态与虚析构的配合**：凡「通过基类指针删除对象」的代码，基类析构必须 virtual——多态容器 `vector<shared_ptr<Quote>>` 之所以安全，是因为 `Quote::~Quote()` 是虚的、且 `shared_ptr` 在析构时调用「实际类型的析构」（通过 vptr 找到派生析构）。反过来，**没有虚析构却放进多态容器，删除时只析构基类部分**——这是「继承了多态、却没继承析构纪律」的典型泄漏源。

**接口类（interface class）的设计惯例**：纯接口类的四件套——**全部成员函数为纯虚**、**析构函数为虚**（或纯虚 + 函数体）、**不携带数据成员**、**通过工厂函数创建**（返回 `shared_ptr<Interface>` 而非让客户直接 new）。这套惯例保证「接口稳定、实现可换」，是插件架构与依赖注入的基础。它也解释了 Effective C++ 第34条「接口继承 vs 实现继承」为何把纯虚函数当作「只继承接口」的标记。<span class="marginnote">为什么「接口类」偏好工厂函数？因为接口类的构造函数是 protected 的、且多态对象必须经指针创建——<strong>工厂函数把「new 哪个实现」的决定权交给一处</strong>，客户代码只依赖抽象，这是依赖倒置原则在 C++ 里的标准落地。</span>

**一个最小多态设计示例的复盘**：从「折扣报价系统」回看本章全部概念——抽象基类 `Quote` 用纯虚 `net_price` 定义接口、虚析构保证安全删除；`Bulk_quote` 覆盖 `net_price` 实现批量折扣、`override` 由编译器校验签名；多态容器装 `shared_ptr<Quote>`、`total` 遍历时按各对象动态类型分发；新增 `Discount_quote` 只需 `push_back`，`total` 零改动。这趟「从接口到实现到扩展」的完整闭环，正是多态设计在真实项目中的模样。

**补充一条「多态与容器」的边界**：`std::vector<Quote>` 能装派生对象，但装进去的是「被切片的 Quote」——多态失效、虚函数退化。要「混装多态对象」必须存指针或智能指针：`std::vector<std::shared_ptr<Quote>>`。这也是「对象值 vs 对象句柄」的根本差别：**按值容器保存的是对象本身，指针容器保存的是对多态对象的引用**。想混装就选后者，并记得让基类析构为虚。

**一个小型「多态容器」的完整示例**（把本章机制串起来）：

```cpp
#include <memory>
#include <vector>

struct Shape {                 // 抽象基类
    virtual double area() const = 0;
    virtual ~Shape() = default;
};
struct Circle : Shape {
    double r_;
    explicit Circle(double r) : r_(r) {}
    double area() const override { return 3.14159 * r_ * r_; }
};
struct Square : Shape {
    double s_;
    explicit Square(double s) : s_(s) {}
    double area() const override { return s_ * s_; }
};

double total(const std::vector<std::shared_ptr<Shape>> &shapes) {
    double sum = 0;
    for (const auto &s : shapes) sum += s->area();   // 各走各的 area
    return sum;
}
```

把 `Circle(2)` 与 `Square(3)` 塞进 `vector<shared_ptr<Shape>>`，`total` 一行不改地按动态类型分发——抽象接口、虚函数、虚析构、多态容器、覆盖，五个概念在一段代码里全部到位。<span class="marginnote">这段示例值得逐行读一遍：`= 0` 纯虚声明接口、`override` 校验覆盖、`= default` 虚析构保证安全删除、`shared_ptr` 让容器能「混装」、`total` 只依赖抽象。它就是「面向接口编程」的最小完整演示。</span>

## 5 小结

- **vtable + vptr**：含虚函数的类有一张虚函数表、每个对象有一个指向它的指针；虚调用 = 间接跳转。
- **纯虚函数** `= 0` 让类成为**抽象基类**：不能实例化、派生类必须实现全部纯虚函数。
- **面向接口编程**：调用方依赖抽象基类，新增派生类不改调用方——开闭原则的落地。
- **静态绑定编译期决定、可内联**；**动态绑定运行期决定、不可内联**；按「类型是否开放」选型。
- 多态容器必须存**指针/引用**（`shared_ptr<Base>`），不能存 `vector<Base>`。
- 继承多态是运行时分发，模板是编译期泛化——二者互补，各司其职。

在下一节，我们把「泛化」推向极致——**函数模板与类模板**：模板的定义与实例化、类型参数与非类型参数、模板与重载的协作。