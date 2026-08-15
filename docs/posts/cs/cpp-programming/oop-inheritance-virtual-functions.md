---
title: 面向对象程序设计：继承与虚函数
date: 2026-08-07
---

# 面向对象程序设计：继承与虚函数

<div class="epigraph">
<p>面向对象编程的威力来自「继承」与「多态」的联姻。</p>
<footer>—— Bjarne Stroustrup（比雅尼 · 斯特劳斯特鲁普）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第15章 ｜ 2026-08-07</p>
</div>

## 为什么从继承开始

第7章的类是「数据 + 行为」的封装，第15章把类与类的关系也结构化：**继承（inheritance）**让新类（派生类）复用并扩展旧类（基类），而 **虚函数（virtual function）** 让「同一句调用、按对象实际类型执行不同行为」成为可能——这就是面向对象编程的核心。C++ 的继承有三处不同于 Java/C# 的细节需要特别留意：**虚函数默认不虚**、**析构函数在基类中必须虚**、**静态类型与动态类型分离**。这一章先把「继承关系怎么搭、virtual 怎么声明、构造/析构怎么传递」讲清楚，下一节再展开动态绑定与抽象基类的多态设计。<span class="marginnote">Java 里方法默认 virtual、C++ 里默认不 virtual——这个差异让 C++ 新人极易踩坑。C++ 的设计理由是对性能的执念：<strong>虚函数调用要走 vtable 间接跳转</strong>，默认不虚=默认零开销；想要多态，显式声明。</span>

## 1 基类与派生类

**基类（base class）**与**派生类（derived class）**通过冒号声明继承关系：

```cpp
class Quote {                          // 基类：普通报价
public:
    Quote() = default;
    Quote(const std::string &book, double price)
        : bookNo(book), price(price) {}
    std::string isbn() const { return bookNo; }
    virtual double net_price(std::size_t n) const {
        return n * price;              // 基类版本
    }
    virtual ~Quote() = default;        // 析构必须 virtual
private:
    std::string bookNo;
protected:
    double price = 0.0;
};

class Bulk_quote : public Quote {       // 派生类：批量折扣
public:
    double net_price(std::size_t n) const override {
        return n * price * (n >= min_qty ? 0.8 : 1.0);  // 覆盖基类
    }
private:
    std::size_t min_qty = 0;
};
```

**访问级别三兄弟**在这里齐登场：基类的 `private` 成员派生类**不可访问**；`protected` 成员派生类**可访问**、外部不可访问；`public` 成员全部可访问。<span class="marginnote">`protected` 的存在就是为了继承：它把「自己的私有」开放给「子孙」，但不开放给外人。Effective C++ 有一条提醒：`protected` 成员一旦进入，接口就「半开放」，改动会波及所有派生类——能用 private + 虚函数就别急着 protected。</span>

## 2 virtual 与 override

**虚函数（virtual function）**：基类中声明为 `virtual` 的成员函数，派生类可以**覆盖（override）**它。调用虚函数时，实际执行哪一版**由对象的动态类型决定**（下一节细讲）。关键语法：

- 派生类覆盖基类虚函数时，签名（形参、返回类型）**必须完全一致**，否则是「隐藏」而非「覆盖」。
- **`override` 关键字**（C++11）：显式标注「我就是要覆盖基类的虚函数」——签名不匹配时编译期报错，而不是悄悄变成一个隐藏函数。
- 基类的虚函数若在派生类里被覆盖，仍可通过 `Quote::net_price` **显式调用基类版本**（例如实现 `operator<<` 打印基类部分时）。

**易错点：** 虚函数**默认参数值**不被动态绑定继承——调用时用的默认参数由**静态类型**决定（Effective C++ 第37条「绝不重新定义继承而来的默认参数值」）。<span class="marginnote">为什么默认参数跟静态类型走、函数体跟动态类型走？因为默认参数在<strong>编译期</strong>就要填进调用点，而函数体要运行时才查 vtable——两者时机不同，只能各管各的。这个分裂极易制造「调了虚函数、却用了错误的默认值」的诡异 bug。</span>

## 3 静态类型与动态类型

**静态类型（static type）**：变量声明时的类型，编译期确定。**动态类型（dynamic type）**：变量实际指向的对象类型，运行期才确定。二者只在「引用/指针指向派生类对象」时才会分离：

```cpp
Quote *p = new Bulk_quote;    // 静态类型 Quote*，动态类型 Bulk_quote
p->net_price(10);             // 运行期调用 Bulk_quote::net_price
```

**重点：** 只有通过**引用或指针**调用虚函数才发生动态绑定。直接按值 `Quote q = bq;` 会发生**对象切片（object slicing）**——派生类部分被切掉，只剩基类子对象，动态类型退化为 Quote。<span class="marginnote">对象切片是 C++ 特有的陷阱：把派生类对象按值赋给基类变量，只是「拷贝基类那部分」——虚函数调用的将是基类版本。所以多态必须靠指针/引用，<strong>按值传参 = 放弃多态</strong>。这也是为什么「多态容器」要存 `shared_ptr<Base>` 而非 `vector<Base>`。</span>

## 4 构造与析构在继承中的秩序

**构造顺序**：派生类构造函数先构造**基类子对象**（自动调用基类构造函数，或显式在初始化列表里指定），再构造自己的成员。**析构顺序**相反：先析构派生类自己的成员，再析构基类子对象。

```cpp
Bulk_quote(const std::string &book, double price, std::size_t qty)
    : Quote(book, price), min_qty(qty) {}   // 先 Quote 后自己
```

**析构的两条铁律：**

- **基类析构函数必须声明为 `virtual`**：`delete` 一个「指向派生类的基类指针」时，只有虚析构才能保证**派生类析构先执行**。不虚，则只析构基类部分、派生类资源泄漏。
- **构造与析构期间调用虚函数不会多态**：基类构造期间，对象还是「基类」——调用虚函数走的是基类版本（Effective C++ 第9条「绝不在构造/析构期间调用虚函数」）。<span class="marginnote">为什么构造期间虚调用是静态的？因为基类构造时派生类成员还没构造好——若此刻调用派生类的虚函数，它可能访问尚未初始化的成员，属未定义行为。标准干脆规定：构造/析构期间的动态类型「就是当前正在构造的那一层」。</span>

## 5 核心对比表：继承中的默认行为

| 行为 | 默认 | 规则 |
| --- | --- | --- |
| 派生类能否访问基类 `private` | 否 | 只能访问 `public`/`protected` |
| 基类析构是否虚 | **不虚** | 多态删除必须手动 `virtual ~Base()` |
| 基类成员函数是否虚 | 不虚 | 想多态就显式 `virtual` |
| 派生类覆盖虚函数 | 需同名同签名 | 用 `override` 让编译器校验 |
| 通过值传递派生对象 | 切片 | 多态一律走指针/引用 |
| 构造/析构期调用虚函数 | 静态绑定 | 走当前层版本 |

**辨析：** `=default` 的虚析构（`virtual ~Quote() = default;`）是推荐写法——既要虚析构，又不需要自定义清理逻辑时，用默认实现最干净。<span class="marginnote">第4篇 Effective C++ 第7条「多态基类声明虚析构函数」与这里完全一致：只要类里有一个虚函数，就把析构也声明为 virtual——否则删除派生对象时析构只走基类层，派生类资源悄无声息地泄漏。</span>

**final 与纯虚的补充**：`final` 关键字（C++11）能同时「锁定类」与「锁定虚函数」——`class Bulk_quote final : public Quote` 表示不再允许派生；`double net_price(...) const override final` 表示「这是最后一个覆盖」，编译器对任何再次覆盖报错。`final` 与 `override` 是同一枚硬币的两面：`override` 声明「我确实覆盖了」，`final` 声明「不允许再被覆盖」，二者都能把「本想在运行期发现的错」提前到编译期。<span class="marginnote">另外，纯虚函数也可以有<strong>函数体</strong>——`virtual void f() = 0 { ... }` 合法，派生类必须覆盖、但覆盖里可以显式调用 `Base::f()` 复用那段缺省实现。这与第4篇 Effective C++ 第34条「接口继承 vs 实现继承」的 NVI 思路相通。</span>

**继承里的名称查找**：C++ 成员查找按「名字」进行——派生类成员函数调用一个名字时，先在派生类作用域里找，找不到才去基类。这意味着**派生类里声明一个与基类同名（哪怕签名不同）的函数，会把基类的同名一族全部藏起来**——这就是第4篇 Effective C++ 第33条「避免遮蔽继承而来的名字」的机制根源。想「既保留基类版本、又加自己的」，用 `using Base::f;` 把基类版本引入派生类作用域。这条规则不区分虚函数与非虚函数，是纯作用域行为。<span class="marginnote">名称查找与虚绑定的次序值得记住：<strong>先按名字定位「哪一层的作用域」、再在该层决定「虚不虚、绑哪版」</strong>。如果派生类里根本没声明那个名字，虚函数链依旧有效；一旦声明了，即使签名相同，基类版本也要靠 `using` 或 `Base::` 全限定才能显式调。这是「藏名」与「多态」两个机制在实践中的交互。</span>

**继承是把双刃剑的再强调**：本章给了机制（继承、虚函数、构造析构秩序），但「该不该继承」的问题要到第4篇 Effective C++ 才给出完整答案。记住一句：**继承传递的是「是」的关系，不是「用」的关系**——想复用实现，优先组合；想表达「派生类是基类的一种」，才轮到 public 继承与虚函数登场。

**一张「本章地图」**：构造顺序（基类→派生）→ 析构顺序（派生→基类）→ 虚析构的必要性 → 静态/动态类型的分野 → 虚函数绑定规则 → 名称隐藏 → override/final 校验。顺着这条链走一遍，继承这一章就算真正贯通了。

**一个「本章全概念」速查**：`class Derived : public Base`（继承）→ `virtual` 成员（多态接口）→ `override`（校验覆盖）→ `virtual ~Base() = default`（安全删除）→ `protected`（派生可访问）→ `Base *p = &d`（静态/动态类型分离）。六个记号串联起来，就是 C++ 继承的完整骨架，也是下一节「动态绑定与抽象基类」的直接入口。

## 6 小结

- **继承**让派生类复用/扩展基类；访问级别分 public/protected/private，`protected` 专为继承开放。
- **虚函数**默认不虚，用 `virtual` 声明、用 `override` 标注覆盖；签名不符时 `override` 编译期报错。
- **静态类型与动态类型**可分离；只有指针/引用调用虚函数才动态绑定，按值传递会**切片**。
- **构造**先基类后自身，**析构**先自身后基类；基类析构**必须虚**，多态删除才安全。
- **构造/析构期间调用虚函数走当前层**，不动态绑定。
- 继承 + 虚函数只是多态的一半，另一半是动态绑定与抽象基类——下一节见。

在下一节，我们把多态设计补全——**动态绑定、抽象基类与多态设计**：虚函数的查找机制、纯虚函数与抽象基类、面向接口的编程设计。