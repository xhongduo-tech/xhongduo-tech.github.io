---
title: 模板与泛型编程
date: 2026-08-07
---

# 模板与泛型编程

<div class="epigraph">
<p>编译期多态让「同样的调用，最快的分支」成为可能。</p>
<footer>—— Scott Meyers（斯科特 · 迈耶斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ Effective C++ 第7章 条款41–48 ｜ 2026-08-07</p>
</div>

## 为什么从模板进阶开始

第16章讲了模板的语法与机制，第7章（条款41–48）把模板升维成**一门独立的编程思想**：隐式接口与编译期多态（41）、typename 的双重身份（42）、模板基类名字访问（43）、剥离参数无关代码（44）、成员函数模板的兼容类型（45）、模板内非成员函数的转换（46）、traits 类（47）、模板元编程（48）。这一章是 Effective C++ 里最「硬核」的一章——它让读者从「用模板」走向「设计模板」，也为现代 C++ 的 constexpr、概念（concepts）做了思想铺垫。<span class="marginnote">模板编程与面向对象是两种「多态」：<strong>运行期多态（虚函数）看类型、编译期多态（模板）看接口</strong>。模板的世界里没有「vtable」，一切在编译期解析完毕——这是它能做到「零运行时开销」的原因，也是它调试困难的来源。</span>

## 1 条款41：了解隐式接口和编译期多态

**重点：** 面向对象谈「显式接口」——类定义里白纸黑字的函数签名；模板谈「**隐式接口（implicit interface）**」——模板函数体里对 `T` 做的**每一处操作**就是要求（`+`、`<`、`.size()`……），T 必须「碰巧都支持」。

```cpp
template <typename T>
T max(const T &a, const T &b) {
    return a < b ? b : a;     // T 必须支持 < 与拷贝 —— 这就是隐式接口
}
```

**编译期多态（compile-time polymorphism）**：虚函数在运行期查 vtable 决定调用；模板在编译期按类型**实例化**、直接内联解析——所以模板调用「更快、更特化」，代价是**每个类型一份代码**（代码膨胀）与**编译期报错**（出错信息可能又长又玄）。

**辨析：** 隐式接口有个容易被忽略的维度——**返回类型**：`a < b` 的结果不要求是 `bool`，只要是能转 `bool` 的类型即可。这给「表达式模板」「延迟求值」留了门，也让「接口是什么」从「签名清单」变成了「表达式约束」——C++20 的 **concepts** 正是把这种隐式接口显式化的机制。<span class="marginnote">「隐式接口」与「鸭子类型」（duck typing）神似但不同：Python 在运行期发现「对象不支持某操作」，C++ 模板在<strong>编译期</strong>就发现——所以 C++ 的模板错误「提前、但报错信息痛苦」。C++20 concepts 出现后，「要求 T 支持 < 」可以直接写成约束，报错也变友好。这是本条款的现代续篇。</span>

## 2 条款42：了解 typename 的双重意义

**易错点：** `typename` 在模板里有两个身份——**声明类型参数**（`template <typename T>`）与**声明嵌套依赖类型**。后者是新手最容易漏的：

```cpp
template <typename C>
void printSecond(const C &c) {
    C::const_iterator it = c.begin();   // 编译错误：C::const_iterator 是依赖类型
}
template <typename C>
void printSecond(const C &c) {
    typename C::const_iterator it = c.begin();   // 加 typename 才行
}
```

**规则：** 在模板中引用「**依赖模板参数的嵌套类型**」时，必须写 `typename`——因为编译器**无法预知** `C::const_iterator` 是个类型还是一个静态数据成员（不同 C 实例化结果可能不同）。**例外**：基类声明里不必写 `typename`（`class Derived : public Base<T>::Type` 中 `typename` 被省略）。

**重点：** 这条规则是「模板是两阶段编译」的直接后果——第一阶段（模板定义处）不知道 `C` 是什么，任何「依赖 C 的类型名」都必须显式标注，否则编译器把 `C::const_iterator` 当成**值**解析，报出令人困惑的错误。<span class="marginnote">这条是模板新手的第一道坎：<strong>「为什么我照着书抄，编译就报 'expected ;'」</strong>——多半是漏了嵌套类型的 `typename`。一个记忆锚点：`typename` 出现在模板里「跟作用域运算符 `::` 后面」时，基本就是要它。</span>

## 3 条款43：学会处理模板化基类内的名称

**易错点：** 模板派生类里调用基类成员，可能**找不到**：

```cpp
class MsgInfo { public: int maxSize() const { return 1024; } };
template <typename T>
class Base {
public:
    void send(T msg) { /* ... */ }
};
template <typename T>
class Derived : public Base<T> {
public:
    void sendMsg(T msg) {
        send(msg);       // 编译错误：send 不可见！
    }
};
```

**为什么**：`Base<T>` 依赖模板参数 `T`——编译器在定义 `Derived` 时**不知道 `Base<T>` 会实例化成什么**（偏特化可能让 `send` 不存在），所以不会去基类里找名字。**对策**三种：用 `this->send(msg)`、用 `using Base<T>::send`、或全限定 `Base<T>::send(msg)`——任选其一，明确告诉编译器「这个名字来自基类」。<span class="marginnote">第三条的原因很务实：<strong>「拒绝到依赖基类里查名」是 C++ 为了保住「模板偏特化」的自由</strong>——若编译器主动到 `Base<T>` 里找，就无法支持「某些 T 的 `Base<T>` 偏特化没有 send」的情形。代价是「模板继承要显式 this->」这条别扭却必要的纪律。</span>

## 4 条款44：将与参数无关的代码抽离 templates

**易错点：** 模板「每类型一份代码」，但有些代码**与类型无关**——把它留在模板里，每个实例化都复制一份，白白膨胀：

```cpp
template <typename T, std::size_t N>
class SquareMatrix {
public:
    void invert() { /* 与 N、T 都无关的求逆算法本体 */ }
};
SquareMatrix<double, 5> m1;   // 两份几乎相同的 invert
SquareMatrix<double, 10> m2;  // 代码膨胀翻倍
```

**对策**：把「与类型无关」的核心逻辑抽成**非模板函数**，模板只做「类型适配」：

```cpp
void invertImpl(double *data, std::size_t size);   // 非模板：一份实现
template <typename T, std::size_t N>
class SquareMatrix {
public:
    void invert() { invertImpl(data_, N); }        // 模板薄壳：转发
};
```

**重点：** 代码膨胀的两种来源——**类型无关**（抽离成普通函数）与**参数无关**（N 不同、T 相同）。抽离时注意「避免过度抽离」：把一切都抽成通用版本，可能引入额外间接调用、反而更慢。**权衡**：代码体积 vs 调用开销。<span class="marginnote">膨胀的极致是「每个 `vector<int>`、`vector<double>` 都有一份自己的 `push_back`」——但现代编译器的链接器（如 LTO）能合并「代码完全相同」的实例化。工程上真正的对策仍是「抽离」，因为 LTO 的合并有前提、且编译时间与内存仍按实例化数增长。</span>

## 5 条款45、46、47、48：成员函数模板、转换、traits 与元编程

**条款45——运用成员函数模板接受「所有兼容类型」**：`shared_ptr<Base>` 应能从 `shared_ptr<Derived>` 构造——但模板的「构造」不是隐式转换链能表达的，需要**成员函数模板**：

```cpp
template <class T>
class shared_ptr {
public:
    template <class U>
    shared_ptr(const shared_ptr<U> &other);   // 接受所有「可转换」的 U
};
```

**辨析：** 成员函数模板**不改变**拷贝控制规则——它不能替代拷贝构造/拷贝赋值；想让「兼容类型」也可移动，还得给移动版本同样加模板。

**条款46——需要类型转换时，在类模板内定义非成员函数**：模板的**隐式转换**不参与实参推导——`Rational<int> r = 2 * r` 想让 `2` 转成 `Rational<int>`，非成员 `operator*` 必须在模板内定义（否则推导不出 `T`）：

```cpp
template <typename T>
class Rational {
public:
    Rational(T n, T d) : n_(n), d_(d) {}
    T n_, d_;
};
template <typename T>
const Rational<T> operator*(const Rational<T> &lhs, const Rational<T> &rhs) { ... }
// 但 2 * r 仍不推导 → 需把 operator* 放进类内作为 friend
```

**重点：** 模板实参推导不做隐式类型转换——这是「模板内非成员函数 + friend」模式的根本原因（该模式叫「隐藏友元」）。

**条款47——请使用 traits classes 表现类型信息**：**traits（特征）**是编译期查询「类型有什么能力」的机制：`std::iterator_traits`、`std::is_integral`、`std::remove_reference`。实现套路 =「主模板给默认 + 偏特化给特例」：

```cpp
template <typename T> struct is_pointer { static const bool value = false; };
template <typename T> struct is_pointer<T*> { static const bool value = true; };
```

**条款48——认识 template metaprogramming（TMP）**：TMP 在**编译期**完成「类型计算」——第17章我们见过的编译期阶乘就是它的 Hello World。TMP 的价值：把「运行期才能做的决策」提前到编译期、把「容易错的手写」交给编译器自动化（`std::tuple`、`std::function`、类型萃取全靠它）。**代价**：编译变慢、代码难读、报错折磨。<span class="marginnote">TMP 在「必须极致性能 + 类型要适配」的场合（图库、矩阵库、序列化）无可替代——它把「为每种类型手写一份」变成「让编译器算出一份」。C++14 后 constexpr 函数让大部分「算值型 TMP」变得直白；而「类型变换型 TMP」（`tuple_element`、`conditional`）至今仍靠类模板偏特化。</span>

**核心对比表：两种多态的对照**（条款41）——

| 维度 | 运行期多态（虚函数） | 编译期多态（模板） |
| --- | --- | --- |
| 接口 | 显式（类定义） | **隐式**（对 T 的操作） |
| 决定时机 | 运行期（vtable） | 编译期（实例化） |
| 开销 | 间接跳转 | 零运行时开销 |
| 类型检查 | 运行期绑定 | 编译期报错 |
| 典型 | 插件、接口类 | 算法、容器 |

## 6 小结

- **条款41**：模板是**隐式接口 + 编译期多态**；「T 支持哪些操作」就是接口，C++20 concepts 将其显式化。
- **条款42**：嵌套依赖类型前要写 **`typename`**；否则编译器当「值」解析。
- **条款43**：模板派生类访问基类名要 `this->` 或 `using` 或全限定——依赖基类不查名。
- **条款44**：把**与类型无关**的核心逻辑抽成非模板函数，对抗代码膨胀。
- **条款45、46**：成员函数模板接兼容类型；模板内非成员函数（friend）补上隐式转换。
- **条款47、48**：**traits = 主模板 + 偏特化**做编译期类型查询；TMP 在编译期算类型/值，代价是编译时间。

在下一节，我们收尾 Effective C++——**定制 new/delete 与杂项**：new-handler 的行为、何时该替换 new/delete、placement new/delete、编译器警告与标准库/Boost。