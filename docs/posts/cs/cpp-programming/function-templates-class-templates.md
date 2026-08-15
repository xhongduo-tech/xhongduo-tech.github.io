---
title: 函数模板与类模板
date: 2026-08-07
---

# 函数模板与类模板

<div class="epigraph">
<p>模板让「一份代码、对任意类型正确」成为可能。</p>
<footer>—— Alexander Stepanov（亚历山大 · 斯捷潘诺夫）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第16章 ｜ 2026-08-07</p>
</div>

## 为什么从模板开始

第10章的泛型算法让我们尝到了「一份 `find` 服务所有容器」的甜头，但那只是**标准库**做好的模板；这章把「写模板」的能力交给你自己。**模板（template）**是 C++ 的**编译期代码生成器**：你写一个「带类型参数」的蓝图，编译器按每个实际类型**实例化（instantiate）**出对应版本。函数模板如 `std::max`、`std::sort` 的比较器；类模板如 `std::vector<int>`、`std::shared_ptr<T>`。模板是 C++ 三范式之一「泛型编程」的核心工具，也是元编程（第17、19章的 tuple、类型萃取）的地基。<span class="marginnote">模板 vs 宏：宏是「文本替换」，不知道类型、不做检查；模板是「带类型的源码生成」，编译期推导 + 类型检查。模板 vs 重载：重载为「已知的有限类型」各写一份，模板为「未知的任意类型」写一份蓝图——两者可以协作，模板失败时重载顶上。</span>

## 1 函数模板

**函数模板（function template）**：带类型参数的函数蓝图，参数写在 `template <typename T>` 里：

```cpp
template <typename T>
T max(const T &a, const T &b) {
    return a < b ? b : a;
}

int m1 = max(3, 7);            // T 推导为 int：实例化 max(const int&, const int&)
double m2 = max(3.14, 2.7);    // T 推导为 double
std::string m3 = max(std::string("a"), std::string("b"));
```

**模板实参推导（template argument deduction）**：调用 `max(3, 7)` 时编译器从实参推出 `T = int`，然后实例化一份 `max` 的 `int` 版本。**每个不同的类型各自实例化一份独立代码**——`max<int>` 与 `max<double>` 是**两个不同函数**。<span class="marginnote">「每类型一份代码」是模板的性能与代价：没有运行时抽象开销（调用方得到的是普通函数），但代码体积随实例化类型数量增长——这就是 Effective C++ 第44条「将参数无关代码从模板中剥离」要解决的「代码膨胀」问题。</span>

**易错点：** `max(3, 3.14)` 推导失败——`T` 既可能是 `int` 又可能是 `double`，模板不自动做类型转换。修法：显式指定 `max<double>(3, 3.14)`（把 3 转成 double），或让两个形参类型独立 `template <typename T, typename U> auto max(const T&, const U&)`。

## 2 非类型参数与模板参数的多样性

模板参数不只有「类型」，还有**非类型参数（nontype parameter）**——编译期常量：

```cpp
template <unsigned N>
void print_fixed(double v) {
    std::cout << std::setprecision(N) << v;   // N 是编译期常量
}
print_fixed<3>(3.14159);    // 输出 3.14

template <size_t N>
constexpr size_t array_size = N;   // 变量模板（C++14）
```

**重点：** 非类型参数必须是**编译期常量表达式**（字面量、constexpr 变量、`sizeof` 结果）。`std::array<T, N>` 与 `std::bitset<N>` 的尖括号里的数字，正是非类型参数。<span class="marginnote">非类型参数是「模板把编译期计算带进类型系统」的桥梁——`array<int, 4>` 与 `array<int, 5>` 是<strong>两个不同类型</strong>，编译器在编译期就锁定长度。这也解释了为什么 C 风格数组传参退化指针，而 `std::array` 能把长度「钉死」在类型里。</span>

## 3 类模板

**类模板（class template）**：类的蓝图，实例化时指定类型：

```cpp
template <typename T>
class Stack {
public:
    void push(const T &v) { data_.push_back(v); }
    T pop() { T v = data_.back(); data_.pop_back(); return v; }
    bool empty() const { return data_.empty(); }
private:
    std::vector<T> data_;
};

Stack<int> si;              // T = int
Stack<std::string> ss;      // T = string
```

**要点：** 类模板的**成员函数在类外定义时也要写 `template <typename T>`**；类模板**不能像函数模板那样靠实参自动推导类型**（C++17 的 CTAD 让 `Stack s{...}` 可以推导，但通常仍显式写）。<span class="marginnote">类模板与函数模板的推导规则差异：函数模板「调用即推导」，类模板「须显式指定或靠 CTAD（C++17 类模板实参推导）」。CTAD 让 `std::pair p{1, "a"}` 这种写法成为可能，但它依赖「推导指引（deduction guide）」，有些类型需要自定义指引。</span>

## 4 模板与重载的协作

函数模板和普通函数可以同名并存——**重载决议**时模板参与竞争：

```cpp
int max(int, int);                     // 普通函数
template <typename T> T max(const T&, const T&);   // 模板

max(3, 7);        // 精确匹配普通函数（非模板优先）
max(3.14, 2.7);   // 只有模板能匹配 → 用模板
```

**规则：** 参数完全相同时，**非模板函数优先于模板**；模板之间按特化程度竞争（更特化的胜出）。这是标准库大量「非模板为主 + 模板兜底」设计的基础。<span class="marginnote">重载决议对模板有一套「部分排序（partial ordering）」规则，判断哪个模板「更特化」。理解它需要实践经验——但有一个实用结论：<strong>能用非模板函数就不写模板，模板只处理「类型要泛化」的部分</strong>，可读性最好。</span>

## 5 公式解析：模板实例化的时机

模板的「两阶段编译」是初学者最难理解的点：

- **第一阶段，模板定义处**：编译器只检查**不依赖模板参数**的语法（拼写、括号、无关类型错误）；`T` 的成员函数调用此刻**不检查**。
- **第二阶段，实例化处**：用具体类型替换 `T` 后，才做**完整检查**——所有错误在这一刻暴露。

**重点推论**：模板「本体」必须在使用它的翻译单元里可见——所以函数模板通常**定义在头文件里**（而非 `.cpp`），否则链接阶段找不到实例化体。这是「模板是头文件公民」的由来，也是它与普通函数「声明进头文件、定义进源文件」惯例的根本区别。<span class="marginnote">两阶段检查解释了模板最常见的困惑：<strong>「模板定义没错、一实例化就报错」</strong>。比如 `T` 没有 `operator<`，`max` 定义时合法、`max(obj, obj)` 实例化时爆错。而「头文件里放模板定义」让每个 include 它的翻译单元都能独立实例化，也带来重复实例化的开销——编译器靠 ODR（单一定义规则）合并它们。</span>

**模板别名与类型推导的补充**：C++11 的 **`using` 模板别名**能给「模板化的类型」起短名——`template <typename T> using Vec = std::vector<T>;` 之后 `Vec<int>` 就是 `vector<int>`。别名的价值在「复杂模板类型」（如 `std::map<std::string, std::vector<T>>`）上尤其明显。而 **`auto` 占位**（C++14 起函数返回类型可写 `auto`）让模板的「返回类型」不再需要手写——`template <typename T, typename U> auto mul(T a, U b) { return a * b; }` 自动推导乘积类型。这三件套（模板、别名、auto）把「写类型」的负担降到最低。<span class="marginnote">注意 `auto` 推导会去掉引用与顶层 const（`auto x = ref` 得到的是值），想要引用就用 `auto &`。这与模板实参推导的「忽略顶层 const、保留底层 const」是同一套规则——<strong>模板推导与 auto 推导在 C++11 之后被统一了</strong>，学一个等于学两个。</span>

**模板与重载的竞争细节**：当「模板实例化」与「非模板函数」都能匹配时，规则是「**非模板优先**」；两个模板都能匹配时，「更特化的模板优先」——这一条靠**部分排序（partial ordering）**在编译期裁决。一个实用推论：标准库里 `std::max`、`std::swap` 常同时提供「非模板版本」与「模板版本」，让「内置类型走非模板、类类型走模板」各得其所。写自定义类型时，为 `swap` 提供**成员 `swap` + 非成员特化**（第4篇条款25），就能让 `std::swap` 的模板框架自动调到你的高效版本。<span class="marginnote">部分排序有一条新手常踩的坑：<strong>模板特化的「偏特化 vs 主模板」与「重载模板」是两套正交机制</strong>——重载选的是「哪个模板」，特化选的是「哪个类型」。两者叠加时编译器先重载决议、再选特化，顺序别记反。</span>

**模板与「零开销抽象」**：模板在编译期实例化、调用点内联，运行时没有任何抽象层——这正是「零开销抽象」的核心兑现：你付出的是编译时间与代码体积，得到的是「手写每一版」级别的性能。理解了这条交换，也就理解了为什么「模板 + constexpr」是现代高性能库（线性代数、图库、序列化）的标准答案。

**一条阅读提醒**：模板代码的报错信息以「长且嵌套」著称——编译器会把整条实例化链打出来。习惯它、学会从最后一行（真正的错误）往前读，是写模板的第一步。`static_assert` 与 concepts（C++20）能把「模板用错了」变成「一句人话」，是现代 C++ 改善模板体验的主力。

**一个小对照**：为什么「函数模板」能自动推导、而「类模板」通常要手写类型？因为函数有「实参」可供推导，类没有。C++17 的 CTAD 让 `std::vector v{1, 2, 3}` 也能推导——但它的推导指引（deduction guide）机制与函数模板推导并不完全相同，复杂类型上仍有边界。理解这个「推导的对称性」，是读模板代码时的背景知识——模板编程的许多「意外」，都源于「推导何时发生、何时不发生」这条边界。

**承上启下**：模板的「推导、实例化、重载、两阶段编译」是下一节「实参推导、特化与可变参数模板」的入场券——那里会把这四个机制推向更深的组合，直至模板元编程。

## 6 小结

- **函数模板** `template <typename T> T f(...)` 按实参**推导**类型、按类型**实例化**；每类型一份代码。
- 模板不自动做类型转换；**显式指定** `f<double>(...)` 可强制统一类型。
- **非类型参数**（编译期常量）让 `array<int, N>`、`bitset<N>` 把长度钉进类型。
- **类模板**须显式指定类型（或靠 C++17 CTAD）；成员函数类外定义也要 `template <...>`。
- **模板与重载**协作：非模板优先、更特化的模板优先。
- 模板**两阶段编译**、定义须进头文件——这是它与普通函数的最大组织差异。

在下一节，我们把模板的「推导」与「定制」讲深——**模板实参推导、特化与可变参数模板**：类型转换与引用折叠、显式特化与偏特化、参数包与递归实例化。