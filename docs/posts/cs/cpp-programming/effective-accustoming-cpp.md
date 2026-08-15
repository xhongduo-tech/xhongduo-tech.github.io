---
title: 让自己习惯 C++：const、enum、inline 与对象初始化
date: 2026-08-07
---

# 让自己习惯 C++：const、enum、inline 与对象初始化

<div class="epigraph">
<p>C++ 语言的一个目标，就是「比 C 更安全、比 Java 更接近机器」。</p>
<footer>—— Scott Meyers（斯科特 · 迈耶斯）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ Effective C++ 第1章 条款1–4 ｜ 2026-08-07</p>
</div>

## 为什么从这里开始

前四篇以 C++ Primer 为主线，把语言机制完整走了一遍；从这一篇起，我们换一本书，把视角从「语言是什么」切换到「**怎么用才对**」——Scott Meyers 的《Effective C++》用 55 条实践经验总结出 C++ 社区几十年的血泪教训。第1章「让自己习惯 C++」的四条是全书的地基：**把 C++ 当成四种语言的联盟**（条款1）、**用 const/enum/inline 替代 #define**（条款2）、**处处用 const**（条款3）、**保证对象在使用前已初始化**（条款4）。它们篇幅最短，却塑造了一个 C++ 程序员的基本审美。<span class="marginnote">《Effective C++》的每一条都遵循同一模式：<strong>先给「为什么」（底层机制或失败模式），再给「怎么做」（具体写法）</strong>。它的对象是「懂语法、缺经验」的程序员——正好接在 C++ Primer 之后读。全书条款编号 1–55，本专题按章展开。</span>

## 1 条款1：把 C++ 视为一个语言联邦

**重点：** 不要把 C++ 当成一种语言，而要当成**四种子语言的联邦**，每种有各自的规则与价值观：

- **C 部分**：过程式、指针、内置类型——没有类、没有模板、没有异常。
- **Object-Oriented C++**：类、封装、继承、多态、虚函数、动态绑定。
- **Template C++**：泛型编程与模板元编程，规则自成体系。
- **STL**：容器、迭代器、算法、函数对象——一套「模板 C++」之上的惯例集合。

**为什么这条是「最高条款」**：绝大多数「这个写法是对是错」的争论，根源是「你在用哪种语言的规则」。`*p = 0` 在 C 部分天经地义；「按值传递 vs 按引用传递」在 C 部分和 STL 部分答案相反；`const` 在 C++ 部分和模板部分的语义深浅也不同。<span class="marginnote">联邦视角解释了经典争论：<strong>「C++ 里该用指针还是引用？」「该按值还是按引用传参？」「该用继承还是模板？」</strong>——答案几乎都是「看你站在联邦的哪个省」。Meyers 建议：跨子语言边界时，把「换了一条法律」当成常态，而不是迷信某一条全局教条。</span>

## 2 条款2：尽量以 const、enum、inline 替换 #define

**#define 的麻烦**：宏在**预处理期**做纯文本替换，没有任何类型检查，也不会进入符号表。

```cpp
#define ASPECT_RATIO 1.653
```

`ASPECT_RATIO` 这个名字在编译器的符号表里**不存在**——出错信息里只有 `1.653` 这个数字，你根本不知道它从哪来。替换方案：

```cpp
const double AspectRatio = 1.653;          // 常量：有类型、进符号表

class GamePlayer {
    static const int NumTurns = 5;         // 类内常量（整型可类内初始化）
    int scores[NumTurns];                  // 编译期常量，可用于数组大小
};
```

**enum hack**：不想让别人拿地址或想确保「编译期常量」时，用 `enum` 伪装常量：

```cpp
class GamePlayer {
    enum { NumTurns = 5 };   // enum hack：NumTurns 是编译期常量、取不到地址
    int scores[NumTurns];
};
```

**宏函数**同样危险——`#define CALL(x) f(x)` 每处调用都展开一份代码，参数还可能有副作用；**inline 函数**才是正解：

```cpp
template <typename T>
inline T callWithMax(const T &a, const T &b) {   // 有类型检查的「宏函数」
    return a > b ? a : b;
}
```

**辨析：** 宏仅剩的合理用途是「日志/断言」里需要 `__FILE__`、`__LINE__` 这类预处理符号；其余场景，常量用 `const`/`enum`、函数用 `inline`。<span class="marginnote">这条的底层道理：<strong>把「编译期可验证」的名字交给编译器，别让预处理期悄悄替换</strong>。`const` 常量有作用域、有类型、进调试符号；宏是全局文本污染。C++11 之后 `constexpr` 更进一步——编译期就求值，比 `const` 更能进常量上下文。</span>

## 3 条款3：尽可能使用 const

**`const`** 的含义是「承诺不修改」。它可以修饰**顶层**（变量本身）、**指针所指**、**成员函数**：

```cpp
char greeting[] = "Hello";
char *p = greeting;            // 指针可变，所指可变
const char *cp = greeting;     // 所指不可变（顶层 const 于所指）
char *const pc = greeting;     // 指针不可变
const char *const cpc = greeting;  // 两者都不可变
```

**const 成员函数**：声明「这个成员函数不会修改对象」。它在两方面有用：**接口层面**——告诉调用方「读操作」；**性能与正确性层面**——`const` 对象只能调用 `const` 版本，且 `const` 版本可以安全并发。

**重点：** const 有**传播力**——`const` 指针传给函数，函数内部再调用成员函数时，只能调用该成员的 `const` 版本。于是「返回内部数据的指针/引用」若不加 const，会让调用方绕过封装改到内部状态——这正是 Effective C++ 第28条「避免返回对象内部的句柄」的前奏。<span class="marginnote">const 的取舍哲学：<strong>「能 const 就 const」</strong>（const-correctness）。它能让你在编译期抓住「误修改内部状态」的错误，而不是运行期调试。唯一要警惕的是 const_cast 去掉 const 后修改真 const 对象——那是未定义行为，const 承诺不该被轻易打破。</span>

**const 与 mutable**：`mutable` 成员即使在 `const` 成员函数里也可修改——用于「缓存、计数器」这类「逻辑上不影响状态、物理上要改」的成员。这是 const 体系里少数被允许的例外。

## 4 条款4：确定对象被使用前已先被初始化

**易错点：** C++ 对「不初始化」的惩罚很隐蔽——**内置类型局部变量默认初始化 = 垃圾值**，而「恰好碰上有用的旧值」纯属运气。Meyers 的铁律：**永远在使用对象前初始化它**。对内置类型，手写初值：

```cpp
int x = 0;
const char *text = "A C-style string";
double d = 0.0;
```

**构造函数的初始化陷阱**：在**函数体内赋值**不是初始化——成员先默认构造、再被赋值，多一步且对 const/引用成员直接编译失败。**初始化列表**才是初始化：

```cpp
class PhoneNumber { /* ... */ };
class ABEntry {
public:
    ABEntry(const std::string &name, const std::string &address)
        : theName(name), theAddress(address),  // 初始化列表：真正的初始化
          theNum(0), theId(0) { }              // 顺序与声明顺序一致
private:
    std::string theName;
    std::string theAddress;
    int theNum;
    int theId;
};
```

**重点：** 初始化列表里成员的初始化**顺序由声明顺序决定，不是列表书写顺序**——所以「列表里按声明顺序写」能避免「后声明先初始化」的隐藏依赖。**静态对象**（跨翻译单元）的初始化顺序未定义——这正是「把非局部静态对象换成局部 static 对象（函数内首次调用时构造）」这条实践（Meyers 单例模式）的动机。<span class="marginnote">「跨翻译单元的静态对象初始化顺序未定义」是 C++ 的著名暗礁：a.cpp 的全局对象构造时若用到 b.cpp 的全局对象，而 b 的构造还没跑，就是未定义行为。解法是 Meyers 的单例惯用法：<strong>把静态对象藏进函数里，首次调用时才构造</strong>——构造函数内部先构造依赖、再构造自己，顺序自然正确。</span>

**核心对比表：四种「常量/函数」的替代品**——条款2 的完整对照：

| 方案 | 类型检查 | 进符号表 | 编译期常量 | 可否取地址 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `#define ASPECT 1.653` | 无 | 否 | 是 | 否 | 弃用（仅保留日志/断言） |
| `const double Aspect` | 有 | 是 | 视语境 | 是 | 一般常量 |
| `enum { N = 5 }` | 有 | 是 | 是 | **否** | 类内编译期常量 |
| `inline 函数` | 有 | 是 | 可 constexpr | 是 | 替代宏函数 |

## 5 小结

- **条款1**：把 C++ 当「C / 面向对象 C++ / 模板 C++ / STL」四种语言组成的联邦，规则分省而立。
- **条款2**：用 `const`/`enum` 替 `#define` 常量、用 `inline` 函数替宏函数——把名字交给编译器。
- **条款3**：**处处 const**：指针分「指针 const / 所指 const」，const 成员函数声明只读，`mutable` 是唯一例外。
- **条款4**：**使用前必初始化**：内置类型手写初值，成员用初始化列表，且按声明顺序写。
- 这一章的审美主线：**用编译期可验证的机制（const、初始化列表）替代运行期靠运气的行为**。

在下一节，我们聚焦对象的「生、老、病、死」——**构造、析构与赋值运算**：编译器自动生成什么、多态基类析构为何要虚、自赋值怎么防、异常怎么不泄漏。