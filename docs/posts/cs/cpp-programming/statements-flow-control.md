---
title: 语句与流程控制
date: 2026-08-07
---

# 语句与流程控制

<div class="epigraph">
<p>程序是给读它的人看的，只是顺带让机器执行一次。</p>
<footer>—— Harold Abelson（哈罗德 · 阿贝尔森）</footer>
</div>

<div class="article-byline">
<p>第三级 · C++ 编程 ｜ C++ Primer 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从语句开始

表达式能算值，但程序是由**语句（statement）**组成的：一条语句是「以分号结尾的一个动作或一次判断」。第5章看似琐碎，却是把前四章的知识变成「会思考的程序」的最后一环——分支、循环、跳转、异常，这四种控制结构几乎覆盖了所有命令式程序。C++ 的流程控制与 C 一脉相承，语法上和其他语言也大同小异，但有几处细节（悬垂 else、switch 的 fallthrough、范围 for 的迭代器语义）值得单独拆开讲。<span class="marginnote">「顺序、选择、循环」是结构化程序设计的三大基石，Dijkstra 1968 年那篇著名的《Go To Statement Considered Harmful》就是在为「少用 goto、多用结构化控制」背书——C++ 保留 goto 却不推荐，正是这段历史的痕迹。</span>

## 1 表达式语句、复合语句与悬垂 else

**表达式语句（expression statement）**：一个表达式后加分号，`x = y;`、`++i;`、`cout << s;` 都是。**空语句** `;` 什么都不做，偶尔用于「循环体恰好是空」的场合。**复合语句（compound statement）**（块）用花括号包裹多条语句，既当语句用、又划定作用域——块内声明的变量出块即失效。

**if 语句**按条件选择执行分支：

```cpp
if (score >= 90)
    grade = "A";
else if (score >= 80)
    grade = "B";
else
    grade = "C";
```

**悬垂 else（dangling else）**：else 总是与**最近的、尚未配对的 if** 结合。`if (a) if (b) x = 1; else x = 2;` 里的 else 配给内层 `if (b)`，不是外层——想改变配对必须用花括号。<span class="marginnote">悬垂 else 是 C 系语言最经典的坑之一，C++ Primer 第5章专门提醒：<strong>只要 if 里再嵌 if，就给内层加花括号</strong>，把「配对关系」写死，别让读者（和编译器）猜。</span>

## 2 switch 语句

**switch 语句**按整型表达式的值跳转到对应 **case 标签**：

```cpp
switch (c) {
    case 'a': case 'e': case 'i': case 'o': case 'u':
        vowelCount++;
        break;
    case ' ':
    case '\t':
    case '\n':
        whitespaceCount++;
        break;
    default:
        otherCount++;
        break;
}
```

要点有三：**case 标签必须是整型常量表达式**（字面量、constexpr 均可）；**case 之间可以叠放**共享同一段代码（如上例的元音）；**匹配后从该 case 一路向下执行，直到 break**——不写 break 就「漏下去」（fallthrough），这是 switch 与 if 的最大区别。<span class="marginnote">fallthrough 偶尔有用（多个 case 共享代码块），但绝大多数是 bug：忘了写 `break`，结果匹配 `case 'a'` 却把空格计数也加了。现代 C++ 编译器会警告「疑似 fallthrough」，可以用 `[[fallthrough]]` 属性显式声明「我就是要漏下去」。</span>`default` 分支处理「没有匹配的 case」，相当于 if 的 else。

## 3 while、for 与范围 for

**while 循环**：先判断条件、条件为真才执行体，适合「事前不知道要循环几次」：

```cpp
int sum = 0, v;
while (std::cin >> v)       // 读到 EOF 或非法输入时停止
    sum += v;
```

**for 循环**把「初始化、条件、步进」写在一行：

```cpp
for (int i = 0; i != 10; ++i)
    sum += i;
```

三部分都能省略，但要注意：**条件留空 = 恒真 = 死循环**（`for (;;)`）。

**范围 for（range for）** 是第3章的正式形态：

```cpp
for (auto &x : vec)         // 引用：可修改元素
    x *= 2;
```

范围 for 在底层会把容器展开成「begin 到 end 的迭代器循环」。因此有个隐含约束：**循环体内不能改变容器的尺寸**——`for (auto x : v) v.push_back(...)` 会让迭代器失效，是未定义行为。<span class="marginnote">范围 for 与普通 for 的另一差别是：它不暴露下标。想「知道当前是第几个」，要么手动计数、要么回到下标 for。C++ 选择用迭代器抽象下标，背后的统一视角是「算法只依赖迭代器，不依赖容器」——第10章泛型算法就是这个思路的全面展开。</span>

## 4 break、continue 与 goto

**break** 立刻跳出**最近的循环或 switch**；**continue** 跳过本次循环的剩余语句、进入下一次迭代。二者都只作用于一层，不会穿透嵌套循环。<span class="marginnote">想在嵌套循环里「一下子跳出两层」，C++ 没有 Java 那样的带标签 break。惯用解法是：把内层循环包进函数提前 `return`，或用标志变量加条件判断——这也是「能不用 goto 就不用 goto」的日常体现。</span>

**goto** 是无条件跳转，C++ 保留它却几乎不推荐用它。它的存在主要是为了与 C 兼容，以及极少数性能关键路径。**解析：** goto 能跳进块、但不能跳过「带初始化的变量声明」跳到其作用域内——会编译报错。

## 5 try/catch：异常处理语句

**异常（exception）**把「错误发生了」与「怎么处理」解耦。抛出异常用 `throw`，捕获用 `try`/`catch`：

```cpp
try {
    int n = std::stoi(input);      // 解析失败会抛异常
    use(n);
} catch (const std::invalid_argument &e) {
    std::cerr << "不是数字：" << e.what() << '\n';
} catch (const std::out_of_range &e) {
    std::cerr << "超出范围：" << e.what() << '\n';
}
```

`throw` 之后，函数调用栈一层层退栈（**栈展开，stack unwinding**），沿途的局部对象被析构，直到某个 `catch` 能匹配异常类型为止；没有任何 catch 匹配，程序调用 `std::terminate` 直接终止。<span class="marginnote">异常不是「万能的错误处理」——它在「错误频率低、跨层传递」时最好用；对高频、可预期的失败（如文件不存在）用返回值或 `std::optional` 更合适。第18章《异常、命名空间与多继承》会把这套机制讲透。</span>

## 6 核心对比表：四类循环

| 维度 | while | 基本 for | 范围 for | do-while |
| --- | --- | --- | --- | --- |
| 判断时机 | 先判断 | 先判断 | 先判断（基于迭代器） | **先执行后判断** |
| 最少执行 | 0 次 | 0 次 | 0 次 | **至少 1 次** |
| 适用场景 | 次数未知 | 次数已知、需下标 | 遍历整个容器 | 至少执行一次 |
| 能否改元素 | 下标/引用 | 下标 | 引用（需 `&`） | 下标 |
| 风险 | 忘了步进 | 边界写错 | 循环内改容器尺寸 | 条件恒真则死循环 |

**易错点：** do-while 的花括号**不能省略**——`do x++; while (cond);` 中 while 后面的分号是语句的一部分，省略花括号会把后续语句「卷」进循环体，逻辑完全走样。

## 7 小结

- **表达式语句**以分号结尾；**复合语句**用花括号划作用域。
- **悬垂 else** 配对最近 if；内层嵌套就加花括号。
- **switch** 的 case 是整型常量表达式，**不写 break 就 fallthrough**；`default` 处理无匹配。
- **while/for** 先判断、do-while 先执行；**范围 for** 遍历容器、不能在循环内改容器尺寸。
- **break** 跳出最近循环/switch，**continue** 跳本次迭代；goto 能用但别用。
- **异常**把错误抛出与处理解耦：`throw` + `try/catch` + 栈展开。

在下一节，我们开始把程序拆成可复用的部件——**函数、重载与默认实参**：参数传递的三种方式、返回类型、重载决议、inline 与函数指针。