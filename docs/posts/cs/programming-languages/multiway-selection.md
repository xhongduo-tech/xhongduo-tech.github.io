---
title: 多重选择结构：if-elif 与 switch/match
date: 2026-08-07
---

# 多重选择结构：if-elif 与 switch/match

<div class="epigraph">
<p>当岔路口超过两个，<code>if-elif</code>` 链是笨重的闸门，<code>switch</code>` 是精准的转盘。
<footer>—— 佚名（PLT 格言）</footer>
</p></div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第8章 §8.6 ｜ 2026-08-07</p>
</div>

## 为什么从多重选择开始

双路选择靠 `if/else`；当分支超过两个，两种策略各据一方：**if-elif 链**（对多个条件逐一测试）与 **switch/case**（按一个表达式的值跳转）。这一节比较这两条路，重点看现代 `switch` 的演化：C 的 `switch` 从「跳转表 + fall-through」走到 Rust 的 `match`——后者把「模式匹配 + 穷尽检查 + 表达式语义」融为一体，成为最强大的选择结构。理解这段演化，就理解了「选择」从语句走向表达式、从跳转走向模式匹配的当代趋势。<span class="marginnote">if-elif 与 switch 的本质区别：if-elif 测试的是<strong>布尔条件序列</strong>（每个分支独立判断），switch 测试的是<strong>一个值等于哪个字面量</strong>（一次取值、多次比较）。前者适合「范围、复合条件」，后者适合「离散值匹配」——选错结构是可读性的隐性损耗。</span>

## 1 if-elif 链：布尔条件序列

**if-elif 链（if-else-if chain）**：依次测试多个条件，命中第一个真条件执行对应分支，全部为假走 else（可选）。

```c
if (score >= 90) {
    grade = 'A';
} else if (score >= 80) {
    grade = 'B';
} else if (score >= 70) {
    grade = 'C';
} else {
    grade = 'D';
}
```

**优点**：每个分支是完整布尔条件——适合范围判断、复合条件、不同变量的比较。

**缺点**：分支多时线性扫描（$O(n)$）；条件互相依赖（先测 90 才轮到 80），顺序错了逻辑就错。

**辨析｜易错点：** elif 链的**顺序敏感**：更严格的条件（如 `score >= 90`）若写在更宽松的条件（如 `score >= 80`）前面，后者的分支永远走不到。多重选择的分支「有重叠」时，顺序决定结果——这是 if-elif 链最隐蔽的错误来源。

## 2 switch：按值的跳转

**switch 语句（switch statement）**：对**一个表达式**求值，按其值匹配 `case` 标签，跳转到对应分支。

```c
switch (day) {
    case 1: printf("Monday");    break;
    case 2: printf("Tuesday");   break;
    case 7: printf("Sunday");    break;
    default: printf("Invalid");
}
```

C 系 switch 的设计特点：

**离散匹配**：`case` 只放**常量**，不做范围或复合条件。
**fall-through（穿透）**：分支末尾不写 `break` 会继续执行下一 case——这是 C 的著名特性，也是 bug 温床。<span class="marginnote">fall-through 的原意是「多个值共享一段代码」（如 `case 1:`、`case 2:` 连续标注共享同一分支），但漏写 `break` 会让执行「穿透」到下一分支。现代语言要么禁止穿透（Rust），要么强制显式（`break`/`fallthrough`）。C 的选择是「默认穿透、显式 break」——历史包袱。</span>
**default**：都不匹配时的兜底分支。

## 3 现代 match：模式匹配的进化

Rust、Swift、Kotlin、现代 C++ 的 `switch` 已经从「值匹配」进化为**模式匹配（pattern matching）**：

```rust
match code {
    200 => println!("OK"),
    404 => println!("Not Found"),
    other => println!("Status: {other}"),
}
```

现代 match 的三大升级：

**表达式语义**：`match` 产生值，每个分支都是表达式（Rust/Swift）。
**穷尽性检查**：编译器强制所有可能值都有分支，缺了不编译——「漏分支」从运行时错误变编译期错误。<span class="marginnote">穷尽检查是 match 对 switch 的革命性改进：C 的 switch 漏了某个值会静默走 default（或什么都不做），Rust 的 match 漏掉变体直接编译失败。「不可达状态不可表示」与「可达状态必被处理」双管齐下，是类型系统与模式匹配结合的威力。</span>
- **模式绑定**：分支可以**解构**数据——`Some(x)` 把值从容器里拆出来绑定到 `x`（判别联合一节已见）。

## 4 公式解析：switch 与 if-elif 的等价

从语义上讲，switch 是 if-elif 链的「特例」。对取值表达式 $E$ 与常量 $c_i$：

$$
\text{switch}(E) \;\equiv\; \begin{cases}
\text{if } E = c_1 \text{ then } S_1 \\
\text{else if } E = c_2 \text{ then } S_2 \\
\vdots \\
\text{else if } E = c_k \text{ then } S_k \\
\text{else } S_{\text{default}}
\end{cases}
$$

三步拆解：

- **第一步，翻译**：每个 `case` 翻译成「若 $E = c_i$ 则执行 $S_i$」。switch 是「等值测试的 elif 链」——这是它表达力的边界：**只能做相等比较**。
- **第二步，看差异**：switch 的优势不在表达力（与 elif 等价），而在**实现**——编译器可建**跳转表（jump table）**或二分查找，$O(1)$ 或 $O(\log k)$ 而非线性扫描。值分布稠密时跳转表极快。
- **第三步，看演化**：现代 match 超越了「等值测试」——它能匹配**模式**（结构、类型、守卫条件），表达力严格超过 switch 与 elif 链。**「等价」是 C switch 的边界，「超越」是 match 的方向**。

**辨析｜易错点：** `case` 标签在 C 里必须是**编译期常量表达式**——`case n`（变量）非法；且各 `case` 值必须唯一。现代 match 无此限制（可匹配动态值、区间、守卫）。「switch 只能匹配常量」是新手常撞的墙。

## 5 从 switch 到 match：语言设计趋势

| 维度 | C switch | Rust match |
| --- | --- | --- |
| 语义 | 语句 | 表达式 |
| 匹配 | 常量等值 | 任意模式 |
| 穷尽性 | 不检查 | 编译器强制 |
| 穿透 | 默认穿透 | 每臂自带 break |
| 解构 | 不支持 | 内建 |
| 守卫 | 不支持 | `if` 守卫 |

<span class="marginnote">现代语言的「选择」正在全面向模式匹配靠拢：Kotlin 的 <code>when</code>`、Swift 的 <code>switch</code>`、Java 的「switch 表达式 + 模式匹配」、Python 3.10 的 <code>match</code>。核心动力是同一句话：<strong>把「选择」从跳转变成「按形状取值」</strong>——更安全、更可读、更声明式。</span>


## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| if-elif 链（if-else-if chain） | if-elif 链（if-else-if chain）：依次测试多个条件，命中第一个真条件执行对应分支，全部为假走 else（可选）。 |
| 优点 | 优点：每个分支是完整布尔条件——适合范围判断、复合条件、不同变量的比较。 |
| 缺点 | 缺点：分支多时线性扫描（O(n)）；条件互相依赖（先测 90 才轮到 80），顺序错了逻辑就错。 |
| switch 语句（switch statement） | switch 语句（switch statement）：对一个表达式求值，按其值匹配 case 标签，跳转到对应分支。 |
| 一个表达式 | switch 语句（switch statement）：对一个表达式求值，按其值匹配 case 标签，跳转到对应分支。 |
| 离散匹配 | 离散匹配：case 只放常量，不做范围或复合条件。 |
| 常量 | 离散匹配：case 只放常量，不做范围或复合条件。 |
| default | default: name = "Unknown"; break; |
| 模式匹配（pattern matching） | Rust、Swift、Kotlin、现代 C++ 的 switch 已经从「值匹配」进化为模式匹配（pattern matching）： |
| 表达式语义 | 表达式语义：match 产生值，每个分支都是表达式（Rust/Swift）。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **if-elif 链**测布尔条件序列，适合范围与复合条件；顺序敏感是陷阱。
- **switch** 按一个值匹配常量，跳转表实现高效；C 的 **fall-through** 是 bug 温床。
- 现代 **match** 升级为模式匹配：表达式语义 + 穷尽性检查 + 模式绑定/解构。
- switch 与 elif 链语义等价（等值测试），match 超越二者（任意模式）；选择结构正全面走向模式匹配。

在下一节，我们将看「重复」的形态——**迭代语句：计数循环、逻辑循环与迭代器**。
