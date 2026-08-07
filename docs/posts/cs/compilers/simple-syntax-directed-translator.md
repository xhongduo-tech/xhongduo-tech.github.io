---
title: 一个简单的语法制导翻译器实战
date: 2026-08-07
---

# 一个简单的语法制导翻译器实战

<div class="epigraph">
<p>听过千百遍，不如亲手写一遍。</p>
<footer>—— 仿自中华传统治学箴言</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）§2.8、§5.5 ｜ 2026-08-07</p>
</div>

## 为什么从语法制导翻译器实战开始

理论滚到此处已经堆了很多层：文法、推导、FIRST/FOLLOW、LL/LR、SDD、SDT。这些概念单独看都清楚，可它们怎么**咬合成一台能跑的翻译器**？教科书龙书的经典答案是第二章末尾那台「中缀转后缀」的小翻译器——几十行代码，把词法、语法、翻译一气呵成。

**实战的价值在于把抽象对齐到代码**：文法的一行变成函数的一个分支，属性的一条规则变成一次返回值或传参。本节我们亲手造一台「中缀表达式 → 后缀表达式」的语法制导翻译器，并把它扩展成能求值的计算器。<span class="marginnote">「能跑的最小编译器」是理解全部编译理论的最短路径：它麻雀虽小，五脏俱全——词法读入、递归下降、SDT 动作、后缀输出一应俱全。读者若想验证自己真懂了，最好的测试就是把这台翻译器独立写出来。</span>

## 1 目标与文法

**目标**：输入 `9-5+2`，输出 `95-2+`（后缀表示），且按左结合正确计算（若扩展为求值）。后缀输出的关键：**运算符延后到两个操作数之后输出**。

文法（已去左递归、可 LL 分析）：

$$\begin{aligned} \textbf{expr} &\to \textbf{term}\ \textbf{rest} \\ \textbf{rest} &\to +\ \textbf{term}\ \{\text{print } '+'\}\ \textbf{rest} \mid -\ \textbf{term}\ \{\text{print } '-'\}\ \textbf{rest} \mid \varepsilon \\ \textbf{term} &\to \textbf{digit}\ \{\text{print } \textbf{digit}.lexval\} \end{aligned}$$

SDL 设计：操作数（digit）在读到时就打印，运算符在「右侧操作数已分析完」后打印——这正是后缀语义：操作数先出，运算符后出。<span class="marginnote">`rest` 里动作放 `term` 之后：先打印右侧操作数，再打印运算符——于是 `9-5` 输出 `95-`。动作位置选择在这里直接决定输出正确性，是「用时已求值」判据的活例子。</span>

## 2 词法：极简记号流

翻译器只需两种记号：数字（digit）与运算符。用一个「前看一个字符」的词法循环即可：

```
next_token():
    跳过空白
    if 字符是数字: 记下值; 返回 DIGIT
    if 字符是 + - ( ) ×: 返回该运算符
    if 文件结束: 返回 EOF
    else: 词法错误
```

真正的词法分析器（flex 版）把「数字」定义为 `[0-9]+` 的正则——但原理无差别：**把字符流切成记号流**。这里为聚焦 SDT，用最简实现。<span class="marginnote">这台翻译器的词法虽然只有几行，却已经具备「记号、模式、词素」的全部要素：`[0-9]+` 是模式，`9` 是词素，DIGIT 是记号。前两篇的理论在此落地。</span>

## 3 递归下降 + SDT 的实现

把文法翻译成函数，属性走「返回/传参」：

```c
void expr()  { term(); rest(); }
void rest() {
    if (lookahead == '+') {
        match('+'); term();
        printf("+");
        rest();
    } else if (lookahead == '-') {
        match('-'); term();
        printf("-");
        rest();
    }
    /* else: ε，什么都不做 */
}
void term() {
    if (lookahead == DIGIT) {
        printf("%d", val);
        match(DIGIT);
    } else error("期待数字");
}
```

主循环调用 `expr()`。对 `9-5+2`，输出 `95-2+`。<span class="marginnote">注意 `printf("+")` 的位置：它在 `term()`（右侧操作数）之后、递归调用 `rest()` 之前。这个「先右操作数、后运算符」的顺序，就是把中缀换成后缀的全部魔法——SDT 的动作位置即翻译语义。</span>

## 4 公式解析：后缀输出的时序论证

为什么这样放动作，输出就是后缀？看产生式：

$$\textbf{rest} \to +\ \textbf{term}\ \{\text{print } '+'\}\ \textbf{rest}$$

- **第一步，DFS 顺序**：递归下降对 `+ term {print} rest` 的执行顺序是：匹配 `+` → 分析 `term`（打印右侧操作数）→ 打印 `+` → 继续 `rest`。
- **第二步，序言式论证**：假设 `term` 已按后缀打印出右操作数的后缀（归纳假设），那么此刻输出「左前缀 + 右操作数后缀 + 运算符」——恰好是整棵 `+` 子树的后缀。
- **第三步，归纳基底**：`term → digit {print digit}` 单个数字打印自身，显然正确。由归纳，任意表达式输出后缀。

**这其实就是对「动作位置正确性」的一次形式化验证**——归纳地证明「动作放这里 = 后缀语义」。

## 5 扩展：从打印到求值

把「打印后缀」升级为「计算结果」，只需让 SDD 携带**值属性**，而非只有输出副作用：

$$\begin{aligned} \textbf{expr} &\to \textbf{term}\ \textbf{rest} & \textbf{rest}.\text{val} &= \textbf{term}.\text{val} \\ \textbf{rest} &\to +\ \textbf{term}\ \textbf{rest}_1 & \textbf{rest}.\text{val} &= \textbf{rest}_1.\text{val} + \textbf{term}.\text{val} \\ \textbf{rest} &\to \varepsilon & \textbf{rest}.\text{val} &= 0 \end{aligned}$$

函数变成「返回 int 的综合属性」：`rest(左值)` 接收左侧已算值，右结合地累加。这就是「继承属性作参数、综合属性作返回值」的直接应用——左侧值作为继承属性传入，结果作为综合属性返回。<span class="marginnote">从「打印」到「求值」的升级只动了属性与动作，没动分析框架——这验证了 SDT 的设计哲学：<strong>翻译逻辑与语法结构解耦</strong>。想要生成三地址码，同样只需换动作。</span>

## 6 小结

- 一台完整的最小翻译器 = **词法（切记号）+ 递归下降（结构）+ SDT 动作（翻译）**三件套。
- 中缀转后缀的关键：**运算符动作放在右操作数之后**，用「先右后运算符」的顺序重排输出。
- 动作位置正确性可**归纳证明**：基底（单个数字）成立、组合步骤成立，则整体成立。
- 从打印到求值：只改属性与动作，不动分析框架——SDT 的翻译逻辑与语法结构**解耦**。
- 这台几十行的翻译器是「能跑的最小编译器」，是理解全部编译理论的最短路径。

在下一节，我们进入第五篇：**中间代码生成**——从「知道句子怎么拆」到「生成机器能进一步加工的代码」。
