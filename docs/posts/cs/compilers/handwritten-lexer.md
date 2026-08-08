---
title: 手写词法分析器：从字符流到记号流
date: 2026-08-07
---

# 手写词法分析器：从字符流到记号流

<div class="epigraph">
<p>词法分析是编译器的门卫：把字符的洪流，归类成一个个清晰的记号。</p>
<footer>—— 仿自阿尔弗雷德 · 艾侯（Alfred V. Aho）对词法分析器的描述</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · 玩具编译器实战 ｜ 2026-08-07</p>
</div>

## 为什么从手写词法分析器开始

ToyLang 编译器动工的第一步是**词法分析器**：把源代码的字符流切成记号流。上一节我们已在设计中定义了所有记号（NUM、IDENT、KEYWORD、OP、运算符、括号），现在把字符流变记号流。手写而非用 flex，是为了看清「词法分析器到底在做什么」——记号的识别、最长匹配、空白与注释的跳过，全是第二篇理论的活体演示。<span class="marginnote">「手写词法 vs flex」是教材路线之争：用 flex 是「写规则、工具生成」，手写是「写代码、亲历规则」。玩具编译器建议<strong>手写</strong>——词法分析器只有两三百行，手写一遍，第二篇的「记号、模式、词素」「最长匹配」「输入缓冲」全部落地；之后再学 flex，你才知道它替你做了什么。</span>

## 1 记号的设计

**记号（token）** 是词法分析器的输出单元。ToyLang 的记号定义：

```cpp
enum TokenType { NUM, IDENT, KEYWORD, OP, EOF };

struct Token {
    TokenType type;      // 记号类型
    std::string value;   // 词素（原文）
    int line, col;       // 位置：行号、列号
};
```

| 类型 | 例子 | 值 |
| --- | --- | --- |
| NUM | 42 | 42 |
| IDENT | fib | "fib" |
| KEYWORD | if、while、return、let、print、fn | 关键字名 |
| OP | + - * / = == != < <= > >= && \|\| ! ( ) | 运算符文本 |
| EOF | 文件结束 | - |

**位置**（行号 + 列号）记录在记号里——后续错误信息「第几行第几列出错」全靠它。

**一个设计**：关键字与标识符怎么区分？词法层把标识符读出来后，查关键字表——命中即 KEYWORD，否则 IDENT。这是第二篇「关键字规则写在标识符前面」的等价做法。<span class="marginnote">「先读标识符、再查关键字表」是手写词法区分关键字的标准做法：if 被读成 IDENT，查表发现它是关键字，改判 KEYWORD。它等价于 flex 里「关键字规则写在标识符规则前面」——两种实现，同一语义：关键字优先。</span>

## 2 词法分析器的结构

手写词法分析器的骨架（C++/伪码）：

```cpp
class Lexer {
public:
    Token nextToken();          // 返回下一个记号
private:
    char peek() const;          // 偷看当前字符，不前进
    char advance();             // 读走当前字符并前进
    void skipWhitespace();      // 跳过空白与注释
    Token readNumber();         // 读数字
    Token readIdent();          // 读标识符/关键字
    Token readOperator();       // 读运算符
};
```

三个核心方法：skipWhitespace（忽略）、readNumber（读数字）、readIdent（读标识符/关键字）、readOperator（读运算符）。<span class="marginnote">「词法分析器 = 一个 peek/advance + 按首字符分流」——词法分析器的结构极简：一个当前位置、一对字符读写、一个「按当前字符类型分流」的调度。第二篇的「记号、模式、词素」在这里一一对应：digit+ 是数字的模式，readNumber 读出的 42 是词素，NUM 是记号。</span>

## 3 识别规则与最长匹配

**数字** digit+：

```cpp
Token Lexer::readNumber() {
    std::string text;
    while (isdigit(peek())) {
        text += advance();
    }
    return Token{NUM, text, line, col};
}
```

**标识符/关键字** letter(letter|digit)*：

```cpp
Token Lexer::readIdent() {
    std::string text;
    while (isalnum(peek()) || peek() == '_') {
        text += advance();
    }
    if (keywords.count(text)) return Token{KEYWORD, text, line, col};
    return Token{IDENT, text, line, col};
}
```

**运算符** 运算符表——这里要处理**多字符运算符**与**最长匹配**：

```cpp
Token Lexer::readOperator() {
    char c = advance();
    // 多字符运算符：偷看下一个字符做最长匹配
    if (c == '=' && peek() == '=') { advance(); return Token{OP, "==", line, col}; }
    if (c == '!' && peek() == '=') { advance(); return Token{OP, "!=", line, col}; }
    if (c == '<' && peek() == '=') { advance(); return Token{OP, "<=", line, col}; }
    if (c == '>' && peek() == '=') { advance(); return Token{OP, ">=", line, col}; }
    if (c == '&' && peek() == '&') { advance(); return Token{OP, "&&", line, col}; }
    if (c == '|' && peek() == '|') { advance(); return Token{OP, "||", line, col}; }
    return Token{OP, std::string(1, c), line, col};
}
```

**最长匹配**：读到 = 时，不能立刻返回 =——要**偷看下一个字符**，若是 = 就组成 ==。这是第二篇「能读多长读多长」的具体实现。

**空白与注释**：

```cpp
void Lexer::skipWhitespace() {
    for (;;) {
        while (isspace(peek())) advance();              // 跳过空白
        if (peek() == '/' && peekAhead(1) == '/') {     // 跳过 // 行注释
            while (peek() != '\n' && peek() != EOF) advance();
            continue;
        }
        break;
    }
}
```

注释被「吃掉」——它们不产生记号。<span class="marginnote">「最长匹配的手写实现 = 偷看下一个字符」：==、!=、<= 这些多字符运算符，词法层必须在读到首字符后「多看一眼」决定是单字符还是双字符。这就是第二篇「最长匹配」原则的落地——== 绝不能读成 =、= 两个记号。多字符运算符少时，逐个 if 偷看即可；多了可用「运算符前缀表」统一处理。</span>

## 4 公式解析：数字与标识符的正则

词法规则与第二篇的正则一一对应：

$$\text{digit} \to [0-9], \qquad \text{num} \to \text{digit}^+, \qquad \text{letter} \to [a-zA-Z\_]$$

$$\text{ident} \to \text{letter}\ (\text{letter} \mid \text{digit})^*$$

- **第一步，数字**：digit+——一个或多个数字；手写即「while isdigit: 读」。
- **第二步，标识符**：letter(letter|digit)*——首字母/下划线，后续字母/数字/下划线；手写即「首字符判定 + 循环读」。
- **第三步，实现 = 正则的翻译**：每一行正则规则，手写成一个循环——digit+ 是 while 循环，letter(letter|digit)* 是 while 循环。正则描述「长什么样」，代码实现「怎么读」。

**「手写词法的代码 = 正则规则的直接翻译」**——第二篇的正则理论，在这里变成几行循环。

## 5 错误处理与健壮性

手写词法要处理「坏输入」：

- **未知字符**：如 @、#——报「未知字符」，带行号列号。
- **未闭合的运算符**：& 后没有另一个 &——报错（或把 & 当单字符，视设计）。
- **数字后的非法字符**：12abc 怎么办？两种策略：报「数字后跟非法字符」或把 12 和 abc 分两个记号。玩具语言选**报错**更严格。

**错误恢复**：词法错误后，跳过该字符继续——让语法层能一次报告更多错误。

**健壮性**：词法器绝不能「卡死」——每个分支要么前进、要么报错，保证词法器总是有进展。<span class="marginnote">「词法器必须总是前进或报错」是一条工程铁律：若某分支既不消费字符也不报错，词法器会无限循环。手写词法最容易的 bug 就是「忘记前进」——每写一个分支都检查「这个分支前进没有」。玩具编译器的调试，一半时间花在这种「一步没走」的 bug 上。</span>

## 6 小结

- **记号** = (类型, 值, 位置)；ToyLang 的记号族：NUM、IDENT、KEYWORD、OP、EOF。
- 词法器结构：**peek/advance + 按首字符分流**——数字、标识符、运算符三个识别器。
- 关键字 = 先读成 IDENT、再查关键字表；等价于 flex 的「关键字规则靠前」。
- **最长匹配**靠「偷看下一个字符」实现：==、!= 不拆成两个单字符。
- 正则规则直接翻译成 while 循环；词法器必须「每次前进或报错」，绝不卡死。

在下一节，我们实现第二阶段：**手写递归下降语法分析器**。
