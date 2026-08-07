---
title: 手写递归下降语法分析器
date: 2026-08-07
---

# 手写递归下降语法分析器

<div class="epigraph">
<p>文法即代码：每个非终结符一个函数，EBNF 的每一行翻译成一个控制流。</p>
<footer>—— 仿自尼基劳斯 · 沃斯（Niklaus Wirth）对递归下降的推崇</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · 玩具编译器实战 ｜ 2026-08-07</p>
</div>

## 为什么从手写递归下降语法分析器开始

词法层把字符流切成记号流，语法层要把记号流**按文法组装成树**。ToyLang 的语法分析器用**手写递归下降**——每个非终结符一个函数，EBNF 的元符号直接翻译成控制流。它不需要分析表（那是 LL(1) 表驱动），因为它用「代码」而非「表」来表达决策——这正是第七十八节那张「EBNF → 控制流」翻译表的实践。<span class="marginnote">「递归下降 = 用代码当分析表」：LL(1) 表驱动分析器把「选哪条产生式」写进表，递归下降把同样的决策写进 if/while——两者做同样的决定（按当前记号选候选），实现方式不同。手写递归下降的优势：可读、可调、错误信息人性化，是 Clang、rustc 的真实选择。</span>

## 1 从 EBNF 到函数

ToyLang 的每个非终结符对应一个函数（伪码）：

```cpp
Program  → { Function }              parseProgram() { while (check(KEYWORD, "fn")) parseFunction(); }
Function → "fn" IDENT "(" [params] ")" Block
Block    → "{" { Statement } "}"
Statement→ let_stmt | if_stmt | while_stmt | return_stmt | expr_stmt
```

**元符号翻译**：

- `{ X }`（重复）→ `while (lookahead 属于 First(X)) parseX();`
- `[ X ]`（可选）→ `if (lookahead 属于 First(X)) parseX();`
- `A B C`（序列）→ `parseA(); parseB(); parseC();`
- 终结符 → `match(TokenType)`（吃当前记号，不匹配则报错）。

`match` 是递归下降的原子操作：

```cpp
Token match(TokenType expected) {
    if (lookahead.type != expected) error("期望 " + name(expected) + "，得到 " + lookahead);
    Token t = lookahead;  advance();  return t;
}
```

**重点是**：EBNF 的每行，机械地变成函数的一小段——这是「文法即代码」的字面实现。<span class="marginnote">「match 是递归下降的原子动作」：每个终结符的匹配、每次 lookahead 的检查，都由 match 完成。它同时是错误报告的集中点——「期望 X 得到 Y」的所有语法错误都从 match 抛出。写递归下降，match 写得顺，一半就顺了。</span>

## 2 前瞻与匹配

递归下降是**预测分析**：每个决策点看「当前记号」（前瞻一个 token）决定走哪条候选。

```
parseStatement():
    if check(KEYWORD, "let")    → parseLet()
    elif check(KEYWORD, "if")   → parseIf()
    elif check(KEYWORD, "while")→ parseWhile()
    elif check(KEYWORD, "return")→ parseReturn()
    else                        → parseExprStmt()   # 表达式语句
```

**关键**：每个候选的「首记号」必须不同（LL(1) 条件的直觉）——`let`/`if`/`while`/`return` 首记号各不相同，所以一眼能选。若两个候选首记号相同，就需要「提取左公因子」改写文法（第九节的改写理论）。

**`check(type)`**：不消费、只问「当前记号是不是这个类型」——决策用 check，消费用 match。

**示例**：解析 `let x = expr;`：

```
parseLet():
    match(KEYWORD, "let")
    Token name = match(IDENT)
    match(OPERATOR, "=")
    ExprAST* e = parseExpression()
    match(OPERATOR, ";")
    return new LetStmt(name, e)
```

**辨析｜易错点：** `check` 与 `match` 的区别是「看 vs 吃」——`check` 只问、不消费；`match` 问+消费。决策点用 `check`，确认后前进用 `match`。混用会导致「多吃一个记号」或「决策后忘前进」两种 bug。<span class="marginnote">「check 与 match 的分工」是递归下降的纪律：check 是「看门」，match 是「进门」。若决策时用 match 吃了记号，后续的分支就没记号可看了；若确认了却忘了 match，分析器就卡在原地。每个函数写完后检查「每个分支前进没有」——与词法器的铁律同源。</span>

## 3 表达式分析：优先级爬升

表达式是递归下降里最精巧的部分。ToyLang 的分层文法（expression → comparison → additive → term → factor）可以直接用**逐层函数**实现：

```cpp
ExprAST* parseExpression()  { return parseComparison(); }   // 含 && ||

ExprAST* parseComparison() {
    ExprAST* lhs = parseAdditive();
    if (checkOp("<") || checkOp(">") || checkOp("==")) {
        Token op = advance();
        ExprAST* rhs = parseAdditive();
        return new BinaryExpr(op, lhs, rhs);
    }
    return lhs;
}

ExprAST* parseAdditive() {
    ExprAST* lhs = parseTerm();
    while (checkOp("+") || checkOp("-")) {   // 左结合循环
        Token op = advance();
        ExprAST* rhs = parseTerm();
        lhs = new BinaryExpr(op, lhs, rhs);
    }
    return lhs;
}
```

**左结合**用「循环」实现：`a + b + c` 循环三次，生成 `(a+b)+c`。**优先级**用「分层调用」实现：`a + b * c` 中 `parseAdditive` 调 `parseTerm`，`*` 先被消化——更深、先算。

**factor** 处理括号、数字、标识符、调用：

```
parseFactor():
    if check("(")  → 吃 ( → parseExpression → 吃 )
    if check(NUMBER) → 数字字面量
    if check(IDENT)  → 标识符，若下一个是 ( 则是函数调用
```

**重点是**：表达式的「优先级」=「函数调用的层级」——越深层的函数，处理的运算符优先级越高。<span class="marginnote">「优先级 = 嵌套深度」是表达式分析的精髓：`*` 的解析函数（parseTerm）比 `+` 的（parseAdditive）深一层，所以 `a+b*c` 里 `*` 先被构造进树、先算。这套「分层函数」与 LL(1) 表驱动是同一文法的两种执行——第九节「分层强制优先级」的理论，在这里就是函数的嵌套调用。</span>

## 4 公式解析：决策的正确性条件

递归下降「看一眼就决定」正确，需要 LL(1) 条件（第十二节）：

$$\forall A \to \alpha \mid \beta: \quad \text{FIRST}(\alpha) \cap \text{FIRST}(\beta) = \emptyset$$

$$\text{且若 } \beta \Rightarrow^* \varepsilon: \quad \text{FIRST}(\alpha) \cap \text{FOLLOW}(A) = \emptyset$$

- **第一步，候选可分**：`let`/`if`/`while` 首记号不同——决策点能唯一选择。
- **第二步，ε 的兜底**：若某候选可空（如语句列表的结束），它的 FOLLOW（`}`）不能与别的候选冲突。
- **第三步，违反的后果**：条件不满足 → 决策点「看着当前记号无法确定」→ 递归下降只能猜或报错。

**「先查 FIRST 再写递归下降」是工程纪律**——写代码前先验一遍文法的 LL(1) 性质，能免掉大量「跑起来才发现选错」的调试。

## 5 错误处理：让分析器爬起来

递归下降的错误恢复（第十节的恐慌模式）：

- **match 失败**：报「期望 X 得到 Y」（带行号列号）。
- **同步（synchronization）**：报错后，跳到「安全记号」再继续——ToyLang 用「跳到下一个 `;` 或 `}`」作为同步点。
- **设计**：错误恢复的目标是「一次运行多报几个错」，不是「修复错误」。

```
parseStatement():
    try:
        ...
    catch SyntaxError:
        跳过直到 ';' 或 '}'
        继续
```

**重点是**：错误恢复是「工程韧性」——分析器被坏输入打乱后，能爬起来继续找错，比「一个错就停」对开发者友好得多。<span class="marginnote">「同步点选在 `;`/`}` 是教科书经典」：语句以 `;` 结尾、块以 `}` 结尾，所以「跳到下一个 `;` 或 `}`」总能回到「下一个语句的起点」。Clang 的「recovery」机制比这精细得多，但思想一致——<strong>报错后要能回到可继续的位置</strong>。</span>

## 6 小结

- **递归下降 = 每个非终结符一个函数**；EBNF 的 `{ }`/`[ ]`/序列/终结符分别翻译成循环/if/调用/match。
- **check（看）与 match（吃）分工**：决策用 check，消费用 match——混用是高频 bug 源。
- 表达式分析：**优先级 = 函数嵌套深度**，左结合 = 循环——`a+b*c` 的树构造靠分层调用。
- 正确性前提是 **LL(1) 条件**：候选 FIRST 不相交、ε 候选对 FOLLOW 避让——先验文法再写代码。
- 错误恢复：**match 失败报错 + 跳到同步点继续**——一次运行多报几个错。

在下一节，我们把「记号流」变成「树」：**构建抽象语法树（AST）与符号表**。
