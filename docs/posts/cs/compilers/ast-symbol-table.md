---
title: 构建抽象语法树（AST）与符号表
date: 2026-08-07
---

# 构建抽象语法树（AST）与符号表

<div class="epigraph">
<p>语法分析器的产物不是「认出了程序」，而是一棵树——程序的结构被钉在树上，供所有后续阶段攀爬。</p>
<footer>—— 仿自阿尔弗雷德 · 艾侯（Alfred V. Aho）对语法树的描述</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · 玩具编译器实战 ｜ 2026-08-07</p>
</div>

## 为什么从 AST 与符号表开始

递归下降分析器「认出了」记号流的结构，但它不保留结构——它要**返回一棵树**。**抽象语法树（AST）** 是程序的「结构档案」：去掉了括号、分隔符等纯语法杂物，保留语义要点。同时，分析过程中要登记每个名字——**符号表（symbol table）** 记录「每个标识符是什么（变量/函数）、在哪、什么类型」。AST 供语义分析与代码生成遍历，符号表供名字解析与查错，两者是编译前端的两大资产。<span class="marginnote">「AST 是语义的地图、符号表是名字的户口本」：语义分析在 AST 上走（检查类型、作用域），代码生成也在 AST 上走（算值、发指令）；而每次遇到一个名字，都要查符号表「它是谁」。第八节「语法分析树」与第六节「符号表管理」在此合体——ToyLang 的 AST 是语法树的瘦身版，符号表是编译前端的记忆库。</span>

## 1 AST 的设计

ToyLang 的 AST 用「基类 + 子类」表达「结点的类型」（C++ 风格）：

```cpp
enum NodeKind { kNumber, kVar, kBinary, kAssign, kIf, kWhile, kCall };

class Node {
  int line, col;                    // 位置（报错用）
  virtual ~Node() {}
};

class Stmt : public Node {};        // 语句基类
class Expr : public Node {};        // 表达式基类

class NumberExpr : public Expr { int value; };
class VarExpr     : public Expr { std::string name; };
class BinaryExpr  : public Expr { char op; Expr *lhs, *rhs; };

class AssignStmt : public Stmt { VarExpr *lhs; Expr *rhs; };
class IfStmt     : public Stmt { Expr *cond; Stmt *thenStmt, *elseStmt; };
class WhileStmt  : public Stmt { Expr *cond; Stmt *body; };
class CallStmt   : public Stmt { std::string callee; std::vector<Expr*> args; };
```

**设计要点**：

**语句与表达式分开**：语句有副作用、表达式是值——语义检查与代码生成对两者处理不同。
**递归结构**：`WhileStmt`/`IfStmt` 的子树是 `Stmt`——树的递归遍历全靠这个。
**每个结点都带位置**（行号列号）——错误信息定位。

**AST 是「瘦身后的语法树」**：`(` 与 `)` 的括号、`;`、`{`、`}` 这些纯语法记号不进 AST——它们在分析时被消费，只留结构。<span class="marginnote">「AST 去掉了什么，比保留了什么都重要」：括号（`(` 与 `)` 同一棵树）、分隔符（`;`、`,`）、块括号（`{`、`}`）全被「吃掉」——因为它们不携带语义，只服务语法。AST 只留「算子的树形结构」，这让后续的语义检查与代码生成不必处理语法噪音。</span>

## 2 从分析器到 AST

递归下降函数返回 AST 结点——分析器一边「认结构」一边「建树」：

```cpp
Expr* parseExpr() {
    Expr* left = parseTerm();                    // 先解析一层 term
    while (lookahead == '+' || lookahead == '-') {
        char op = lookahead; advance();          // 吃掉运算符
        Expr* right = parseTerm();
        left = new BinaryExpr(op, left, right);  // 建一个二元结点
    }
    return left;
}
```

**要点**：分析器「边分析边建树」——每个产生式对应一个 AST 结点。分析完，整棵 AST 就在手里，可传给语义分析/代码生成。

**重点是**：AST 是「分析器的输出、语义分析器的输入」——它是前后阶段的**契约结构**。<span class="marginnote">「边分析边建树」让 AST 构造零额外开销：不需要「分析完再走一遍树来建树」，分析过程本身就是建树过程。这对应第四篇「一趟分析、一趟翻译」的思想——语法结构（分析）与语义结构（AST）同步生成。</span>

## 3 符号表的设计

**符号表（symbol table）** 记录名字信息。ToyLang 的符号表设计：

```cpp
struct Symbol {
    std::string name;
    enum { kVar, kFunc } kind;   // 是变量还是函数
    Type type;                   // 类型（TinyLang 只有 int）
    int offset;                  // 栈帧偏移（代码生成用）
    int nParams;                 // 若为函数：参数个数
};

class SymbolTable {
    std::vector<std::unordered_map<std::string, Symbol>> scopes;
public:
    void pushScope();            // 进入块
    void popScope();             // 离开块
    void define(const Symbol& s);
    Symbol* lookup(const std::string& name);
};
```

**作用域链（scope chain）**：ToyLang 是块级作用域，符号表用「嵌套作用域」实现：

```cpp
void pushScope() { scopes.emplace_back(); }        // 压入一层空作用域

void popScope()  { scopes.pop_back(); }            // 弹出最内层作用域

Symbol* lookup(const std::string& name) {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return &found->second;  // 从内向外，命中即返回
    }
    return nullptr;                                 // 未声明
}
```

**lookup 的规则**：从最内层作用域向外查——内层声明的名字遮蔽外层同名（作用域遮蔽）。

**两个阶段的用途**：

**语义分析**：查「变量是否已声明」「函数是否存在」「参数个数对不对」。
**代码生成**：查「变量的存储位置」（栈帧偏移或寄存器槽位）。<span class="marginnote">「符号表是名字的户口本」：编译器在声明 `变量` 时在符号表登记，之后每次用 `名字` 都查表找到它的记录。作用域栈让「内层遮蔽外层」变成「从栈顶往下找」——与第六节「活动记录、作用域」的理论呼应。玩具编译器的符号表可以做得很朴素（一个 `vector` 作用域栈），但机制与真实编译器（GCC 的作用域栈、Clang 的 DeclContext）一致。</span>

## 4 公式解析：作用域查找

设符号表的作用域栈为 $S_0, S_1, \ldots, S_k$（$S_k$ 最内层），查找名字 $n$：

$$\text{lookup}(n) = \text{第一个} S_i \ (i = k, k-1, \ldots, 0) \text{ 使 } n \in S_i$$

$$\text{未找到} \Rightarrow \text{「未声明」错误}$$

- **第一步，从内向外**：从最内层 $S_k$ 开始逐层查——内层优先，天然实现「遮蔽」。
- **第二步，命中即返回**：找到第一层含 $n$ 的作用域，返回其符号——**遮蔽**的语义：「最内层的那个」。
- **第三步，落空报错**：查遍所有作用域没有——「未声明的标识符」错误，带位置。

**「从内向外、内层优先」是作用域查找的全部规则**——遮蔽、全局可见性、块级隔离都由它决定。

## 5 AST 遍历：语义与代码生成的公共底座

AST 建好、符号表填好，后续阶段靠**遍历 AST** 工作：

- **语义分析**：递归遍历每个结点——检查变量已声明、类型正确、函数参数匹配（下一节）。
- **代码生成**：递归遍历——对每个表达式发指令、对每个语句发控制流（第六节）。

**两种遍历模式**：

- **后序遍历**（先子后父）：表达式求值——先算子树再组合（综合属性）。
- **先序遍历**（先父后子）：作用域管理——进块 pushScope、出块 popScope（继承属性）。

**设计提示**：AST 的结点类可以加 `codegen`/`typecheck` 方法，让遍历变成「每个结点自己会干活」——Kaleidoscope 的 OOP 风格（第七十六节）。

**重点是**：AST 是「结构契约」，符号表是「名字档案」——两者配合，语义分析与代码生成才能站在可靠的地基上。<span class="marginnote">「AST 遍历的两种模式 = 综合属性与继承属性的执行顺序」：后序（先子后父）对应 S-属性（算值），先序（先父后子）对应 L-属性的继承传递（作用域）。第四篇的属性理论，在这里就是「遍历顺序的选择」——理解了遍历，就理解了属性求值。</span>

## 6 小结

- **AST** 是语法树的「瘦身版」：去掉括号/分隔符，保留语义结构；语句与表达式分开建模。
- 分析器**边分析边建树**：每个产生式对应一个 AST 结点——分析结束，AST 在手。
- **符号表**记录名字信息，用**作用域栈**实现块级作用域与遮蔽。
- **lookup 从内向外**：内层优先、遮蔽语义、落空报错——名字解析的全部规则。
- AST 供语义分析/代码生成**遍历**：后序（算值）与先序（作用域）对应综合/继承属性。

在下一节，我们让语义「活」起来：**语义分析与简单类型检查**。
