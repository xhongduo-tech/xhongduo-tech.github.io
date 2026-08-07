---
title: 基于 LLVM 实现一门语言的前端：Kaleidoscope 教程解读
date: 2026-08-07
---

# 基于 LLVM 实现一门语言的前端：Kaleidoscope 教程解读

<div class="epigraph">
<p>两千行代码，从零实现一门能跑出机器码的编程语言——这就是 Kaleidoscope 的承诺。</p>
<footer>—— 仿自克里斯 · 拉特纳（Chris Lattner）的《Kaleidoscope》教程前言</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · LLVM 专题 ｜ 2026-08-07</p>
</div>

## 为什么从 Kaleidoscope 教程解读开始

前面的课程把编译原理拆成了几十个章节，而 **Kaleidoscope**（LLVM 官方的「我的第一个语言前端」教程）把它们**重新拼成一台能跑的编译器**——用不到两千行 C++，从零实现一门完整语言：词法、语法、AST、语义、LLVM IR 生成、优化、甚至 JIT。它是第七篇「一个简单的语法制导翻译器」的工业级放大版：语言更完整、后端是真实的 LLVM。<span class="marginnote">「Kaleidoscope 是编译原理的毕业设计」：教程分 10 章，每章加一个特性——词法 → 语法 → AST → IR 生成 → 优化 → JIT → 控制流 → 变量 → 用户运算符 → 复合类型。跟着写完它，等于把本专题第三到第八篇的所有概念亲手实现了一遍。它也是「新语言 + LLVM」路线的标准起手式。</span>

## 1 Kaleidoscope 语言

Kaleidoscope 是一门极简的**过程式语言**：

```
# 计算斐波那契
def fib(x)
  if x < 3 then
    1
  else
    fib(x-1) + fib(x-2);

fib(20)
```

特性：

- **函数**：`def name(args) expr;`——函数体是一个表达式。
- **表达式**：数字、变量、二元运算（`+ - * / <`）、函数调用、`if/then/else`、`for/in`。
- **顶层表达式**：直接写 `fib(20)` 就求值并打印（配合 JIT 即 REPL）。

**极简的意义**：语言小到「半天能实现」，又完整到「覆盖编译器前端的所有核心」。教程用「加法」起步，每章加一个特性——每个特性都是编译原理的一个知识点。<span class="marginnote">「把语言设计到极小但完整」是 Kaleidoscope 的教学智慧：`def fib(x) if x<3 then 1 else fib(x-1)+fib(x-2)` 这一行就包含了函数、条件、递归、算术——编译器前端的所有结构。语言设计者学它「怎么把一个功能砍到最小」，编译器学习者学它「怎么实现每一个最小的功能」。</span>

## 2 词法与语法：递归下降重演

教程的**词法**（第 1 章）用最简单的「每字符分类」：标识符、数字、运算符、注释——与第七篇的翻译器词法同构，只是更完整。

教程的**语法**（第 2-3 章）用**递归下降**实现，每个非终结符一个函数：

```cpp
// 解析表达式: primary → unary → binary → expression
std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParseUnary();                    // 先解析一元/primary
    return ParseBinOpRHS(0, std::move(LHS));    // 再处理二元（优先级）
}
```

**运算符优先级**用「优先级爬升（precedence climbing）」处理——一张运算符优先级表 + 循环，避免了深层递归。这正是第十篇讲过的「表达式用优先级表 + 循环」的实践。

**AST**：每个语法结构一个类（`NumberExprAST`、`BinaryExprAST`、`CallExprAST`、`FunctionAST`），递归下降直接「建 AST」。<span class="marginnote">「递归下降 + 优先级爬升」是 Kaleidoscope 语法层的核心：`ParseBinOpRHS(exprPrec, LHS)` 用一个循环按优先级表消化运算符——`a + b * c` 在这里被正确构造成 `a + (b * c)` 的树。读者可以对照第十节「递归下降」与第十三节「表达式分析」的知识，看教程如何把理论变成代码。</span>

## 3 从 AST 到 LLVM IR：CodeGen

**IR 生成**（第 3 章）是教程的心脏：给每个 AST 结点一个「生成 IR」的方法。

```cpp
Value *NumberExprAST::codegen() {
    return ConstantFP::get(TheContext, APFloat(Val));  // 数字 → 常量
}

Value *BinaryExprAST::codegen() {
    Value *L = LHS->codegen(), *R = RHS->codegen();
    // 根据运算符生成 fadd/fmul/fcmp 等指令
    return Builder.CreateFAdd(L, R, "addtmp");
}
```

- **`Builder`**：LLVM 的 `IRBuilder`——按顺序在基本块末尾追加指令。
- **常量**：`ConstantFP`。
- **二元运算**：`CreateFAdd`、`CreateFMul`、`CreateFCmp`。

**递归下降 + codegen** 的组合：语法分析建 AST，AST 的 `codegen()` 递归生成 IR——这是「语法制导翻译」的面向对象实现。<span class="marginnote">「AST 每个结点一个 codegen()」是语法制导翻译的 OOP 形态：语法结构（AST 结点）与语义动作（codegen 方法）绑定在一起。对照第四篇的 SDD——`codegen()` 就是综合属性 `E.code` 的面向对象实现；`Builder` 负责指令的「拼接」，对应 SDD 里的 `||` 拼接操作。</span>

## 4 公式解析：优先级爬升的核心

二元表达式解析的核心函数：

$$\text{ParseBinOpRHS}(\text{ExprPrec}, \text{LHS}) = \begin{cases} \text{若下一运算符优先级 } < \text{ExprPrec} & \text{返回 LHS} \\ \text{否则} & \text{ParseBinOpRHS}(\text{新优先级}, \text{LHS op RHS}) \end{cases}$$

- **第一步，停止条件**：下一个运算符的优先级低于当前级别——说明该「升级给外面处理」，返回已累积的左子树。
- **第二步，吸收**：运算符优先级足够，就解析右侧操作数，合成新的 LHS，递归继续。
- **第三步，右结合**：若运算符右结合（如赋值），右侧递归用**更低的**优先级继续——让右边吃得更深。

**「优先级表 + 递归循环」把 `a+b*c` 构造成正确的树**——它是递归下降处理表达式的最优雅写法，也是 Kaleidoscope 语法层的点睛之笔。

## 5 优化与 JIT：白送的收益

教程的优化（第 4 章）几乎免费：

```cpp
TheFPM->add(createInstructionCombiningPass());   // 指令化简
TheFPM->add(createReassociatePass());            // 重结合
TheFPM->add(createGVNPass());                    // 全局值编号
TheFPM->add(createCFGSimplificationPass());      // CFG 化简
```

**优化 pass 直接复用 LLVM**——`fib(20)` 这样的递归程序在 -O 下会被深度优化。

**JIT（第 5 章）** 更是白送：LLVM 的 **ORC JIT** 直接把「顶层表达式」编译成机器码并执行——`fib(20)` 输入就返回 6765，语言瞬间变成 REPL。<span class="marginnote">「优化与 JIT 白送」是 LLVM 生态的最大红利：Kaleidoscope 教程前 4 章只需写前端（词法 + 语法 + AST + IR 生成），优化器与 JIT 全由 LLVM 提供。这再次印证了「N×M 变 N+M」——写前端的人不需要懂寄存器分配、指令选择，LLVM 后端全部代劳。</span>

## 6 小结

- **Kaleidoscope** 是 LLVM 官方的「一门语言的完整前端」教程：不到两千行实现词法、语法、AST、IR、优化、JIT。
- 语言设计极简但完整：函数、表达式、条件、递归、`for`、运算符——覆盖前端所有核心。
- 语法层用**递归下降 + 优先级爬升**处理表达式；AST 每个结点一个类。
- **IR 生成**：AST 结点的 `codegen()` + `IRBuilder` 递归生成 LLVM IR——SDD 的 OOP 实现。
- **优化与 JIT 由 LLVM 白送**：`fib(20)` 输入即得 6765——「写前端、后端白拿」的 LLVM 承诺。

在下一节，我们鸟瞰 LLVM 生态：**LLVM 与现代编译器生态——Clang、Rust 与 Swift**。
