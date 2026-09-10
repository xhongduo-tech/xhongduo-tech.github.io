---
title: yacc / bison
date: 2026-09-08
section: cs
---

# yacc / bison

<div class="epigraph">
<p>产生式旁边写动作；LALR 生成器交出移进归约表，冲突用优先级声明消解，而不是改写全部左递归。</p>
<footer>—— 据 Johnson, Yacc: Yet Another Compiler-Compiler；Levine, Flex &amp; Bison；Knuth, 1965；龙书第 4 章整理</footer>
</div>

上一课[lex / flex](/cs/lex-flex)把字符变成记号。主干已有[LR 与移进归约](/cs/lr-shift-reduce)、[SLR 与 LALR](/cs/slr-lalr)。缺口不是再构造项集闭包，而是**生成器**：yacc / bison 读 `.y`，发出 `yyparse`，并在归约时跑嵌入动作。本课钉接口、冲突报告与 `%left` 习惯；不手填一张表达式文法的 LALR 表。

## 问题

手写递归下降要消左递归；表达式用 LR 可保留左结合产生式。yacc：非终结符、产生式、`$$`/`$1` 写语义值。生成期建 LALR(1) 表；运行期栈上是状态与值。缺口是这份**规则到表**的工程，不是再定义 CFG。

与 lex 的契约：`yylex` 供记号，`YYSTYPE` 是语义值联合。错误：`yyerror`。本课不把错误恢复写完，那是后课。

### `%left` 不是 CFG 的定理

二义文法（if-else、加减乘）在 LALR 上会移进/归约冲突。bison 用优先级与结合性给冲突格一个确定动作。这是工具约定，语言的无二义性并未被证明——只是表变成函数。

<span class="marginnote"> Johnson 的 yacc 服务 Unix C 编译器。GNU bison 兼容并扩展（`%glr-parser` 点名，下一课才讲 GLR）。龙书 4.8–4.9 节把生成器当 LALR 的落地。</span> 

## 方法

写 `.y`：记号声明、`%%`、文法、辅助函数。归约动作里建 AST 节点，不要在动作里做优化。冲突：bison 报告 shift/reduce、reduce/reduce；先看 `.output` 的状态，再决定改文法还是加优先级。

```mermaid
flowchart TD
  Y["文法 .y"] --\gt  GEN["yacc / bison"]
  GEN --\gt  TBL["LALR 表 + yyparse"]
  LEX["yylex"] --\gt  TBL
  TBL --\gt  AST["语义值 / AST"]
```

中期：把词法规则与语法规则分成两个文件，用共享的记号枚举。不要在 lex 动作里解析表达式。

## 机制

LALR 合并同心项集，表比规范 LR 小，可能引入额外冲突。reduce/reduce 通常是文法真有问题；shift/reduce 在表达式与悬挂 else 上常见。bison 的 `%expect` 只是把已知冲突数钉死，不是证明正确。

语义值栈与状态栈同步：归约弹出右部长度，压入 `$$`。动作里若释放 `$1` 又把指针放进 `$$`，所有权要一次说清，否则错误恢复会双释放——后课再收。

## 边界

本课不讲 Earley，不把 GLR 当默认。不抄 bison 手册的全部指令。后课默认：实用语言的确定文法用 LALR 生成器；认不了的二义与无限前瞻，交给更一般的解析策略。下一课 Earley：任意 CFG，不要表冲突消解。

优先级声明解决的是生成器的表，不是语言标准里的唯一语法树定义。标准若另有消二义规则，生成器必须与之对齐。

## 小结

- yacc / bison：CFG + 动作 → LALR 分析器。
- 与 lex 经记号种别与 `yylval` 对接。
- 冲突用改文法或优先级；`%expect` 不是证明。
- 出处：Johnson, Yacc；Levine；Knuth, 1965；Aho et al., 龙书。
