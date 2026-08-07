---
title: Prolog 的基本元素：项、事实、规则与目标
date: 2026-08-07
---

# Prolog 的基本元素：项、事实、规则与目标

<div class="epigraph">
<p>Prolog 程序不是指令序列，而是一份关于世界的声明书；运行它，就是向世界提问。</p>
<footer>—— 佚名（Prolog 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第16章 §16.4–16.5 ｜ 2026-08-07</p>
</div>

## 为什么从 Prolog 元素开始

前两节讲了谓词演算与归结原理的理论；这一节进入 **Prolog** 本身——把理论落成可运行的程序。Prolog 的四个基本元素：**项**（数据）、**事实**（无条件真的命题）、**规则**（条件命题）、**目标**（查询）。一个 Prolog 程序就是「事实 + 规则」的集合，而「运行」就是「查询目标」。这一节把这些元素的语法、语义、以及它们如何构成「声明式程序」讲透——Prolog 是逻辑式编程的工业标准，理解它，就能触类旁通 Datalog、规则引擎、以及知识图谱查询。<span class="marginnote">Prolog 的名字来自「Programming in Logic」。它的程序结构完全对应霍恩子句：事实 = 无前提子句，规则 = 有前提子句，目标 = 无结论子句。理解 Prolog 的钥匙：<strong>程序是知识库，运行是问询</strong>。</span>

## 1 项：Prolog 的数据

**项（term）**是 Prolog 的唯一数据形态，四种：

```prolog
alice            % 原子（常量）：小写开头
X, Parent        % 变量：大写开头或下划线
likes(alice, music)   % 复合项：函子 + 参数（结构）
[1, 2, 3]        % 列表：语法糖，等价于 .(1, .(2, .(3, [])))
```

- **原子**：小写开头的符号串——`alice`、`music`、`likes`。
- **变量**：大写开头——`X`、`Person`；`_` 是匿名变量（每次独立）。
- **复合项**：`functor(arg1, arg2, ...)`——函子 + 参数，本质是「带标签的树」。
- **列表**：`[H|T]` 头尾结构（同 Scheme），`[a, b]` 是糖衣。<span class="marginnote">Prolog 的「一切皆项」极简得优雅：程序与数据都是项——`likes(alice, music)` 既可作为「事实」放在程序里，也可作为「查询」被询问，还可作为「结构数据」被构造与分解。这种「代码即数据」与 Lisp 同源（homoiconicity），只是 Prolog 的项天然带「可合一」的语义。</span>

## 2 事实与规则：知识库的两种陈述

**事实（fact）**：无条件为真的项——一个原子或复合项，以句号结束。

```prolog
human(socrates).
likes(alice, music).
```

**规则（rule）**：条件命题——`结论 :- 前提1, 前提2, ...`（「若前提都成立，则结论成立」）。

```prolog
mortal(X) :- human(X).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

规则的读法：`grandparent(X,Z)` 当 `parent(X,Y)` 且 `parent(Y,Z)`——**`,` 是「且」，`;` 是「或」**。<span class="marginnote">规则里的变量是「共享的约束」：`grandparent(X,Z) :- parent(X,Y), parent(Y,Z)` 中，同一个 `Y` 出现在两个前提里——它必须是「同一个中间人」。Prolog 的变量共享 = 「合一约束」：求解时 `Y` 被统一实例化为同一个人。</span>

## 3 目标：向知识库提问

**目标（goal / query）**：询问某个命题是否为真——在交互环境输入 `?- ...`：

```prolog
?- human(socrates).
true.

?- mortal(X).
X = socrates.

?- likes(alice, Who).
Who = music.
```

查询的行为：

- `human(socrates)` 问「这是已知事实吗」——归结验证。
- `mortal(X)` 问「谁符合 mortal」——**X 被实例化为所有解**（通过合一与回溯逐个给出）。
- 变量在查询中 = 「求值对象」——回答是「X 取什么值使查询为真」。

## 4 公式解析：查询求解的递归过程

Prolog 查询求解（SLD 归结）可以理解为「目标分解」：

$$
\text{solve}(G_1, \dots, G_k) = \text{找一个子句 } C \text{ 使 } \text{head}(C) \text{ 与 } G_1 \text{ 合一，} \text{然后解 } \text{body}(C), G_2, \dots, G_k
$$

以 `?- grandparent(alice, X)` 为例：

```prolog
grandparent(X,Z) :- parent(X,Y), parent(Y,Z).   % 规则
parent(alice, bob).   parent(bob, carol).       % 事实
```

三步拆解：

- **第一步，匹配规则**：目标 `grandparent(alice, X)` 与规则头 `grandparent(X, Y)` 合一：`X=alice`、`Y=X`（目标里的 X 是查询变量）。
- **第二步，分解前提**：新目标变成 `parent(alice, Y), parent(Y, X)`——两个子目标，变量共享。
- **第三步，逐个求解**：`parent(alice, Y)` 与事实合一得 `Y=bob`；再解 `parent(bob, X)` 得 `X=carol`。**「目标分解 + 合一 + 回溯」三件套完成一次查询**。

**辨析｜易错点：** 子目标按**书写顺序**（从左到右）求解——这个顺序影响性能与结果。`parent(X,Y), parent(Y,Z)` 与 `parent(Y,Z), parent(X,Y)` 逻辑等价但**搜索顺序不同**（可能一个飞快一个死循环）。**「声明式语义一样，过程式语义（搜索顺序）不同」**——这是 Prolog 初学者最需要适应的一点。

## 5 Prolog 的典型应用

- **专家系统 / 规则引擎**：事实 + 规则的「知识库 + 推理」模式。
- **自然语言处理**：Prolog 的 DCG（定从句文法）表达语法分析。
- **约束求解 / 图算法**：`member`、`append` 等列表谓词天然支持「多方向」使用。
- **知识图谱查询**：RDF/SPARQL 与 Datalog 的推理（Prolog 的近亲）。<span class="marginnote">Prolog 谓词的「多方向性」令人着迷：`append([1,2], [3], R)` 拼接（R=[1,2,3]），而 `append(X, Y, [1,2,3])` 能<strong>枚举所有拆分方式</strong>——同一个谓词，问「拼接结果」或「所有拆分」，都由声明式定义直接给出。命令式函数做不到这种「可逆」。</span>



## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 项（term） | 项（term）是 Prolog 的唯一数据形态，四种： |
| 原子 | alice % 原子（常量）：小写开头 |
| 变量 | X, Parent % 变量：大写开头或下划线 |
| 复合项 | likes(alice, music) % 复合项：函子 + 参数（结构） |
| 事实（fact） | 事实（fact）：无条件为真的项——一个原子或复合项，以句号结束。 |
| 规则（rule） | 规则（rule）：条件命题——结论 :- 前提1, 前提2, ...（「若前提都成立，则结论成立」）。 |
| 目标（goal / query） | 目标（goal / query）：询问某个命题是否为真——在交互环境输入 ?- ...： |
| X 被实例化为所有解 | mortal(X) 问「谁符合 mortal」——X 被实例化为所有解（通过合一与回溯逐个给出）。 |
| 「目标分解 + 合一 + 回溯」三件套完成一次查询 | 第三步，逐个求解：parent(alice, Y) 与事实合一得 Y=bob；再解 parent(bob, X) 得 X=carol。「目标分解 + 合 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **项**是 Prolog 唯一数据形态：原子、变量、复合项、列表——一切皆项。
- **事实** = 无条件真的命题；**规则** = 条件命题（`结论 :- 前提`），`,` 是且、`;` 是或。
- **目标** = 查询；变量在查询中「求值对象」，经合一与回溯逐个给出解。
- 查询求解 = 目标分解 + 合一 + 回溯；子目标顺序影响搜索（声明等价、过程不同）；谓词可「多方向」使用。

在下一节，我们将深入 Prolog 的执行机制与缺陷——**合一、回溯与 Prolog 的缺陷**。
