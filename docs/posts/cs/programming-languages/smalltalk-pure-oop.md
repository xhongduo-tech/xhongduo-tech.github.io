---
title: Smalltalk 的纯面向对象模型
date: 2026-08-07
---

# Smalltalk 的纯面向对象模型

<div class="epigraph">
<p>Smalltalk 不是一门语言，而是一种看待世界的方式：一切都是对象，一切皆可编程。</p>
<footer>—— 艾伦 · 凯（Alan Kay）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第11章 §11.9 + Smalltalk 历史 ｜ 2026-08-07</p>
</div>

## 为什么从 Smalltalk 开始

在 C++/Java 之前，Smalltalk 于 1970 年代在 Xerox PARC 由艾伦·凯的团队创造，是**第一个完整的面向对象语言**，也是「纯 OO」的典范。说它「纯」，因为它的两个原则贯彻到底：**一切皆对象**（连数字、类、代码块都是对象）、**一切皆消息**（连 `1 + 2` 都是消息发送）。理解 Smalltalk，就是理解 OOP 的「纯粹形态」——它没有 C++/Java 为了兼容过程式世界而做的妥协（如基本类型、非对象操作符）。虽然 Smalltalk 本身式微，但它的思想（消息传递、动态绑定、图像化环境）至今在 Objective-C、Ruby、以及现代 IDE 中回响。<span class="marginnote">Smalltalk 的历史地位：它把 Simula 的「类」与「继承」从「仿真语言的扩展」提升为「整个世界的模型」。艾伦·凯说 OOP 的核心是「消息传递」而非「对象」——对象是载体，消息是灵魂。这个观点至今仍是理解 OOP 设计的最深刻视角。</span>

## 1 一切皆对象

Smalltalk 的第一个原则：**每一个值都是对象**——数字、字符串、布尔、类、代码块、栈帧，全部是「有类、能响应消息」的对象。

```smalltalk
5 class        "→ SmallInteger：数字是对象"
true class     "→ True：布尔是对象"
```

推论：**没有「基本类型/对象」之分**。C++/Java 有 `int`（非对象）与 `Integer`（对象）之分，Smalltalk 没有——数字也能接收消息（`5 factorial`、`3 between: 1 and: 5`）。<span class="marginnote">Java 的「基本类型 vs 包装类型」双轨（`int` vs `Integer`）、自动装箱/拆箱，正是「向纯对象靠拢但未走到底」的折中。Smalltalk 从一开始就没有这道裂缝——「一切皆对象」消除了基本类型特例。</span>

## 2 一切皆消息

Smalltalk 的第二个原则：**所有操作都是消息发送**。语法 `receiver message` 或 `receiver message: argument`：

```smalltalk
1 + 2        "向对象 1 发送 + 消息，参数 2"
aCollection size
obj perform: #methodName    "连方法名本身都可以是数据"
```

连 `ifTrue:`、`whileTrue:` 都是**发送给布尔对象的消息**——`true` 对象响应 `ifTrue: [ ... ]`，`false` 对象响应 `ifFalse: [ ... ]`。控制结构不是语法，而是方法调用！<span class="marginnote">Smalltalk 的 `ifTrue:`/`whileTrue:` 是「控制结构即消息」的极致：`[ ... ]` 是<strong>代码块对象</strong>（closure），作为参数传给布尔对象。这让「条件」本身就是对象——`(x > 0) ifTrue: [ ... ]` 中，`x > 0` 产生布尔对象，再向它发送 `ifTrue:`。没有语法关键字，只有消息。</span>

## 3 类与元类：Smalltalk 的类系统

Smalltalk 中**类本身也是对象**（`Dog` 是 `Dog class` 的实例）。类对象的方法（如 `Dog new`、`Dog name`）定义在**元类（metaclass）**中。

```
Dog（对象）——实例 of——> Dog class（元类）
Dog class ——实例 of——> Metaclass
```

**元类**是 Smalltalk 的独特设计：类方法（静态方法）不再是「挂在类上的函数」，而是「类对象响应消息」的自然结果——因为类也是对象，它也有自己的类（元类）。<span class="marginnote">「类也是对象」让 Smalltalk 的反射能力浑然天成：`Dog methods`、`Dog superclass`、`Dog allInstances` 都是类对象可响应的消息。「元类」统一了「实例方法」与「类方法」的概念——后者不过是「类对象」的「实例方法」。</span>

## 4 公式解析：消息分派的最简模型

Smalltalk 的消息分派是「纯对象模型」下动态绑定的极致体现。发送消息 `m` 给对象 `o` 的分派：

$$
\text{dispatch}(o, m) = \text{lookup}(\text{class}(o), m)
$$

其中 `class(o)` 是 `o` 的类，`lookup` 沿继承链查找：

$$
\text{lookup}(C, m) = \begin{cases}
m \in \text{methods}(C) & \Rightarrow \text{methods}(C)[m] \\
C \text{ 有父类} & \Rightarrow \text{lookup}(\text{superclass}(C), m) \\
\text{否则} & \Rightarrow \text{error: doesNotUnderstand}
\end{cases}
$$

三步拆解：

- **第一步，问类**：消息发给谁 → 查它的类（`class(o)`）——动态类型在此显形。
- **第二步，沿继承链**：类的 `methods` 里没有 → 到父类找，直到根（`Object`）。动态绑定 = 这条链上的查找。
- **第三步，兜底 `doesNotUnderstand`**：全链找不到 → 发送 `doesNotUnderstand:` 消息给对象（它可重写！）——**「方法不存在」本身也通过消息机制处理**，保持「一切皆消息」的闭环。

**辨析｜易错点：** Smalltalk 没有「静态绑定」——所有消息都动态分派，无 `virtual` 关键字（不需要：默认全虚）。这与 C++（默认静态绑定、`virtual` 显式）形成鲜明对照。**「纯 OO = 无静/虚之分，因为一切皆动态」**——Smalltalk 的纯粹性体现在这里。

## 5 Smalltalk 的遗产与现代回响

- **Objective-C / Swift**：消息传递语法（`[obj method]`）、动态分派直接继承 Smalltalk。
- **Ruby**：一切皆对象（数字是对象）、方法调用即消息、`send` 方法。
- **IDE 革命**：Smalltalk 的「图像（image）+ 浏览器」环境启发了现代 IDE 的调试器、检查器、热重载。
- **测试文化**：Smalltalk 的 SUnit 是 xUnit 测试框架的祖先（Kent Beck）。<span class="marginnote">Smalltalk 最被低估的遗产是<strong>环境哲学</strong>：程序不是「编辑-编译-运行」的文本，而是「活的对象图」，随时可检视、可修改、可恢复。这种「活环境」思想如今在 Python/Node 的 REPL、Jupyter、以及 WebAssembly 组件模型中重新生长。</span>

## 6 消息 vs 方法：从 Smalltalk 看 OO 本质

Smalltalk 的「消息」视角，给理解整个 OO 提供了一把钥匙——**「消息」与「方法」的区别**：

**方法（method）**：函数定义——「一段可调用的代码」。它是**静态**的（写在类里）。

**消息（message）**：一次「请求」——「请接收者做某事」。它是**动态**的（发给某个对象）。

`a.speak()` 可以两种视角看：

- **函数视角**：调用 `speak` 这个方法，参数 `a`。
- **消息视角**：向对象 `a` 发送 `speak` 消息——**`a` 决定如何响应**。

两种视角的差别在**接收者的主动权**：函数视角下「调用者决定一切」；消息视角下「接收者决定响应」——这正是动态绑定的哲学根源（接收者的类型决定实现）。**「OO 的本质是『消息传递』而非『函数调用』」**——艾伦·凯反复强调这一点：对象是「自主的个体」，不是「被调用的数据结构」。

**辨析｜易错点：** 主流 OO 语言（Java/C++）把「消息」语法化为「方法调用」，消息视角被掩盖——但**「方法调用」的真正语义是「向接收者发送消息」**。理解这一点，「动态绑定」「多态」「接口」都豁然开朗：它们都是「接收者如何响应消息」的不同方面。「方法调用」是语法，「消息传递」是语义——这正是 Smalltalk 留给 OO 的最深刻遗产。



## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 每一个值都是对象 | Smalltalk 的第一个原则：每一个值都是对象——数字、字符串、布尔、类、代码块、栈帧，全部是「有类、能响应消息」的对象。 |
| 所有操作都是消息发送 | Smalltalk 的第二个原则：所有操作都是消息发送。语法 receiver message 或 receiver message: argument |
| 类本身也是对象 | Smalltalk 中类本身也是对象（Dog 是 Dog class 的实例）。类对象的方法（如 Dog new、Dog name）定义在元类（meta |
| 元类（metaclass） | Smalltalk 中类本身也是对象（Dog 是 Dog class 的实例）。类对象的方法（如 Dog new、Dog name）定义在元类（meta |
| 「方法不存在」本身也通过消息机制处理 | 第三步，兜底 doesNotUnderstand：全链找不到 → 发送 doesNotUnderstand: 消息给对象（它可重写！）——「方法不存在」 |
| Objective-C / Swift | Objective-C / Swift：消息传递语法（[obj method]）、动态分派直接继承 Smalltalk。 |
| Ruby | Ruby：一切皆对象（数字是对象）、方法调用即消息、send 方法。 |
| IDE 革命 | IDE 革命：Smalltalk 的「图像（image）+ 浏览器」环境启发了现代 IDE 的调试器、检查器、热重载。 |
| 「消息」与「方法」的区别 | Smalltalk 的「消息」视角，给理解整个 OO 提供了一把钥匙——「消息」与「方法」的区别： |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 7 小结

- Smalltalk 的两个原则：**一切皆对象**（无基本类型特例）与**一切皆消息**（连控制结构都是消息）。
- **类也是对象**，元类承载类方法——「类方法」不过是「类对象的方法」。
- 消息分派：`dispatch(o,m) = lookup(class(o), m)`，沿继承链查找，`doesNotUnderstand` 兜底。
- Smalltalk 无「静/虚」之分（一切动态绑定）；其消息模型、图像环境、测试文化深刻影响现代语言与 IDE。

在下一节，我们将看 OOP 的「契约层」——**接口（Interface）、协议（Protocol）与混入（Mixin）**。
