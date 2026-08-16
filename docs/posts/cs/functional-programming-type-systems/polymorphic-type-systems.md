---
title: 多态类型系统（System F、参数多态、ML 类型推断）
date: 2026-08-07
---

# 多态类型系统（System F、参数多态、ML 类型推断）

<div class="epigraph">
<p>计算机科学之于计算机，并不比天文学之于望远镜更多。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · 函数式编程与类型系统 ｜ Pierce《Types and Programming Languages》第23章 ｜ 2026-08-07</p>
</div>

## 为什么从多态开始

上一节的简单类型 λ 演算留下一个恼人的缺憾：一个如此平凡的函数——恒等函数 $\lambda x. x$——居然要为 `Bool`、`Nat`、`(Bool→Nat)` 每种类型各写一遍。程序员当然不愿这么干，而语言设计者更看到了一个更深的问题：**一个「对所有类型都同样工作」的算法，凭什么要为每一种类型重复表达？** 这一节要讲的**多态（polymorphism）**，就是类型系统对「复用」这个工程刚需给出的理论回答。

多态的历史与两个名字分不开：让-伊夫 · 吉拉德（Jean-Yves Girard）1972 年在博士论文里独立发现 System F，约翰 · 雷诺兹（John C. Reynolds）1974 年又独立提出等价的「多态 λ 演算」。它把类型系统从「一阶」推进到「二阶」——**类型本身开始谈论类型**，语言的自我表达力跃升了一个维度。<span class="marginnote">「System F」的 F 指 second-order（二阶），因为它的类型可以量化所有类型；与之相对，简单类型只量项不量类型，是一阶。Girard 的发现还带出强正规化——System F 的项必然终止，尽管表达力远超简单类型。</span>

这段历史还有一层饶有趣味的注脚：Girard 是为证明逻辑学里的「二阶逻辑强正规化」而构造 System F 的，他**最初根本没想做编程语言**；雷诺兹则是为了给编程语言找多态的语义模型。两条动机完全不同的路，最后撞进同一座建筑——这类「逻辑与编程殊途同归」的桥段，在本专题《依赖类型》一篇还会再上演一次。

## 1 从「重写一遍」到「一次写成」

问题出在类型标注上。简单类型的恒等函数只能写成：

$$\lambda x: \texttt{Bool}.\; x \qquad \lambda x: \texttt{Nat}.\; x \qquad \lambda x: (\texttt{Bool}\to\texttt{Nat}).\; x$$

每一份都是同一个算法、不同的类型皮囊。多态的思路很直接：**让函数抽象类型本身**——既然我们早就会用 $\lambda x. t$ 抽象「值」这个维度，为什么不能用 $\Lambda X. t$ 抽象「类型」这个维度？System F 引入两个新语法构造，一个在类型层、一个在项层：

- 类型层：类型 $T ::= X \mid T_1 \to T_2 \mid \forall X.\; T$——$\forall X. T$ 读作「对所有类型 $X$，$T$ 成立」，是一张**通用模板**；
- 项层：项 $t ::= x \mid \lambda x: T.\; t \mid t_1\; t_2 \mid \Lambda X.\; t \mid t\;[T]$——$\Lambda X. t$ 是**类型抽象**（对类型参数的函数），$t\;[T]$ 是**类型应用**（把类型 $T$ 塞进模板）。<span class="marginnote">用大写 $\Lambda$ 与大写类型变量 $X$ 强调「这是在类型层面做函数」，与项层面的 $\lambda$ 和小写变量 $x$ 严格区分。这样分层后，「类型抽象」与「类型应用」完全镜像「项抽象」与「项应用」。</span>

于是恒等函数可以一次写成、处处使用：

$$\mathrm{id} = \Lambda X.\; \lambda x: X.\; x$$

要 Bool 版？$\mathrm{id}\;[\texttt{Bool}]$。要函数类型版？$\mathrm{id}\;[\texttt{Bool}\to\texttt{Bool}]$。**同一个定义，实例化出无穷多个具体版本**——这就是**参数多态（parametric polymorphism）**：一个对类型参数「一视同仁」的程序。

更典型的例子是列表映射 `map`。它要「把函数 $f : X \to Y$ 应用到列表中每个元素上」，于是类型天然带两个类型参数：

$$\mathrm{map} = \Lambda X.\;\Lambda Y.\; \lambda f: X \to Y.\; \lambda l: \mathrm{List}\;X.\; \cdots$$

`map` 完全不关心 $X$、$Y$ 具体是什么——整数变整数、字符串变布尔、记录变记录，同一份代码全都能用。<span class="marginnote">在 Haskell 里这条签名写作 `map :: (a -> b) -> [a] -> [b]`，在 Rust 里写作 `fn map<A,B>(f: impl Fn(A)->B, xs: Vec<A>) -> Vec<B>`——三种语法，同一个 System F 内核。</span>

## 2 类型推导规则的升级

System F 的类型规则在简单类型的 T-Var、T-Abs、T-App 之上，增加两条新规则：

$$
\frac{\Gamma \vdash t : T}{\Gamma \vdash \Lambda X.\; t : \forall X.\; T}\;(\mathrm{T\text{-}TAbs})
\qquad
\frac{\Gamma \vdash t : \forall X.\; T}{\Gamma \vdash t\;[U] : [X \mapsto U]\,T}\;(\mathrm{T\text{-}TApp})
$$

**T-TAbs**：若在「类型变量 $X$ 可以自由使用」的前提下 $t$ 有类型 $T$，则类型抽象 $\Lambda X. t$ 有类型 $\forall X. T$。**T-TApp**：若 $t$ 是模板 $\forall X. T$，则把具体类型 $U$ 塞进去，得到 $[X\mapsto U]\,T$（把 $T$ 里所有 $X$ 替换成 $U$）。<span class="marginnote">这两条规则是「项层 λ 规则」在类型层的精确镜像：T-Abs 引入 $\lambda$，T-TAbs 引入 $\Lambda$；T-App 消去 $\lambda$，T-TApp 消去 $\Lambda$。类型理论里这种「结构同构」无处不在。</span>

## 3 公式解析：System F 的完整推导

**把「模板实例化」这一整套流程走一遍，是理解 System F 的最佳方式。** 我们证明 $\mathrm{id}\;[\texttt{Bool}]\; \mathrm{true} : \texttt{Bool}$。

- **第一步，写出 id 的类型**：$\mathrm{id} = \Lambda X.\; \lambda x: X.\; x$。由 T-TAbs 与 T-Abs 叠用，其类型是 $\forall X.\; X \to X$——「对任何类型 $X$，从 $X$ 到 $X$ 的函数」。
- **第二步，类型应用**：用 T-TApp 把 $\texttt{Bool}$ 代入：$\mathrm{id}\;[\texttt{Bool}] : \texttt{Bool} \to \texttt{Bool}$。模板 $\forall X. X\to X$ 中的 $X$ 被替换为 `Bool`。
- **第三步，项应用**：再用一次普通 T-App，把实参 $\mathrm{true} : \texttt{Bool}$ 喂给 $\texttt{Bool} \to \texttt{Bool}$ 的函数，得到结果类型 $\texttt{Bool}$。
- **第四步，直觉**：三步对应三次「接口咬合」——先给类型参数供上具体类型，再给值参数供上具体值。**模板在类型层面被「填充」一次，在值层面再被「填充」一次**，两次填充构成了多态调用的全部。

这个例子还展示了 System F 的一个优雅性质：**类型应用发生在编译期、被擦除**——运行时的 $\mathrm{id}\;[\texttt{Bool}]$ 与 $\mathrm{id}\;[\texttt{Nat}]$ 是完全相同的一段代码。关于「擦除」，详见《类型系统与编译器》。

值得一提：System F 能类型化丘奇数——在简单类型系统里 $Y$ 组合子无法类型化，但**多态的量词恰好给了 Church 编码一个家**。丘奇数 $\overline{n} = \lambda f.\lambda x. f^n x$ 在 System F 里有类型：

$$\overline{n} : \forall X.\; (X \to X) \to X \to X$$

于是自然数成为一个**可静态检查的抽象数据类型**，这为后续《代数数据类型》与《类型系统与编译器》里的「数据即模板」视角埋下了伏笔。

## 4 参数多态与自由定理

System F 最强的地方不是表达能力，而是**参数性（parametricity）**——一个只有类型抽象、没有针对类型做出任何判断的程序，对「不同的类型实例」行为必然一致。雷诺兹 1983 年用关系模型证明了这一性质，菲利普 · 瓦德勒（Philip Wadler）1989 年把它变成著名的《免费的定理》（Theorems for Free!）。<span class="marginnote">「免费的定理」指：只要函数有类型 $\forall X.\; X \to X$，你不看实现就知道它<strong>几乎必然是恒等函数</strong>——这是类型本身携带的、无需证明的定理。瓦德勒的论文标题即由此而来。</span>

这种「由类型反推行为」的能力极其罕见：在命令式语言里，同样签名可能做了任何事；而在 System F 里，类型签名把实现空间压缩到几乎唯一。这是多态「一视同仁」的直接红利——**类型越通用，行为越受限，程序越可预测**。<span class="marginnote">这种「通用性换可预测性」的权衡，正是 Rust 泛型、Haskell 的 `forall` 背后共享的哲学；也是类型理论对大模型「提示词可预测性」难题的一种抽象对照。</span>

来看一个自由定理的实操案例。给定一个完全未知的函数 $f : \forall X.\; \mathrm{List}\;X \to \mathrm{List}\;X$，我们能凭类型断言些什么？它不能对元素做任何「类型判断」（因为 $X$ 未知），所以它**只能**重排、丢弃或复制元素——`reverse`、`take 3`、`id` 都是合法实现，而「把数字翻倍」不可能。这一句话就过滤掉了一大类 bug。

**辨析｜易错点：**「多态」与「重载（overload）」不是一回事。多态是一个定义对**所有**类型工作（如 `id`、`map`）；重载是同一名字对**特定几个**类型各有一份实现（如 `+` 对 `int`、对 `float`）。System F 只提供前者；重载需要另外的类型类或特化机制（Haskell 的 typeclass、Rust 的 trait），那是《现代语言实践》一篇的内容。

## 5 局限与后续：谓词 vs 非谓词

System F 的表达力是有代价的。它引入的 $\forall$ 使类型推断从可判定变成不可判定——**通用的 System F 类型推断不存在算法**（Frank Pfenning 1988 证明）。一个直接后果是：如果程序员不写类型标注，机器无法自动补全——这与简单类型的「可推断」形成鲜明对照。这正是为什么主流语言不直接用 System F，而是退一步采用它的受限版本：

- **ML 风格**：程序员不写 $\Lambda$、不写 $\forall$，类型变量隐式全称量化在**最外层**，推断可判定——这就是下一篇《Hindley-Milner 类型推断》；
- **谓词多态（predicative）**：$\forall$ 只能出现在类型箭头的最外层左侧（rank-1），像 Rust、Haskell 的默认 `forall`；
- **非谓词（impredicative）**：$\forall$ 可以出现在任何位置（System F 属于此类），表达力最强但推断不可判定。

三种形态的取舍，可以用一张表收束：

| 系统 | $\forall$ 出现位置 | 类型推断 | 代表性语言 |
| --- | --- | --- | --- |
| 简单类型 | 无多态 | 可判定、全自动 | 早期 ML 雏形 |
| HM（ML） | 仅最外层（rank-1） | 可判定、全自动 | OCaml、Haskell |
| System F | 任意位置（rank-n） | 不可判定 | 理论语言、依赖类型 |

「推断可判定」之所以珍贵，是因为它意味着**程序员可以完全不写类型标注**——机器替你推导。这直接催生了下一篇的主题。

还有一个工程上的折衷值得记录：现代语言往往**混合**两种策略——对普通代码用可判定的 HM 推断（不用写类型），对特殊构造允许显式标注 rank-n 类型（自己写类型）。GHC 的 `RankNTypes` 扩展、Rust 的 trait 对象、TypeScript 的 `any` 逃生门，都是在这条「自动化与表达力」的谱系上取点。

这条谱系在依赖类型语言（Agda、Idris）里走得最远：那里的类型几乎完全由程序员书写，但换来的是「类型即证明」的全部表达力——详见本专题《依赖类型》一篇。

## 6 小结

- **多态**解决「同一算法为每种类型重写一遍」的复用难题，其理论形态是 System F。
- System F 新增**类型抽象 $\Lambda X. t$** 与**类型应用 $t\;[T]$**，使类型系统升级为二阶。
- 类型规则 T-TAbs / T-TApp 是项层 λ 规则的精确镜像，推导即「两次接口咬合」。
- **参数性/自由定理**：类型越通用，行为越受限——`∀X. X→X` 几乎只能是恒等函数。
- System F 的类型推断不可判定，主流语言改用受限的 rank-1 多态。
- Girard（1972）与 Reynolds（1974）独立发现 System F，后者还给出参数性定理。
- 类型应用发生在编译期并被擦除，`∀X. (X→X)→X→X` 也让 Church 编码在 System F 里终于有了家。
- **多态**（参数多态）让一个定义服务所有类型，是复用与抽象的理论基石。

在下一节，我们将直面「System F 推断不可判定」这个坎，走一条实用路线：**Hindley-Milner 类型推断**——让程序员不写任何类型标注，机器照样把类型算出来。
