---
title: Hindley-Milner 类型推断（合一算法、Algorithm W）
date: 2026-08-07
---

# Hindley-Milner 类型推断（合一算法、Algorithm W）

<div class="epigraph">
<p>简单不先于复杂，而是跟随复杂。</p>
<footer>—— 艾伦 · 佩利斯（Alan Perlis，《编程格言》Epigrams on Programming，1982）</footer>
</div>

<div class="article-byline">
<p>第三级 · 函数式编程与类型系统 ｜ Pierce《Types and Programming Languages》第22章 ｜ 2026-08-07</p>
</div>

## 为什么从类型推断开始

上一节停在 System F 的尴尬处：它表达力强，但**类型推断不可判定**——如果程序员不写类型，机器就永远算不出来。可回到 1970 年代的 ML 语言，程序员们却有一个几乎不加约束的体验：**完全不写类型标注，机器照样把最精确的类型算给你**。这中间的落差，由一个精妙的折衷填平：**Hindley-Milner（HM）类型推断**。

这也是「从极限到大模型」主线上「自动化 vs 表达力」主题的又一站：与优化器、自动微分把「推导」外包给机器同构，HM 把「写类型」这件苦活外包给了合一算法——**你只描述结构，机器替你完成约束求解**。

三个名字对应三段历史：罗杰 · 欣德利（Roger Hindley）1969 年、罗宾 · 米尔纳（Robin Milner）1978 年先后给出算法，卢卡 · 达马斯（Luis Damas）与米尔纳 1982 年给出完整证明。Milner 为之担保的那句话「类型正确的程序不会出错」，也在这里第一次成为可机械执行的现实。<span class="marginnote">HM 推断是「编程语言给人类的最大福利」之一：OCaml、Haskell、Rust、Swift、TypeScript、Go 的类型推断，全都是 HM 思想的后代。它的核心是两步——<strong>生成约束</strong>（从语法看它必须满足哪些类型等式）与<strong>解约束</strong>（用合一算法求出最一般解）。</span>

## 1 思路：让「未知类型」现形

HM 的关键想法朴素到令人惊讶：**把没写出来的类型当成未知数，列方程，再解方程**。程序员写 `let f x = x + 1`，推断器自动展开为：

- 给 `f`、`x`、`x+1` 各安排一个**类型变量**：$f : \alpha$，$x : \beta$，结果 : $\gamma$；
- 从语法读出**约束**：`+` 的两个操作数类型相同（$\beta = \beta$），结果 $\gamma$ 是 `+` 的结果类型；
- 从函数结构读出**箭头约束**：$f$ 是函数，$f : \beta \to \gamma$。

于是问题变成「找一组类型变量到具体类型的替换，使所有约束同时成立」。这一步类比方程求解：**类型推断 = 给程序列一个类型方程组，然后解方程组**。<span class="marginnote">这种「把未知数当变量、把要求当方程」的建模，与中学解应用题、与线性代数解方程组是同一个心智模型；只是这里的「变量」替换的是类型而非数字。</span>

值得注意的是「未知类型」究竟出现在哪：它出现在**你懒得写**的地方（参数类型、局部变量类型），也出现在**算法才能算出来**的地方（`map` 组合后的结果类型）。HM 把两处一并解决——这正是「推断」比「检查」更强的含义：检查是验证你写对了，推断是替你把它写出来。

## 2 合一算法：解类型方程

解方程组的核心工具叫**合一（unification）**——约翰 · 罗宾逊（John Alan Robinson）1965 年为了解决一阶逻辑的机械化证明而发明。**合一**：给定两个类型表达式，找把它们变成同一个类型的替换（substitution）$\sigma$。

先定义**替换**：替换是从类型变量到类型的有限映射，写作 $\sigma = [\alpha \mapsto \texttt{Bool},\; \beta \mapsto \texttt{Bool}]$；把替换作用到类型上记作 $\sigma(T)$。若 $\sigma(T_1) = \sigma(T_2)$，则 $\sigma$ 是 $T_1, T_2$ 的一个**合一子（unifier）**。通常我们希望解是**最一般**的：最一般合一子（MGU，most general unifier）——它不加任何多余限制，任何其他合一子都是它的实例。<span class="marginnote">「最一般」的直觉：解 $\alpha \mapsto \texttt{Bool}$ 与解 $\alpha \mapsto \texttt{Nat}$ 都对某些方程成立，但若方程只要求「$\alpha$ 与 $\beta$ 相等」，那么 $\alpha \mapsto \beta$ 是最一般的——它保留最大的自由度，后面还能再实例化。</span>

## 3 公式解析：合一的两条主规则

**合一算法本身只有两条反复应用的主规则。** 设要合一 $T_1$ 与 $T_2$：

- **变量-类型**：若 $T_1$ 是类型变量 $\alpha$ 而 $T_2$ 不含 $\alpha$，则令 $\alpha \mapsto T_2$（若 $T_2$ 含 $\alpha$ 则失败——这是「循环类型」，要专门拒绝）；反过来对称。
- **结构递归**：若 $T_1 = T_1' \to T_1''$、$T_2 = T_2' \to T_2''$，则分别合一 $T_1'$ 与 $T_2'$、$T_1''$ 与 $T_2''$，合成两边的结果。
- **常数匹配**：`Bool` 只能与 `Bool` 合一，`Bool` 与 `Nat` 合一失败——这就是**类型错误**的判定。

直观地说，合一就是把两棵类型树**对齐**：树顶不同直接报错，树顶相同就递归对齐子树，碰到未知变量就先欠着、记录下来。整个过程是确定性的、多项式时间的。<span class="marginnote">「$T_2$ 含 $\alpha$ 则失败」这条叫<strong>出现检查（occurs check）</strong>，防止解出无限类型 $\alpha = \alpha \to \alpha$。没有出现检查的合一器是实践中大量无限循环 bug 的源头。</span>

举一个具体求解：合一 $\alpha \to \texttt{Bool}$ 与 $\texttt{Nat} \to \beta$。

- **第一步，看树顶**：两侧都是箭头 $\to$，结构匹配，于是递归合一首部 $\alpha$ 与 `Nat`、尾部 `Bool` 与 $\beta$。
- **第二步，变量-类型**：$\alpha$ 与 `Nat` 合一得 $\alpha \mapsto \texttt{Nat}$；`Bool` 与 $\beta$ 合一得 $\beta \mapsto \texttt{Bool}$。
- **第三步，合成**：解为 $\sigma = [\alpha \mapsto \texttt{Nat},\; \beta \mapsto \texttt{Bool}]$，两树都对齐成 $\texttt{Nat} \to \texttt{Bool}$。

若换成合一 $\alpha \to \texttt{Bool}$ 与 $\texttt{Nat} \to \texttt{Nat}$，尾部 `Bool` 与 `Nat` 匹配失败——**报类型错误**，推断停止。

## 4 Algorithm W：一整棵程序怎么推

有了合一，剩下的就是设计一个递归遍历语法树的算法——**Algorithm W**（Damas–Milner 1982）。它的口号是：**从左到右、自底向上**地对每个子表达式收集约束并解出替换。伪代码骨架：

```
infer(Γ, e):
  若 e = x:         查 Γ，返回其类型（已由泛化得到的模式）
  若 e = λx. t:     生成新变量 α，infer(Γ∪{x:α}, t)，返回 α→(其类型)
  若 e = t1 t2:     infer(Γ,t1) 得 T1；infer(Γ,t2) 得 T2；
                    新变量 β，合一 T1 与 (T2→β)；返回 β
  若 e = let x = t1 in t2:   infer(Γ,t1) 得 T1；
                    对 T1 中自由类型变量做泛化 → 模式 σ；
                    infer(Γ∪{x:σ}, t2)；返回其类型
```

每一步返回的不仅是类型，还有一路积累下来的替换。W 的关键在最后一条 `let` 规则——它把「泛化」安放在语言里唯一的位置上，这正是 HM 与「无限制多态」的分水岭。<span class="marginnote">Milner 的算法之所以要嵌在 `let` 里，是因为「变量在 lambda 体内可以用多次，在 let 右侧只计算一次」——多次使用要同一类型，一次使用可以推广。这个区别构成了 HM 的整个表达能力。</span>

走一个完整的小例子：推断 `\x -> x` 的类型。第一遍，$\lambda$ 体里的 `x` 类型未知，记为 $\alpha$；由 T-Abs 得整个抽象类型 $\alpha \to \alpha$。没有 `let`，没有泛化，$\alpha$ 仍是**自由的**——所以单靠一个 $\lambda$，HM 只能给你「$\alpha \to \alpha$」，而不能写 $\forall \alpha. \alpha\to\alpha$。

再看 `let id = \x -> x in id` 又如何：`let` 规则把 $\alpha \to \alpha$ 中自由变量 $\alpha$ 全称量化，得模式 $\forall \alpha. \alpha \to \alpha$。**泛化与否，一字之差，就决定了一个函数能不能复用**。这个「唯一窗口」的设计，是整个 HM 的精妙所在：能力有限，但恰好够用，且永远可判定。

## 5 let-泛化：多态从哪来

HM 只允许在 `let x = t1 in t2` 处引入多态。具体规则是：先推断 `t1` 的类型 $T$，把 $T$ 中**不受环境限制的自由类型变量**全称量化，得到**类型模式（type scheme）** $\forall \vec{\alpha}.\; T$，放进上下文供 `t2` 反复以不同实例使用：

$$f = \lambda x.\; x \quad\Longrightarrow\quad f : \forall \alpha.\; \alpha \to \alpha$$

于是 `let id = \x -> x in (id 1, id True)` 才能成立：`id` 既当 `Nat→Nat` 用，又当 `Bool→Bool` 用——因为每处引用都各自实例化 $\alpha$。<span class="marginnote">这里有一处微妙的<strong>值限制（value restriction）</strong>：只对「语法上的值」（lambda 抽象、常量、构造子）做泛化，对会执行副作用的表达式（如引用赋值、异常）不泛化——否则多态会被副作用破坏。OCaml 的 `let x = ref []` 报「weak type」正是这个机制。</span>

值限制的必要性能用一个反例说明。若允许 `let r = ref []` 泛化，则 `r` 既被当作 `ref (int list)` 又当作 `ref (bool list)` 使用——可 `ref` 是可变引用，先塞整数、后当布尔取出来，类型安全瞬间崩盘。**多态与可变状态天然互斥**，这是类型系统史上反复上演的主题（《效应系统》一篇会再相遇）。

## 6 HM 的成就与天花板

HM 推断在实践上已经近乎完美：OCaml、Haskell、F# 让程序员几乎不写类型，同时保有**可判定**与**多项式时间**的推断。但它的天花板同样清晰——只支持 **rank-1 多态**：$\forall$ 只能出现在类型最外层。这意味着「把多态函数作为参数传给另一个函数」（rank-2 及以上）在纯 HM 里写不出来，需要显式标注或语言扩展。

现代语言的解决方案分两派：**隐式**（HM 推断 + 少数扩展）如 Haskell 的 `RankNTypes`、OCaml 的 polymorphic variants；**显式**（用户写 $\forall$）如 Rust 的 `trait` 泛型、Swift 的 protocol。TypeScript 则做了个有趣的取舍：全用 HM 推断但允许类型标注放宽到 `any`，牺牲一点安全性换开发速度。<span class="marginnote">这四行对照也预示了《现代语言实践》一篇的主题：同一个 HM 内核，各家在「推断能力 × 表达力 × 安全性」三个角上各取所爱。</span>

把「推断力」这张谱系拉通，能看清整个领域的版图：

| 系统 | 程序员写类型？ | 推断可判定？ | 多态位置 | 典型语言 |
| --- | --- | --- | --- | --- |
| 简单类型 | 必写 | 可判定 | 无 | 教学语言 |
| HM | 几乎不写 | 可判定、多项式时间 | 仅 let 处、rank-1 | OCaml、Haskell |
| System F | 必写 $\Lambda$ | 不可判定 | 任意位置 | 理论研究 |
| 依赖类型 | 类型=命题，大量书写 | 半可判定 | 任意 + 依赖 | Agda、Idris |

这张表读出的规律很清晰：**推断力越强，越难自动化**。HM 恰好站在「几乎不用写类型」与「可判定」的交叉点上，这是它六十年不衰的根本原因。

## 7 小结

- **HM 推断** = 生成类型约束 + 用**合一算法**解约束，把「写类型」变成「解方程」。
- **合一**（Robinson 1965）找到最一般替换 MGU；出现检查阻止无限类型。
- 推断是「替你写出类型」而检查只是「验证你写对了」——HM 提供了前者。
- **Algorithm W**（Damas–Milner 1982）自底向上遍历语法树，逐步累积替换。
- **let-泛化**是 HM 引入多态的唯一窗口，配**值限制**防止副作用破坏多态。
- HM 只支持 rank-1 多态，推断可判定、多项式时间——这是它长盛不衰的原因。
- System F 推断不可判定，HM 用「限制位置」换回可判定，是最成功的工程折衷。
- 值限制提醒我们：多态与可变状态天然互斥，二者相遇必须精打细算。

在下一节，我们将暂时放下函数与推断，转向**数据**：用**代数数据类型**把积类型、和类型与模式匹配组织成一个整体，为 `map`、`Maybe`、`List` 等真实数据结构立起类型骨架。
