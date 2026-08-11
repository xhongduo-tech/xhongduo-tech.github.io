---
title: 类型推导
date: 2026-08-11
---

# 类型推导

<div class="epigraph">
<p>良类型的程序不会出错。</p>
<footer>—— 罗宾 · 米尔纳（Robin Milner），1978 年</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 程序设计语言理论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从类型推导开始

《类型系统与类型规则》里我们学会了 $\Gamma \vdash t: T$——但那里要求程序员把类型写全。真实语言凭什么能让 ML、Haskell 的程序员**几乎不写类型**？答案是**类型推导（type inference）**：从项的结构反推出它唯一（最一般）的类型。这是米尔纳 1970 年代为 ML 设计的 Hindley-Milner 系统，皮尔斯在《类型与程序设计语言》§22 把它拆成两半：**约束生成**（类型规则机械化地把类型变出来）+ **统一化**（解类型方程）。本课讲这两半，以及一个极其精巧的设计决策：**let 多态**。

## 1 类型重建问题：从标注回到规则

回顾应用规则——它原来长这样：$\frac{\Gamma \vdash t_1 : T_1 \to T_2 \quad \Gamma \vdash t_2 : T_1}{\Gamma \vdash t_1\;t_2 : T_2}$。推导时，$T_1, T_2$ 是「必须对上」的未知量。类型重建（type reconstruction）的思路朴素而有效：**遇到未知类型就造一个新类型变量，把「必须对上」写成方程**。

以项 $\lambda f.\, f\;\overline{0}$ 为例，走一遍：

1. 给 $f$ 一个未知类型 $X_1$，给 $\overline{0}$ 已知类型 $\texttt{Nat}$，给 $f\;\overline{0}$ 一个结果类型 $X_2$。
2. 应用规则要求 $f : X_1$ 必须是「某输入 $\to X_2$」，即 $X_1 = \texttt{Nat} \to X_2$。
3. 于是 $\lambda f.\, f\;\overline{0} : X_1 \to X_2 = (\texttt{Nat} \to X_2) \to X_2$。

$X_2$ 没被约束死，结论是**一族类型**：对任何 $X_2$，$(\texttt{Nat}\to X_2)\to X_2$ 都成立。推导算法要回答的正是：**这一族类型里，哪一个是最一般的（主类型）？**

## 2 统一化：解类型方程

类型推导的下半身是**统一化（unification）**：给定类型方程 $T_1 \sim T_2$（读作「$T_1$ 与 $T_2$ 必须相等」），求一个**置换（substitution）**$\sigma$（一个「类型变量 → 类型」的替换表），使得 $\sigma(T_1) = \sigma(T_2)$。Robinson 的统一化算法递归分解：

$$\texttt{Nat} \sim \texttt{Nat} \Rightarrow \text{成功};\qquad T_1\to T_2 \sim T'_1 \to T'_2 \Rightarrow \text{分别统一 } T_1\!\sim\! T'_1 \text{ 与 } T_2\!\sim\!T'_2$$
$$X \sim T \text{ 或 } T \sim X \Rightarrow \text{记录 } X \mapsto T,\ \text{若 } X \text{ 不在 } T \text{ 中出现}$$

**重点：** 若求出的置换不只一种，取**最一般合一子（most general unifier, MGU）**——其他一切解都能由它进一步实例化得到。<span class="marginnote">统一化的「出现检查（occur check）」至关重要：解方程 $X \sim X \to \texttt{Bool}$ 时若直接写 $X \mapsto X\to\texttt{Bool}$ 会得到无穷展开的类型，编译器会当场转晕。出现检查就是「不许把变量自身塞进自己的展开」——与《递归类型》里负递归要小心的原因如出一辙。</span>**主类型存在性定理**保证：良类型项必有主类型，且主类型唯一（至多差一个重命名）。

## 3 let 多态：推导皇冠上的宝石

朴素推导有个致命限制：$\lambda f.\, f\;(\overline{0})\;(f\;\texttt{true})$ 里 $f$ 先用成「吃 Nat」再用成「吃 Bool」，而应用规则要求 $f$ 的类型处处一致——于是推导失败。可直觉上 $f$ 明明可以是「对任何类型都能用」的恒等函数。Hindley-Milner 的解法惊世地简单：**只有 let 绑定的变量才允许多态**。

$$\frac{\Gamma \vdash t_1 : T_1 \quad \Gamma, x: \forall \overline{X}.\, T_1 \vdash t_2 : T_2}{\Gamma \vdash \texttt{let } x = t_1 \texttt{ in } t_2 : T_2}
\quad \text{（} \overline{X} \text{ 为 } T_1 \text{ 中未受约束的类型变量）}$$

**let 绑定的变量被全称泛化（generalize）成 $\forall X.\,T$，使用时可各自实例化成不同类型；λ 绑定的变量保持单态（monomorphic），处处一致。** 这就是「let 多态（let-polymorphism）」——正是它让 `let id = fun x -> x in (id 0, id true)` 通过而 `(fun id -> (id 0, id true)) (fun x -> x)` 不通过（后者是 λ 绑定）。**辨析｜易错点：** ML 的**值限制（value restriction）**规定只有**语法值**（λ、常量、构造子应用）才能泛化——因为可变引用让泛化不再安全：若 `let r = ref []` 泛化成 $\forall X.\, \texttt{ref } X$，往 `r` 里塞了 `Nat` 再读成 `Bool` 就会在类型系统的眼皮底下炸开。Haskell 的**单态限制（monomorphism restriction）**也是同一类取舍的变体。<span class="marginnote">「只泛化 let、不泛化 λ」常被初学 ML 者吐槽为「魔法区别」，但它是<strong>健全性</strong>的必要条件：没有值限制，HM 系统就会接受会在运行时出错的程序——这恰好是米尔纳格言「良类型程序不会出错」要誓死捍卫的那条线。</span>

## 4 算法 W 与主类型

把约束生成与统一化串成单一算法，就是 Damas–Milner 的 **算法 W**：输入环境与项，输出「类型 + 置换」。它逐结点走：

1. 变量：查环境；若环境给的是 $\forall X.\,T$，则实例化（用新变量替换 $X$）。
2. 应用：递归推 $t_1, t_2$，得 $T_1, T_2$，用**新变量** $X$ 统一 $T_1$ 与 $T_2 \to X$。
3. 抽象：推函数体，把参数类型取成体里的那个变量。
4. let：推 $t_1$，对其主类型做**泛化**，加入环境再推 $t_2$。

每一步产生的方程在统一化中就地求解，产生的置换向前传播——**整个算法是确定性的，且对良类型项总能终止并给出主类型**。这就是现代强类型函数式语言「零标注」背后的引擎。

## 5 公式解析：主类型从一次推导中长出来

我们把 §1 的例子做完整，看主类型如何被唯一锁定：

$$
\lambda f.\, f\;\overline{0} \quad:\quad (\texttt{Nat} \to X_2) \to X_2
$$

- **变量 $X_1, X_2$**：$f$ 的类型 $X_1$ 与应用的返回类型 $X_2$ 都是推导引入的**未知量**，分别代表「f 的类型」与「结果类型」。
- **方程 $X_1 = \texttt{Nat} \to X_2$**：来自应用规则——$f$ 必须能吃掉 $\overline{0}:\texttt{Nat}$。这条方程把 $X_1$ 锁定成箭头类型。
- **未约束的 $X_2$**：没有其它方程碰它，于是它留在结果里，成为「自由」参数——主类型是 $(\texttt{Nat}\to X_2)\to X_2$，其中 $X_2$ 可取任何类型。
- **为什么是「最一般」**：任何「更具体」的类型（如 $(\texttt{Nat}\to\texttt{Bool})\to\texttt{Bool}$）都能由 $X_2 \mapsto \texttt{Bool}$ 实例化得到；反之不成立。**主类型 = 未被多余约束削弱的那个解**——这就是推导「不多不少、恰好给出最通用签名」的含义。

## 6 小结

- **类型推导** = 约束生成（规则机械化地引入类型变量）+ **统一化**（解方程）。
- **统一化**递归分解类型、记录变量→类型替换，并做**出现检查**防止无穷类型；解取**最一般合一子**。
- 良类型项必有**主类型**；**算法 W**（Damas–Milner）把整个流程实现为确定性算法。
- **let 多态**是 HM 的关键：只有 let 绑定被全称泛化，λ 绑定保持单态；**值限制/单态限制**是为健全性付出的代价。
- 推导让强类型语言摆脱标注负担，是 ML、Haskell、OCaml、Rust 的日常体验。

在下一节，我们把「多态」本身升格为语言里的头等公民——类型上的抽象（$\Lambda X.t$）与全称类型（$\forall X.T$），这就是 **System F 参数化多态**。
