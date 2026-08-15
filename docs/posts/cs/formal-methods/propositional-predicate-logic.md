---
title: 命题逻辑与谓词逻辑
date: 2026-08-07
---

# 命题逻辑与谓词逻辑

<div class="epigraph">
<p>逻辑是思想的外显结构，它告诉我们哪些说法可以成立，哪些说法注定不能成立。</p>
<footer>—— 戈特洛布 · 弗雷格（Gottlob Frege）</footer>
</div>

<div class="article-byline">
<p>第三级 · 形式化方法 ｜ Huth &amp; Ryan《Logic in Computer Science》§1–2 ｜ 2026-08-07</p>
</div>

## 为什么从逻辑开始

形式化方法的核心信念只有一句话：**把「系统是否正确」这个问题，翻译成「一个数学命题是否为真」的问题，再用机械的方法去判定。** 而要翻译任何系统——从一段并发程序到一条芯片总线——你需要的第一个工具，就是把「论证」写成精确符号的语言。这就是逻辑。<span class="marginnote">整个形式化方法专题将在这条主线上展开：命题/谓词逻辑是语言，时序逻辑给语言加入「时间」，模型检测负责自动判定，霍尔逻辑把语言对准程序。今天的这两课是全部地基。</span>

逻辑不是冷冰冰的符号游戏。今天的大模型能理解「如果……那么……」的推理，Transformer 里的注意力机制也在学习句子的逻辑结构；而把「推理」机械化这件事，从布尔代数到 SAT 求解器（本专题第 9 篇），恰恰是计算机最早的动机。<span class="marginnote">布尔 (George Boole) 1854 年《思维定律》把「真/假」代数化，香农 (Claude Shannon) 1938 年把布尔代数接上继电器电路——逻辑由此成为计算机的物理基础，也是本专题最后讲 SAT/SMT 求解器时回来接续的线。</span>

## 1 命题逻辑的语法

**命题（proposition）**：一个有真假含义的陈述句。「今天是星期一」是命题，「你好吗？」不是。不再细分、整体作为最小单位的命题叫**原子命题（atomic proposition）**，记为 $p, q, r, \dots$。

原子命题可以通过**逻辑联结词（connectives）**组合成复合命题。Huth §1 使用五个联结词：

| 联结词 | 记号 | 读法 | 直觉 |
| --- | --- | --- | --- |
| 否定 | $\neg$ | 非 | 「不是……」，一元 |
| 合取 | $\land$ | 且 | 「两者都成立」 |
| 析取 | $\lor$ | 或 | 「至少一个成立」（可兼或） |
| 蕴含 | $\to$ | 蕴含 / 推出 | 「如果……那么……」 |
| 等价 | $\leftrightarrow$ | 当且仅当 | 「两边同真同假」 |

有了联结词还不够，还得规定**什么组合是合法的**。**合式公式（well-formed formula, WFF）**由归纳定义给出：原子命题是公式；若 $\phi, \psi$ 是公式，则 $\neg\phi$、$(\phi \land \psi)$、$(\phi \lor \psi)$、$(\phi \to \psi)$、$(\phi \leftrightarrow \psi)$ 都是公式；除此外无他。<span class="marginnote">注意这是「最小不动点」式的归纳定义——它保证每个公式都能唯一地分解成一棵语法树。你在《编译原理》里学的递归下降、语法树，根子上就是这个定义。</span>

**辨析｜易错点：** 数学里「或」分可兼（inclusive）与排斥（exclusive），命题逻辑的 $\lor$ 是**可兼或**，即「$p \lor q$」允许两者同真。「鱼香肉丝或宫保鸡丁」这种二选一要用 $(p \land \neg q) \lor (\neg p \land q)$ 表达。

## 2 命题逻辑的语义：真值表

语法规定「长什么样」，语义规定「什么意思」。一个**赋值（valuation）**给每个原子命题指定真假；复合公式的真假由真值表逐步算出。核心的五张表：

$$
\begin{array}{c|cc|ccccc}
p & q & p \land q & p \lor q & p \to q & p \leftrightarrow q \\
\hline
\mathrm{T} & \mathrm{T} & \mathrm{T} & \mathrm{T} & \mathrm{T} & \mathrm{T}\\
\mathrm{T} & \mathrm{F} & \mathrm{F} & \mathrm{T} & \mathrm{F} & \mathrm{F}\\
\mathrm{F} & \mathrm{T} & \mathrm{F} & \mathrm{T} & \mathrm{T} & \mathrm{F}\\
\mathrm{F} & \mathrm{F} & \mathrm{F} & \mathrm{F} & \mathrm{T} & \mathrm{T}
\end{array}
$$

**重点：蕴含 $p \to q$ 只在「前件真而后件假」时为假。** 这常被初学者误解为「蕴含就是因果」，其实它是材料蕴含（material implication）——「前件为假时整个蕴含为真」不是语义怪癖，而是保证「**从假的前提推不出反例**」这个原则。说「如果今天是周六，我们就爬山」，如果今天不是周六，这句话并没有被违反。

由真值表引出两个贯穿全书的概念：

**可满足（satisfiable）**：存在某个赋值使公式为真；**永真式 / 重言式（tautology）**：所有赋值下都为真，记作 $\models \phi$。<span class="marginnote">「$\models$」与后文的语义蕴含是同一个符号：单参数时表示「恒真」，多参数时表示「前提推出结论」。它是语义层面的「真」，对应下一节语法层面的证明「$\vdash$」。</span>

## 3 语义蕴含、等价与自然演绎

**语义蕴含（semantic entailment）**：公式集 $\Phi$ 蕴含公式 $\phi$，记作 $\Phi \models \phi$，意思是「使 $\Phi$ 中每个公式都为真的赋值，必然使 $\phi$ 为真」——也就是说，**前提全真时结论不可能假**。<span class="marginnote">这与第 6 篇《程序验证与霍尔逻辑》的可靠性/完备性、以及第 9 篇 SAT 求解都直接挂钩：SAT 判定可满足性，「蕴含」不过是在所有赋值上做「若前提真则结论真」的检查。</span>

**逻辑等价（logical equivalence）**：$\phi \equiv \psi$ 当且仅当 $\phi \models \psi$ 且 $\psi \models \phi$，即两者在所有赋值下同真同假。等价式可以当重写规则用：

$$
\begin{aligned}
\neg(p \land q) &\equiv \neg p \lor \neg q \qquad \text{（德摩根律）}\\
p \to q &\equiv \neg p \lor q \qquad \text{（蕴含定义）}\\
p \lor (q \land r) &\equiv (p \lor q) \land (p \lor r) \qquad \text{（分配律）}
\end{aligned}
$$

这些等价式可以当作代数恒等式随手改写公式，其中几条值得单独记住：

| 等价式 | 名称 |
| --- | --- |
| $p \lor \neg p$ | 排中律（law of excluded middle） |
| $p \land \neg p$ | 矛盾律（law of contradiction） |
| $\neg\neg p \equiv p$ | 双重否定消去 |
| $p \to q \equiv \neg q \to \neg p$ | 逆否命题（contraposition） |
| $p \leftrightarrow q \equiv (p \to q) \land (q \to p)$ | 等价的分解 |

语义告诉我们「什么为真」，但**人类和机器要能一步步地『写』出推理**。Huth §1.2 给出**自然演绎（natural deduction）**系统：由引入规则（introduction rules）与消去规则（elimination rules）组成，每条规则对应一个联结词的「如何得到它」与「如何使用它」。例如合取引入 $\dfrac{\phi \quad \psi}{\phi \land \psi}$、蕴含消去（即**分离规则 modus ponens**）$\dfrac{\phi \quad \phi \to \psi}{\psi}$。当存在一棵从前提 $\Phi$ 到结论 $\phi$ 的推导树时，记作 $\Phi \vdash \phi$。

**辨析｜易错点：** 语义的 $\models$ 与语法的 $\vdash$ 是两套语言。**可靠性（soundness）**断言「能证明的都真」：$\Phi \vdash \phi \Rightarrow \Phi \models \phi$；**完备性（completeness）**断言「真的都能证明」：$\Phi \models \phi \Rightarrow \Phi \vdash \phi$。对命题逻辑，两者都成立——这不是平凡结论，而是哥德尔完备性定理（1930）说的「逻辑的证明能力与语义吻合」。

## 4 谓词逻辑：量词登场

命题逻辑的原子命题是不可再分的，这限制了它的表达力——「所有偶数都能被 2 整除」无法用 $p \land q$ 这类式子写出，因为它在谈论「所有」。**谓词逻辑（first-order / predicate logic）**在命题逻辑之上加入了：**个体（terms）**、**谓词（predicates）**和**量词（quantifiers）**。

**量词（quantifier）**两个：全称量词 $\forall$（对论域中所有个体成立）与存在量词 $\exists$（论域中存在个体成立）。例如「每个偶数都是 2 的倍数」写成 $\forall x\,(E(x) \to M(x,2))$，其中 $E, M$ 是谓词；「存在素数是偶数」写成 $\exists x\,(P(x) \land E(x))$。<span class="marginnote">注意两句的形态差别：全称句的躯体是蕴含 $E(x) \to M(x,2)$，存在句的躯体是合取 $P(x) \land E(x)$。把这两类「自然语言翻译模式」记牢，后面翻译规格说明（specification）时能少犯一半错误。</span>

**自由变量与约束变量**：$x$ 出现在量词作用域内叫**约束（bound）**，否则叫**自由（free）**。只有闭公式（无自由变量）才有确定真假；含自由变量的公式叫**开公式**，它更像一个「模板」，在语义学里要通过赋值把自由变量绑定到个体。<span class="marginnote">「约束变量」这个名字正是《编译原理》里变量作用域、以及 λ 演算里绑定子的原型——词法作用域、捕获、替换，都是同一个思想的移植。</span>

**辨析｜易错点：量词的顺序不能交换。** $\forall x \exists y\,(y > x)$ 表示「每个数都有比它大的数」（实数域下为真）；而 $\exists y \forall x\,(y > x)$ 表示「存在一个数比所有数都大」（为假）。**顺序不同，真值可以完全不同**——这是谓词逻辑里最高频的陷阱，翻译规格、书写断言时务必警惕。

## 5 公式解析：德摩根律到底在说什么

选取一条贯穿本课、也贯穿整个形式化方法的等价式做拆解：

$$
\neg(p \land q) \equiv \neg p \lor \neg q
$$

对这条式子做三步拆解：

- **第一步，读符号**：左边读作「并非 $p$ 且 $q$」，即「$p$ 和 $q$ 不都成立」；右边读作「非 $p$ 或非 $q$」，即「$p$、$q$ 至少有一个不成立」。
- **第二步，验证直觉**：两种说法描述的是同一个事实——「至少有一个是假的」。这正是德摩根律的内容：**否定一个合取，等于析取各自的否定**。这对应集合论的补集对交/并的分配（第二级《离散数学》与《集合》专题都会再见），也是数字电路里「与非门 = 或非门」的代数来源。
- **第三步，机械化**：验证永真性不需要直觉，枚举 4 种赋值即可：$p,q \in \{\mathrm{T},\mathrm{F}\}$。当 $p=\mathrm{T},q=\mathrm{F}$ 时左边为 $\neg \mathrm{F}=\mathrm{T}$，右边为 $\neg\mathrm{T} \lor \neg\mathrm{F} = \mathrm{F} \lor \mathrm{T} = \mathrm{T}$，两边一致；其余三种赋值同理。**真值表穷举是语义层面最朴素、也最可靠的验证手段**——它正是第 9 篇 SAT 求解器要规模化（scale up）的东西。

这条等价式也示范了「蕴含」如何降格为「可满足」：$p \to q \equiv \neg p \lor q$，于是「验证一个蕴含式恒真」变成了「验证一个合取式不可满足」，把证明问题归约为搜索问题——这就是本专题第 9 篇「SAT/SMT」的伏笔。

## 6 小结

- **命题逻辑**由原子命题与五个联结词 $\neg, \land, \lor, \to, \leftrightarrow$ 组成；**合式公式**由归纳定义严格界定。
- **语义**由真值表给出；**可满足性**（至少一个赋值使其真）与**永真性**（所有赋值使其真）是两条主线，$\models$ 与 $\vdash$ 分别在语义层与证明层标记「成立」，可靠性 + 完备性把两层焊在一起。
- **蕴含**是材料蕴含：只在「前件真、后件假」时为假；把推理机械化要用**自然演绎**这类证明系统。
- **谓词逻辑**增加个体、谓词与量词 $\forall, \exists$；**全称配蕴含、存在配合取**是自然语言翻译的两大模板，量词顺序不可交换。
- 德摩根律 $\neg(p \land q) \equiv \neg p \lor \neg q$ 示范了「证明问题可归约为可满足性判定」，指向第 9 篇的 SAT 求解。

在下一节，我们将给这套逻辑加上「时间」——命题不再只是「此刻为真」，而是「在某个未来状态为真」或「沿着所有可能路径都为真」，这就是**时序逻辑 LTL/CTL**。
