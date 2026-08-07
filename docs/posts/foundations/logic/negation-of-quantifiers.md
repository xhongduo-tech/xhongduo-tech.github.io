---
title: 量词的否定与对偶关系
date: 2026-08-07
---

# 量词的否定与对偶关系

<div class="epigraph">
<p>「并非每一个人都…」的意思是「有一个人不…」。</p>
<footer>—— 量词否定的日常直觉</footer>
</div>

<div class="article-byline">
<p>第一级 · 逻辑学 ｜ 陈波《逻辑学导论》第3章 §3.4 ｜ 2026-08-07</p>
</div>

## 为什么从量词否定开始

上一课的翻译训练留下了一个悬案：否定号遇到量词时会发生什么？「并非所有人都爱逻辑」到底该怎么说？
直觉告诉你它等于「有人不爱逻辑」。本课把这条直觉严格化为**量词的否定律**——命题逻辑德摩根律的量词版本。
理解量词否定，是谓词逻辑推理的钥匙：数学证明里「不存在」「并非所有」「不唯一」的表述，全都要靠它。

## 1 量词的否定律

两条基本定律，是德摩根律在量词层面的翻版：

$$
\neg\forall x\,P(x) \equiv \exists x\,\neg P(x)
$$

$$
\neg\exists x\,P(x) \equiv \forall x\,\neg P(x)
$$

用自然语言读一遍：

- **否定全称 = 存在否定**：「并非所有 $x$ 都满足 $P$」=「存在某个 $x$ 不满足 $P$」。要反驳「所有人都爱逻辑」，只需找出一个不爱逻辑的人。
- **否定存在 = 全称否定**：「不存在 $x$ 满足 $P$」=「所有 $x$ 都不满足 $P$」。说「没有不劳而获的事」，等于说「所有事都不是不劳而获的」。

<span class="marginnote">把德摩根律与量词否定律对照，会发现完美的平行：$\neg(P\land Q) \equiv \neg P \lor \neg Q$ 对应 $\neg\forall x P \equiv \exists x\neg P$，$\neg(P\lor Q) \equiv \neg P \land \neg Q$ 对应 $\neg\exists x P \equiv \forall x\neg P$。只要把 $\forall$ 想成「大合取」（对每个个体都要），把 $\exists$ 想成「大析取」（对某个个体就行），两条定律就自动浮现。</span>

## 2 为什么成立：从有限域到一般域

量词否定律的直觉可以用**有限论域**看清。假设宇宙只有两个个体 $a, b$，那么：

$$
\forall x\,P(x) \;\equiv\; P(a) \land P(b)
, \qquad \exists x\,P(x)
 \;\equiv\; P(a) \lor P(b)
$$

于是「否定全称」就变成了「否定合取」，用德摩根律：

$$
\neg\forall x\,P(x) \equiv \neg(P(a)
 \land P(b)) \equiv \neg P(a)
 \lor \neg P(b) \equiv \exists x\,\neg P(x)
$$

有限论域里，量词否定律完全还原为德摩根律；而一般论域（可能无限）正是把「对每个个体」看作无限合取、「对某个个体」看作无限析取的推广。
这个「量词 = 广义的 ∧/∨」的视角，是整个逻辑分析中最有用的洞见之一。

## 3 双重否定与深入否定

运用量词否定律，可以把否定号**穿过任意多个量词**。否定号每穿过一个量词，量词就翻转一次：

$$
\neg\forall x\,\exists y\,R(x,y) \;\equiv\; \exists x\,\neg\exists y\,R(x,y)
 \;\equiv\; \exists x\,\forall y\,\neg R(x,y)
$$

规则口诀：**否定号向右穿过量词时，$\forall$ 与 $\exists$ 互换，再继续推进到谓词**。
这本质上是一种「对偶」操作：给公式取否定，等价于把每个量词换成它的对偶、把每个联结词换成它的对偶、把谓词取否定。

<strong>对偶关系（duality）</strong>：$\forall$ 与 $\exists$ 互为对偶，正如 $\land$ 与 $\lor$ 互为对偶。一个公式的对偶式，是把其中所有 $\forall \leftrightarrow \exists$、$\land \leftrightarrow \lor$ 互换得到的公式。对偶概念在数学里处处出现——分析里的「上确界/下确界」、范畴论里的「极限/余极限」都是同一结构。

**辨析｜易错点：** 量词否定律的经典误用是**忘记翻转量词**。$\neg\forall x P(x)$ 绝不等于 $\forall x \neg P(x)$！「并非所有人都爱逻辑」不是「所有人都不爱逻辑」——后者强太多了。每次取否定，先问自己：**量词翻了吗？** 翻错了，整个证明就崩了。

## 4 公式解析：否定一个带量词的数学命题

数学里处处需要否定带量词的命题，最经典的例子是「连续性」的否定。连续性的（$\varepsilon$-$\delta$）定义是：

$$
\forall\varepsilon>0\;\exists\delta>0\;\forall x\;\big(|x-a|<\delta \to |f(x)
-f(a)|<\varepsilon\big)
$$

要写它的否定，即「$f$ 在 $a$ 不连续」，按三步走：

- **第一步，加否定号**：在最外层写上 $\neg$。
- **第二步，逐层穿入**：$\neg\forall\varepsilon \Rightarrow \exists\varepsilon\neg$；$\neg\exists\delta \Rightarrow \forall\delta\neg$；$\neg\forall x \Rightarrow \exists x\neg$。
- **第三步，否定内核**：$\neg(P \to Q) \equiv P \land \neg Q$，于是 $|x-a|<\delta$ 保持，$|f(x)-f(a)|<\varepsilon$ 取否。

结果：

$$
\exists\varepsilon>0\;\forall\delta>0\;\exists x\;\big(|x-a|<\delta \land |f(x)
-f(a)|\ge\varepsilon\big)
$$

读作「存在一个 $\varepsilon>0$，对任意 $\delta>0$，都能找到 $x$ 满足 $|x-a|<\delta$ 但 $|f(x)-f(a)|\ge\varepsilon$」——这正是「存在一个震荡点」的严格表述。<span class="marginnote">这个例子的完整链条，是把命题逻辑的否定规则（$\neg(P\to Q)\equiv P\land\neg Q$）与量词否定律组合使用。它说明：要真正掌握否定，必须同时会用<strong>联结词层面的否定</strong>与<strong>量词层面的否定</strong>两套工具。数学分析（第二级《数学分析》）开篇的 $\varepsilon$-$\delta$ 训练，本质就是量词嵌套与量词否定的反复操练。</span>

## 5 例题演练

**例 1**：把「并非所有学生都通过了考试」等价改写成存在形式。

- $\neg\forall x(S(x)\to P(x)) \equiv \exists x(S(x)\land\neg P(x))$——「存在一个学生没通过考试」。否定号穿过全称量词变存在，且内核「若则」变成「且」。

**例 2**：为什么「否定一个含两个量词的公式」要逐个翻转？

- 否定号每穿过一个量词就翻转一次：$\neg\forall x\exists y R(x,y) \equiv \exists x\neg\exists y R(x,y) \equiv \exists x\forall y\neg R(x,y)$。逐层处理，最后推进到谓词。

**例 3**：数学里的「函数不连续」为什么常写成「存在 ε 对任意 δ 存在 x…」？

- 因为连续性的定义是「对任意 ε 存在 δ 对任意 x…」，取否定就变成「存在 ε 对任意 δ 存在 x…」——量词逐层翻转后的结果正是「存在震荡点」的严格表述。

**本节要点自检**：否定穿过量词必翻转（$\forall\leftrightarrow\exists$）；有限论域下它就是德摩根律；翻错量词是经典错误。

## 6 小结

- **量词否定律**：$\neg\forall xP \equiv \exists x\neg P$，$\neg\exists xP \equiv \forall x\neg P$。
- 有限论域下，量词否定律**还原为德摩根律**；一般论域是无限合取/析取的推广。
- 否定号穿过量词时**量词必须翻转**：$\forall \leftrightarrow \exists$。
- **对偶关系**：$\forall$ 与 $\exists$ 互为对偶，正如 $\land$ 与 $\lor$。
- 经典误用：把 $\neg\forall xP$ 写成 $\forall x\neg P$——否定时先检查量词是否翻转。

在下一节，我们处理量词与变元的第三种纠缠：**辖域、自由变元与约束变元**——「$x$ 到底被哪个量词管住」。
