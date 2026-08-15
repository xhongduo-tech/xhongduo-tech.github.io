---
title: 等式逻辑的完备性
date: 2026-08-07
---

# 等式逻辑的完备性

<div class="epigraph">
<p>凡在一切模型中都为真的等式，都能从公理机械地推出——等式逻辑是完整自洽的。</p>
<footer>—— 阿尔弗雷德 · 塔斯基（Alfred Tarski）学派等式逻辑传统</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛代数（万有代数） ｜ Burris &amp; Sankappanavar《A Course in Universal Algebra》第III章 §8 ｜ 2026-08-07</p>
</div>

## 为什么从「等式逻辑的完备性」继续

Birkhoff HSP 定理回答的是**语义**问题：簇在模型层面是什么。这一篇回答**语法**问题：给定等式集 $\Sigma$，哪些等式能从 $\Sigma$ **机械地推出**？这需要给「推出」定一套证明规则。令人欣慰的是，这套规则只有五条，朴素得像加减法——而它居然**完备**：凡是被 $\Sigma$ 的一切模型共同满足的等式，都逃不出这五条规则的推导。这就是**等式逻辑的完备性定理**。它是「语法 = 语义」在等式层面的完美复刻，也是计算机科学里重写系统、项重写与自动定理证明的数学基石。

## 1 等式逻辑的五条推理规则

给类型 $\tau$、变量集 $X$。**等式逻辑**的证明规则作用在等式 $s \approx t$ 上，共五条：

1. **自反（reflexivity）**：$t \approx t$ 恒可推出。
2. **对称（symmetry）**：由 $s \approx t$ 推出 $t \approx s$。
3. **传递（transitivity）**：由 $s \approx t$ 与 $t \approx u$ 推出 $s \approx u$。
4. **代入（substitution）**：由 $s \approx t$ 推出 $s' \approx t'$，其中 $s', t'$ 是把 $s, t$ 中的变量**一致地**替换为任意项的结果。<span class="marginnote">「一致地」三个字是灵魂：替换必须同时作用于 $s$ 与 $t$ 中所有同名变量，例如由 $x + y \approx y + x$ 推出 $(z \cdot z) + w \approx w + (z \cdot z)$——把 $x$ 换成 $z \cdot z$、$y$ 换成 $w$，两边同步执行。</span>
5. **替换（congruence / replacement）**：由 $s_i \approx t_i$（$i = 1, \dots, n$）推出 $f(s_1, \dots, s_n) \approx f(t_1, \dots, t_n)$，对任意 $n$ 元运算符号 $f$。<span class="marginnote">这条规则对应同态/同余的「一致作用」：把等式逐个嵌进运算符号的各个参数位置。名字「congruence」提醒我们它正是同余关系对运算封闭的那条性质的语法版本。</span>

由 $\Sigma$ 能**推出（derive）**等式 $s \approx t$，记 $\Sigma \vdash s \approx t$，指存在有限步应用上述规则、以 $\Sigma$ 中等式与自反式为公理、终于 $s \approx t$ 的推导序列。<span class="marginnote">「有限步」是逻辑的底线——无穷推导不被承认。这与第一级《集合》里「自然数归纳」、第二级《数理逻辑》里「证明是有限对象」的立场完全一致。</span>

## 2 语义蕴含与语法推演的对偶

现在两套记号同台：

- **语义蕴含** $\Sigma \models s \approx t$：每个满足 $\Sigma$ 的代数都满足 $s \approx t$。这是「世界」层面的事。
- **语法推演** $\Sigma \vdash s \approx t$：从 $\Sigma$ 出发用五条规则能推出来。这是「纸面」层面的事。

两者方向自然地有一个成立：**可靠性与完备性**的两个半句。

- **可靠性（soundness）**：若 $\Sigma \vdash s \approx t$，则 $\Sigma \models s \approx t$。即五条规则**不会推出假话**——规则每条都保真，组合起来也保真。证明是逐一验证：自反、对称、传递是等价的自身性质；代入与替换在任意模型里都保持相等。
- **完备性（completeness）**：若 $\Sigma \models s \approx t$，则 $\Sigma \vdash s \approx t$。即五条规则**不会漏掉真话**——凡语义上必然的等式，语法上都推得出。这是深刻的一半。

**定理（等式逻辑的完备性，Birkhoff 1935）**：对任意等式集 $\Sigma$ 与等式 $s \approx t$：

$$\Sigma \models s \approx t \iff \Sigma \vdash s \approx t$$

## 3 完备性的证明思路

完备性（$\Leftarrow$ 的反方向，即 $\models \Rightarrow \vdash$）的证明可以用自由代数干净地收束。思路分三步：

1. **构造典型模型**：取变量集 $X$ 为「足够大」的集合，在簇 $\mathcal{V} = K(\Sigma)$ 中取自由代数 $\mathbf{F}_{\mathcal{V}}(X)$。
2. **搭起推导**：对每个不在 $\Sigma \vdash$ 意义下相等的项对，设法在 $\mathbf{F}_{\mathcal{V}}(X)$ 里把 $X$ 赋值成「区分它们」的向量——这一步正是用自由代数的万有性质：任何赋值都是唯一同态，而 $\mathbf{F}_{\mathcal{V}}(X)$ 的元素按「$\Sigma$ 可证相等」分类，恰是项代数的商。
3. **反证收束**：若 $\Sigma \not\vdash s \approx t$，则在 $\mathbf{F}_{\mathcal{V}}(X)$ 中 $s, t$ 落在不同的等价类，于是存在赋值使 $s^{\mathbf{A}}[v] \neq t^{\mathbf{A}}[v]$，即 $\Sigma \not\models s \approx t$。逆否得证。

**重点：自由代数在完备性证明里同时扮演「反例生成器」与「模型」。** 它把「没有推导」翻译成「存在反例」——语法的漏洞，被语义的反例精确填补。这正是「语法 = 语义」最深刻的体现。<span class="marginnote">这套「用自由对象造反例」的方法不限于等式：模型论里 Henkin 构造、代数数论里用分裂环造反例，都是同一精神的远亲。理解它，等于理解整个逻辑完备性证明的通用骨架。</span>

## 4 公式解析：代入规则与替换规则的分工

两条最容易被混淆的规则，放在一起看它们的差异：

$$
\dfrac{s \approx t}{s'\approx t'}\ \text{（代入）} \qquad
\dfrac{s_1 \approx t_1 \quad \cdots \quad s_n \approx t_n}{f(s_1,\dots,s_n) \approx f(t_1,\dots,t_n)}\ \text{（替换）}
$$

- **第一步，看代入规则**：它作用在**变量**上。$s', t'$ 由把 $s, t$ 中的变量换为任意项得到。等式的「外形」可以任意放大——把 $x$ 换成巨型项。
- **第二步，看替换规则**：它作用在**运算符号**上。多个等式分别从各个参数位置「焊进」同一个 $f$。外形不变，只是把 $f$ 的参数逐个替换。
- **第三步，对比本质**：代入改变变量绑定（全称实例化），替换改变结构骨架（子项替换）。前者是「量的放大」，后者是「位置的渗透」。
- **第四步，合起来**：两条规则联手，才能模拟「任意的项重写」。重写系统里的一次重写步 = 若干次代入 + 若干次替换的组合。这也是为什么项重写理论以等式逻辑为公理化基础。

## 5 辨析｜易错点：完备性与一致性的边界

**辨析｜易错点：** 等式逻辑的完备性有不少「看起来显然、其实微妙」的边界：

- **完备性 ≠ 平凡成立**：$\Sigma \models s \approx t$ 是语义事实，涉及**所有**（可能无穷多个、任意大的）模型；完备性说五条规则就能穷尽这些语义事实——这绝非显然。
- **可靠性不能被完备性替代**：完备性给「够用」，可靠性给「不越界」。没有可靠性，规则可能推出假等式，完备性就失去意义。
- **等式逻辑与一阶逻辑的关系**：等式逻辑是一阶逻辑（第二级《数理逻辑》）里「纯等式语句」片段的特例。一阶逻辑完备性（Gödel）蕴含等式逻辑完备性；反过来等式逻辑的完备性更早（Birkhoff 1935 亦发表于一阶完备性前夜），且不需要选择公理级别的大炮，证明更直接。
- **「有限步」不可省**：若允许无穷推导，「完备性」会退化为空洞的套套逻辑。所以证明序列必须有限——这也是计算理论里「可判定性」的伏笔。

## 6 小结

- 等式逻辑有五条规则：**自反、对称、传递、代入、替换**。
- **可靠性**：$\Sigma \vdash s \approx t \Rightarrow \Sigma \models s \approx t$，规则不说谎。
- **完备性**：$\Sigma \models s \approx t \Rightarrow \Sigma \vdash s \approx t$，规则不漏真。
- **Birkhoff 完备性定理**：$\Sigma \models s \approx t \iff \Sigma \vdash s \approx t$