---
title: 标准部分映射
date: 2026-08-07
---

# 标准部分映射

<div class="epigraph">
<p>每个有限超实数都悬停在一个标准实数的晕圈之中，那个标准实数就是它的灵魂。</p>
<footer>—— 彼得 · 卢布（Peter A. Loeb，非标准测度论奠基人）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非标准分析 ｜ R. Goldblatt《Lectures on the Hyperreals》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从标准部分映射开始

我们已经拥有两样法宝：会「翻译」的转移原理，和会「比较」的无穷小语言。但还缺一件关键工具——**把超实数结论送回标准实数的出口**。毕竟非标准分析的全部目的，是证明关于标准实数的定理；而无穷小 $dx$、无穷大 $\omega$ 再漂亮，最终都要化成一个标准实数作为答案。这个出口就是**标准部分映射**。

沿用上一座城的比喻：城中心是标准实数，每个标准点 $r$ 周围悬着一圈无穷小尘埃（单子 $\mu(r)$）。标准部分映射做的事是**把每个单子整体投影回城中心**：尘埃的每一粒，都塌缩成它们共同的圆心。数学上，$\operatorname{st}$ 就是商映射 ${}^*\!\mathbb{R} \to \mathbb{R}$ 对「$\approx$ 等价类」取代表元——它把无穷多个彼此只差无穷小的点认作同一个标准点。<span class="marginnote">这个「投影」是分析学里最朴素的舍入：浮点运算里把一个实数舍入到最近的机器数，是它的离散表亲；而标准部分映射是把无穷小误差全部抹平、保留「真实值」的连续版舍入。后面的微积分定理证明里，你会一次次看到它出场。</span>

## 1 存在唯一性：每个有限超实数都有标准部分

**核心概念：标准部分（standard part）**。设 $x$ 是有限超实数，则存在**唯一**的标准实数 $\operatorname{st}(x)$，使得

$$x - \operatorname{st}(x) \text{ 是无穷小}, \qquad \text{即} \quad \operatorname{st}(x) \approx x$$

记作 $\operatorname{st}(x)$（有时也写作 ${}^\circ x$ 或 $\operatorname{sh}(x)$，shadow，取自「影子」一词）。若 $x$ 是无穷大，则 $\operatorname{st}(x)$ 无定义。<span class="marginnote">名字的来历：非标准分析创始文献里常把 $\operatorname{st}(x)$ 叫 $x$ 的「影子」（shadow）——$x$ 在标准平面上的投影。中文译作「标准部分」，点明它是「$x$ 里属于标准世界的那一部分」。</span>

**唯一性**容易证：若 $r, s$ 都是标准实数且 $x - r$、$x - s$ 都是无穷小，则 $r - s = (x - s) - (x - r)$ 是无穷小之差，仍是无穷小；但 $r - s$ 是标准实数，一个标准实数若是无穷小就只能等于 $0$，故 $r = s$。

**存在性**需要实数完备性。思路如下：考虑集合

$$L = \{r \in \mathbb{R} \mid r \leq x\}$$

由于 $x$ 有限，$L$ 非空（有下界）且有上界，由确界原理它有一个实数上确界。**重点：$\operatorname{st}(x) = \sup L$ 就是我们要的标准部分。** 证明的关键一步是：对任何标准实数 $r > 0$，$x$ 必然落在 $(\sup L - r,\, \sup L + r)$ 内——否则会与「$\sup L$ 是 $L$ 的最小上界」矛盾。于是 $|x - \sup L| \lt  r$ 对每个正实数 $r$ 成立，即 $x - \sup L$ 是无穷小。<span class="marginnote">这里第一次用到 $\mathbb{R}$ 的<strong>完备性</strong>：确界原理。它正是在《数学分析》里被当作实数公理的那条性质，也是超实数里「被转移原理挡在外面」的那条性质——于是存在性证明必须回到标准实数 $\mathbb{R}$ 来做，不能指望 ${}^*\!\mathbb{R}$ 自带完备性。</span>

## 2 标准部分的运算性质

标准部分映射是「从超实数回标准实数」的代数同态（在有限范围内）。设 $x, y$ 有限，$r$ 为标准实数，则：

| 性质 | 公式 | 条件 |
| --- | --- | --- |
| 恒等 | $\operatorname{st}(r) = r$ | $r$ 标准 |
| 幂等 | $\operatorname{st}(\operatorname{st}(x)) = \operatorname{st}(x)$ | $x$ 有限 |
| 加法 | $\operatorname{st}(x + y) = \operatorname{st}(x) + \operatorname{st}(y)$ | $x, y$ 有限 |
| 减法 | $\operatorname{st}(x - y) = \operatorname{st}(x) - \operatorname{st}(y)$ | $x, y$ 有限 |
| 乘法 | $\operatorname{st}(xy) = \operatorname{st}(x)\operatorname{st}(y)$ | $x, y$ 有限 |
| 除法 | $\operatorname{st}(x/y) = \operatorname{st}(x)/\operatorname{st}(y)$ | $\operatorname{st}(y) \neq 0$ |
| 保序 | $x \leq y \implies \operatorname{st}(x) \leq \operatorname{st}(y)$ | $x, y$ 有限 |

这七条性质合起来，保证 $\operatorname{st}$ 在有限超实数范围内表现得像一个「连续的同态」。要特别强调的是：**$\operatorname{st}$ 的定义域不是整个 ${}^*\!\mathbb{R}$，而只是有限超实数之集 $\operatorname{Fin}({}^*\!\mathbb{R})$**。无穷大被挡在门外，这是「有限」条件的分量所在。

**公式解析：加法如何穿透无穷小。** 把 $x = \operatorname{st}(x) + \varepsilon$，$y = \operatorname{st}(y) + \delta$ 代入：

$$x + y = \bigl(\operatorname{st}(x) + \operatorname{st}(y)\bigr) + (\varepsilon + \delta)$$

因为 $\varepsilon + \delta$ 是无穷小（两个无穷小之和仍是无穷小），所以 $x + y$ 与 $\operatorname{st}(x) + \operatorname{st}(y)$ 之差是无穷小。由唯一性，标准部分就是 $\operatorname{st}(x) + \operatorname{st}(y)$。乘法同理，只是多一项 $\varepsilon\delta$——它比 $\varepsilon$ 还低一阶，同样是无穷小。**整个性质的证明套路是固定的：拆成「标准部分 + 无穷小」，再验证「无穷小部分仍然无穷小」**。<span class="marginnote">这种「拆分—验证」二步法是全部非标准证明的模板，往后会不断复现：求导数、算极限、证连续性，都是先把对象拆成标准部分与无穷小，再让无穷小互相抵消或塌缩。</span>

## 3 用标准部分重写极限

标准部分最重要的用武之地，是把「极限」翻译成一句无 $\varepsilon$ 的话。

**重点：函数极限的非标准刻画（预告）**。设 $f$ 是标准实函数，$a, L$ 是标准实数，则

$$\lim_{x \to a} f(x) = L \iff \forall x \approx a,\ x \neq a: \quad f(x) \approx L$$

即：**$x$ 无限接近（但不等于）$a$ 时，$f(x)$ 无限接近 $L$**。这里 $f(x)$ 需要被理解成超实数上的值——把 $f$ 的序列表达成 $\bar{f} = [f(a_n)]$，或者直接用非标准延拓 ${}^*f$，两者是一回事。<span class="marginnote">这句话是整个非标准分析最有感染力的一行：标准分析的 $\varepsilon$–$\delta$ 像层层剥笋，而这句断言一次到位。它的严格证明（为什么与 $\varepsilon$–$\delta$ 等价）正是下一篇《连续性与极限的非标准刻画》的主题。</span>

**数列极限同理**：$\lim_{n \to \infty} a_n = L$ 当且仅当对每个无穷大超自然数 $N$，都有 $a_N \approx L$。注意这里 $a_N$ 的下标 $N$ 可以是无穷大——序列的「第无穷项」在超自然数框架下是合法的对象。

**导数的新写法**：导数 $f'(a) = \lim_{h \to 0} \frac{f(a+h)-f(a)}{h}$ 可以写成：对任何非零无穷小 $dx$，

$$f'(a) = \operatorname{st}\!\left(\frac{f(a + dx) - f(a)}{dx}\right)$$

只要右端的标准部分存在且不依赖 $dx$ 的具体选择。这就是莱布尼茨当年梦寐以求的写法：先算差商，再「丢掉无穷小尾巴」——差别在于，如今每一步都有严格的逻辑支撑。<span class="marginnote">对照标准定义：差商 $\frac{f(a+h)-f(a)}{h}$ 在 $h$ 趋近 0 时的极限。两者完全等价，但非标准写法把「极限」换成「先取任意无穷小、再取标准部分」，把「存在 $\delta$」换成了「所有足够小的扰动」，分析难度骤降。</span>

注意这里有一个隐藏的微妙点：「不依赖 $dx$ 的具体选择」不能省。若 $f'(a)$ 存在，则任意两个非零无穷小 $dx_1, dx_2$ 给出的差商标准部分相同；反过来，若存在某个非零无穷小 $dx$ 让标准部分等于 $L$，还不能立刻断言 $f'(a) = L$——必须对所有 $dx$ 一致。这个「对所有无穷小一致」的要求，正是标准定义里「对任意 $h \to 0$ 的路径一致」的非标准版本。

## 4 实战：用标准部分算一个导数

理论的试金石是动手。取 $f(x) = x^2$，$a = 3$，任取非零无穷小 $dx$，按新写法算：

$$\frac{f(3 + dx) - f(3)}{dx} = \frac{(3 + dx)^2 - 9}{dx} = \frac{6dx + dx^2}{dx} = 6 + dx$$

于是

$$f'(3) = \operatorname{st}(6 + dx) = 6$$

**公式解析，分四步：**

1. **代入并展开**：$(3+dx)^2 = 9 + 6dx + dx^2$，减去 $9$ 得 $6dx + dx^2$。
2. **约去 $dx$**：因为 $dx \neq 0$，可以安全地除以 $dx$，得 $6 + dx$。
3. **取标准部分**：$dx$ 是无穷小，$6 + dx \approx 6$，所以标准部分是 $6$。
4. **与标准答案对账**：$\frac{d}{dx}x^2\big|_{x=3} = 2 \cdot 3 = 6$，一致。

注意第三步的魔法：标准分析里「约分后令 $h \to 0$」时那句含糊的「$h$ 趋于 0 所以只剩 $6$」，在这里被严格化为「$dx$ 是无穷小，故 $6 + dx$ 的标准部分是 $6$」。**$dx$ 从头到尾都是一个具体的、非零的数**，不需要再假装它「趋近」什么。这就是莱布尼茨的梦想在现代数学里被兑现的方式。<span class="marginnote">同样地，$f(x) = x^3$ 在 $a$ 处的差商展开为 $3a^2 + 3a\,dx + dx^2$，取标准部分得 $3a^2$——每一项的 $dx$ 幂次越高，越早被标准部分吞掉。这个「丢掉高阶无穷小」的机械动作，正是微分运算 $(dx)^2 = 0$ 记号背后的真实含义。</span>

再算一个容易出错的反例：$f(x) = |x|$ 在 $a = 0$ 处。取 $dx > 0$，差商为 $|dx|/dx = 1$；取 $dx \lt  0$，差商为 $|dx|/dx = -1$。两者标准部分分别是 $1$ 与 $-1$，**随 $dx$ 的选择而变**，因此「标准部分与 $dx$ 无关」的条件不满足，$f'(0)$ 不存在——与标准分析结论一致，但这次「为何不存在」在超实数里看得一清二楚：**单子 $\mu(0)$ 在 $0$ 两侧给出不同的差商极限**。

## 5 辨析：标准部分映射的四条红线

**辨析｜易错点：** 标准部分映射看着像「取整数的整数部分」，实则有三条易踩的红线：

1. **无穷大没有标准部分**。$\operatorname{st}(\omega)$ 无定义。只有有限超实数才配拥有标准部分——这正是「有限」这一条件存在的意义。
2. **保序不保严格不等**。$x \lt  y$ 只能推出 $\operatorname{st}(x) \leq \operatorname{st}(y)$，可能相等。例：$x = 0$，$y = [1/n]$，则 $x \lt  y$ 但两者标准部分都是 $0$。**严格不等式在取标准部分后可能塌缩成等式**。
3. **除法有条件**。$\operatorname{st}(x/y)$ 需要 $\operatorname{st}(y) \neq 0$。若 $y$ 是无穷小，$x/y$ 可能是无穷大、有限或无穷小，标准部分不再由 $x, y$ 的标准部分单独决定。
4. **不能逐点交换**。$\operatorname{st}\bigl(f(x)\bigr)$ 一般不等于 $f(\operatorname{st}(x))$——这正是「$f$ 连续」的定义内容！**函数与标准部分是否可交换，恰好刻画了函数的连续性**，我们下一篇会专门展开。

如果觉得第四条太抽象，把它翻译成一句人话：**「先取极限再代入函数」与「先代入函数再取极限」在什么时候可以互换，是分析学反复追问的问题**。$\lim_{n\to\infty} f(a_n) = f(\lim_{n\to\infty} a_n)$ 正是这条交换律的序列版本，而它在 $f$ 连续时成立、不连续时可能失败。标准部分映射把这个古老的追问压成了一个等式：$\operatorname{st}(f(x)) \stackrel{?}{=} f(\operatorname{st}(x))$。<span class="marginnote">第四条是理解「为什么有些函数不连续」的最深入口：$f(x) = [x]$ 取整函数在 $x=1$ 处不连续，正是因为 $\operatorname{st}(f(1+\varepsilon)) = 1$ 而 $f(\operatorname{st}(1+\varepsilon)) = f(1) = 1$……不对，真正的问题是另一侧：$f(1-\varepsilon) = 0$，其标准部分为 0，而 $f(\operatorname{st}(1-\varepsilon)) = f(1) = 1$，两者不同。这个例子留作下一篇的引子。</span>

## 6 小结

- **标准部分存在唯一**：每个有限超实数 $x$ 有唯一标准实数 $\operatorname{st}(x) \approx x$；存在性依赖 $\mathbb{R}$ 的确界原理，唯一性来自「标准无穷小只能是 $0$」。
- **运算性质**：加、减、乘、除（分母标准部分非零）都在有限范围内保持；保序但不保严格不等；对标准实数恒等、幂等。
- **极限的新语言**：$\lim_{x\to a} f(x) = L \iff \forall x \approx a,\ x \neq a: f(x) \approx L$；导数写成差商再取标准部分。
- **四条红线**：无穷大无标准部分；严格不等可塌缩；除法需分母标准部分非零；$\operatorname{st}$ 与函数可交换恰是连续性。
- **实战**：$f(x) = x^2$ 的导数用「展开、约分、取标准部分」三步得出 $2x$；$|x|$ 在 $0$ 处的导数不存在，因为标准部分随 $dx$ 的选择而变。

有了标准部分映射这把「翻译器」，下一篇我们就能正式做分析：用「$x \approx a \Rightarrow f(x) \approx f(a)$」一句话重新刻画**连续性**，并顺便看清取整函数、狄利克雷函数这些标准分析里的「病态例子」到底病在哪里。到那时你会发现：标准分析里需要长篇论证的连续性，在超实数语言里只是一个「保持 $\approx$