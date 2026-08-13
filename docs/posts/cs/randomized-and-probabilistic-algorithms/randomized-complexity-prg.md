---
title: 随机复杂性类与伪随机发生器
date: 2026-08-07
---

# 随机复杂性类与伪随机发生器

<div class="epigraph">
<p>随机性也许只是「计算上不可区分」的另一副面孔。</p>
<footer>—— 安德鲁 · 姚期智（Andrew Chi-Chih Yao）</footer>
</div>

<div class="article-byline">
<p>第三级 · 随机算法与概率方法 ｜ Motwani & Raghavan 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从复杂性类与伪随机开始

本专题的终章回答两个根本问题：**随机算法到底多「强」？随机性到底「是什么」？** 第一个问题由**随机复杂性类**回答——$\mathsf{RP}$、$\mathsf{co}$-$\mathsf{RP}$、$\mathsf{ZPP}$、$\mathsf{BPP}$ 用概率把问题的困难度分级；第二个问题由**伪随机发生器（PRG）**回答——一小段真随机种子能否膨胀出「计算上不可区分」的长串。两者交汇于一个惊人的猜想：**如果有足够难的函数，随机算法就不比确定性算法强**（$\mathsf{BPP} = \mathsf{P}$）。这一节把四类、它们的包含关系、以及 PRG 的「计算不可区分」范式讲清，为整个专题画上句号。

## 1 随机复杂性类：概率版的 P 与 NP

**随机复杂性类** 用「接受/拒绝概率」刻画决策问题的难度：

- **$\mathsf{RP}$（随机多项式）**：语言 $L \subseteq \{0,1\}^*$，存在多项式时间算法 $A$ 与随机位，满足
  - $x \in L \Rightarrow \Pr[A(x) \text{ 接受}] \ge 1/2$
  - $x \notin L \Rightarrow \Pr[A(x) \text{ 接受}] = 0$

  单侧错误：在语言的实例上可能误拒，但不在语言的实例**绝不误收**。<span class="marginnote">$\mathsf{RP}$ 的「错误方向」：正确实例以 $1/2$ 概率被拒（可以重复跑放大，$k$ 次全拒概率 $\le 2^{-k}$），错误实例永远不被接受。对照第 6 节——$\mathsf{RP}$ 正是「单侧错误的 Monte Carlo」的复杂度表述。随机素性测试（Miller-Rabin）属于 $\mathsf{co}$-$\mathsf{RP}$。</span>
- **$\mathsf{co}$-$\mathsf{RP}$**：对称版本，$x \notin L$ 时接受概率 $= 0$、$x \in L$ 时 $\ge 1/2$。
- **$\mathsf{ZPP}$**：期望多项式时间、总是正确的 Las Vegas 类。**$\mathsf{ZPP} = \mathsf{RP} \cap \mathsf{co}$-$\mathsf{RP}$**——两个方向都能验，就跑成「零错误概率」。
- **$\mathsf{BPP}$（有界错误概率多项式）**：双侧错误，
  - $x \in L \Rightarrow \Pr[\text{接受}] \ge 3/4$；$x \notin L \Rightarrow \Pr[\text{接受}] \le 1/4$。

  $\mathsf{BPP}$ 是「双侧错误的 Monte Carlo」的复杂度表述，也是实际随机算法最常用的归属类。

## 2 包含关系：随机性在复杂性世界的位置

这些类之间有清晰的包含链：

$$\mathsf{P} \subseteq \mathsf{ZPP} \subseteq \mathsf{RP} \subseteq \mathsf{BPP} \subseteq \mathsf{P}/\mathrm{poly}, \qquad \mathsf{RP} \subseteq \mathsf{NP}$$

<span class="marginnote">几个值得驻足的结论：$\mathsf{RP} \subseteq \mathsf{NP}$（随机接受可当「见证」：接受时记录随机位作见证）；$\mathsf{BPP} \subseteq \mathsf{P}/\mathrm{poly}$（非均匀电路能模拟 BPP，Adleman 定理——随机算法可被「非均匀确定性电路」替代）；$\mathsf{BPP} \subseteq \Sigma_2 \cap \Pi_2$（Sipser-Gács 定理，多项式谱系第二层）。这些结果都指向同一个悬念：$\mathsf{BPP}$ 或许就等于 $\mathsf{P}$。</span>

**关键猜想**：$\mathsf{BPP} = \mathsf{P}$。这不是已证定理，但它有坚实的支撑——下一节的「困难度 vs 随机性」：如果存在足够强的伪随机发生器（或足够难的函数），则 $\mathsf{BPP} = \mathsf{P}$，随机算法可以被**彻底去随机化**。

## 3 伪随机发生器：计算不可区分

**伪随机发生器（pseudorandom generator, PRG）**：确定性多项式时间函数 $G: \{0,1\}^s \to \{0,1\}^m$（$m \gg s$），把 $s$ 位种子膨胀成 $m$ 位输出，且对**所有多项式时间判别器** $D$ 都「看起来随机」：

$$\left|\Pr_{x \sim \{0,1\}^s}[D(G(x)) = 1] - \Pr_{y \sim \{0,1\}^m}[D(y) = 1]\right| \leq \varepsilon$$

<span class="marginnote">「计算不可区分」是伪随机的定义核心：真正的随机 $y$ 与 PRG 输出 $G(x)$，没有任何高效算法能分辨。注意这与「统计上接近」截然不同——PRG 的膨胀比意味着输出集合只是真随机集合的 $2^{s-m}$ 分之一，统计上差得远；但计算上，只要判别器是多项式时间的，就分辨不出。</span>

**Nisan-Wigderson PRG**：若存在「近似困难」的显式函数（如特定复杂度假设下的硬布尔函数），就能构造 $G$ 把 $s$ 位种子膨胀到多项式长，且分辨不了。它把「随机位」问题归结为「函数困难度」问题。

## 4 公式解析：BPP 的定义与放大

$$
\mathsf{BPP}:\; \begin{cases} x \in L \Rightarrow \Pr[A(x)=1] \ge 3/4 \\ x \notin L \Rightarrow \Pr[A(x)=1] \le 1/4 \end{cases}
$$

- **第一步，常数界**：$3/4$ 与 $1/4$ 是约定俗成的常数，取任何「大于 1/2 且远离 1/2」的值都可。
- **第二步，放大**：独立运行 $k$ 次取多数。由切尔诺夫界，错误概率降到 $2^{-\Omega(k)}$——常数错误率可放大到任意小。
- **第三步，为什么不是 $1/2$**：恰在 $1/2$ 处（无界错误）就是 $\mathsf{PP}$——它甚至包含 $\mathsf{NP}$，比 $\mathsf{BPP}$ 强得多。「有界」错误是 $\mathsf{BPP}$ 的命门：允许重试放大。
- **第四步，与 RP 对照**：$\mathsf{RP}$ 只在一侧容错且另一侧零错误；$\mathsf{BPP}$ 两侧都有界。两者都是「可放大的常数概率」。

**直觉**：$\mathsf{BPP}$ 的 $3/4$ 是「明显优于抛硬币」的门槛——一旦越过，重复就能把误差压到指数小；而 $\mathsf{PP}$ 的「$> 1/2$」任意微弱优势无法放大（你可能需要指数次重复才知道优势存在）。

## 5 困难度 vs 随机性：全专题的收官主题

本专题从「概率空间」走到「伪随机」，最后一条线把一切串起来——**困难度 vs 随机性（hardness vs randomness）**：

- 若存在「足够难」的显式函数，则存在「足够好」的 PRG。
- 若存在「足够好」的 PRG，则 $\mathsf{BPP} = \mathsf{P}$（用 PRG 种子替代全部随机位，枚举种子去随机化）。

<span class="marginnote">这条蕴含链把随机算法的命运绑定在「函数的困难度」上：随机性不是宇宙的恩赐，而是「我们还不会算太难函数」的次生现象。正如姚期智所猜想、Impagliazzo 与 Wigderson 等所深化：<strong>在极强困难度假设下，随机性可以完全被伪随机替代</strong>——这是本专题所有内容在复杂性理论里的终极回声。</span>

**全专题收束**：从概率空间 → 期望与矩 → 集中不等式（马尔可夫、切比雪夫、切尔诺夫、霍夫丁）→ Las Vegas / Monte Carlo → 随机分治与平摊 → 随机数据结构（Treap、跳表、哈希、布隆过滤器）→ 概率方法（期望、变化、第二矩、LLL、大偏差）→ 马尔可夫链（平稳分布、混合、游走、MCMC、鞅）→ 去随机化 → 伪随机。每一步都为下一步铺路，最终回到「随机性本身是什么」。

## 6 更多复杂性类与开放问题

随机复杂性类的地图不止 $\mathsf{RP}/\mathsf{BPP}$ 四个，周围还有一圈重要邻居：

**$\mathsf{PP}$（无界错误概率）**：接受概率 $> 1/2$（可任意接近 $1/2$）。因为没有「远离 $1/2$」的保证，无法放大，$\mathsf{PP}$ 反而很强：$\mathsf{NP} \subseteq \mathsf{PP}$，甚至 $\mathsf{PP}$ 对补封闭。它与 $\mathsf{BPP}$ 的分界正是「界住错误」与「界不住」的天堑。

**$\mathsf{MA}$ 与 $\mathsf{AM}$（Merlin-Arthur 类）**：交互证明的随机版本。$\mathsf{MA}$：Merlin 先给一个多项式长的证明，Arthur 用随机算法验证；$\mathsf{AM}$：Arthur 先抛硬币，Merlin 看到随机串后再给证明。已知 $\mathsf{NP} \subseteq \mathsf{MA} \subseteq \mathsf{AM}$，且若 $\mathsf{co}$-$\mathsf{NP} \subseteq \mathsf{AM}$ 则多项式谱系坍塌。<span class="marginnote">$\mathsf{MA}/\mathsf{AM}$ 是「概率方法 + 交互证明」的产物，它们通向更深的猜想：图非同构 $\mathsf{GNI}$ 在 $\mathsf{AM}$ 里（Goldwasser-Sipser），这给出「概率交互证明」不可约的经典证据。随机性在这里不是算法内部工具，而是<strong>协议</strong>的语言。</span>

**开放问题**：$\mathsf{BPP} = \mathsf{P}$？$\mathsf{BPP} = \mathsf{NP}$？$\mathsf{RP} \subseteq \mathsf{NP}$ 是否严格？这些问题的解答都等价于「能否构造足够强的伪随机发生器」，即「足够难的函数是否存在」。它们是随机算法理论留给未来的核心悬念，也是本专题从「概率空间」一路走到「复杂度与伪随机」的终极注脚。

## 7 从 PRG 到密码学的桥：计算不可区分的力量

伪随机发生器的「计算不可区分」定义，正是现代密码学的语言。理解这条桥，随机算法就与信息安全接上了轨：

**密码学伪随机（CSPRNG）**：密码学要求的 PRG 更强——输出必须对**所有**多项式时间判别器不可区分，且具备「前向安全」（知道当前输出也不能预测下一个）。这与算法理论里「对 BPP 够用的 PRG」同源，只是安全参数更大、假设更硬。<span class="marginnote">两者共享同一核心：计算不可区分。算法理论用 PRG 去随机化（把 BPP 变 P），密码学用 PRG 生成密钥流（把短种子膨胀成一次一密）。区别只在「敌人」是谁——算法理论假设对手只是多项式时间，密码学假设对手可以访问更多（选择明文、量子等）。这条桥解释了为什么「伪随机」是算法与密码学的共同地基。</span>

**困难度假设的角色**：PRG 的存在依赖「单向函数存在」（或其弱化）。若单向函数不存在，一切密码学与伪随机都塌方；反之，从单向函数可构造任意强度的 PRG。这与「困难度 → 随机性」的主线完全一致——伪随机与困难是一体两面。

**对读者的启示**：学完本专题，再看「密码学随机性」「机器学习随机性」「算法随机性」，它们其实是同一棵树的枝叶——「如何用少量真随机 + 计算困难，模拟大量随机」。这正是随机算法理论在最前沿的回响。

**困难度假设的谱系**：构造伪随机发生器所需的「困难函数」，从弱到强是一条谱系——单向函数（最弱，足以构造所有密码学 PRG）、指数级困难函数（足以证明 $\mathsf{BPP} = \mathsf{P}$）、乃至「无零知识」等更强假设。这条谱系的价值在于分层：工程上若只想要「看起来随机」，最弱的单向函数就够；理论上若想「彻底去随机化」，需要更强的困难度。每一层假设对应一类构造，也对应一类可能的反例——整个「困难度 vs 随机性」的研究，就是在这条谱系上逐层攻坚。

**对读者的终章寄语**：本专题从「概率空间」起步，途经「集中不等式、随机算法、随机数据结构、概率方法、马尔可夫链、去随机化」，最终抵达「伪随机与复杂性」——这条路径完整回答了「随机性在算法里是什么、能做什么、边界在哪」。你如今拥有的工具链，已足以阅读随机算法领域的大部分论文，也是通往密码学、机器学习理论、量子算法的重要基石。

**尾声**：从概率空间到伪随机，本专题的完整弧线已经画圆。随机性在算法里的三重身份——平均化的工具（Las Vegas / Monte Carlo）、存在性的证据（概率方法）、不可区分的资源（伪随机）——你如今都有了第一手的理解。随机算法不是「确定性算法加上噪声」，而是一门「在不确定性中提取确定性保证」的精确科学。

## 8 小结

- PRG 的存在等价于困难函数的存在，这是 $\mathsf{BPP}=\mathsf{P}$ 猜想的枢纽。
- 随机性三重身份：平均化工具、存在性证据、不可区分资源。
- 随机复杂性类：$\mathsf{RP}$、$\mathsf{co}$-$\mathsf{RP}$、$\mathsf{ZPP}$、$\mathsf{BPP}$，分别对应单侧/双侧/零错误的 Monte Carlo 与 Las Vegas。
- 包含链：$\mathsf{P} \subseteq \mathsf{ZPP} \subseteq \mathsf{RP} \subseteq \mathsf{BPP} \subseteq \mathsf{P}/\mathrm{poly}$，$\mathsf{RP} \subseteq \mathsf{NP}$。
- **$\mathsf{BPP} = \mathsf{P}$