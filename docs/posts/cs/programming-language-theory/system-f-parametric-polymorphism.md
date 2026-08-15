---
title: System F 参数化多态
date: 2026-08-07
---

# System F 参数化多态

<div class="epigraph">
<p>若一个操作能对多种类型的实参工作，就说它是多态的。</p>
<footer>—— 克里斯托弗 · 斯特雷奇（Christopher Strachey），1967 年</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言理论 ｜ Pierce《类型与程序设计语言》§23 ｜ 2026-08-07</p>
</div>

## 为什么从 System F 开始

上一课的 let 多态给了我们「$id$ 对任意类型都能用」，但它是个特例——多态只发生在 let 位置，而且类型变量藏在系统内部。皮尔斯《类型与程序设计语言》§23 引入的 **System F（二阶 λ 演算，λ₂）** 把多态从「特性」升格为「语言本身」：让**类型抽象（type abstraction）**成为一等公民。这是 Girard 1971 年为证明逻辑一致性发明的系统，也是 Haskell 的 `∀a.`、Java 泛型、Rust 泛型的理论祖先。理解 System F，就理解了「参数化多态」为什么能给出如此强的不变量（参数性定理），乃至「自由定理」。

## 1 新的语法：类型抽象与应用

System F 在 $\lambda_\to$ 上新增两件事：**类型抽象（type abstraction）**与**类型应用（type application）**，以及相应的**全称类型（universal type）** $\forall X.\,T$：

$$t ::= \cdots \;\big|\; \lambda X.\, t \;\big|\; t\;[T] \qquad T ::= \cdots \;\big|\; \forall X.\, T$$

读法很自然：$\lambda X.\,t$ 是「以类型为参数、体为 $t$ 的抽象」，$t\;[T]$ 是「把类型 $T$ 传给类型函数 $t$」，$\forall X.\,T$ 是「对所有类型 $X$ 都成立的 $T$」。于是恒等函数有了明确的类型：

$$\texttt{id} = \lambda X.\,\lambda x: X.\, x \quad : \quad \forall X.\, X \to X$$

**重点：** 类型应用 $\texttt{id}\;[\texttt{Nat}]$ 得到 $\lambda x:\texttt{Nat}.\,x$——类型是运行时参数之外的第二类参数，但它在**编译期**就被消掉（见 §4 擦除）。System F 的程序员写「多态函数」时，其实是在写**一个接受类型参数、返回普通函数的函数**。<span class="marginnote">皮尔斯 §23 的记号：项 $\lambda X.t$ 用小写 lambda、类型 $T$ 用 $\forall X.T$，与「值抽象/类型抽象」一一对应。System F 被称为「二阶」是因为类型变量可以取「类型」为值，而类型本身又可以是「对类型做全称量化」——这开启了<strong>不可判定域</strong>（见 §5）。</span>

**System F 的「二阶」指什么**：λ 演算是「值 → 值」的函数，System F 多了一层「类型 → 值」的抽象。它能写 $\lambda X.\,t$（对类型抽象），而 $X$ 本身又可以取「含 $\forall$ 的类型」——于是系统可以对「对类型做量化的类型」再做量化，这就是「二阶」（second-order）的含义：**类型变量可以取「泛化类型」为值**。正是这一层升级，让 System F 的表达力远超 $\lambda_\to$，却也撞上了类型检查的不可判定性。

## 2 类型规则：抽象与应用的配对

两条新规则与《类型系统与类型规则》里的抽象/应用规则完全平行：

$$
\frac{\Gamma, X \vdash t : T}{\Gamma \vdash \lambda X.\,t : \forall X.\,T}
\qquad
\frac{\Gamma \vdash t : \forall X.\,T}{\Gamma \vdash t\;[T'] : [X \mapsto T']\,T}
$$

类型抽象规则说：**要在空想世界「假设一个类型变量 $X$」下证明 $t$，然后把它包装成 $\forall X.T$**；类型应用规则说：**全称类型 $\forall X.T$ 可以实例化到任意类型 $T'$，把 $X$ 替换掉**。两者互为逆操作——构造多态用抽象，使用多态用应用。这是一对「一般的」（对一切类型成立）与「特定的」（取定某个类型）之间的电梯。

## 3 程序写在 System F 里：丘奇编码重出江湖

System F 里多态编码有多强？看布尔值——注意类型里出现了 $\forall$，值是「选择器」：

$$\texttt{Bool} = \forall X.\, X \to X \to X; \qquad \texttt{true} = \lambda X.\,\lambda t: X.\,\lambda f: X.\, t$$

自然数同样可以编码（邱奇数的多态版）：

$$\texttt{Nat} = \forall X.\, (X \to X) \to X \to X; \qquad \overline{n} = \lambda X.\,\lambda s: X\to X.\,\lambda z: X.\, s^n\,z$$

这个编码的真正威力在于**主类型与自指**：自然数 $n$ 被定义为「对任何 $X$，把 $s$ 用 $n$ 次」——于是「$+$」只需组合两个这样的函数，不需要在语言里特判加法。<span class="marginnote">§23 里皮尔斯用 System F 证明了重要结论：<strong>类型擦除保真</strong>——把 $\lambda X.t$ 的 $\lambda X.$ 删掉、把 $t[T]$ 删成 $t$，得到的无类型项与原项行为一致。这给了「多态是编译期概念」最严格的表述：Java 泛型的擦除、Haskell 的类型擦除，全是这条定理的实现。</span>

**多态编码的实际计算：加法**。丘奇数的「加一」是 $\texttt{succ} = \lambda n.\,\lambda X.\,\lambda s.\,\lambda z.\, s\,(n\,X\,s\,z)$——先跑 $n$ 次、再补一次 $s$；于是「$+$」组合两个数：$\texttt{plus} = \lambda m.\,\lambda n.\,\lambda X.\,\lambda s.\,\lambda z.\, m\,X\,s\,(n\,X\,s\,z)$，`2 + 3` 就是「先跑 2 次再跑 3 次 = 跑 5 次」。**System F 不需要内建加法，因为它能『组合』出加法**——编码的表现力来自类型层的多态，这正是「一切可编码」的日常证明。

**擦除的一次具体演示**：`id = λX. λx:X. x` 对任何 `X` 都是恒等函数，运行时根本不需要知道 `X` 是什么——把它擦成 `λx. x`，行为不变。于是 `id[Nat] 3` 与 `id[Bool] true` 擦除后分别是 `(λx.x) 3` 与 `(λx.x) true`，同一个函数服务两种类型。**这就是「多态是编译期概念」的最直接证据**，也是为什么 Java 泛型可以安全地只在编译期做检查。

## 4 参数性：多态类型把行为锁死

System F 最令人惊叹的副产品是**参数性（parametricity）**——雷纳德的抽象定理：**类型的多态程度越高，能写出的行为越少**。看类型 $\forall X.\, X \to X$：函数只能有输入输出各一个 $X$，它**必须**是恒等函数（或发散）——不可能「看穿」 $X$ 是什么，于是无差别地返回输入。再看 $\forall X.\, X \to X \to X$，它只能是「选第一个」或「选第二个」——$\texttt{true}$ 与 $\texttt{false}$ 穷尽了它！

这引出一条工程真理：**接口类型本身就在给行为上锁**。泛型 `T f(T)` 在 Java 里「不能凭空造出 `T`、只能来自参数」，正是参数性的日常回响。所谓**自由定理（free theorems）**，就是从类型免费导出等式（如 `map f ∘ map g = map (f∘g)`），不需要读实现——这是「类型即规格」最漂亮的证据。<span class="marginnote">参数性由 John C. Reynolds 在 1983 年的"Types, Abstraction and Parametric Polymorphism"中确立；自由定理则由 Philip Wadler 1990 年的同名论文普及。它把「抽象」从编程习惯提升为可证明的性质：模块实现的自由被类型锁住，用户看到的唯一东西就是类型承诺的行为。</span>

**参数性在日常 API 里的回声**：为什么 `T f(T x)` 在 Java 里「不能凭空造 T」？因为多态的 `T` 对实现者不透明——实现拿不到任何 `T` 的信息，只能把传入的 `T` 原样返回（或存进容器）。这正是参数性「锁死行为」的工程直觉：**类型越泛，实现越受限**。反过来，这也解释了为什么「类型即文档」在泛型场景下格外可信——读签名几乎就等于读实现，不用翻实现也能猜出九成行为。

## 5 不可判定性与表达力

System F 并非没有代价。**类型检查与类型推导在 System F 里都是不可判定的**（Wells 1994）：类型标注可以任意复杂，无法保证「补全标注」总能终止；类型等价判定同样不可判定。而且 $\forall$ 是**不可直谓的（impredicative）**：$\forall X.T$ 自己可以被实例化到含 $\forall$ 的类型（$X$ 可以取到「含 $\forall$ 的类型」），形成了一个会自指的宇宙。因此工程上大家都做了让步：

- **Haskell** 的 `RankNTypes` 允许高阶多态，但默认类型推导仍限定在 rank-1（let 多态）。
- **Java / C#** 泛型走「擦除」路线，类型参数被抹掉，换来类型检查的可判定与向后兼容。
- **ML 家族**用 let 多态保住可判定推导，多态表达力弱于完整 System F。

**辨析｜易错点：** 不要把 $\forall X.\,T$ 与 $\exists X.\,T$ 混同。《对象与封装》里的 $\exists$ 说「存在某个类型，但它是哪个我不知道」；$\forall$ 说「任给一个类型都可以」——**$\forall$ 是用户的自由选择，$\exists$ 是实现者的保密承诺**。$\forall X.(X\to X)\to X\to X$ 与 $\exists X.\{ \dots \}$ 一个泛化、一个隐藏，方向截然相反。

**rank 是什么**：rank 指「$\forall$ 允许出现在哪一层」。rank-0 类型不含任何 $\forall$；rank-1 允许 $\forall$ 出现在箭头右侧顶层（`∀X. X→X` 是 rank-1）；rank-2 允许 $\forall$ 作为某个 $\forall$ 的实参（如 `(∀X. X→X) → Nat`）。**Hindley–Milner 推导只能处理 rank-1 的类型**，rank-2 及以上的类型标注需要程序员手写，且类型检查复杂度剧增——这正是 Haskell 的 `RankNTypes` 扩展存在的理由：想要完整 System F 的表达力，就必须付出标注与复杂度的代价。

## 6 公式解析：一对规则就是一座电梯

$$
\frac{\Gamma \vdash t : \forall X.\,T}{\Gamma \vdash t\;[T'] : [X \mapsto T']\,T}
$$

- **前提 $t : \forall X.\,T$**：$t$ 是一个「类型参数化」的项——它对每个 $X$ 都准备好了行为，像一份万能模板。
- **类型应用 $t\;[T']$**：给模板填入具体类型 $T'$，得到「$X$ 已经替换成 $T'$」的专门版本。
- **替换 $[X \mapsto T']$**：$T$ 中所有自由 $X$ 都被 $T'$ 取代——模板在类型层面被实例化。
- **与值层面平行**：若把 $\forall X.T$ 想成「函数类型」，那 $\lambda X.t$ 是它的构造、$t[T']$ 是它的调用。System F 的对称之美正在于此：**值上有 λ（抽象/应用），类型上也有 λ（抽象/应用），两套机制同构**。
- **实例**：$\texttt{id}\;[\texttt{Nat}] : \texttt{Nat}\to\texttt{Nat}$，从 $\forall X.X\to X$ 一步跨到具体箭头类型——电梯从「一般」降落到「特定」。

## 7 术语速查

| 术语 | 含义 | 一句话直觉 |
| --- | --- | --- |
| System F（λ₂） | λ→ + 类型抽象 + 类型应用 | 二阶 λ 演算 |
| 类型抽象 $\lambda X.\,t$ | 以类型为参数的函数 | 万能模板 |
| 类型应用 $t[T]$ | 把具体类型填进模板 | 实例化 |
| 全称类型 $\forall X.\,T$ | 对一切类型成立 | 用户自由选择 |
| 类型擦除 | 删去类型参数行为不变 | 多态是编译期概念 |
| 参数性 / 自由定理 | 多态程度越高行为越少 | 类型锁死实现 |
| 不可直谓（impredicative） | ∀ 可实例化到含 ∀ 的类型 | 自指的宇宙 |
| rank | ∀ 允许出现的层级 | 多态嵌套深度 |
| 擦除保真 | 擦除后与原项行为一致 | 无类型项不亏 |

**记忆锚点**：System F = 给 λ 演算加上「类型上的 λ」；参数性 = 「类型越泛、实现越受限」；不可判定 = 「表达力买不到免费午餐」——三句话收束全课。

## 8 小结

- **System F** = $\lambda_\to$ + 类型抽象 $\lambda X.t$ + 类型应用 $t[T]$ + 全称类型 $\forall X.T$，两条规则与值层面完全平行。
- 丘奇编码在 System F 里长成了「多态编码」：$\texttt{Bool} = \forall X.X\to X\to X$，自然数同理。
- **类型擦除保真**：多态是纯编译期概念，擦掉类型运行时行为不变（Java/Haskell 泛型的理论依据）。
- **参数性/自由定理**：多态程度越高、能写的实现越少，类型直接锁定行为，`id : ∀X.X→X` 只能是恒等函数。
- 类型检查与推导在完整 System F 中不可判定，工程上退化为 rank-1 与擦除。
- $\forall$（用户自由选择）与 $\exists$（实现者保密承诺）方向相反，不可混淆。

在下一节，我们从「类型与证明」的云端落地，回到一台真实程序运行的地方——看看执行环境如何组织内存、回收垃圾：**运行时系统与存储管理**。
