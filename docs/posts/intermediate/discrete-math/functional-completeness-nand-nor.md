---
title: 函数完备性：与非门与或非门
date: 2026-08-07
---

# 函数完备性：与非门与或非门

<div class="epigraph">
<p>只用一种门也能算出一切——与非门一只，足以构造整个数字世界。</p>
<footer>—— 改编自 Kenneth H. Rosen《离散数学及其应用》（Discrete Mathematics and Its Applications）</footer>
</div>

<div class="article-byline">
<p>第二级 · 离散数学 ｜ Rosen《离散数学及其应用》 第12章 §12.2 ｜ 2026-08-07</p>
</div>

## 为什么"够用"是个数学命题

上一节我们从真值表造出积之和，用了 $\land$、$\lor$、$\neg$ 三种运算。这引出一个深刻问题：**最少需要哪些运算才能表示所有布尔函数？** 答案是令人惊讶的：只用**与非门（NAND）**或**或非门（NOR）**一种就能造出全部。这叫**函数完备性（functional completeness）**。物理上芯片几乎全用 NAND/NOR 门（制造便宜、速度快），完备性保证"一种门就能实现一切逻辑"——这是数字电路设计的第一条数学依据。<span class="marginnote">函数完备性的直觉也通向逻辑学的"归约"：命题逻辑里 $\land$、$\lor$ 都可由 $\neg$ 与 $\to$ 定义，而"只用 $\uparrow$（NAND）"是归约的极端。在第三级《数字逻辑》里，"用 NAND 搭出与非/或非/反相器"是必会的基本功——完备性就是它的理论保证。</span>

## 1 什么是函数完备性

**函数完备（functionally complete）**：一组布尔运算若能**表示所有布尔函数**（对任意 $n$，任意 $n$ 元布尔函数都能用这组运算的表达式表示），则这组运算函数完备。

**已知完备的集合**：

- $\{\land, \lor, \neg\}$：由积之和展开，任何布尔函数都可用这三者表达——完备。
- $\{\land, \neg\}$：$\lor$ 可由德摩根律定义：$x \lor y = \overline{\bar x \land \bar y}$——完备。
- $\{\lor, \neg\}$：对称地，$\land$ 可由 $\overline{\bar x \lor \bar y}$ 定义——完备。
- $\{\text{NAND}\}$、$\{\text{NOR}\}$：单个运算也完备（下面证明）。

**为什么需要否定**：只用 $\land$ 和 $\lor$ 不完备——它们单调，无法造出非单调的 $\bar x$。

## 2 公式解析：NAND 与 NOR 的完备性

**NAND**（与非）：$x \uparrow y = \overline{x \land y}$；**NOR**（或非）：$x \downarrow y = \overline{x \lor y}$。

**证明 $\{\uparrow\}$ 完备**：只需用 NAND 定义 $\neg$、$\land$、$\lor$。

$$
\bar x = x \uparrow x
$$

（NAND 两个相同输入：$\overline{x \land x} = \bar x$。）

$$
x \land y = (x \uparrow y) \uparrow (x \uparrow y)
$$

（先 NAND 得 $\overline{x\land y}$，再 NAND 自己得 $\overline{\overline{x\land y}} = x\land y$——**双重否定**。）

$$
x \lor y = (x \uparrow x) \uparrow (y \uparrow y) = \bar x \uparrow \bar y = \overline{\bar x \land \bar y} = x \lor y
$$

（德摩根律。）

- **第一步，造非**：$x \uparrow x = \bar x$——NAND 接同一输入即反相器。
- **第二步，造与**：先 NAND 再 NAND 自身，双重否定还原 AND。
- **第三步，造或**：用德摩根律 $\overline{\bar x \land \bar y} = x \lor y$，配合第一步的反相。∎

既然 $\neg$、$\land$、$\lor$ 都可由 NAND 表达，而这三者完备，故 $\{\uparrow\}$ 完备。**NOR 的证明完全对称**。

**辨析｜易错点：** 用 NAND 实现 AND 时**不能省第二层 NAND**。$x \uparrow y$ 直接就是 $\overline{x\land y}$，要 AND 必须"再非一次"（再 NAND 自己）。**"取非两次"是最容易漏的一步**。

## 3 为什么电路世界偏爱 NAND/NOR

- **物理便宜**：晶体管实现上，NAND/NOR 门比 AND/OR 门更省晶体管（AND 需要额外的反相器）。
- **速度快**：CMOS 工艺里 NAND/NOR 的延迟更小。
- **存储**：静态存储单元（锁存器、触发器）本质由交叉 NAND/NOR 构成。

**例（全用 NAND 的电路）**：任何组合电路都能翻译成"全是 NAND 门"的两级或任意级实现——EDA 工具自动做这个翻译（NAND 映射）。

**工程含义**：函数完备性保证**不必每种逻辑都造一种门**——一种 NAND 门就能拼出全部，芯片设计因此极大简化。

## 4 不完备的集合与"最小完备集"

**为什么 $\{\land, \lor\}$ 不完备**：这两者都是**单调**的——输入从 0 变 1 不会让输出从 1 变 0。而 $\bar x$ 非单调，无法表达。完备集**必须包含某种"翻转"能力**。

**寻找最小完备集**：恰好"能表达 $\neg$ + 一个单调运算"的集合通常完备。例如 $\{\neg, \land\}$、$\{\neg, \lor\}$、$\{\text{NAND}\}$、$\{\text{NOR}\}$——都恰好包含"翻转 + 一种合并"。

**例**：$\{\to\}$（蕴含）单独**不完备**——蕴含无法表达否定（需要常量 0）。而 $\{\to, 0\}$ 完备。

<span class="marginnote">完备性的思想延伸到函数式编程与逻辑：λ 演算里"只用一两个组合子（S、K）就足够表达一切计算"——这就是<strong>组合子完备性</strong>。而"一套原语生成全部"的抽象，从逻辑门到指令集（RISC 精简指令）到神经网络激活函数的选择，都是同一个完备性思维。</span>

## 5 用 NAND 搭出全部逻辑门：一个实战练习

把"NAND 完备"落到工程：用 NAND 门搭出 NOT、AND、OR、XOR 门。

**搭 NOT**：$x \uparrow x = \bar x$——NAND 两输入短接即反相器。

**搭 AND**：$\bar{\bar{x \land y}} = (x \uparrow y) \uparrow (x \uparrow y)$——NAND 后接 NAND 反相。

**搭 OR**：$x \lor y = \overline{\bar x \land \bar y} = (x \uparrow x) \uparrow (y \uparrow y)$——先各自取反，再 NAND。

**搭 XOR**（异或）：$x \oplus y = (x \lor y) \land \overline{x \land y}$，用上面搭出的门组合。

**公式解析：XOR 的 NAND 实现**

$$
x \oplus y = (x \uparrow (x \uparrow y)) \uparrow (y \uparrow (x \uparrow y))
$$

- **第一步，读异或定义**：$x \oplus y = 1$ 当且仅当 $x \ne y$——"一个为 1 且不同时为 1"。
- **第二步，读标准构造**：$x \oplus y = (x \lor y) \land \overline{(x \land y)}$——或的结果与"不同时为 1"相交。
- **第三步，读 NAND 化**：$\lor$、$\land$、$\neg$ 全部替换成 NAND 组合。∎

**为什么工程只造 NAND 门**：CMOS 工艺里 NAND 是"最自然的门"（晶体管最少、速度最快）。**函数完备性保证：只要一种门，就能造出 ALU 里的一切**——加法器、比较器、译码器全用 NAND 拼装。

**辨析｜易错点：** 搭 XOR 时最易错的写法是漏掉"不同时为 1"这一层。$x \lor y$ 在 $x=y=1$ 时也输出 1，但那不是异或——必须 $\land \overline{x \land y}$ 排除。**异或 = 或 且 非与**，三件缺一不可。

**这个例子的要点**：函数完备性不是抽象口号——**它是"一种门造一切"的许可证**。从"3 种门"到"1 种门"，完备性告诉你信息没有丢失，代价只是门数增加。

## 6 小结

- **函数完备**：一组运算能表示所有布尔函数。
- 完备集：$\{\land,\lor,\neg\}$、$\{\land,\neg\}$、$\{\lor,\neg\}$、$\{\text{NAND}\}$、$\{\text{NOR}\}$。
- NAND 完备：$\bar x = x\uparrow x$、$x\land y = (x\uparrow y)\uparrow(x\uparrow y)$、$x\lor y$ 由德摩根。
- **单调性障碍**：只用 $\land,\lor$ 表达不了非单调的 $\neg$。
- 物理上 NAND/NOR 更便宜更快，完备性让"一种门造一切"成立。

在下一节，把布尔函数接到物理世界——**逻辑门电路：组合电路的设计**。
