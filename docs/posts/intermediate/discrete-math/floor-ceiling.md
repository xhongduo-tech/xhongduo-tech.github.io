---
title: 取整函数：下取整（floor）与上取整（ceiling）
date: 2026-08-07
---

# 取整函数：下取整（floor）与上取整（ceiling）

<div class="epigraph">
<p>上帝创造了整数，其余的都是人的作品。</p>
<footer>—— 利奥波德 · 克罗内克（Leopold Kronecker）</footer>
</div>

<div class="article-byline">
<p>第二级 · 离散数学 ｜ Rosen《离散数学》§2.3 ｜ 2026-08-07</p>
</div>

## 为什么从取整函数开始

上一节我们把函数推广到了「任意集合到任意集合」，这一节要介绍的是一对**把实数投影到整数**的特殊函数：下取整 $\lfloor x \rfloor$ 与上取整 $\lceil x \rceil$。它们太平凡了——小学就会「四舍五入」——却恰恰是**连续世界通往离散世界的第一道闸门**。<span class="marginnote">「从极限到大模型」的主线在这里分叉：极限研究连续地逼近，而离散数学研究精确的整数。取整函数站在两条路交汇的地方：把一个连续的实数，敲定成一个离散的整数。</span>

在计算机科学里，取整无处不在：分页要算「需要几页」，二分搜索要算「中间下标」，哈希要算「落在哪个桶」，大模型量化要把连续权重压到几个离散档位。**凡是要把「连续」塞进「离散」的容器，都绕不开这对函数。** 而且它们自带一套奇妙的恒等式，能让许多计数问题一秒钟算完。

## 1 从连续到离散：floor 与 ceiling 的定义

**下取整（floor）**：对任意实数 $x$，$\lfloor x \rfloor$ 表示**不大于 $x$ 的最大整数**。**上取整（ceiling）**：对任意实数 $x$，$\lceil x \rceil$ 表示**不小于 $x$ 的最小整数**。<span class="marginnote">符号 $\lfloor \cdot \rfloor$ 与 $\lceil \cdot \rceil$ 由计算机科学家肯尼斯 · 艾弗森（Kenneth Iverson）在 1962 年提出——他后来还发明了 APL 语言与数组编程思想。把「地板」「天花板」画在数轴上，含义一目了然。</span>

看几个例子：

$$
\lfloor 3.14 \rfloor = 3, \qquad \lfloor -3.14 \rfloor = -4, \qquad \lceil 3.14 \rceil = 4, \qquad \lceil -3.14 \rceil = -3
$$

注意 $-3.14$ 的下取整是 $-4$ 而不是 $-3$：因为 $-4 < -3.14 < -3$，「不大于 $-3.14$ 的最大整数」是 $-4$。**取整不是「去掉小数部分」，而是「往数轴左（floor）/ 右（ceiling）取最近的整数」。** 这一条直觉能避掉后面一半的坑。

当 $x$ 本身是整数时，两个函数重合：

$$
x \in \mathbb{Z} \iff \lfloor x \rfloor = \lceil x \rceil = x
$$

而当 $x$ 不是整数时，它们恰好相差 1：$\lceil x \rceil = \lfloor x \rfloor + 1$。

## 2 取整函数的基本性质

取整函数虽然简单，却有一条条可以当工具用的恒等式。设 $n$ 是整数，$x$ 是实数：

- **平移**：$\lfloor x + n \rfloor = \lfloor x \rfloor + n$，$\lceil x + n \rceil = \lceil x \rceil + n$。<span class="marginnote">平移公式的直觉：整个数轴往右挪 $n$ 个单位，「地板」「天花板」也一起挪 $n$ 个单位。它让「把小数部分剥离出来」成为可能：$\{x\} = x - \lfloor x \rfloor$ 就叫 $x$ 的小数部分。</span>
**与负号交换**：$\lfloor -x \rfloor = -\lceil x \rceil$，$\lceil -x \rceil = -\lfloor x \rfloor$。数轴对折之后，地板变天花板。
**夹逼**：$x - 1 < \lfloor x \rfloor \le x \le \lceil x \rceil < x + 1$。取整不会把 $x$ 挪出「左右各 1」的范围内。
**与整数比较**：对整数 $n$，$\lfloor x \rfloor \le n \iff x < n + 1$；$\lceil x \rceil \le n \iff x \le n$。这个「把取整符号翻译成普通不等式」的技巧，是解带取整的方程的标准手段。
**区间里的整数个数**：闭区间 $[a, b]$（$a \le b$，实数）内的整数个数是 $\lfloor b \rfloor - \lceil a \rceil + 1$。<span class="marginnote">这条公式常被忽视却极有用：数「某区间里有几个整数」是计数题与算法题的基本动作。比如 $[1.2, 4.8]$ 里有 $\lfloor 4.8 \rfloor - \lceil 1.2 \rceil + 1 = 4 - 2 + 1 = 3$ 个整数：`$2`, 3, 4$。</span>

**重点：取整是一种「与整数的加减法可交换」的运算。** 公式 $\lfloor x + n \rfloor = \lfloor x \rfloor + n$ 说明整数可以自由「进出」取整号；反之，任何非整数的数都不能自由进出——$n$ 必须是整数，这是所有取整恒等式的通用前提。

## 3 计算机科学里的取整

取整函数在算法与系统里出现的频率，仅次于加减乘除。

**分页与分组**：$n$ 条数据每页放 $k$ 条，需要的页数是 $\lceil n / k \rceil$。9 条数据每页 5 条，要 $\lceil 9/5 \rceil = 2$ 页——「多出来的一点也要占一页」，这正是上取整。
**二分搜索的中间点**：经典写法 `mid = (lo + hi) // 2`，即 $\lfloor (lo + hi) / 2 \rfloor$。下取整保证 $mid$ 不会越界，也保证区间一定收敛。
**哈希表的桶下标**：把连续的关键字 $x$ 映射到 `$0` \dots m-1$ 的桶，常用 $x \bmod m$ 或 $\lfloor (x \cdot A) \rfloor \bmod m$，取整是其中一环。
**大模型量化**：把浮点权重 $w$ 压到低比特整数时，先做缩放再取整：$w_q = \lfloor w / s + 0.5 \rfloor$（四舍五入）或直接 $\lfloor w / s \rfloor$。**一个模型的「离散化误差」就由取整这一下决定。**<span class="marginnote">量化是「从极限到大模型」的又一个落点：大模型在训练时用连续的浮点数，推理压缩时却要切成离散档位。取整函数就是把「连续」切成「离散」的那把刀。</span>

在程序语言里，取整的行为各不相同，这也是最现实的坑。

## 4 辨析｜易错点：负数取整与「向零取整」

**辨析｜易错点：数学的 $\lfloor x \rfloor$ 是「向负无穷取整」，而许多编程语言的整数除法是「向零取整」。** 两者在正数上一致，在负数上分道扬镳。

数学里 $\lfloor -3.7 \rfloor = -4$；而 C 语言 $\lfloor -7/3 \rfloor = -3$ 得到 $-3$（向零截断）。C 的整数除法 $\lfloor -7/3 \rfloor = -3$ 得 $-2$，数学的 $\lfloor -7/3 \rfloor = -3$。Python 里 $-3$ 得 $-3$（Python 的 $-2$ 恰好是向负无穷取整），但 `int(-7 / 3)` 得 $-2$（先做浮点除法再向零截断）。<span class="marginnote">同一条代码在不同语言里语义不同：C 的整数除法向零，Python 的 `//` 向下取整。跨语言移植取整逻辑时，这是最容易踩的隐性地雷。</span>

```c
int a = -7 / 3;   /* C 语言整数除法：向零截断，a == -2 */
```

```python
b = -7 // 3       # Python 地板除：向负无穷取整，b == -3
```

还有一个浮点精度陷阱：`0.1 * 3` 在有些机器上可能返回 `0.30000000000000004` 而不是 `0.3`，因为 `0.1` 在二进制里是无限循环小数（无法精确表示）。**处理金额、页码这类对精度敏感的量时，尽量用整数运算（把 `$10`^{-k}$ 倍的小数换成整数），把取整留给真正需要的最后一步。**

## 5 公式解析：分页公式 $\left\lceil \frac{n}{k} \right\rceil = \left\lfloor \frac{n + k - 1}{k} \right\rfloor$

程序员口中的「整除向上取整」有一个等价写法，把上取整换成了下取整：

$$
\left\lceil \frac{n}{k} \right\rceil = \left\lfloor \frac{n + k - 1}{k} \right\rfloor \qquad (n, k \in \mathbb{Z}^{+})
$$

对这条公式做三步拆解：

- **第一步，理解要算什么**：$n$ 条数据每 $k$ 条一页，页数是 $\lceil n/k \rceil$。它是「$n$ 除以 $k$ 后，有余数就进一位」——一个典型的离散计数。
- **第二步，把「有余数就进位」翻译成不等式**：令 $q = \lceil n/k \rceil$，则 $q$ 是满足 $n \le qk$ 的最小整数，即 $(q-1)k < n \le qk$。这正是上取整的定义。
- **第三步，换成下取整**：$(q-1)k < n \iff qk < n + k \iff n + k - 1 \ge qk$（因为 $qk$ 是整数），于是 $q = \lfloor (n + k - 1)/k \rfloor$。分子上加的 $k-1$，作用就是「把任何一个正余数顶到进一位」。

用一个具体数字验证：$n = 9, k = 5$。左边 $\lceil 9/5 \rceil = 2$；右边 $\lfloor (9 + 4)/5 \rfloor = \lfloor 13/5 \rfloor = 2$。再试 $n = 10, k = 5$：左边 $\lceil 2 \rceil = 2$，右边 $\lfloor 14/5 \rfloor = 2$——刚好整除时不多算一页。<span class="marginnote">这条公式的妙处在于，很多语言里没有原生上取整除法，却有下取整除法（如 Python 的 `//`）。用 `(n + k - 1) // k` 一行实现「向上取整的整除」，正是这条恒等式的工程价值。</span>

同样的手法也出现在区间计数：数 $[a, b]$ 内整数个数 $\lfloor b \rfloor - \lceil a \rceil + 1$，本质是把「区间的两个端点」各自取整，再用「首末整数相减加一」完成计数。

## 6 取整的亲戚：四舍五入、截断与小数部分

$\lfloor \cdot \rfloor$ 与 $\lceil \cdot \rceil$ 不是唯一的取整方式。程序语言里最常见的另外三种都能用它们表示：

**四舍五入（round）**：$\mathrm{round}(x) = \lfloor x + 0.5 \rfloor$（正数情形）。直觉是「取更近的那个整数」：把半程决策点 $x + 0.5$ 顶到整数上，再用 floor 落回整数。
**截断（truncate）**：$\mathrm{trunc}(x)$ 是「直接去掉小数部分」——正数时等于 $\lfloor x \rfloor$，负数时等于 $\lceil x \rceil$。它正是 C 语言整数除法（`/`）的行为，也就是第 4 节辨析里那个「向零取整」的家伙。
**小数部分（fractional part）**：$\{x\} = x - \lfloor x \rfloor \in [0, 1)$。它把 $x$ 拆成「整数部分 + 小数部分」，是随机数生成、谐波分析里的基础工具。

四舍五入有一个微妙处：**半程点 $x = 2.5$ 离 2 和 3 一样近，「往哪走」并没有唯一答案。** 常见约定有「四舍五入到 3」（取远离 0 的一方）与**银行家舍入**「五取偶」——2.5 → 2、3.5 → 4、4.5 → 4。银行家舍入在金融与统计里能消除「恒向上取」带来的系统性偏差，Python 的 `round()` 用的正是它：

```python
round(2.5)  # 2
round(3.5)  # 4
round(4.5)  # 4
```

**辨析｜易错点：floor、truncate、round 在负半轴各走各的路。** $\lfloor -2.5 \rfloor = -3$（向负无穷），$\mathrm{trunc}(-2.5) = -2$（向零），$\mathrm{round}(-2.5) = -2$（Python 五取偶）。三者在 $x \ge 0$ 时大多一致，一旦 $x$ 变负就分道扬镳。<span class="marginnote">量化、音频采样、图像处理这类「把连续压成离散」的工程里，选错取整规则会累积出可见的偏差：恒向上取会整体偏大，五取偶能保持统计上的对称。规则的选择，本身就是一种有偏估计 vs 无偏估计的权衡。</span>

## 7 小结

- **下取整** $\lfloor x \rfloor$ 是「不大于 $x$ 的最大整数」；**上取整** $\lceil x \rceil$ 是「不小于 $x$ 的最小整数」；$x$ 为整数时二者相等，否则 $\lceil x \rceil = \lfloor x \rfloor + 1$。
- **取整不是去小数部分**：$\lfloor -3.14 \rfloor = -4$；对负数尤其要小心。
- 关键恒等式：$\lfloor x + n \rfloor = \lfloor x \rfloor + n$（整数可自由进出取整号）、$\lfloor -x \rfloor = -\lceil x \rceil$、区间整数个数 $\lfloor b \rfloor - \lceil a \rceil + 1$。
- 工程上：分页用 $\lceil n/k \rceil$；二分、哈希、量化都要取整；**数学取整向负无穷，C 的整数除法向零，Python 的 `//` 向负无穷**——跨语言移植必踩的坑。
- 分页公式 $\lceil n/k \rceil = \lfloor (n + k - 1)/k \rfloor$ 把「有余数就进位」变成一行整除代码。

在下一节，我们将回答一个更「大」的问题：**无穷集合能不能比大小？** 自然数、整数、有理数、实数，谁的「个数」更多——这就是**基数与可数性**。
