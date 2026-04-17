## 核心结论

组合数学研究的是离散对象的计数问题：在给定规则下，到底有多少种不同结果。这里的“离散对象”指能够明确区分、不能再无限细分的结构，例如字符串、集合、二进制码字、排班方案、任务分配结果。

最常用的三类工具是排列、组合、生成函数；处理重叠条件时，经常还要补上容斥原理。

| 工具 | 关心什么 | 典型问题 | 核心公式 |
| --- | --- | --- | --- |
| 排列 | 顺序是否不同 | 3 个位置怎么排 2 个字母 | $P(n,k)=\frac{n!}{(n-k)!}$ |
| 组合 | 只看选了谁，不看顺序 | 3 个字母里选 2 个 | $C(n,k)=\frac{n!}{k!(n-k)!}$ |
| 生成函数 | 把“选法数量”编码成多项式系数 | 同时统计取 0 个到取 $n$ 个的数量 | $(1+x)^n=\sum_{k=0}^n C(n,k)x^k$ |
| 容斥原理 | 多个条件重叠时如何去重 | 至少满足一个坏条件的对象有多少 | $\left|\bigcup_i A_i\right|=\sum |A_i|-\sum |A_i\cap A_j|+\cdots$ |

先看最小规模例子。给定 3 个不同字母 A、B、C，从中取 2 个：

- 如果关心顺序，结果是 AB、AC、BA、BC、CA、CB，共 6 种，所以 $P(3,2)=6$
- 如果不关心顺序，结果是 $\{A,B\}$、$\{A,C\}$、$\{B,C\}$，共 3 种，所以 $C(3,2)=3$
- 如果想一次性看到“取 0 个、1 个、2 个、3 个”的全部数量，可以写生成函数
  $$
  (1+x)^3=1+3x+3x^2+x^3
  $$
  其中 $x^2$ 的系数是 3，正好表示“从 3 个元素中取 2 个”的组合数

这个例子已经包含了四个核心判断：

| 问题 | 应该问什么 | 对应工具 |
| --- | --- | --- |
| 我在数什么 | 是字符串、集合还是位置集合 | 先定义对象 |
| 顺序重要吗 | AB 和 BA 是否算不同 | 排列 / 组合 |
| 要单层还是整体分布 | 只求取 2 个，还是求所有层数 | 组合 / 生成函数 |
| 条件会重叠吗 | 两个限制会不会同时满足 | 容斥原理 |

因此，核心结论可以压缩成一句话：排列解决“排法数”，组合解决“取法数”，生成函数解决“整层结构如何统一计数”，容斥原理解决“多个条件重叠时如何去重”。

---

## 问题定义与边界

计数题最容易出错的地方通常不是公式，而是对象定义不清。真正开始计算之前，先把边界说清楚。

一个计数问题至少要回答下面 3 个问题：

| 问题 | 必须明确的边界 | 适用工具 |
| --- | --- | --- |
| 我要数什么对象 | 字符串、集合、路径、码字、任务分配 | 全部都依赖对象定义 |
| 顺序重不重要 | AB 和 BA 是否算不同 | 排列或组合 |
| 我要单层数量还是整体分布 | 只求“选 2 个”还是求“所有选法” | 组合或生成函数 |

### 1. 对象是什么

“从 4 个人里选 2 个”与“把 4 个人安排到 2 个岗位”不是同一个问题。前者的结果是集合，后者的结果是有序二元组。对象一变，公式就变。

看一个对比：

| 题目表述 | 数学对象 | 正确计数 |
| --- | --- | --- |
| 从 A、B、C、D 里选 2 个 | 2 元子集 | $C(4,2)=6$ |
| 从 A、B、C、D 里选 2 个并排先后 | 长度为 2 的有序序列 | $P(4,2)=12$ |
| 从 4 个位置里挑 2 个放 1 | 位置集合 | $C(4,2)=6$ |
| 给岗位 1、岗位 2 分别分配不同的人 | 岗位到人的单射 | $P(4,2)=12$ |

### 2. 顺序是否重要

顺序是第一条分界线。可以用最直接的判断方法：交换两个元素的位置后，结果是否改变。

- 排座位、安排发言顺序、给编号位置填字符：顺序重要
- 选项目成员、选中奖号码、挑出故障机器：顺序通常不重要

再看一个最小例子。元素集为 $\{A,B,C\}$，取 2 个：

- 若结果写成有序对，则 $(A,B)$ 和 $(B,A)$ 不同
- 若结果写成集合，则 $\{A,B\}$ 和 $\{B,A\}$ 相同

很多错误都出在这里：题目在自然语言里没写“顺序”，但对象本身已经隐含了顺序。

### 3. 只求某一层，还是要求整体分布

“只求从 10 个元素中选 3 个的数量”是单层问题，直接求 $C(10,3)$ 即可。

“要知道从 10 个元素中选 0 个到选 10 个分别有多少种”是整体分布问题，这时生成函数更自然，因为它能一次性编码所有层数：
$$
(1+x)^{10}=\sum_{k=0}^{10} C(10,k)x^k
$$

这里 $x^k$ 的系数就是“选 $k$ 个”的数量。

### 4. 工程例子：码字权重分布

在编码理论里，长度为 $n$ 的二进制码字可以看成一个长度为 $n$ 的 0-1 串。若恰好有 $k$ 位为 1，则这样的码字数量是
$$
C(n,k)
$$
原因很直接：本质是在 $n$ 个位置里挑出 $k$ 个位置放 1。

例如，长度 7 的码字中恰好有 3 个 1：
$$
C(7,3)=35
$$

如果进一步想统计所有可能权重的分布，权重就是码字里 1 的个数，对应生成函数
$$
(1+x)^n
$$
因为每一位都只有两种状态：

- 取 0，贡献 1
- 取 1，贡献 $x$

全部位置相乘后，$x^k$ 的系数就是权重为 $k$ 的码字数量。

### 5. 重叠条件与去重边界

如果问题里出现“满足条件 A 或条件 B”的表述，就必须检查是否重叠。设：

- $A$ 表示“第 1 位出错”
- $B$ 表示“第 2 位出错”

则“第 1 位或第 2 位出错”的数量不是简单相加，而是
$$
|A\cup B|=|A|+|B|-|A\cap B|
$$

原因是同时满足 A 和 B 的对象被计算了两次。

这就是容斥原理最基本的形式。它的作用不是“让公式更复杂”，而是纠正重叠计数。

---

## 核心机制与推导

这一节把四个核心工具的推导串起来。重点不是记住公式，而是理解每个因子为什么出现。

### 1. 排列：逐位置乘法

设有 $n$ 个不同元素，从中取出 $k$ 个并按顺序排好。

- 第 1 个位置有 $n$ 种选择
- 第 2 个位置剩下 $n-1$ 种
- 第 3 个位置剩下 $n-2$ 种
- ...
- 第 $k$ 个位置有 $n-k+1$ 种

因此：
$$
P(n,k)=n(n-1)(n-2)\cdots(n-k+1)=\frac{n!}{(n-k)!}
$$

其中阶乘定义为
$$
n!=n(n-1)(n-2)\cdots 2\cdot 1,\qquad 0!=1
$$

把排列公式写成阶乘形式的好处是简洁，也方便后续和组合联系起来。

举例：从 5 个字母中取 3 个并排序，
$$
P(5,3)=5\cdot 4\cdot 3=60
$$

### 2. 组合：在排列基础上去掉内部顺序重复

若只关心“选了哪几个”，不关心内部顺序，那么每一组被排列公式重复计算了 $k!$ 次。

例如，选出 $\{A,B,C\}$ 这一组，在排列里会对应：

- ABC
- ACB
- BAC
- BCA
- CAB
- CBA

一共 $3!=6$ 次。

所以组合数应当是：
$$
C(n,k)=\frac{P(n,k)}{k!}=\frac{n!}{k!(n-k)!}
$$

这两个分母的含义可以拆开看：

| 因子 | 作用 |
| --- | --- |
| $(n-k)!$ | 表示只取前 $k$ 个位置，不继续安排剩余元素 |
| $k!$ | 去掉同一组选法内部的顺序重复 |

再看一个常用性质：
$$
C(n,k)=C(n,n-k)
$$
这叫对称性。它表示“选出 $k$ 个”与“舍弃 $n-k$ 个”是一回事。  
例如：
$$
C(10,3)=C(10,7)=120
$$

### 3. 二项式系数与递推关系

组合数不仅有闭式公式，还有递推公式：
$$
C(n,k)=C(n-1,k-1)+C(n-1,k)
$$

解释很直接。固定某个元素，比如元素 $a$，那么大小为 $k$ 的选法分两类：

- 选了 $a$：还要从剩余 $n-1$ 个元素里再选 $k-1$ 个，共 $C(n-1,k-1)$ 种
- 没选 $a$：要从剩余 $n-1$ 个元素里选 $k$ 个，共 $C(n-1,k)$ 种

两类互斥，所以相加。

这条递推对应杨辉三角：

$$
\begin{aligned}
&1 \\
&1\quad 1 \\
&1\quad 2\quad 1 \\
&1\quad 3\quad 3\quad 1 \\
&1\quad 4\quad 6\quad 4\quad 1
\end{aligned}
$$

第 $n$ 行第 $k$ 项就是 $C(n,k)$。

### 4. 生成函数：把计数写进系数

给定数列 $a_0,a_1,a_2,\dots$，它的普通生成函数定义为
$$
G(x)=\sum_{n\ge 0} a_n x^n
$$

这里的含义是：

- $x^n$ 只负责标记“规模是 $n$”
- 系数 $a_n$ 才是真正的“第 $n$ 类对象有多少个”

也就是说，生成函数是“用多项式保存一整列计数结果”的方法。

### 5. 为什么 $(1+x)^n$ 对应“任意选取”

每个元素只有两种状态：

- 不选它，贡献 1
- 选它，贡献 $x$

所以一个元素对应因子 $(1+x)$；$n$ 个独立元素一起考虑，就得到
$$
(1+x)^n
$$

展开时，如果最终得到一项 $x^k$，说明恰好有 $k$ 个括号提供了 $x$，也就是恰好选了 $k$ 个元素。因此：
$$
(1+x)^n=\sum_{k=0}^n C(n,k)x^k
$$

这就是二项式定理。

### 6. 用最小例子把系数读出来

当 $n=3$ 时：
$$
(1+x)^3=1+3x+3x^2+x^3
$$

对应关系如下：

| 项 | 系数 | 实际含义 | 具体选择 |
| --- | --- | --- | --- |
| $1$ | 1 | 选 0 个 | $\varnothing$ |
| $3x$ | 3 | 选 1 个 | A、B、C |
| $3x^2$ | 3 | 选 2 个 | AB、AC、BC |
| $x^3$ | 1 | 选 3 个 | ABC |

这里有一个很重要的阅读方式：不要把它只看成代数恒等式，而要看成“分层统计表”。

### 7. 更一般的生成函数例子

如果每个元素最多可以选 2 次，那么单个元素对应的因子不是 $(1+x)$，而是
$$
1+x+x^2
$$

因为它有 3 种状态：

- 选 0 次，贡献 1
- 选 1 次，贡献 $x$
- 选 2 次，贡献 $x^2$

若有 4 类独立元素，每类最多选 2 次，总体生成函数就是
$$
(1+x+x^2)^4
$$

其中 $x^5$ 的系数表示“总共选 5 次”的方案数。

这正是生成函数的优势：局部规则写成小多项式，整体规则用乘法组合起来。

### 8. 容斥原理：把重复算回去

设 $A_i$ 表示“满足第 $i$ 个坏条件”的对象集合，那么满足“至少一个坏条件”的对象数量是
$$
\left|\bigcup_i A_i\right|
$$

直接相加 $\sum |A_i|$ 会重复统计交集对象，所以要交替加减：
$$
\left|\bigcup_{i=1}^m A_i\right|
=
\sum_i |A_i|
-
\sum_{i<j} |A_i\cap A_j|
+
\sum_{i<j<k} |A_i\cap A_j\cap A_k|
-\cdots
+(-1)^{m+1}|A_1\cap\cdots\cap A_m|
$$

最容易理解的是 3 个集合的情况：
$$
|A\cup B\cup C|
=
|A|+|B|+|C|
-|A\cap B|-|A\cap C|-|B\cap C|
+|A\cap B\cap C|
$$

其中：

- 单集合项：把所有满足坏条件的对象都加进来
- 两两交集项：修正“被加了两次”的对象
- 三重交集项：前一步又被减多了，需要加回来

容斥原理本质上是在修正重复计数，不是单独的魔法公式。

---

## 代码实现

下面给出一个可直接运行的 Python 版本，覆盖：

- 排列数 `perm`
- 组合数 `comb`
- 二项式生成函数系数 `gf_binomial_coeffs`
- 一般多项式卷积 `poly_mul`
- 一类简单容斥计数 `count_binary_strings_with_at_least_one_fixed_one`

代码只依赖 Python 标准库。

```python
from __future__ import annotations

from itertools import combinations, permutations
from math import factorial
from typing import Iterable, List


def perm(n: int, k: int) -> int:
    """P(n, k): from n distinct items, choose k and arrange them."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    return factorial(n) // factorial(n - k)


def comb(n: int, k: int) -> int:
    """C(n, k): from n distinct items, choose k without order."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    k = min(k, n - k)  # use symmetry C(n, k) = C(n, n-k)
    ans = 1
    for i in range(1, k + 1):
        ans = ans * (n - k + i) // i
    return ans


def gf_binomial_coeffs(n: int) -> List[int]:
    """Coefficients of (1 + x)^n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    coeffs = [1]
    for _ in range(n):
        nxt = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            nxt[i] += c       # choose 0 from this item
            nxt[i + 1] += c   # choose 1 from this item
        coeffs = nxt
    return coeffs


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials represented by coefficient arrays."""
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            res[i + j] += ai * bj
    return res


def gf_bounded_choice_coeffs(num_items: int, max_take_per_item: int) -> List[int]:
    """
    Coefficients of (1 + x + x^2 + ... + x^max_take_per_item)^num_items.
    coefficient[t] = number of ways to take total t items.
    """
    if num_items < 0 or max_take_per_item < 0:
        raise ValueError("arguments must be non-negative")

    base = [1] * (max_take_per_item + 1)
    coeffs = [1]
    for _ in range(num_items):
        coeffs = poly_mul(coeffs, base)
    return coeffs


def inclusion_exclusion_union_size(universe_size: int, event_predicates: Iterable) -> int:
    """
    Generic inclusion-exclusion over an explicit finite universe [0, universe_size).
    Suitable for small examples and verification, not for large-scale production.
    """
    predicates = list(event_predicates)
    m = len(predicates)
    total = 0

    for r in range(1, m + 1):
        sign = 1 if r % 2 == 1 else -1
        for idxs in combinations(range(m), r):
            count = 0
            for x in range(universe_size):
                if all(predicates[i](x) for i in idxs):
                    count += 1
            total += sign * count
    return total


def count_binary_strings_with_at_least_one_fixed_one(n: int, positions: List[int]) -> int:
    """
    Number of binary strings of length n where at least one listed position is 1.
    Positions are 0-based.
    Closed form: 2^n - 2^(n-len(positions)).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if any(p < 0 or p >= n for p in positions):
        raise ValueError("position out of range")

    unique_positions = sorted(set(positions))
    if not unique_positions:
        return 0
    return (1 << n) - (1 << (n - len(unique_positions)))


def main() -> None:
    # 1) Basic permutation / combination checks
    assert perm(3, 2) == 6
    assert comb(3, 2) == 3

    # 2) Binomial generating function
    coeffs3 = gf_binomial_coeffs(3)
    assert coeffs3 == [1, 3, 3, 1]

    # 3) Cross-check with brute force enumeration
    items = ["A", "B", "C"]
    assert len(list(permutations(items, 2))) == perm(3, 2)
    assert len(list(combinations(items, 2))) == comb(3, 2)

    # 4) Coding-theory style example:
    #    binary words of length 7 and weight 3
    assert comb(7, 3) == 35
    assert gf_binomial_coeffs(7)[3] == 35

    # 5) Bounded-choice generating function:
    #    4 item types, each can be chosen 0/1/2 times
    #    coefficient of x^3 in (1 + x + x^2)^4
    bounded = gf_bounded_choice_coeffs(num_items=4, max_take_per_item=2)
    assert bounded[3] == 16

    # 6) Simple inclusion-exclusion example:
    #    length-5 bit strings where position 0 or position 1 is 1
    #    direct formula: 2^5 - 2^3 = 24
    assert count_binary_strings_with_at_least_one_fixed_one(5, [0, 1]) == 24

    print("perm(3, 2) =", perm(3, 2))
    print("comb(3, 2) =", comb(3, 2))
    print("coeffs of (1+x)^3 =", gf_binomial_coeffs(3))
    print("C(7, 3) =", comb(7, 3))
    print("coeff of x^3 in (1+x+x^2)^4 =", bounded[3])
    print("bit strings of length 5 with bit0=1 or bit1=1 =", count_binary_strings_with_at_least_one_fixed_one(5, [0, 1]))


if __name__ == "__main__":
    main()
```

这段代码的输入输出关系可以先用下面这张表记住：

| 函数 | 输入 | 输出 | 含义 |
| --- | --- | --- | --- |
| `perm(n, k)` | `n=3, k=2` | `6` | 从 3 个不同元素中取 2 个并排序 |
| `comb(n, k)` | `n=3, k=2` | `3` | 从 3 个不同元素中取 2 个，不看顺序 |
| `gf_binomial_coeffs(n)` | `n=3` | `[1, 3, 3, 1]` | $(1+x)^3$ 的系数 |
| `poly_mul(a, b)` | `[1,1]`, `[1,1,1]` | `[1,2,2,1]` | 多项式卷积 |
| `gf_bounded_choice_coeffs(4, 2)` | `4, 2` | 长度 9 的系数数组 | $(1+x+x^2)^4$ 的系数 |
| `count_binary_strings_with_at_least_one_fixed_one(5,[0,1])` | `5,[0,1]` | `24` | 长度 5 串中第 0 或第 1 位为 1 的数量 |

### 代码背后的数学对应

| 代码对象 | 数学对象 | 说明 |
| --- | --- | --- |
| `coeffs[i]` | $x^i$ 的系数 | 表示规模为 $i$ 的对象数量 |
| `poly_mul` | 多项式乘法 | 对应独立选择规则的组合 |
| `gf_binomial_coeffs` | $(1+x)^n$ | 每个元素只能选 0 次或 1 次 |
| `gf_bounded_choice_coeffs` | $(1+x+\cdots+x^m)^n$ | 每个元素最多选 $m$ 次 |

### 为什么卷积会出现

若
$$
A(x)=\sum_i a_i x^i,\qquad B(x)=\sum_j b_j x^j
$$
则
$$
A(x)B(x)=\sum_k \left(\sum_{i+j=k} a_i b_j\right)x^k
$$

这说明乘积中 $x^k$ 的系数，是所有“局部规模相加为 $k$”的方案数之和。这正是“把两个独立选择过程合起来”的计数规则。

例如：
$$
(1+x)(1+x+x^2)=1+2x+2x^2+x^3
$$

其中 $x^2$ 的系数为 2，表示总规模为 2 的方案有两种：

- 前者选 0，后者选 2
- 前者选 1，后者选 1

### 一个更像工程的例子

假设有 5 个任务槽位，需要安排 3 个不同任务进入其中 3 个位置，则方案数是
$$
P(5,3)=60
$$

如果任务本身相同，只关心哪 3 个位置被占用，则方案数变成
$$
C(5,3)=10
$$

如果进一步要求“每个槽位可空、可放 1 个任务、可放 2 个重复子任务”，并统计总放入量的所有分布，那么对应生成函数是
$$
(1+x+x^2)^5
$$

这三类问题的对象不同，所以工具也不同。

---

## 工程权衡与常见坑

组合数学在工程里最大的问题通常不是不会套公式，而是把问题建模错了。下面几类错误最常见。

| 常见坑 | 错误原因 | 后果 | 规避策略 |
| --- | --- | --- | --- |
| 把组合当排列 | 忘了顺序是否重要 | 结果偏大，常多乘一个 $k!$ | 先问交换位置后结果是否改变 |
| 把排列当组合 | 忽略位置区别 | 结果偏小 | 只要位置有业务含义，就用排列 |
| 忽略对象是否可重复 | 没分清“不同元素”与“允许重复选取” | 公式直接失效 | 先判断是否放回、是否允许重复 |
| 生成函数次数开太大 | 盲目展开高阶多项式 | 时间和内存迅速增长 | 只保留关心的最高次数 |
| 容斥只算前几层 | 忽略高阶交集 | 误差可能很大 | 只在可接受近似时截断 |
| 用浮点数算组合数 | 大整数精度丢失 | 得到非整数或错误值 | 用整数递推或高精度整数 |
| 忽略边界值 | 没处理 $k>n$、$k=0$、$n=0$ | 代码异常或结果错误 | 明确约定并写测试 |

### 1. 组合和排列混淆

假设“从 6 个可用频点里分配 3 个给一个用户”，如果频点集合不区分先后，正确答案是
$$
C(6,3)=20
$$

如果误用排列，则得到
$$
P(6,3)=120
$$

差了 6 倍，正好是 $3!$。原因是同一组频点被不同顺序重复统计了。

### 2. 允许重复时，公式完全不同

“从 4 种颜色里选 3 个球，允许同色重复，且不看顺序”不再是普通组合数，而是“可重复组合”问题，其结果为
$$
C(n+k-1,k)=C(4+3-1,3)=C(6,3)=20
$$

这里已经不是本文主线，但它说明一个关键点：只要“是否允许重复”改变，问题就不是同一个问题。

### 3. 生成函数的实现边界

若只想求不超过 $d$ 次的系数，没有必要把完整多项式全部展开。可以在每一步卷积后截断到 $x^d$：

$$
A(x)\cdot B(x)\pmod{x^{d+1}}
$$

这种做法在工程上很重要，因为很多时候只关心某个有限权重范围。例如只需要 0 到 10 个错误位的分布，就没必要保留更高次项。

### 4. 容斥原理不是总能便宜算

容斥原理本身精确，但项数可能指数增长。若有 $m$ 个条件，理论上会出现 $2^m-1$ 个非空交集项。  
因此容斥适合：

- 条件数量不多
- 交集结构有规律
- 可以推导统一闭式

如果条件很多且结构复杂，直接做容斥往往不划算，DP 或状态压缩更实际。

### 5. 形式幂级数与收敛半径

在离散计数里，生成函数通常先当作形式幂级数使用，即只关心系数，不关心数值收敛。

但如果进一步做解析方法、极限估计、奇点分析，就必须检查收敛半径。  
也就是说：

- 做基本计数时：重点是代数结构
- 做渐近分析时：还要考虑解析性质

这是初学者容易忽略的边界。

---

## 替代方案与适用边界

组合数学不是唯一办法。它适合规则明确、结构离散、限制可表达的问题，但不是所有计数问题都应该直接写生成函数。

| 方法 | 适用规模 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 暴力枚举 | $n\le 6$ 或状态很小 | 直观，方便验证 | 很快爆炸 |
| 排列/组合公式 | 结构简单、约束少 | 快，结论直接 | 难处理复杂依赖 |
| 动态规划 | 有阶段结构和递推关系 | 易落地，适合带约束问题 | 状态设计可能复杂 |
| 生成函数 | 多层选择、和式约束明显 | 统一表达能力强 | 理解和实现门槛更高 |
| 容斥原理 | 条件重叠明确 | 去重精确 | 项数可能指数增长 |

### 1. 暴力枚举什么时候最合适

当规模很小，枚举是最稳的方法。  
例如从 5 个元素里取 3 个：

- 直接列出所有组合，便于人工检查
- 再用公式比对，能确认建模是否正确

所以在写程序时，常见工程做法是：

- 小规模用暴力生成真值
- 大规模用公式或 DP
- 两者交叉验证

### 2. 只问单层数量时，优先公式

如果题目只问：

- 从 20 个元素中取 4 个
- 长度 30 的二进制串中恰好 8 个 1
- 8 个岗位安排 3 个不同人

这类问题通常直接落在排列或组合上，不需要引入生成函数。

例如长度 50 的二进制串中，权重在 20 到 30 之间的数量，可以写成：
$$
\sum_{k=20}^{30} C(50,k)
$$

虽然结果是多项求和，但每一项本身仍然是组合数。

### 3. 有局部规则时，生成函数往往更自然

假设长度 20 的二进制串被分成 4 组，每组 5 位，并要求每组最多出现 2 个 1。  
单组的生成函数是
$$
\sum_{j=0}^{2} C(5,j)x^j = 1+5x+10x^2
$$

四组独立，因此总体生成函数是
$$
(1+5x+10x^2)^4
$$

其中 $x^k$ 的系数表示总权重为 $k$ 的码字数量。  
这种问题如果直接暴力枚举，需要看 $2^{20}$ 个对象；若写成生成函数，结构会清楚得多。

### 4. 有阶段状态时，DP 可能更容易落地

例如：

- 网格路径计数
- 背包式约束
- 合法括号串数量
- 带前缀限制的字符串计数

这类问题常有明显的“前缀到后缀”递推结构。  
此时 DP 的优势在于程序实现直接对应状态转移，而生成函数更像是先建立代数表达，再从系数里读答案。

可以把两者理解为：

| 方法 | 思维方式 |
| --- | --- |
| 动态规划 | 按过程递推 |
| 生成函数 | 把整体规则打包成代数对象 |

### 5. 实际判断原则

工程上可以用一套简单判断：

- 问题很小，先枚举，保证模型对
- 只问单层数量，先看排列或组合
- 要整体分布、总权重、分组限制，优先考虑生成函数
- 条件之间有重叠，检查是否需要容斥
- 有明显阶段结构，优先尝试动态规划

这套顺序的重点不是“谁更高级”，而是谁更贴合对象结构。

---

## 参考资料

- Graham, Knuth, Patashnik, *Concrete Mathematics*  
  适合系统学习二项式系数、递推、生成函数、组合恒等式。优点是推导扎实，但对初学者来说密度较高，建议配合例题阅读。

- Richard A. Brualdi, *Introductory Combinatorics*  
  适合作为基础教材，覆盖排列、组合、容斥、递推、图论中的基础计数问题，章节组织比研究型教材更友好。

- Kenneth H. Rosen, *Discrete Mathematics and Its Applications*  
  适合离散数学入门，内容覆盖组合计数、递推关系、生成函数、容斥原理。优点是面向计算机专业读者，例题比较规整。

- Philippe Flajolet, Robert Sedgewick, *Analytic Combinatorics*  
  适合在掌握普通生成函数之后继续深入，理解为什么生成函数不仅能计数，还能做渐近分析。对初学者偏难，但视角很强。

- F. J. MacWilliams, N. J. A. Sloane, *The Theory of Error-Correcting Codes*  
  适合把组合计数和编码理论连接起来，尤其是权重分布、码字容量、距离结构等问题。

- Stanford CS166 或同类算法课程中的 Combinatorics / Generating Functions 讲义  
  适合从算法实现视角理解“为什么多项式系数能表示计数”，以及如何把组合问题转成卷积、DP、递推。

- Herbert S. Wilf, *Generatingfunctionology*  
  一本专门讲生成函数的经典入门书。优点是主题集中，适合把“会用二项式公式”进一步提升到“会建生成函数模型”。
