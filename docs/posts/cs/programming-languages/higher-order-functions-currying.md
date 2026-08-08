---
title: 高阶函数、柯里化与函数组合
date: 2026-08-07
---

# 高阶函数、柯里化与函数组合

<div class="epigraph">
<p>把函数当作参数与返回值，程序的「积木」就从数据升级为行为。</p>
<footer>—— 佚名（PLT 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第15章 §15.5 ｜ 2026-08-07</p>
</div>

## 为什么从高阶函数开始

λ 演算的「抽象与应用」在语言层面打开了一个新世界：函数可以**接受函数**、**返回函数**——这就是**高阶函数（higher-order function）**。加上柯里化（多参数变单参）与函数组合（管道式拼装），函数式编程获得了它的「积木」体系：不再用语句堆叠控制流，而是用「函数套函数」精确地组装行为。这一节把这三个互相关联的机制讲透——它们是 map/filter/reduce 等一切函数式惯用法的语言基础。<span class="marginnote">一等函数（函数是值）是「高阶函数」的前提；高阶函数是「函数操作函数」的能力。map 里 f 是参数、柯里化返回函数——两者都是高阶的体现。没有高阶函数，map/filter 就只能写进语法（如 Python 的列表推导是语法，map 是高阶函数）。</span>

## 1 高阶函数：接受或返回函数

**高阶函数（higher-order function）**：至少满足一条——① 接受函数作参数；② 返回函数作结果。

```python
def apply_twice(f, x):
    return f(f(x))                # 接受函数 f 作参数

apply_twice(lambda n: n + 1, 5)   # 结果 7
```

最经典的高阶函数三件套：

```python
xs = [1, 2, 3, 4, 5]
list(map(lambda x: x * x, xs))              # map：变换
list(filter(lambda x: x % 2 == 0, xs))      # filter：筛选
functools.reduce(lambda a, b: a + b, xs)    # reduce：折叠
```

**map / filter / reduce** 的图景：map 变换、filter 筛选、reduce 折叠——**三种通用数据操作取代了大量手写循环**。<span class="marginnote">map/filter/reduce 是「循环的抽象」：循环变量、累积、条件被封装进高阶函数，用户只需提供「单元素操作」。这正是指称语义「组合性」的实践——复合表达式的语义由各部分的语义决定，高阶函数可重组、可优化。</span>

## 2 柯里化：多参数函数的分步喂参

**柯里化（currying）**：把「一个多参数函数」转换成「一串单参数函数」——每次调用接收一个参数、返回一个「等待下一个参数」的函数。

$$
\text{curry}(f)(a)(b) = f(a, b), \qquad \text{curry}(f) = \lambda a.\ \lambda b.\ f(a, b)
$$

```haskell
-- Haskell：柯里化函数 add 天然支持部分应用
add :: Int -> Int -> Int
add x y = x + y

add3 = add 3   -- 部分应用：add3 是「等待一个参数」的函数
add3 4         -- 7
```

柯里化的价值：**偏函数应用（partial application）**——喂部分参数，得到「更专门」的函数，可复用、可传递。

**辨析｜易错点：** 柯里化 ≠ 多参数语法糖。柯里化的函数**天然支持部分应用**——add 3 就是合法函数；而多参数函数（Python def add(x, y)）不能只传一半（add(3) 报缺参）。**「柯里化让每个参数都成为一次独立的调用点」**——这是它与普通多参的本质差异。

## 3 函数组合：管道式拼装

**函数组合（function composition）**：把多个函数串成管道——前一个的输出是后一个的输入。

$$
(f \circ g)(x) = f(g(x))
$$

```haskell
-- Haskell：组合用 . 运算符
f = (+1) . (*2)   -- 先 *2 再 +1
f 5               -- 11
```

组合的威力：**声明式地描述「数据处理流程」**——不写循环、不写中间变量，只描述「数据经过哪些变换」。Rust 的迭代器链（iter().filter().map()）就是组合思想的现代形态。

## 4 公式解析：高阶函数的行为语义

高阶函数的行为可以用「函数即值」的语义刻画。以 map 为例：

$$
\text{map}(f, [x_1, \dots, x_n]) = [f(x_1), \dots, f(x_n)]
$$

组合律（关键性质）：

$$
\text{map}(f) \circ \text{map}(g) = \text{map}(f \circ g)
$$

三步拆解：

- **第一步，map 的定义**：map 把 f 逐个应用到每个元素——结果列表的长度与输入相同，每个元素独立变换。
- **第二步，组合律**：先 map g 再 map f = 直接 map (f∘g)。**这条等式成立靠纯函数**——f、g 无副作用，两次遍历可合并为一次。
- **第三步，看优化与并行**：组合律让编译器可以「融合（fusion）」两次遍历为一次（Haskell 的 list fusion）；也让 map 天然可并行（每个元素独立）。**「高阶函数 + 纯函数 = 可优化、可并行的声明式代码」**。

**辨析｜易错点：** 高阶函数的求值时机：**严格求值**下 map 立即生成整个列表；**惰性求值**（Haskell）下 map 只生成「头部」，元素按需计算——这允许处理无限列表。**「高阶函数 + 惰性」是 Haskell 表达无限数据结构（流）的基石**。

## 5 高阶函数在现代语言

- **Python**：map/filter/reduce（functools）、sorted(key=f)、functools.partial。
- **JavaScript**：Array.map/filter/reduce、.bind、箭头函数柯里化。
- **Java**：Stream 的 map/filter/reduce、方法引用 ::。
- **Rust**：迭代器 iterator——组合链式、惰性、零成本。<span class="marginnote">「高阶函数」已从函数式语言的专属变成主流语言的标准能力——Java 8 Stream、JS 数组方法、Rust 迭代器、C++ 的 STL 算法（std::transform 等）都是「用高阶函数替代手写循环」。语言进化的一个显著方向，就是「让函数操作函数」越来越顺手。</span>



## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 高阶函数（higher-order function） | 高阶函数（higher-order function）：至少满足一条——① 接受函数作参数；② 返回函数作结果。 |
| 柯里化（currying） | 柯里化（currying）：把「一个多参数函数」转换成「一串单参数函数」——每次调用接收一个参数、返回一个「等待下一个参数」的函数。 |
| 偏函数应用（partial application） | 柯里化的价值：偏函数应用（partial application）——喂部分参数，得到「更专门」的函数，可复用、可传递。 |
| 函数组合（function composition） | 函数组合（function composition）：把多个函数串成管道——前一个的输出是后一个的输入。 |
| 声明式地描述「数据处理流程」 | 组合的威力：声明式地描述「数据处理流程」——不写循环、不写中间变量，只描述「数据经过哪些变换」。Rust 的迭代器链（iter().filter().m |
| 这条等式成立靠纯函数 | 第二步，组合律：先 map g 再 map f = 直接 map (f∘g)。这条等式成立靠纯函数——f、g 无副作用，两次遍历可合并为一次。 |
| 「高阶函数 + 纯函数 = 可优化、可并行的声明式代码」 | 第三步，看优化与并行：组合律让编译器可以「融合（fusion）」两次遍历为一次（Haskell 的 list fusion）；也让 map 天然可并行（ |
| Python | Python：map/filter/reduce（functools）、sorted(key=f)、functools.partial。 |
| JavaScript | JavaScript：Array.map/filter/reduce、.bind、箭头函数柯里化。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **高阶函数**接受/返回函数；map/filter/reduce 是「循环的抽象」三件套。
- **柯里化**把多参函数变单参链，支持**偏函数应用**（喂一部分参得专门函数）。
- **函数组合**把函数串成管道：f(g(x))，声明式描述数据流。
- 组合律 map(f) ∘ map(g) = map(f∘g) 靠纯函数成立，带来**融合优化**与**天然并行**；高阶函数已全面渗透主流语言。

在下一节，我们将看高阶函数与递归的实战舞台——**Scheme 基础：函数、表与递归**。
