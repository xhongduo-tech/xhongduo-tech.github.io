---
title: 方法：指针接收者、方法与嵌入
date: 2026-08-07
---

# 方法：指针接收者、方法与嵌入

<div class="epigraph">
<p>面向对象编程把「数据」与「对数据的操作」绑在一起，Go 用方法做到这一点，却不用继承。</p>
<footer>—— Go 语言设计哲学（Methods, not classes）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从方法开始

函数是自由的，方法（method）是**绑定到类型的函数**——它让「某个类型的值」拥有「属于自己的操作」。Go 的方法系统有三个关键点：**接收者（receiver）**把函数与类型绑定、**指针接收者**让方法能修改原值、**嵌入（embedding）**让字段与方法自动提升。理解了这三点，Go 的「对象」图景就清晰了——它没有 `class`，却能用 struct + 方法 + 嵌入组合出面向对象的能力。<span class="marginnote">对标《Go语言圣经》第6章：这一章把方法从「语法糖」升级为「语言哲学」。Go 官方反复强调「方法 vs 函数」的选择是设计问题，而指针接收者 vs 值接收者的选择，直接决定了方法的语义与性能。</span>

方法在本专题承上启下：承上一章《函数》——方法是函数的特殊形态；启下一章《接口》——只有「实现了方法的类型」才能满足接口。可以说，**方法是为接口准备的原料**。

## 1 方法与接收者

**方法** 是在 `func` 关键字与函数名之间插入一个**接收者参数（receiver）**的声明：

```go
type Point struct{ X, Y float64 }

func (p Point) Distance(q Point) float64 {
	dx := q.X - p.X
	dy := q.Y - p.Y
	return math.Hypot(dx, dy)
}

p := Point{1, 2}
q := Point{4, 6}
fmt.Println(p.Distance(q))   // 5
```

接收者 `p` 类似其它语言的 `this`/`self`——它让方法可以访问所属类型的字段。调用语法 `p.Distance(q)` 把「谁的数据、什么操作、另一个数据」组织成一个自然的表达。

**重点：** 任何类型都能有方法，不只是 struct。你可以给**自定义类型**（基于 `type` 声明）定义方法：

```go
type Celsius float64

func (c Celsius) String() string {
	return fmt.Sprintf("%g°C", c)
}
```

这为「带语义的单位」提供了可能——`Celsius` 与 `Fahrenheit` 即使底层都是 `float64`，也是不同类型，互相不能直接运算。这与《基础数据类型》篇「int 与 int32 是不同的类型」一脉相承：**类型是编译期的安全边界**。

## 2 值接收者与指针接收者

方法的接收者可以是值，也可以是指针。两者语义截然不同：<span class="marginnote">这是 Go 方法系统最重要的辨析。规则一句话：<strong>要修改原值，用指针接收者；只读、或接收者很小，用值接收者</strong>。《Go语言圣经》第6.2节的建议是「避免在值接收者上把大结构体当参数」，因为值接收者会拷贝整个对象。</span>

```go
func (p Point) ScaleByValue(factor float64) {
	p.X *= factor   // 修改的是副本，无效！
}

func (p *Point) ScaleByPtr(factor float64) {
	p.X *= factor   // 修改原值
}
```

**核心对比：值接收者 vs 指针接收者**

| 维度 | 值接收者 | 指针接收者 |
| --- | --- | --- |
| 修改原值 | 否（拷贝） | 是 |
| 大结构体代价 | 每次调用整份拷贝 | 只拷贝指针 |
| 允许 nil 检查 | 否 | 是 |
| 语法糖 | `p.ScaleByValue` | `p.ScaleByPtr` 等价于 `(&p).ScaleByPtr` |

**辨析｜易错点：** 调用上有一个自动解引用糖：变量 `p` 是值，但 `p.ScaleByPtr(2)` 也能编译——Go 自动取 `&p`。同理，指针 `&p` 也能调用值接收者方法，自动解引用。**这个糖让「接收者类型」的选择变成隐式**：混用两套接收者会导致方法集（method set）差异，影响接口满足的判断——这将在《接口》篇深入。

**一致性规则**：对同一个类型，方法要么全用值接收者、要么全用指针接收者，不要混用。混用会让「类型是否满足接口」的判断变得混乱——这是 Go 官方反复强调的纪律。

## 3 方法集与方法调用

**方法集（method set）** 决定「一个类型拥有哪些方法」。关键规则：

- **值类型 `T` 的方法集**：包含所有接收者为 `T` 的方法。
- **指针类型 `*T` 的方法集**：包含所有接收者为 `T` 与 `*T` 的方法。

这条规则是接口判断的基石：**只有指针类型 `*T` 才拥有全部方法**。<span class="marginnote">理解方法集，是理解《接口》篇「某类型是否满足接口」的关键。`var v T` 只能调用值接收者方法，而 `var p *T` 两者都能调用——因为只有指针才能保证「修改原值」的能力。这个「指针 = 全能力」的直觉，能帮你快速判断代码能否编译。</span>

```go
type Counter struct{ n int }

func (c Counter) Value() int { return c.n }      // 值接收者
func (c *Counter) Inc()      { c.n++ }           // 指针接收者

var c Counter
c.Inc()       // OK：自动取 &c
c.Value()     // OK
```

若某个接口要求 `Inc`，那么 `Counter`（值）不满足，而 `&c`（指针）满足。这个细微差别在《接口》篇会反复出现。

## 4 嵌入：组合而非继承

结构体可以**匿名嵌入**另一个结构体，被嵌入类型的方法与字段**自动提升**到外层：

```go
type ColoredPoint struct {
	Point          // 匿名嵌入，不是字段名 Point，而是类型嵌入
	Color string
}

cp := ColoredPoint{Point{1, 2}, "red"}
fmt.Println(cp.X)            // 1：字段提升
fmt.Println(cp.Distance(q))  // 方法提升：cp 自动获得 Point 的方法
```

**重点：** 嵌入不是继承。`ColoredPoint` 与 `Point` 没有「is-a」关系——你不能把 `ColoredPoint` 当作 `Point` 传给需要 `Point` 的函数。它只是「字段和方法被提升」的组合。Go 的哲学是**组合优于继承**：用嵌入复用字段与方法，用接口表达多态。

**辨析｜易错点：** 嵌入与字段同名时，外层字段遮蔽内层字段；方法的解析规则是「先找本层，再逐层向外」。当一个方法被多层提升时，接收者总是「定义该方法的类型」，而不是最外层类型——`cp.Distance(q)` 的接收者是 `Point`，不是 `ColoredPoint`。

## 5 公式解析：距离方法的方法

**「方法调用 = 普通函数调用」** 这一等价关系是理解方法的钥匙。Go 规范保证 `p.Distance(q)` 与 `Distance(p, q)` 完全等价：

$$
p.\text{Dist}(q) \equiv \text{Dist}(p, q)
$$

对 `Distance` 方法（欧氏距离）：

- **第一步，展开为函数**：`p.Distance(q)` 编译为 `Distance(p, q)`，其中 `p` 是接收者。
- **第二步，代入坐标**：`dx = q.X - p.X`，`dy = q.Y - p.Y`。
- **第三步，计算距离**：$\text{dist} = \sqrt{dx^2 + dy^2}$，即 `math.Hypot(dx, dy)`。
- **第四步，验证数值**：$p=(1,2), q=(4,6)$ 时 $dx=3, dy=4$，得 $\sqrt{9+16}=5$。

这条等价式揭示了两件事：**方法只是「带接收者的函数」**（没有魔法，语义与函数一致），以及**接收者就是普通参数**（值接收者拷贝、指针接收者共享）。理解了等价式，方法的一切行为——提升、遮蔽、方法集——都能从「函数 + 类型」推出。

## 6 小结

- **方法** 是绑定到类型的函数：`func (p Point) Dist(q Point) float64`。
- 接收者类似 `this`；**任何自定义类型都能有方法**，不只是 struct。
- 值接收者拷贝、指针接收者修改原值；**一致性规则**：同类型不混用两种接收者。
- 方法集：值类型 `T` 只有值接收者方法，指针类型 `*T` 拥有全部方法。
- **嵌入**让字段与方法提升，实现组合而非继承——`ColoredPoint` 不是 `Point`。
- `p.Dist(q)` 与 `Dist(p, q)` 等价，方法是「带接收者的函数」。

在下一节，我们让不同类型的值能够统一被对待：**接口——接口约定、类型断言与类型开关**，实现 Go 版的多态。
