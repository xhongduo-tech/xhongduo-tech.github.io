---
title: 接口：接口约定、类型断言与类型开关
date: 2026-08-07
---

# 接口：接口约定、类型断言与类型开关

<div class="epigraph">
<p>接口的强大之处在于它小；越小越好。</p>
<footer>—— 罗勃 · 派克（Rob Pike，Go 接口设计理念）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从接口开始

方法让「类型拥有行为」，接口（interface）则让「行为本身成为类型」。Go 的接口是一种**隐式契约**：只要一个类型的方法集「恰好包含」接口要求的全部方法，它就自动满足该接口——**不需要显式声明**「我实现了这个接口」。这种「鸭子类型」由编译器静态检查，把「多态」从继承层次里解放出来，让代码按行为组织，而不是按类型血缘组织。<span class="marginnote">对标《Go语言圣经》第7章：第7.1节开门见山——「Go 的接口提供了比传统面向对象语言更灵活的组合方式」。标准库的 `io.Reader`、`io.Writer` 只有一两个方法，却是整个 I/O 生态的轴心，这正是「接口越小越强」的证明。</span>

接口在本专题是「面向对象能力」的收尾：struct 打包数据、方法挂行为、接口定契约，三者拼出 Go 的对象模型。接下来的第2篇将进入并发——而接口是理解 `io.Reader`、`net/http`、`context` 等标准库设计的钥匙，它们几乎全是「小而美」的接口。

## 1 接口：行为即类型

**接口类型** 定义一组方法签名，作为「行为的约定」：

```go
type Shape interface {
	Area() float64
	Perimeter() float64
}
```

**核心概念：任何实现了 `Area()` 与 `Perimeter()` 方法的类型，都自动满足 `Shape` 接口**——无需任何显式声明。这种「结构化满足」（structural typing）是 Go 接口的精髓：

```go
type Circle struct{ R float64 }
func (c Circle) Area() float64      { return math.Pi * c.R * c.R }
func (c Circle) Perimeter() float64  { return 2 * math.Pi * c.R }

type Square struct{ S float64 }
func (s Square) Area() float64      { return s.S * s.S }
func (s Square) Perimeter() float64  { return 4 * s.S }

func PrintShape(sh Shape) {
	fmt.Printf("面积 %.2f，周长 %.2f\n", sh.Area(), sh.Perimeter())
}

PrintShape(Circle{2})   // OK：Circle 满足 Shape
PrintShape(Square{3})   // OK：Square 满足 Shape
```

<span class="marginnote">「接口是否被满足」由编译器在<strong>编译期</strong>静态检查。与 Java 的 `implements`、C++ 的虚继承相比，Go 不需要类型「知道」接口的存在——只要方法集对得上，接口就成立。这让「给别人的类型补上接口」成为可能：你不能改 `time.Duration`，却可以给它定义方法（在方法篇见过）让它满足自己的接口。</span>

**接口值（interface value）** 是「具体类型 + 具体值」的二元组：接口变量存储的动态类型与动态值。`var sh Shape = Circle{2}` 时，`sh` 的动态类型是 `Circle`，动态值是 `Circle{2}`。

## 2 接口约定的规则：方法集

一个类型是否满足接口，由**方法集（method set）**决定——这正是《方法》篇「值 vs 指针接收者」的实战场景：

**重点：** 若接口要求的方法中有指针接收者方法，则只有**指针类型**满足该接口。

```go
type Areaer interface {
	Area() float64
}

type Rect struct{ W, H float64 }
func (r Rect) Area() float64 { return r.W * r.H }   // 值接收者

var a Areaer = Rect{2, 3}   // OK：Rect 的方法集包含 Area()
// var a Areaer = &Rect{2, 3}   // 也 OK：指针的方法集包含全部
```

若改为 `func (r *Rect) Area() float64`（指针接收者），则 `Rect`（值）**不再**满足 `Areaer`，而 `&Rect` 满足。

**辨析｜易错点：** 这个「指针才拥有全方法集」的规则常让人困惑。直觉是：值接收者方法的副本语义意味着「值也能调用」；而指针接收者方法可能修改原值，「只拿到副本的值」无法保证这一语义，所以值类型不满足。**当接口方法包含指针接收者时，把接口变量声明为指针类型**，是惯用法。

**空接口 `interface{}`**（Go 1.18 起可用别名 `any`）没有方法，因此**所有类型都满足它**。它等价于「不承诺任何行为」，常用于「装任何类型」的容器——但代价是失去了编译期类型信息，取出时必须用类型断言（见下节）。

## 3 类型断言：取回具体类型

**类型断言（type assertion）** 从接口值中取出动态类型的值：`x.(T)`，其中 `x` 是接口值，`T` 是目标类型。

```go
var sh Shape = Circle{2}
c, ok := sh.(Circle)    // 断言动态类型是 Circle
if ok {
	fmt.Println("它是圆，半径", c.R)
}
```

类型断言有两种形态：

| 形态 | 失败时 | 适用 |
| --- | --- | --- |
| `v := x.(T)` | panic | 确信类型正确 |
| `v, ok := x.(T)` | `ok=false`，`v` 为零值 | 需要容错 |

**重点：** 单返回值断言失败会 **panic**，用 `, ok` 双返回值形式更安全。这是「断言必须验证」的纪律——与《程序结构》篇「`v, ok := m[k]` 检查 map 键存在」是同一精神。

**类型开关（type switch）** 是类型断言的批处理版本，用 `switch x.(type)` 按动态类型分派：

```go
func describe(x any) string {
	switch v := x.(type) {
	case nil:
		return "nil"
	case int:
		return fmt.Sprintf("整数 %d", v)
	case string:
		return fmt.Sprintf("字符串 %q", v)
	case Circle:
		return fmt.Sprintf("圆半径 %.2f", v.R)
	default:
		return "未知类型"
	}
}
```

在 `case` 分支里，变量 `v` 被**自动转换为该分支的具体类型**——这是类型开关比手写一串断言更优雅的原因。

## 4 接口组合与 error

接口可以**嵌入**接口，组合出更大契约——与结构体嵌入同理：

```go
type Reader interface { Read(p []byte) (n int, err error) }
type Writer interface { Write(p []byte) (n int, err error) }

type ReadWriter interface {
	Reader
	Writer
}
```

`ReadWriter` 要求同时实现 `Read` 与 `Write`。标准库正是用这种方式，从 `io.Reader`、`io.Writer`、`io.Closer` 组合出 `io.ReadCloser` 等常用接口。<span class="marginnote">接口组合让「依赖倒置」成为语言惯例：高层代码依赖小而窄的接口（如 `io.Reader`），底层实现五花八门（文件、网络、内存、压缩流），但都能被同一个接口接住。这就是《软件架构》课程里「面向接口编程」的 Go 表达。</span>

**`error` 接口**是 Go 最重要的接口——它只声明一个方法：

```go
type error interface {
	Error() string
}
```

任何实现 `Error() string` 方法的类型都可以作为错误返回。这正是《函数》篇「错误即值」的根基：错误不是内建魔法，而是「一个实现了 `Error()` 的普通类型」——这让错误可以携带任意结构化信息（超时、重试次数、内部错误链），《错误处理与 errors 包》篇会系统展开。

## 5 公式解析：接口满足的判定

**「一个类型满足接口」的判定可用集合语言精确表达**：设 `M(T)` 是类型 `T` 的方法集，`M(I)` 是接口 `I` 要求的方法集，则

$$
T \text{ 满足 } I \iff M(I) \subseteq M(T)
$$

对 `Shape` 接口（要求 `Area`、`Perimeter`）与 `Circle` 类型逐步验证：

- **第一步，列方法集**：`M(Circle) = {Area, Perimeter}`，`M(Shape) = {Area, Perimeter}`。
- **第二步，求包含关系**：`{Area, Perimeter} ⊆ {Area, Perimeter}` 成立，故满足。
- **第三步，考虑指针**：若 `Area` 改为指针接收者，`M(Circle)`（值）就只剩 `Perimeter`，包含关系不成立——`Circle` 不满足，`*Circle` 才满足。
- **第四步，动态判定**：运行时接口值存储的动态类型必须满足接口，否则赋值即 panic。

这条判定式的直觉是：**接口就是「最低方法要求」**，类型只要「方法多到覆盖要求」就合格，多出来的方法无关紧要。这也是为什么「接口越小越好」——要求的方法越少，能被满足的类型越多，代码的适用面越广。

## 6 小结

- **接口** 是行为契约：类型的方法集包含接口的全部方法即自动满足，无需显式声明。
- **接口值** 是「动态类型 + 动态值」二元组；空接口 `any` 满足所有类型。
- 方法集规则：接口含指针接收者方法时，**只有指针类型满足接口**。
- **类型断言** `x.(T)` 取回具体类型；`v, ok := x.(T)` 容错，单值形式失败会 panic。
- **类型开关** `switch x.(type)` 按动态类型分派，分支内变量自动类型化。
- 接口可嵌入组合；`error` 接口只有一个 `Error()` 方法，是「错误即值」的根基。

在下一节，我们进入 Go 最鲜明的名片：**goroutine 与并发基础**——用 `go` 关键字让程序真正「同时做很多事」。
