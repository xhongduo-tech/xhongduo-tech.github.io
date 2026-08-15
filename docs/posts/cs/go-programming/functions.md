---
title: 函数：多返回值、匿名函数、可变参数与 defer
date: 2026-08-07
---

# 函数：多返回值、匿名函数、可变参数与 defer

<div class="epigraph">
<p>函数是程序的组织单位：一个函数做好一件事，程序就清晰了。</p>
<footer>—— 布赖恩 · 克尼汉（Brian W. Kernighan）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从函数开始

前面的篇章里函数已经反复出现——`main`、`fmt.Println`、`json.Marshal`——现在到了把它当作一等公民的时候。**函数（function）** 是 Go 程序的基本组织单位：一段可命名、可复用、可传递的代码块。Go 的函数设计里有四个特色：**多返回值**（让错误处理成为值）、**匿名函数与闭包**（把代码当数据传）、**可变参数**（`...`）、**defer**（资源释放的优雅方式）。<span class="marginnote">对标《Go语言圣经》第5章：这一章专门讲函数，因为函数是「把数据变成行为」的枢纽。多返回值支撑了 Go 全生态的 `(value, err)` 约定，defer 则把「清理」写到了使用处旁边——这两点深刻影响了每个 Go 项目的写法。</span>

函数在本课程的坐标：它是《程序结构》中「声明」的落地，也是后续《方法》《接口》的基石——方法本质上是「绑定到类型的函数」。理解函数，才谈得上理解 Go 的一切抽象。

## 1 函数声明与多返回值

**函数声明**由 `func` 关键字、函数名、参数列表、返回值列表与函数体组成：

```go
func add(a, b int) int {
	return a + b
}
```

**多返回值**是 Go 函数的标志性能力——最常见的形态就是 `(value, err)`：

```go
func sqrt(x float64) (float64, error) {
	if x < 0 {
		return 0, fmt.Errorf("sqrt of negative %g", x)
	}
	return math.Sqrt(x), nil
}
```

`(float64, error)` 表示返回两个值：结果与错误。<span class="marginnote">多返回值让「函数调用即赋值」成为可能：`v, err := sqrt(9)` 一次拿到结果与错误。这与 C 的「通过指针参数返回错误码」、与 Python 的「异常」形成鲜明对比——Go 把错误当成普通值显式传递。</span>

**命名返回值（named result）**：返回值可以命名，在函数体内当作局部变量使用，`return` 不带参数即可返回它们的当前值：

```go
func f(a int) (x int) {
	x = a * 2
	return   // 返回 x 的当前值
}
```

命名返回值让「填充式」构建返回值的代码更清晰，但要小心：裸 `return` 容易让读者不知道返回什么，规范建议仅在返回逻辑简单时使用。

## 2 错误处理：错误即值

Go 没有异常（exception）。错误以**返回值**的形式显式传递，调用者必须处理：

```go
f, err := os.Open("data.txt")
if err != nil {
	return err     // 向上传播
}
defer f.Close()
```

**约定**：函数返回多个值时，**错误约定为最后一个**；成功时错误为 `nil`。调用者**必须检查错误**——忽略错误意味着用错误的数据继续跑，这正是《错误处理与 errors 包》篇系统展开的主题。

**辨析｜易错点：** 检查错误的惯用写法是 `if err != nil { return ... }`。初学者的常见错误是「忘记检查」或「把错误当字符串比较」。用 `errors.Is(err, os.ErrNotExist)` 而不是 `err.Error() == "..."` 来判断错误类型——前者语义化、可嵌套，后者脆弱。

## 3 匿名函数与闭包

**匿名函数**没有名字，可以赋值给变量、作为参数传递、立即调用。当匿名函数**引用外层函数的变量**时，它就构成**闭包（closure）**——这个函数「捕获」了外层变量，即使外层函数已返回，变量仍存活：<span class="marginnote">闭包是「函数 + 捕获的环境」。对照第一级《基础数学》的「函数是映射」：闭包把「规则」与「规则要用到的上下文」打包成一体。这一概念在《函数式编程》课程中会进一步深化。</span>

```go
func adder() func(int) int {
	sum := 0
	return func(x int) int {
		sum += x
		return sum
	}
}

add := adder()
fmt.Println(add(1))   // 1
fmt.Println(add(2))   // 3：sum 被捕获，持续累积
```

**重点：** 闭包捕获的是**变量本身**，不是其值快照。若循环里创建闭包，注意变量的捕获时机：

```go
for _, v := range []int{1, 2, 3} {
	defer func() { fmt.Println(v) }()   // 打印 3,3,3（Go 1.22 前）
}
```

Go 1.22 起 `range` 变量每次迭代创建新实例，此问题已修复；但旧代码里这种陷阱依然存在——这正是《单元测试与表驱动测试》篇提到「循环变量捕获」的根源。识别这个模式，是读 Go 老代码的基本功。

## 4 可变参数与 defer

**可变参数（variadic）** 用 `...` 表示「任意多个同类型参数」：

```go
func sum(nums ...int) int {
	total := 0
	for _, n := range nums {
		total += n
	}
	return total
}

fmt.Println(sum(1, 2, 3))    // 6
fmt.Println(sum())           // 0
```

在函数内部，`nums` 是一个 `[]int` 切片。向可变参数传递一个已有切片，用 `nums...` 展开：

```go
values := []int{1, 2, 3}
fmt.Println(sum(values...))   // 6
```

**`defer`** 延迟执行一个函数调用，直到**外层函数返回之前**执行。最典型的用途是释放资源——打开即安排关闭：<span class="marginnote">defer 的执行顺序是<strong>后进先出（LIFO）</strong>：后 defer 的先执行。`defer` 在函数返回前运行，无论函数是正常返回还是 panic。这保证了「打开就要关、加锁就要解锁」的成对操作永不遗漏。</span>

```go
func copyFile(dst, src string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()          // 函数结束时关闭 in

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()         // 后 defer，先执行

	_, err = io.Copy(out, in)
	return err
}
```

**defer 的常用场景：**

- 资源释放：`Close()`、`Unlock()`。
- 记录耗时：`defer func() { t := time.Since(start); log.Println(t) }()`。
- 捕获 panic：`defer func() { if r := recover(); r != nil { ... } }()`。

**辨析｜易错点：** defer 的**参数在 defer 语句处求值**，而函数体在返回时才执行：

```go
func f() {
	x := 1
	defer fmt.Println(x)   // 打印 1（参数已求值）
	x = 2
}
```

若想打印「返回时的 x」，需用闭包捕获变量：`defer func() { fmt.Println(x) }()`。

## 5 核心对比：Go 函数与 C、Python 的函数设计

| 维度 | Go | C | Python |
| --- | --- | --- | --- |
| 多返回值 | 一等公民 | 无（用指针参数） | 元组（隐式） |
| 错误处理 | 返回 `(value, err)` | 错误码 + `errno` | `raise` 异常 |
| 匿名函数/闭包 | 支持 | 支持（函数指针） | 支持 |
| 可变参数 | `...` 切片 | `va_arg` 宏 | `*args`/`**kwargs` |
| 延迟执行 | `defer` | 无（手动清理） | `with` 上下文 / `try-finally` |

这张表揭示 Go 的设计取舍：**显式 > 隐式**。错误不靠异常「悄悄抛出」，而是靠返回值「摆在明面」；资源清理不靠「析构函数」，而是靠 `defer` 写在打开处旁边。它牺牲了部分简洁，换来了「代码路径全可见」——这在大型工程里价值极大。

## 6 小结

- 函数声明：`func 名(参数) 返回`；**多返回值**让 `(value, err)` 成为全生态约定。
- 错误即值：函数返回多个值，**错误约定为最后一个**，调用者必须检查。
- 匿名函数可赋值、可传参；引用外层变量即构成**闭包**，捕获变量本身而非快照。
- 可变参数 `...` 收集为切片；已有切片用 `x...` 展开。
- `defer` 延迟到函数返回前执行，**后进先出**，用于资源释放与 panic 恢复。
- defer 的参数在语句处求值，取「返回时的值」需用闭包。

在下一节，我们把函数绑定到类型：**方法——指针接收者、方法与嵌入**，给结构体挂上行为。
