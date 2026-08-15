---
title: 复合数据类型：结构体与 JSON 序列化
date: 2026-08-07
---

# 复合数据类型：结构体与 JSON 序列化

<div class="epigraph">
<p>数据比程序更重要；好的程序遇上坏的数据，坏的程序遇上好的数据，前者的结局通常更糟。</p>
<footer>—— 传统软件工程格言（Data dominates）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从结构体开始

slice 与 map 装的是「同质数据」，而现实对象往往是「异质」的：一个用户有姓名、年龄、邮箱、注册时间——类型各不相同。**结构体（struct）** 把不同类型的字段打包成一个整体，是 Go 里建模「对象」的主要工具。而 **JSON 序列化** 是结构体与外部世界对话的通用语言：任何 Web 服务、任何配置文件、任何分布式系统之间的消息，几乎都以 JSON 为中间格式。<span class="marginnote">对标《Go语言圣经》第4.4 节「结构体」与 4.5 节「JSON」：把这两节合成一篇，是因为它们共同回答「如何给数据建模、如何把模型搬出进程」。结构体是内存中的模型，JSON 是线上的模型。</span>

结构体在本专题是分水岭：有了它，我们才能写《函数》接收结构化参数、写《方法》给结构体挂行为、写《接口》让不同类型满足同一约定。可以说，**结构体是 Go 面向对象思想的载体**——虽然 Go 没有「类」，但 struct + 方法 + 接口组合出了等价的表达力。

## 1 结构体：字段的打包

**结构体（struct）** 用 `type` 声明，包含一组**字段（field）**，每个字段有名字与类型：<span class="marginnote">对照 Python 的 `dataclass` 与 C 的 struct：Go 的结构体语法更接近 C，但「字段名首字母大写=导出」的规则让结构体天然具备封装边界——外部包只能读写导出的字段。</span>

```go
type Employee struct {
	ID        int
	Name      string
	Address   string
	Salary    int
	ManagerID int
}

var dilbert Employee            // 全字段零值
dilbert.Name = "Dilbert"
dilbert.Salary -= 5000          // 字段可读可写
```

**关键操作：**

| 操作 | 写法 |
| --- | --- |
| 复合字面量（按序） | `p := Employee{1, "Alice", "addr", 5000, 0}` |
| 复合字面量（按名） | `p := Employee{ID: 1, Name: "Alice"}` |
| 字段访问 | `p.Salary` |
| 取地址再访问 | `&p.Salary` |

**重点：** 按名（field: value）的字面量比按序的更安全——字段增删不会悄悄错位，且漏掉的字段自动取零值。Go 规范甚至要求**不可导出类型的外部包只能用按名方式**构造。

**结构体是值类型**：赋给另一个变量会整体拷贝。`q := p` 之后改 `q` 不影响 `p`——除非用指针 `q := &p`。这与 slice 的共享语义形成对照：struct 拷贝便宜（通常几十字节），而 slice 拷贝只复制头。

## 2 结构体指针与组合

在函数间传递大型结构体时，通常用**指针** `*Employee` 避免拷贝。结构体指针的字段访问自动解引用：<span class="marginnote">`p.Salary` 与 `(*p).Salary` 等价——Go 在「指针的字段访问」上做了语法糖，省去 C 里 `(*p).x` 的啰嗦。但这层糖在初学时常引发困惑：到底什么时候 `p` 是值、什么时候是指针？规则是：看声明类型。</span>

```go
func GiveRaise(p *Employee, amount int) {
	p.Salary += amount   // 等价于 (*p).Salary += amount
}
```

**组合（composition）** 而非继承，是 Go 组织结构的核心思想。一个结构体可以嵌入另一个结构体：

```go
type Point struct{ X, Y int }

type Circle struct {
	Point        // 匿名嵌入
	Radius int
}

c := Circle{Point{1, 2}, 5}
fmt.Println(c.X)      // 1：字段提升（promotion）
```

嵌入字段 `Point` 的字段 `X`、`Y` 被**提升**到外层 `Circle`，可以直接 `c.X` 访问。这构成了 Go 版本的「继承」：嵌入让内层结构体的字段与方法「冒泡」到外层，但类型关系是「有一个（has-a）」而非「是一个（is-a）」。这将在《方法：指针接收者、方法与嵌入》与《接口》两篇中进一步展开。

**核心对比：结构体组合 vs 类继承**

| 维度 | Go 结构体组合 | 传统类继承 |
| --- | --- | --- |
| 关系 | has-a（嵌入） | is-a（继承） |
| 运行时类型改变 | 静态 | 可多态（虚表） |
| 字段提升 | 自动冒泡 | 需显式继承 |
| 设计哲学 | 组合优于继承 | 继承层次 |

## 3 JSON：结构与线上的桥

**JSON（JavaScript Object Notation）** 是一种文本数据交换格式。Go 用 **`encoding/json`** 包在结构体与 JSON 之间互相转换：<span class="marginnote">对标《Go语言圣经》4.5 节：JSON 之所以在 Go 里「一等公民」，是因为 `encoding/json` 利用<strong>反射（reflect）</strong>自动读取结构体的字段标签（tag），把「结构体 ↔ JSON」的样板代码压缩到近乎为零。反射机制我们会在《反射：reflect 包与动态类型操作》篇专门讲解。</span>

```go
type Movie struct {
	Title  string
	Year   int  `json:"released"`
	Color  bool `json:"color,omitempty"`
	Actors []string
}

movies := []Movie{
	{Title: "Casablanca", Year: 1942, Color: false, Actors: []string{"Bogart", "Bergman"}},
}

data, err := json.Marshal(movies)
if err != nil {
	log.Fatalf("JSON marshaling failed: %s", err)
}
fmt.Printf("%s\n", data)
```

输出：

```json
[{"Title":"Casablanca","released":1942,"Actors":["Bogart","Bergman"]}]
```

**字段标签（struct tag）** 控制 JSON 键名：`json:"released"` 把字段名从 `Year` 映射为 `released`；`json:"color,omitempty"` 表示零值时省略该字段（所以 `false` 没出现）。<span class="marginnote">struct tag 是字符串字面量，用反引号包裹：`json:"released"`。它在运行时对 `reflect` 可见，因此成了 Go 生态最流行的「元数据」约定——从 JSON 到 ORM 都靠它驱动。</span>

**反序列化**用 `json.Unmarshal`：

```go
var got []Movie
if err := json.Unmarshal(data, &got); err != nil {
	log.Fatalf("JSON unmarshaling failed: %s", err)
}
fmt.Println(got[0].Title)   // Casablanca
```

## 4 序列化的边界与安全

JSON 序列化不是万能银弹，几个边界要认清：

**只序列化导出字段**：`json.Marshal` 只处理首字母大写的字段。小写字段静默忽略——这不是 bug，而是封装的体现：不想暴露给 JSON 的字段用小写。

**解析错误要检查**：`json.Unmarshal` 对字段类型不匹配、JSON 语法错误会返回 `error`，且会把字段留在零值。忽略错误 = 用错误数据继续跑，这在《错误处理与 errors 包》会系统对待。

**辨析｜易错点：** 对**不匹配的 JSON**（如 `null` 值、未知字段），Go 的默认行为是宽容的——未知字段被忽略，`null` 置零值。若想要严格校验，用 `json.Decoder` 并检查 `decoder.Token()`，或手动校验必需字段。宽容是默认，严格是选择。

**性能提示**：`json.Marshal` 用反射遍历结构体，速度一般。对高吞吐热路径，可用 `jsoniter` 或手写编码。但 95% 场景下标准库足够——先正确，再优化，这是《基准测试与 pprof》篇的教训。

## 5 公式解析：内存中的字段布局

结构体的内存布局可用**对齐公式**描述。`unsafe.Sizeof` 告诉我们每个字段的字节偏移必须是对齐值的倍数：

$$
\text{offset}(\text{field}_i) \equiv 0 \pmod{\text{align}(\text{field}_i)}
$$

以 `struct{ A int8; B int64; C int8 }` 为例：

- **第一步，定基础对齐**：`int64` 的对齐值是 8，`int8` 是 1，整体对齐取最大者 8。
- **第二步，逐个放置**：`A` 在偏移 0；`B` 需偏移是 8 的倍数，故从偏移 8 开始（1–7 是填充）。
- **第三步，尾部补齐**：`C` 在偏移 16，结构体总大小须是 8 的倍数，故补到 24。
- **第四步，重排优化**：改成 `struct{ A int8; C int8; B int64 }`，总大小降到 16——两个小字段合并，填充减少。

这条公式解释了《unsafe-cgo》篇「字段重排省内存」的根源，也解释了为什么结构体大小不一定等于字段大小之和——**对齐与填充是编译器的布局决策**。依赖特定布局的代码必须谨慎，因为它随架构与编译器版本可能变化。

## 6 小结

- **结构体**把异质字段打包：按名字面量更安全，漏字段自动零值。
- 结构体是**值类型**，赋值整体拷贝；传大结构体用指针。
- **组合**（嵌入+字段提升）代替继承，构成 Go 的「对象建模」方式。
- **`encoding/json`** 用反射 + 字段标签完成结构体 ↔ JSON 双向转换。
- JSON 只序列化导出字段；解析错误必须检查；默认宽容、严格是选择。
- 结构体内存布局由**对齐规则**决定，字段重排可显著减小体积。

在下一节，我们把行为挂到数据上：**函数——多返回值、匿名函数、可变参数与 defer**，让代码从「声明」走向「执行」。
