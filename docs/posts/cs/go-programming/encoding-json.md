---
title: encoding/json 与数据序列化
date: 2026-08-07
---

# encoding/json 与数据序列化

<div class="epigraph">
<p>序列化是让内存中的对象能跨进程、跨语言、跨时间存活的通用语言。</p>
<footer>—— 数据交换格言（Serialization bridges systems）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第4章 + encoding/json 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 JSON 序列化开始

结构体是内存中的模型，但程序之间要交换数据，必须先把内存模型翻译成「传输格式」。**JSON** 是当下最通用的数据交换格式——几乎所有 Web API、配置文件、分布式系统消息都以它为中间语言。Go 的 **`encoding/json`** 用反射自动完成「结构体 ↔ JSON」的双向转换，把序列化的样板代码压缩到一行。<span class="marginnote">对标《Go语言圣经》第4.5 节「JSON」与 `encoding/json` 官方文档：圣经详细讲了字段标签、`Marshal`/`Unmarshal`、以及 `json.Decoder`/`json.Encoder` 的流式版本。JSON 在 Go 里是「一等公民」，因为反射让通用序列化成为可能（见《反射》篇）。</span>

JSON 在本专题的定位：它是《结构体与 JSON》篇的深化与《net/http》篇的服务端拼图——HTTP 请求体/响应体大多是 JSON，`json.Decoder`/`json.Encoder` 直接对接 `io.Reader`/`io.Writer`，与 I/O 抽象无缝衔接。

## 1 Marshal：结构体到 JSON

**`json.Marshal`** 把 Go 值编码成 JSON 字节切片：<span class="marginnote">`json.Marshal(v)` 返回 `([]byte, error)`。编码规则：导出字段→JSON 键，字段名即键名（可被标签改写）；值类型映射——`int`→数字、`string`→字符串、`bool`→true/false、`nil`/`[]`→null/数组。</span>

```go
type User struct {
	Name    string   `json:"name"`
	Age     int      `json:"age,omitempty"`
	Emails  []string `json:"emails,omitempty"`
	Secret  string   `json:"-"`
}

u := User{Name: "Alice", Age: 30, Emails: nil, Secret: "hidden"}
data, err := json.Marshal(u)
if err != nil { log.Fatal(err) }
fmt.Println(string(data))
// {"name":"Alice","age":30}
```

**输出解读：**

- `Name` → `"name"`（标签改写键名）。
- `Age` → `"age"`，30 非零值所以保留。
- `Emails` 为 `nil`，`omitempty` 让它**被省略**。
- `Secret` 的标签 `json:"-"` 让字段**永远不序列化**。

**重点：** 标签（struct tag）是 JSON 序列化的控制面板：`json:"name"` 改键名、`json:"age,omitempty"` 省略零值、`json:"-"` 排除字段。多个选项用逗号分隔。

**易错点：** **只有导出的字段会被序列化**——小写字段被静默忽略。若 JSON 里缺少某个键，那不是 bug，是「你没导出」；若想忽略字段又保留导出，用 `json:"-"`。

## 2 Unmarshal：JSON 到结构体

**`json.Unmarshal`** 把 JSON 解码回 Go 值：

```go
data := []byte(`{"name":"Bob","age":25}`)
var u User
if err := json.Unmarshal(data, &u); err != nil {
	log.Fatal(err)
}
fmt.Println(u.Name, u.Age)   // Bob 25
```

**关键行为：**

- JSON 键**按名称匹配**字段——`"name"` 匹配标签为 `name` 的字段，也匹配 `Name`（不区分大小写、忽略下划线）。
- JSON 中**多余的键**被忽略；**缺失的键**留字段零值。
- JSON `null` → Go 零值；JSON 数组 → Go slice；嵌套对象 → 嵌套结构体。

**要点：** `Unmarshal` 的入参是**指针** `&u`——它需要往结构体里写数据。JSON 数字会按目标字段类型转换：`"age":25` 若目标 `Age` 是 `int` 则解码成功；若类型不匹配（如字符串进 int）返回错误。

**易错点：** 必须**检查 `Unmarshal` 的错误**——JSON 语法错误、类型不匹配都会返回 `err`，此时 `u` 只被部分填充。忽略错误 = 用脏数据继续跑（《错误处理》篇的纪律再次生效）。

## 3 流式编解码：Encoder 与 Decoder

**`json.Decoder`** 从 `io.Reader` 流式解码，**`json.Encoder`** 向 `io.Writer` 流式编码——它们是 HTTP、文件等「流」场景的正确工具：<span class="marginnote">`json.NewDecoder(r)` 包装任意 `io.Reader`，`Decode(&v)` 每次读一个 JSON 值；`json.NewEncoder(w)` 包装 `io.Writer`，`Encode(v)` 写一个 JSON 值。流式版本省掉「先 `ReadAll` 再 `Unmarshal`」的内存拷贝，且天然处理「一整个文件是多个 JSON」的场景。</span>

```go
// 从请求体流式解码
dec := json.NewDecoder(r.Body)
var u User
if err := dec.Decode(&u); err != nil {
	http.Error(w, "bad json", http.StatusBadRequest)
	return
}

// 向响应体流式编码
w.Header().Set("Content-Type", "application/json")
enc := json.NewEncoder(w)
enc.Encode(u)
```

**核心对比：Marshal/Unmarshal vs Encoder/Decoder**

| 维度 | Marshal/Unmarshal | Encoder/Decoder |
| --- | --- | --- |
| 输入/输出 | `[]byte` | `io.Reader`/`io.Writer` |
| 内存 | 整体进内存 | 流式、增量 |
| 场景 | 小数据、内存中 | 大文件、网络、管道 |
| 多值 | 一次一个 | 循环 `Decode` 多个 |

**易错点：** `Encoder.Encode` 输出**带换行符**的 JSON；`Decoder` 可以连续 `Decode` 多个值（每行一个 JSON）。判断「流是否读完」用 `Decode` 返回 `io.EOF`。

## 4 自定义 MarshalJSON / UnmarshalJSON

当默认序列化不满足需求时，可以给类型实现 `json.Marshaler` 与 `json.Unmarshaler` 接口，**自定义编码/解码**：

```go
type Duration struct {
	time.Duration
}

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.Duration.String())   // "1h30m0s"
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	d.Duration = parsed
	return nil
}
```

**用途：** 把 `time.Duration` 序列化成人类可读的 `"1h30m"` 而非原始纳秒数；把时间戳格式化成特定时区；对敏感字段加密。实现 `MarshalJSON`/`UnmarshalJSON` 后，这个类型在嵌套结构体里也自动使用自定义逻辑。<span class="marginnote">`MarshalJSON`/`UnmarshalJSON` 是 `json.Marshaler`/`json.Unmarshaler` 接口的方法。标准库在 `time.Time` 上实现了它们（序列化为 RFC 3339 时间串），这就是「Go 时间进 JSON 变成 `"2026-08-07T09:30:00Z"`」的原因——见《time 与日期时间》篇。</span>

**易错点：** 在 `UnmarshalJSON` 内部再调用 `json.Unmarshal` 时要小心**无限递归**——用别名类型（`type alias Duration`）剥离方法集再解码。这是 `UnmarshalJSON` 最常见的坑。

## 5 公式解析：omitempty 的判定

**`omitempty` 选项的判定本质是「该字段是否为空值」。** 空值集合

$$
\text{empty}(v) \iff v = \text{零值} \lor v = \text{空切片} \lor v = \text{nil map} \lor v = \text{nil 指针/接口}
$$

对字段逐项判定是否省略：

- **第一步，取字段值**：如 `Emails []string` 的当前值。
- **第二步，查零值**：`nil` 切片是零值，`omitempty` 命中——省略。
- **第三步，非空保留**：`[]string{"a"}` 非零值，保留并输出 `["a"]`。
- **第四步，特别提醒**：`Age int` 为 `0` 时也会被省略——**若「0」是有意义的业务值，不要给该字段加 `omitempty`**，否则数据无声丢失。

这条判定式的工程启示：`omitempty` 是「优雅输出」的利器，但**不要对「零值有意义」的字段使用**。JSON 编码的每个选项都要想清楚「省略后，接收方能读懂吗」。

## 6 小结

- **`json.Marshal`** 结构体→JSON：标签改键名、`omitempty` 省零值、`json:"-"` 排除字段。
- **`json.Unmarshal`** JSON→结构体：按标签/名称匹配、多余键忽略、缺失留零值、**检查错误**。
- **`json.Encoder`/`Decoder`** 流式编解码：对接 `io.Reader`/`Writer`，适合网络与大文件。
- 自定义 `MarshalJSON`/`UnmarshalJSON` 满足特殊格式（`Duration`→`"1h30m"`）。
- **只有导出字段被序列化**；`omitempty` 对「零值有意义」的字段慎用。
- 反射 + 标签是 JSON 自动化的根基，性能热路径可用 `jsoniter` 或手写编码替代。

到这里，本专题「从语法到工程」的全部 28 篇就收束了。你已从 Hello World 出发，走过了数据类型、复合结构、函数、方法与接口，进入了并发世界（goroutine、channel、select、数据竞争、sync 与模式），又回到工程工具（模块、工具链、测试、性能、反射、底层），最后在 Effective Go、错误处理、泛型、I/O、Web 与 context 中打磨成完整能力。**Go 是一门「少即是多」的语言——愿你带着这套最小但完备的词汇表，去阅读 Docker、Kubernetes、etcd 的源码，去写出真正经得起并发与时间考验的系统。**
