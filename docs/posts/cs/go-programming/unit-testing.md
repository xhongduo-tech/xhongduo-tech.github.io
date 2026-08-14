---
title: 单元测试与表驱动测试
date: 2026-08-07
---

# 单元测试与表驱动测试

<div class="epigraph">
<p>写测试不是在写额外的工作，而是在写对「你的代码应该做什么」的精确记录。</p>
<footer>—— Go 测试哲学（Testing in Go）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从测试开始

代码能编译不等于正确——「看起来能跑」与「在边界情况下依然正确」之间隔着整个测试体系。Go 把测试框架**内置**在标准工具链里（`go test`），用最少的仪式感让写测试成为日常动作。而 **表驱动测试（table-driven tests）** 是 Go 社区标志性的测试风格：把「输入-期望输出」的用例集中成一张表，一个测试函数遍历执行——新增用例只需加一行，而不是复制整个函数。<span class="marginnote">对照第八级《软件工程》课程的测试理论：单元测试对应「验证函数/模块的独立正确性」，边界值分析（边界用例）、等价类划分（典型用例）在这里都有直接体现。Go 的表驱动风格让「一个测试一个关注点、用例平铺成表」成为天然实践。</span>

## 1 测试文件与测试函数

Go 的测试文件以 **`_test.go`** 结尾，与被测代码同包。测试函数以 **`Test`** 开头，接收 `*testing.T`：

```go
// reverse.go
func Reverse(s string) string {
	b := []byte(s)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return string(b)
}

// reverse_test.go
package main

import "testing"

func TestReverse(t *testing.T) {
	got := Reverse("abc")
	want := "cba"
	if got != want {
		t.Errorf("Reverse(%q) = %q, want %q", "abc", got, want)
	}
}
```

运行：

```bash
$ go test
PASS
ok      example/mypkg  0.003s
```

测试函数里的断言方式：

| 方法 | 行为 | 适用 |
| --- | --- | --- |
| `t.Error(...)` / `t.Errorf(...)` | 记录失败但继续执行 | 还想看后续断言 |
| `t.Fatal(...)` / `t.Fatalf(...)` | 记录失败并立即停止 | 后续断言依赖本步 |

**辨析｜易错点：** 用 `t.Errorf` 还是 `t.Fatalf` 要看场景：`Fatal` 在「当前状态已不可信」时使用（如解析配置失败），`Error` 在「还能继续检查其它用例」时使用。表驱动测试里子用例失败应 `t.Errorf`，让一个循环跑完所有用例。

## 2 表驱动测试：用例平铺成表

**表驱动测试**把「输入、期望输出、可选标签」集中到一张匿名结构体切片，循环执行：

```go
func TestReverseTable(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"single", "a", "a"},
		{"palindrome", "abcba", "abcba"},
		{"two words", "hello world", "dlrow olleh"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {   // 子测试
			got := Reverse(tt.in)
			if got != tt.want {
				t.Errorf("Reverse(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
```

`t.Run(name, func)` 创建**子测试**，它带来两个好处：<span class="marginnote">子测试的威力：`go test -run TestReverseTable/empty` 可以<strong>只跑「empty」这一个子用例</strong>；子测试可以单独并行（`t.Parallel()`）；失败时报告精确到用例名。这让「定位到具体失败用例」变成一条命令的事。配合 `-run` 正则，调试体验极佳。</span>

1. **精确定位**：失败输出 `TestReverseTable/empty`，一眼看出哪个用例挂了。
2. **局部执行**：`go test -run 'TestReverseTable/(two words)'` 只跑指定子测试。

**易错点：** 循环变量捕获——子测试闭包里引用 `tt` 时，Go 1.22 之前需 `tt := tt` 拷贝，否则并行子测试可能读到最后一个元素。标准库的新测试风格已默认安全，但读旧代码时要认得这个模式。

## 3 辅助函数与干净断言

复杂测试常需要**辅助函数（helper）**，用 `t.Helper()` 标记后，失败信息会指向**调用者**而不是辅助函数内部：

```go
func assertReverse(t *testing.T, in, want string) {
	t.Helper()                       // 标记为辅助函数
	got := Reverse(in)
	if got != want {
		t.Errorf("Reverse(%q) = %q, want %q", in, got, want)
	}
}
```

调用 `assertReverse(t, "abc", "cba")` 时，若失败，堆栈定位在**这一行**，而不是辅助函数内部——调试时省去一层跳转。

## 4 测试覆盖与边界用例

`go test -cover` 输出语句覆盖率：

```bash
$