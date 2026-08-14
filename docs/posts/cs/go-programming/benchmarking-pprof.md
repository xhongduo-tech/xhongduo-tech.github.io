---
title: 基准测试与 pprof 性能剖析
date: 2026-08-07
---

# 基准测试与 pprof 性能剖析

<div class="epigraph">
<p>先让它正确，再让它快。</p>
<footer>—— 经典性能优化格言（Make it work, make it right, make it fast）</footer>
</div>

<div class="article-byline">
<p>第三级 · Go 语言编程 ｜ The Go Programming Language 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从性能分析开始

程序正确之后，「快不快」成为下一个问题。但「快」不能靠感觉——直觉常常错得离谱。Go 提供了两条**用数据说话**的路径：**基准测试（benchmark）**精确测量一段代码的耗时，**pprof** 剖析整个程序的 CPU 与内存热点。它们让你先找到真正的瓶颈，再针对优化，而不是对着空气猜。<span class="marginnote">对照第一级《基础数学》的「先证明正确再谈计算」与第五级《计算机体系结构》的 Amdahl 定律：优化只能加速「占比大的部分」。pprof 的价值正是告诉你「哪部分占比大」——没有剖析就去优化，是在违反 Amdahl 定律地浪费力气。</span>

## 1 基准测试：测量一段代码

**基准测试（benchmark）** 函数以 `Benchmark` 开头，接收 `*testing.B`：

```go
func BenchmarkReverse(b *testing.B) {
	s := "a quick brown fox jumps over the lazy dog"
	for i := 0; i < b.N; i++ {
		Reverse(s)
	}
}
```

运行：

```bash
$ go test -bench=. -benchmem
BenchmarkReverse-8   3562251   316.3 ns/op   48 B/op   2 allocs/op
```

输出解读：

`3562251`：循环执行次数（框架自动调整 `b.N` 直到计时稳定）。
`316.3 ns/op`：**每次操作平均耗时**。
- `48 B/op` 与 `2 allocs/op`：每次操作的**内存分配量**与**分配次数**——`-benchmem` 开启。分配次数往往比耗时更能揭示性能问题。

**辨析｜易错点：** 基准测试的计时可能被「首次调用预热」污染。应保证被测函数**足够稳定**，且避免编译器把「结果未使用的调用」优化掉——把结果赋给包级变量（`result = Reverse(s)`）即可防止被「死代码消除」。

## 2 基准测试对比：用数据选方案

基准测试最常见的用途是**对比两种实现**。例如对比字符串拼接的两种方式：

```go
func BenchmarkConcatPlus(b *testing.B) {
	s := "go"
	for i := 0; i \lt  b.N; i++ {
		result = s + s + s + s
	}
}

func BenchmarkConcatBuilder(b *testing.B) {
	s := "go"
	for i := 0; i \lt  b.N; i++ {
		var sb strings.Builder
		sb.WriteString(s)
		sb.WriteString(s)
		sb.WriteString(s)
		sb.WriteString(s)
		result = sb.String()
	}
}
```

`-bench` 支持正则选组、`-benchtime` 控制时长：

```bash
$ go test -bench=Concat -benchmem -benchtime=1s
```

结果对比告诉我们：**小字符串拼接 `+` 足够快，大量拼接才需要 `strings.Builder`**——这正是「用数据而不是直觉选方案」的注脚。<span class="marginnote">`strings.Builder` 在内部预分配缓冲区、避免每次拼接都新建字符串，因此在「循环内反复拼接」的场景远快于 `+`。但小规模拼接时 `+` 的优化已足够好——这个「规模决定选型」的结论，只有基准测试能给出来。想深入可对照第三级《数据结构》动态数组的扩容策略。</span>

## 3 pprof：剖析 CPU 与内存

**pprof（profile）** 是 Go 的性能剖析工具，回答「时间花在哪、内存耗在哪」。两种使用方式：

**方式一：命令行生成剖析文件**

```go
import _ "net/http/pprof"

// 在 main 中启动一个 HTTP 端口
go func() {
	log.Println(http.ListenAndServe("localhost:6060", nil))
}()
```

```bash
$ go tool pprof http://localhost:6060/debug/pprof/profile   # CPU 剖析 30s
$ go tool pprof http://localhost:6060/debug/pprof/heap       # 内存剖析
```

**方式二：直接在测试里出剖析文件**

```bash
$ go test -cpuprofile cpu.out -memprofile mem.out .
$