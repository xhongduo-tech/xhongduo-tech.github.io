---
pageClass: plain-doc
---

# Go 语言编程

Go 是 Google 于 2009 年发布的静态类型、编译型系统编程语言，以极简语法、内建并发与工程化工具链著称，是云原生、网络服务与分布式系统领域的核心语言。对标《Go语言圣经》与官方 Go 文档，从基础语法、复合类型、方法接口到并发编程、工具链与标准库循序渐进，学完即建立起完整的 Go 开发能力。

## 对标教材

- Alan A. A. Donovan & Brian W. Kernighan《Go语言圣经》(The Go Programming Language)
- 官方 Go 文档与 Effective Go（go.dev 与 go.dev/blog）
- 官方 Go 博客与 Go 1.18+ 泛型设计文档

## 主题规划

<ProgressGrid cat="cs/go-programming" />

### 第1篇

- [x] [入门：Hello World 与语言设计理念](./hello-world)
- [x] [程序结构：包、声明、变量与赋值](./program-structure)
- [x] [基础数据类型：整数、浮点、复数与字符串](./basic-data-types)
- [x] [复合数据类型：数组、slice 与 map](./arrays-slices-maps)
- [x] [复合数据类型：结构体与 JSON 序列化](./structs-json)
- [x] [函数：多返回值、匿名函数、可变参数与 defer](./functions)
- [x] [方法：指针接收者、方法与嵌入](./methods)
- [x] [接口：接口约定、类型断言与类型开关](./interfaces)

### 第2篇

- [x] [goroutine 与并发基础](./goroutines)
- [x] [channel：无缓冲与有缓冲、通道方向](./channels)
- [x] [select 多路复用、超时与关闭通道](./select-multiplexing)
- [x] [数据竞争与竞态检测](./data-races)
- [x] [sync 包：Mutex、RWMutex、WaitGroup 与 Once](./sync-package)
- [x] [并发模式：工作池、扇出扇入与并发 Web 爬虫](./concurrency-patterns)

### 第3篇

- [x] [包与模块：go mod 依赖管理与版本语义](./packages-modules)
- [x] [go 工具链：build、test、vet、fmt 与 gofmt](./go-toolchain)
- [x] [单元测试与表驱动测试](./unit-testing)
- [x] [基准测试与 pprof 性能剖析](./benchmarking-pprof)
- [x] [反射：reflect 包与动态类型操作](./reflection)
- [x] [底层编程：unsafe、cgo 与汇编](./unsafe-cgo)

### 第4篇

- [x] [Effective Go 惯用法与代码风格](./effective-go)
- [x] [错误处理与 errors 包](./error-handling)
- [x] [泛型：类型参数、类型集合与约束](./generics)
- [x] [io 与 bufio：输入输出抽象](./io-bufio)
- [x] [net/http 与 Web 服务开发](./net-http)
- [x] [context 上下文与并发取消](./context)
- [x] [encoding/json 与数据序列化](./encoding-json)
