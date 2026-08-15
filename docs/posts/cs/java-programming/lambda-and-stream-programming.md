---
title: Lambda 与 Stream 流式编程
date: 2026-08-07
---

# Lambda 与 Stream 流式编程

<div class="epigraph">
<p>Stream 让「怎么算」让位于「算什么」——从命令式到声明式的优雅一跃。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Lambda 与 Stream 开始

接口那章你学了 lambda 的语法，这一章看它如何**改变写代码的方式**。Java 8 引入的 **Stream API** 是集合处理的革命：不再用 `for` 循环一步步「命令」机器怎么算，而是用 `filter`、`map`、`collect` 等算子**声明**「我要什么结果」。命令式写法关注「每一步怎么做」，流式写法关注「数据怎么变换」——后者更贴近人类思考，也更少出错（没有下标、没有中间变量）。Effective Java 第 7 章告诉你何时该用、何时该克制。这一篇把 Stream 的管道模型、常用算子与最佳实践一次讲透。

## 1 Stream 管道：三阶段的流水线

**Stream（流）**不是集合——它**不存数据**，而是一条「数据变换流水线」。一条流的生命周期分三个阶段：

$$

\text{数据源} \to \text{中间操作（惰性）} \to \text{终止操作（立即执行）}

$$

```java
List<String> topNames = names.stream()          // 1. 数据源
        .filter(n -> n.length() > 3)            // 2. 中间操作：过滤
        .map(String::toUpperCase)               // 2. 中间操作：映射
        .sorted()                               // 2. 中间操作：排序
        .limit(3)                               // 2. 中间操作：截断
        .collect(Collectors.toList());          // 3. 终止操作：收成 List
```

三个要点：

- **数据源**：集合 `.stream()`、数组 `Arrays.stream(arr)`、值 `Stream.of(a, b, c)`。
- **中间操作（intermediate）是惰性的**：`filter`、`map` 只是「登记了要做的事」，**不会真正执行**。它们返回新的 Stream，直到终止操作出现才被「点燃」。
- **终止操作（terminal）触发执行**：`collect`、`forEach`、`count`、`reduce` 一出现，整条流水线立即跑起来。<span class="marginnote">惰性不是实现细节，而是设计：它让 `limit(3)` 能做到「短路」——只要凑够 3 个结果就停下，不必处理全量数据。这来自函数式编程的惰性求值传统，也是 Stream 能处理无限流的理论基础。</span>

**一条流的纪律**：**流只能用一次**。终止操作执行后，这条流就「耗尽」了，再操作会抛 `IllegalStateException`。要重跑就得重新从数据源建一条。

## 2 常用算子：filter、map、collect 与 reduce

把最常用的算子分三类记：

**过滤与映射**——流式处理的左右手：

```java
// filter：按条件保留（Predicate<T>）
stream.filter(p -> p.getSalary() > 8000)
// map：逐元素变换（Function<T,R>）
stream.map(Employee::getName)
// flatMap：把「流中的流」展平（合并子流）
stream.flatMap(list -> list.stream())
```

`flatMap` 是处理「集合的集合」的关键：`List<List<Integer>>` 要算所有元素的总和，先 `flatMap` 展平成 `Stream<Integer>`，再 `reduce`。

**收集与归约**——把流变回普通数据：

```java
// collect：收进集合（最常用）
List<String> list = stream.collect(Collectors.toList());
Set<String> set = stream.collect(Collectors.toSet());
Map<String, List<Employee>> byDept =
        stream.collect(Collectors.groupingBy(Employee::getDept));  // 分组
String joined = stream.collect(Collectors.joining(", "));          // 拼接
// reduce：逐元素归约成一个值
int sum = stream.reduce(0, Integer::sum);
```

`groupingBy` 是 SQL 里 `GROUP BY` 的化身——按键把元素分组进 Map。<span class="marginnote">`collect` 家族是 Stream 的「出口」：`toList`、`toSet`、`toMap`、`groupingBy`、`partitioningBy`（按 boolean 分两群）、`joining`。写熟这些收集器，流式代码就顺了——它们对应 SQL 的 SELECT、GROUP BY、字符串聚合。</span>

**重点结论：一个 Stream 用例 = 数据源 + 若干中间操作 + 一个终止操作。** 把这个三段式钉死，任何流式代码都能拆解成可读的流水线。

## 3 公式解析：reduce 的折叠语义

`reduce` 是流式编程里最抽象也最核心的算子——它把整条流**折叠**成一个值。签名是：

$$

\text{reduce}(\text{identity},\; (a, b) \to \text{combiner})

$$

对这条公式做三步拆解：

- **第一步，读签名**：`reduce` 接收两个参数——**`identity`**（起始值/恒等元，如 `0`、`""`）和 **`combiner`**（把两个元素并成一个的二元函数，如 `Integer::sum`）。
- **第二步，折叠过程**：从 identity 出发，逐个与流里的元素做 combiner：`((0 + 1) + 2) + 3`。combiner 满足结合律时，并行流可以分块折叠再合并——这就是并行归约的数学基础。
- **第三步，看结果**：整条流被「折」成一个值——求和、求积、取最大、拼字符串都是它的特例。`Optional` 版 `reduce((a,b) -> ...)`（无 identity）流为空时返回 `Optional.empty()`。

```java
int total = orders.stream()
        .map(Order::getAmount)
        .reduce(0, Integer::sum);         // 求订单总金额

Optional<String> longest = words.stream()
        .reduce((a, b) -> a.length() >= b.length() ? a : b);   // 找最长词
```

**reduce 与 collect 的分工**：`reduce` 把流折成一个**不可变值**（sum、max、拼接），`collect` 把流收进**可变容器**（`List`、`Map`、`StringBuilder`）——**「归约成值」用 reduce，「收集进容器」用 collect**。多数需求 `collect` 更常用，`reduce` 是更底层的折叠原语。

## 4 公式解析：Stream 与循环的等价变换

命令式循环与流式管道是**同一件事的两种写法**，理解等价性才能自如切换：

$$
\text{循环：} \quad \text{中间变量} \xrightarrow{\text{逐元素修改}} \text{结果} \qquad \text{Stream：} \quad \text{数据源} \xrightarrow{\text{算子管道}} \text{终止收集}
$$

```java
// 循环版：计算「工资超过 8000 的员工姓名，按工资降序取前 3」
List<String> names = new ArrayList<>();
List<Employee> rich = new ArrayList<>();
for (Employee e : staff) if (e.getSalary() > 8000) rich.add(e);
rich.sort((a, b) -> Double.compare(b.getSalary(), a.getSalary()));
for (int i = 0; i < Math.min(3, rich.size()); i++) names.add(rich.get(i).getName());

// Stream 版：同样的逻辑，一行管道
List<String> names = staff.stream()
        .filter(e -> e.getSalary() > 8000)          // 过滤
        .sorted(Comparator.comparingDouble(Employee::getSalary).reversed())  // 降序
        .limit(3)                                    // 取前 3
        .map(Employee::getName)                      // 取姓名
        .collect(Collectors.toList());               // 收成 List
```

**重点结论：Stream 的每一步都是「声明要什么」，循环的每一步是「命令怎么做」。** 流式版没有中间变量、没有下标、没有「先收集再排序再截断」的三段样板——它更贴近「筛选 → 排序 → 截断 → 取名」的思考顺序。这也是 Effective Java 第 45 条「流要优先于循环」的理由：**流更不易出错、更可读**。

## 5 何时用流、何时该克制

Effective Java 第 45–48 条给出流的使用边界——**不是所有循环都要改写成流**：

**该用流**：过滤 + 映射 + 收集的管道、分组聚合（`groupingBy`）、需要惰性短路（`limit`、`findFirst`）、并行归约（`.parallel()`）。

**该克制（用循环）**：

- 需要**修改外部状态**（流式写法易写成「副作用」）。
- 需要**多个集合同步下标遍历**（流没有下标）。
- 逻辑是**分支密集**的嵌套控制流（`break`/`continue`/`return` 在流里没有直译）。
- 代码在**性能关键路径**且测出流式是瓶颈（流有对象分配、无法内联时）。

**辨析｜易错点：别为「花哨」而流。** 一条流管道里塞进 `forEach` 里再改外部变量、或一个 `map` 里做三重嵌套分支——那比循环还难读。**判断标准是「可读性」：管道若在一行内说不清「在做什么」，就拆回循环。** Effective Java 第 48 条的总纲：**「谨慎使用流并行」**——`.parallel()` 不是免费提速，它要求元素操作无状态、可重复、combiner 结合，否则结果错误或更慢。

## 6 小结

- Stream 三阶段：**数据源 → 惰性中间操作 → 终止操作**；流只能用一次。
- 常用算子：`filter`/`map`/`flatMap`/`sorted`/`limit` + `collect`/`reduce`/`forEach`。
- `reduce` 折叠成值（identity + combiner），`collect` 收进容器（`toList`/`groupingBy`/`joining`）。
- **优先流**（声明式、少错）；但改外部状态、多集合对齐、分支密集时**用循环**。
- `.parallel()` 慎用：要求无状态、可结合，否则结果错误。

在下一节，我们把「方法怎么设计」的规范提上日程——**方法设计：参数校验、重载与返回值**。