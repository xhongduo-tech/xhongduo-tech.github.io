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