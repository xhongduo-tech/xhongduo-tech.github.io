---
title: 数组与字符串处理
date: 2026-08-07
---

# 数组与字符串处理

<div class="epigraph">
<p>数据是程序的血肉，容器是数据的收纳盒。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第3章 ｜ 2026-08-07</p>
</div>

## 为什么从数组与字符串开始

流程控制让程序「会走」，数据容器让程序「有东西可走」。几乎每一个真实程序的第一步，都是把一堆数据装起来——一个班 40 个学生的成绩、一篇文章的字符流。**数组（array）**是「同类型元素的定长序列」，**字符串（String）**是「字符组成的不可变序列」；前者是容器，后者是 Java 中最常用的对象类型。这一篇讲清数组的创建与拷贝、String 的不可变设计与字符串池，它们是后面集合框架（可变的、可增长的容器）的对照基线。

## 1 数组：声明、创建与初始化

Java 数组是一块**固定长度**的内存，所有元素类型相同。声明与创建分开写：

```java
int[] scores;               // 声明：一个 int 数组变量（尚未分配内存）
scores = new int[5];        // 创建：分配能装 5 个 int 的连续内存
```

**重点结论：`int[] scores` 是「数组变量」，它存的是数组对象的引用**——`new int[5]` 才真正在堆上分配内存。声明与创建也可以合并，并用花括号字面量直接初始化：

```java
int[] primes = {2, 3, 5, 7, 11};        // 字面量初始化，长度由元素个数推断
String[] names = new String[]{"张三", "李四"};
```

**下标从 0 开始**，合法范围是 `0 ~ length-1`。访问 `primes[5]` 会抛出 **`ArrayIndexOutOfBoundsException`**——这是 Java 的边界检查：数组越界在运行期被拦下并报错，而不是像 C 那样静默写出相邻内存。<span class="marginnote">C 里 `int a[5]; a[10] = 1;` 会污染栈上其他变量，酿成难以排查的内存破坏；Java 的边界检查牺牲了一点性能，却换来「越界必报错」的确定性——这是教科书级的「安全优先」设计取舍。</span>

**数组的长度**通过字段 `length` 获取（注意不是方法）：`primes.length` 得 5。**遍历数组**最常用 for-each：

```java
for (int p : primes) {
    System.out.println(p);
}
```

**多维数组**是「数组的数组」。Java 的多维数组可以是**不规则的**（每行长度不同），这与 C 的连续矩阵不同：

```java
int[][] grid = new int[3][];
grid[0] = new int[2];   // 第一行 2 个元素
grid[1] = new int[3];   // 第二行 3 个元素
```

## 2 数组的拷贝与排序

数组是引用类型，直接赋值只是拷贝**引用**而非内容：

```java
int[] a = {1, 2, 3};
int[] b = a;        // b 与 a 指向同一块内存
b[0] = 99;          // a[0] 也变成 99 —— 别名陷阱
```

**辨析｜易错点：别名（aliasing）。** `b = a` 之后，改 `b` 就是改 `a`，因为它们共享同一个数组对象。要真正复制内容，用 `Arrays.copyOf` 或 `System.arraycopy`：

```java
int[] c = Arrays.copyOf(a, a.length);   // 复制全部
int[] d = Arrays.copyOf(a, 2);          // 只复制前 2 个，常用于扩容
```

`java.util.Arrays` 是数组的瑞士军刀，高频方法包括：

`Arrays.toString(arr)`：把数组转成 `[1, 2, 3]` 的可读字符串——直接打印数组只会看到 `[I@1b6d3586` 这种地址，新手必踩。
`Arrays.sort(arr)`：原地排序（对基本类型是快速排序，对对象是稳定的归并排序）。
- `Arrays.fill(arr, 0)`：把数组每个元素填成指定值。
- `Arrays.binarySearch(arr, key)`：二分查找，返回下标；找不到返回负的插入点。<span class="marginnote">`binarySearch` 要求数组已排序，否则结果未定义——「先排序再二分」是算法课上的标准姿势，在第三级《算法设计与分析》的分治篇会有严格复杂度证明。</span>

**扩容技巧**：数组定长，需要增长时用 `Arrays.copyOf` 复制到更大的新数组。这个「复制 + 增长」的模式正是后面集合框架中 `ArrayList` 内部实现的原始版本。

## 3 String：不可变的字符序列

`String` 是 Java 中使用频率最高的类，它有三个必须记住的设计：

**第一，String 是不可变对象（immutable）**。一旦创建，其内容不能被修改。`"abc".toUpperCase()` 不会改掉 `"abc"`，而是返回一个新的 `"ABC"` 字符串。<span class="marginnote">不可变带来了线程安全与缓存友好——多个线程共享同一个 String 不需要加锁；字符串字面量也能安全地被放进「字符串池」复用。代价是拼接大量字符串会频繁创建新对象，所以有了 `StringBuilder`。</span>

**第二，字符串字面量共享「字符串池（string pool）」**。同样的字面量只存一份：

```java
String s1 = "hello";
String s2 = "hello";
System.out.println(s1 == s2);   // true —— 池中同一对象
String s3 = new String("hello");
System.out.println(s1 == s3);   // false —— new 强制新建对象
```

**辨析｜易错点：比较字符串要用 `equals`，不要用 `==`。** `==` 比较的是引用是否指向同一对象；`equals` 比较的是内容是否相等。上面的 `s1 == s3` 为 `false`，但 `s1.equals(s3)` 为 `true`。用 `==` 比较字符串是 Java 新手第一高频错误——即使在池化场景偶尔「碰巧对」，一旦字符串来自拼接或 IO 就会失效。

**第三，常用方法**几乎都返回新字符串：`length()`、`charAt(i)`、`substring(a, b)`、`indexOf(sub)`、`startsWith/endsWith`、`replace`、`trim`、`split(regex)`、`toUpperCase/toLowerCase`。`String` 拼接用 `+`，编译器会把它翻译成 `StringBuilder.append` 的链式调用。

**StringBuilder** 是可变字符串，适合大量拼接：

```java
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.append(i).append(",");
}
String result = sb.toString();   // 最后转成 String
```

循环里用 `+` 拼接 1000 次会创建 1000 个中间 String；用 `StringBuilder` 只在最后产生一个——性能差一个数量级。

## 4 核心对比表：String、StringBuilder 与 StringBuffer

三个字符串相关类型常被放在一起比较：

| 维度 | String | StringBuilder | StringBuffer |
| --- | --- | --- | --- |
| 可变性 | 不可变 | 可变 | 可变 |
| 线程安全 | 安全（不可变） | 不安全 | 安全（方法加锁） |
| 性能 | 拼接慢 | 快 | 稍慢（锁开销） |
| 适用场景 | 固定文本、键值 | 单线程大量拼接 | 多线程共享拼接 |

**重点结论：单线程拼接用 `StringBuilder`，多线程共享才用 `StringBuffer`**——`StringBuffer` 的方法被 `synchronized` 保护，安全但慢；而现代并发编程更推荐用不可变 String 或局部 StringBuilder，避免无谓的锁竞争。这个「不可变 vs 可变」的取舍，在 Effective Java 第 17 条「使可变性最小化」里会上升到设计原则层面。

## 5 公式解析：substring 的开闭区间

`substring` 的参数约定是 Java 里最容易被记错的小细节——**起始包含、结束排除**，即左闭右开区间：

$$

\text{result} = s[\text{beginIndex} \;\ldots\; \text{endIndex} - 1]

$$