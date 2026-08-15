---
title: 泛型程序设计
date: 2026-08-07
---

# 泛型程序设计

<div class="epigraph">
<p>泛型让「算法与数据结构」脱离具体类型而存在——写一次，用于任何类型。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第8章 ｜ 2026-08-07</p>
</div>

## 为什么从泛型开始

写了这么多类，你有没有想过一个尴尬：`ArrayList` 存字符串、存整数、存员工，代码长得一模一样，难道要每种类型写一份？**泛型（generics）**就是答案——它把「类型」本身变成参数，让容器类写一次、装任何类型，且在**编译期**就保证类型正确。没有泛型的 Java（1.4 之前），从 `List` 里取元素要做手工强转，取错类型要等到运行期才炸。泛型把「类型错误」从运行期提前到编译期——这是 Java 类型系统最成功的一次升级。这一篇讲透泛型类、泛型方法与类型擦除。

## 1 泛型类：用类型参数做占位符

**泛型类**在类名后加尖括号声明**类型参数（type parameter）**：

```java
public class Box<T> {
    private T content;

    public void set(T content) { this.content = content; }
    public T get() { return content; }
}
```

`T` 是占位符，使用时用**类型实参（type argument）**替换：

```java
Box<String> sBox = new Box<>();
sBox.set("hello");
String s = sBox.get();        // 无需强转，编译器知道它是 String
// sBox.set(42);              // 编译错误：Box<String> 不接受 Integer
```

**重点结论：泛型容器在编译期就能拒绝错误类型。** 对比 JDK 1.4 时代：`List list = new ArrayList(); String s = (String) list.get(0);`——取出后手工强转，若第 0 个元素其实是 Integer，强转在**运行期**才抛 `ClassCastException`。泛型让这个错误提前到**编译期**：`sBox.set(42)` 直接编译失败。

**类型参数命名约定**：`T`（Type）、`E`（Element，集合元素）、`K`/`V`（Key/Value）、`R`（Return）。这只是约定，但全世界的 Java 代码都遵守，看到 `E` 就知道是元素类型。<span class="marginnote">泛型出现前，Java 用 `Object` 兜底一切类型——`ArrayList` 里存 `Object`，取出来强转。泛型并没有消灭 `Object` 化，而是在编译期<strong>自动插入强转并校验类型</strong>：你看源码是干净的 `String s = list.get(0)`，编译器在幕后帮你补上了 `(String)` 强转。</span>

## 2 泛型方法与有界类型参数

**泛型方法**在返回类型前声明自己的类型参数，与所在类是否泛型无关：

```java
public static <T> T getMiddle(T[] a) {
    return a[a.length / 2];
}
String mid = getMiddle(new String[]{"a", "b", "c"});   // 编译器推断 T = String
```

调用时类型参数由实参**推断**，通常无需显式写出 `<String>getMiddle(...)`。

**有界类型参数（bounded type parameter）**用 `extends` 限制类型参数必须满足的条件：

```java
public static <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) > 0 ? a : b;
}
```

`<T extends Comparable<T>>` 表示「T 必须实现了 `Comparable` 接口」。为什么需要这个界？因为 `max` 体内要调用 `a.compareTo(b)`，编译器必须确认 `a` 有这个能力——`extends` 界保证了这一点。<span class="marginnote">注意这里 `extends` 的语义被拓宽了：不只是「继承类」，还包括「实现接口」。`T extends B` 的含义是「T 是 B 的子类型」，B 既可以是类也可以是接口——通配符的界、后面 Stream 的泛型界都沿用这个读法。</span>

**辨析｜易错点：类型参数不能是基本类型。** `Box<int>` 是编译错误，必须用包装类 `Box<Integer>`。因为泛型在 JVM 里最终要落到 `Object` 上，而 `int` 不是对象。好在你几乎感觉不到——自动装箱/拆箱（`int ↔ Integer`）替你把活儿干了，代价是极小的一次对象分配。

## 3 类型擦除：泛型在 JVM 里的真相

**关键机制：泛型信息在运行期会被擦除（erasure）。** JVM 里根本没有 `Box<String>` 与 `Box<Integer>` 两个类，只有一份 `Box`，类型参数 `T` 被替换为它的上界（无界时是 `Object`）：

$$

\text{源码} \;\; \text{Box<T>} \xrightarrow{\text{编译}} \text{字节码} \;\; \text{Box（T 被擦除为 Object）}

$$

对 `Box<String>` 的 `get()`，编译器在调用点自动插入 `(String)` 强转。所以：

- **泛型类只有一个 Class 对象**：`Box<String>.class` 与 `Box<Integer>.class` 是同一个 `Box.class`。
- **运行期拿不到类型实参**：你不能在方法里写 `if (obj instanceof T)`，因为运行期 `T` 已经没了。
- **不能 `new T()`、不能 `new T[10]`**：创建对象需要知道具体类型，而 `T` 被擦除了。解决方案是传入 `Class<T>` 作为「类型令牌」，或用 `Array.newInstance` 反射创建。

**辨析｜易错点：泛型类型不能直接用在 `instanceof` 与 `new` 上**——`obj instanceof Pair<String>` 编译错误，`new T()` 编译错误。这些都是「擦除」的直接后果，理解擦除，这些限制全部顺理成章。

## 4 公式解析：擦除后方法冲突为何能漏网

擦除带来一个著名陷阱：**两个签名只在泛型参数上不同的方法，擦除后签名相同**，会冲突。

$$

\text{boolean equals(String)} \xrightarrow{\text{擦除}} \text{boolean equals(Object)} \;\Longrightarrow\; \text{与 Object.equals 冲突}

$$

对这条公式做三步拆解：

- **第一步，你写了重载**：一个泛型类里写了 `boolean equals(String)`（想重载 `Object.equals`）。
- **第二步，擦除发生**：编译后类型参数消失，`String` 参数被擦成 `Object`——方法签名变成 `boolean equals(Object)`。
- **第三步，冲突**：这与 `Object.equals(Object)` **签名完全相同**，而你的返回类型、语义都对不上——轻则意外覆盖，重则编译报错。

**擦除带来的四条限制，背住它们：**

- **`instanceof T` 非法**：运行期没有 `T` 这个类型，没法判断。`if (obj instanceof T)` 编译错误。
- **`new T()`、`new T[10]` 非法**：创建对象需要具体类型，`T` 被擦除了。要「按类型建对象」就传入 `Class<T>`（类型令牌）或反射。
- **静态字段不能是类型参数**：`private static T instance;` 非法——静态字段属于类，而类只有一份，无法为每个 `T` 各存一份。
- **泛型类不能直接用于 `instanceof`**：`obj instanceof Pair<String>` 编译错误——擦除后只有一个 `Pair`，无法区分 `Pair<String>` 与 `Pair<Integer>`。要区分就取 `pair.getClass()` 或检查元素。

**桥方法（bridge method）**：擦除还会让编译器**自动生成桥方法**保证多态正确。比如 `Comparable<T>` 的 `compareTo(T)` 被擦成 `compareTo(Object)`，编译器会插入一个 `compareTo(Object)` 转发到你的 `compareTo(Employee)`——所以你重写 `compareTo(Employee)` 也能通过 `Collections.sort` 的多态调用。看到 `.class` 里多出「没见过的方法」，多半就是桥方法。

## 5 泛型的最佳实践小结

泛型的内容到此收束，把「用得对」的要点汇总：

- **类型参数用约定命名**：`T`（Type）、`E`（Element）、`K`/`V`（Key/Value）、`R`（Return）。
- **泛型优先于 `Object` + 强转**：类型安全从运行期提前到编译期。
- **有界类型参数 `T extends B` 表达「T 是 B 的子类型」**——`B` 既可以是类也可以是接口。
- **理解擦除**：运行期只有一份 `Pair`，类型信息只存在于编译期。
- **不能 `new T()`、不能 `instanceof T`、静态字段不能是类型参数**——都是擦除的直接后果。

**重点结论：泛型是「编译期的类型安全」+「运行期的 Object」双层结构。** 源码里干净的类型、字节码里被擦掉的参数、调用点自动插入的强转——这三件事合起来，就是「泛型在 JVM 里的真相」。理解擦除，你不会再被「为什么不能 new T」「为什么 instanceof 报错」卡住；也自然理解下一章为什么「泛型与集合」要绑定使用——`List<T>` 是集合框架类型安全的根基。

## 6 小结

- **泛型类/方法**用类型参数占位，编译期保证类型正确、自动强转。
- **有界类型参数** `T extends Comparable<T>` 保证类型有指定能力。
- **类型擦除**：JVM 里只有一份类，`T` 被替换为上界；调用点自动插入强转。
- 擦除限制：**不能 `new T()`、不能 `instanceof T`、静态字段不能是类型参数**。
- 泛型把「类型错误」从运行期提前到编译期，是集合框架类型安全的根基。

在下一节，我们把泛型应用到一整套数据结构库——**Java 集合框架**。