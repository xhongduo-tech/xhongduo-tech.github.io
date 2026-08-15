---
title: 对象创建与销毁的最佳实践
date: 2026-08-07
---

# 对象创建与销毁的最佳实践

<div class="epigraph">
<p>一个对象从生到死，藏着一门手艺：怎么把它造出来、怎么让它别浪费、怎么让它体面地离开。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Effective Java》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从对象创建与销毁开始

Java 里对象的「生」与「死」看似都被语言包办了——`new` 创建，垃圾收集器回收。但 Effective Java 第 2 章（第 1–9 条）告诉你：**「包办」不等于「不用管」。** 对象怎么创建（构造器？静态工厂？Builder？单例？）、怎么避免重复创建、怎么在对象死后清除它占用的非内存资源——这些决策直接决定程序的性能、内存与正确性。<span class="marginnote">对比 C/C++：C 程序员手动 `malloc/free`，Java 把「内存」交给 GC。但 GC 只回收<strong>堆内存</strong>，不回收文件句柄、连接、锁——这些「非内存资源」必须手动释放。所以「对象销毁」在 Java 里的意思不是删对象，而是<strong>关资源</strong>。</span>这一篇把「对象的生老病死」四件事讲透：静态工厂、Builder、避免重复创建、及时释放资源。

## 1 静态工厂方法：替代构造器的第一选择

Effective Java 第 1 条：**考虑用静态工厂方法替代构造器。** 这里的「静态工厂」不是设计模式里的 Factory Method，而是**类里一个返回自身实例的静态方法**：

```java
public class Order {
    private Order() { }                      // 构造器私有，外部不能 new

    public static Order create() { return new Order(); }            // 简单工厂
    public static Order createPaid() { return new Order(); }        // 语义化工厂
}
Order order = Order.createPaid();
```

**静态工厂相比构造器的四大优势：**

- **有名字**：`BigInteger.probablePrime(...)` 比 `new BigInteger(...)` 清楚得多——构造器只能叫类名，工厂方法能叫 `createPaid`、`fromFile`。
- **不必每次创建新对象**：可以返回「缓存好的实例」（单例、享元）——`Boolean.valueOf(true)` 就返回缓存的常量对象。
- **能返回任何子类型**：`Collections.emptyList()` 返回的其实是内部私有子类，调用方只见到 `List` 接口——接口编程更灵活。
- **返回类型可随参数变化**：如 `EnumSet` 按元素个数返回不同实现。

**辨析｜易错点：静态工厂不是银弹。** 它不能让子类继承（构造器私有时），也难被 IDE 自动发现。惯例是工厂方法用 `of`、`from`、`valueOf`、`getInstance` 等约定命名。**多数场景静态工厂优于构造器——除非你必须让调用方 `new`。**

## 2 Builder：多参数对象的救星

**重叠构造器（telescoping constructor）**是「参数多」的经典丑解法：`new Pizza(1)`、`new Pizza(1, 2)`、`new Pizza(1, 2, 3)`……参数一多，调用方根本记不住哪个位置是什么。**JavaBean 方式**（无参构造 + setter）又允许「先造好再改」——对象在构造期间处于非法状态，且不可变类用不了它。

**Builder 模式**（Effective Java 第 2 条）介于两者之间：构造器只接收必需的参数，其余通过「链式 setter」设置，最后 `build()` 生成**不可变**对象：

```java
public class Pizza {
    private final int size;                    // 必选
    private final boolean cheese;              // 可选
    private Pizza(Builder b) { size = b.size; cheese = b.cheese; }

    public static class Builder {
        private final int size;                // 必选，构造器里传
        private boolean cheese;
        public Builder(int size) { this.size = size; }
        public Builder cheese() { this.cheese = true; return this; }   // 返回 this，链式
        public Pizza build() { return new Pizza(this); }
    }
}
Pizza p = new Pizza.Builder(10).cheese().build();
```

**重点结论：Builder 的优雅在「链式 + 不可变」**——调用链把每个参数名字写清楚，`build()` 之后对象不可变，还有机会在 `build()` 里做整体校验。代价是样板代码多一份。**参数 ≥ 4 个且多数可选时用 Builder**，两三个参数用静态工厂就够了。

## 3 避免创建不必要的对象

Effective Java 第 6 条：**避免创建不必要的对象。** 对象创建本身不贵，但「不必要」的重复创建是纯浪费：

```java
// 反例：每次调用都 new 一个 String（内容相同，纯浪费）
String s = new String("hello");     // "hello" 字面量已在字符串池，new 是重复造

// 反例：循环里重复创建正则 Pattern（编译正则很贵）
for (String line : lines) {
    Matcher m = Pattern.compile("\\d+").matcher(line);   // 每行重新编译一次正则！
}
```

**正例：把昂贵的对象提到循环外、复用不可变实例：**

```java
private static final Pattern DIGITS = Pattern.compile("\\d+");   // 类加载时编译一次
for (String line : lines) {
    Matcher m = DIGITS.matcher(line);        // 只做匹配，不重新编译
}
```

**再注意一个细节：优先使用基本类型，避免无意识的装箱。** `Integer` vs `int`：在「求和 100 万个数」这种循环里，每次 `int` 自动装箱成 `Integer` 都是对象分配。**优先基本类型，把装箱留给「确实需要对象」的场合**（进集合、作泛型参数）。<span class="marginnote">装箱的另一个坑是 `Integer` 缓存：`Integer.valueOf` 对 -128~127 复用常量对象，所以 `Integer a=100, b=100; a==b` 为 true，而 `a=200, b=200` 时 `==` 为 false——「碰巧相等」最容易误导。基本类型比较永远用 `==`，包装类用 `equals` 或 `intValue`。</span>

**辨析｜易错点：别走向另一个极端。** 为「避免创建对象」而维护一个「缓存池」的复杂度，往往超过对象本身的创建成本。原则是：**省掉「无意识」的重复创建**（循环里的 `Pattern`、不必要的 `new String`），别为了省一个廉价的小对象去自造复杂度。

## 4 消除过期的对象引用：内存泄漏的另一面

Effective Java 第 7 条：**消除过期的对象引用。** 垃圾收集器只能回收「不可达」的对象；**仍被引用但不再需要**的对象是「过期引用」，会阻止回收——这就是 Java 里的内存泄漏。

最经典的场景是**自实现栈**：

```java
public class Stack {
    private Object[] elements;
    private int size = 0;
    public Object pop() {
        if (size == 0) throw new EmptyStackException();
        Object result = elements[--size];
        elements[size] = null;      // ★ 置 null，让「出栈但仍在数组里」的元素可被回收
        return result;
    }
}
```

如果不置 `null`，`elements[size]` 还抱着那个「已弹出」的对象——数组仍引用它，GC 永远收不走。**凡是「自己管理内存」的类（数组、缓存、listener 列表），都要在元素失效时显式清引用。**

**三个高频泄漏源：**

- **缓存**：`Map` 里塞了不再用的键。用 `WeakHashMap`（键弱引用）或让缓存带过期清理。
- **监听器/回调**：注册了不注销，对象被监听器列表长期引用。
- **静态集合**：`static` 的 `List` 一旦被塞入对象，永不回收。

**重点结论：内存泄漏在 Java 里是「可避免」的——只要记住「谁持有引用，谁负责清空」。** GC 管理内存的「回收」，但「引用生命周期」是程序员的责任。

**公式解析：对象何时可被回收**

GC 回收的判据是「不可达」。一个对象是否可回收，由「从根出发的引用路径」决定：

$$
\text{对象可回收} \iff \text{不存在任何从 GC 根（GC Roots）出发的引用链到达该对象}
$$

GC 根包括：静态字段、活动线程栈、JNI 引用。栈里 `pop()` 出的对象本身没被根引用，但**数组仍引用它**——所以它「可达」、收不走。置 `null` 就是**切断这条引用链**，让它从「可达但无用」变成「不可达、可回收」。理解这条判据，你就理解了「过期引用 = 未切断的引用链」。

## 5 资源释放：try-with-resources 与终结器之殇

Java 里「对象销毁」主要分两层：内存由 GC 管，**非内存资源（文件、连接、锁）必须手动释放**。

Effective Java 第 8 条警告：**别依赖终结器（finalizer）**——`finalize()` 方法在对象回收前被调用，但它执行时机不确定、可能拖慢回收、异常会被吞掉。**永远不要用 `finalize()` 释放重要资源。**（`finalize` 在 Java 9 已被标记弃用，Java 18 被移除。）

**正确的资源释放是 try-with-resources**（Effective Java 第 9 条）：

```java
try (FileInputStream in = new FileInputStream("a.txt");
     BufferedOutputStream out = new BufferedOutputStream(...)) {
    // 用 in 读、用 out 写……
}   // 无论是否异常，两个资源都自动关闭
```

**为什么 try-with-resources 优于手写 `finally`**：手写 `finally` 时若「主体抛异常、关闭也抛异常」，关闭的异常会**覆盖**主体的异常，丢了真正的原因；try-with-resources 则把关闭异常记为**抑制异常（suppressed）**，主体异常保留——排障信息更完整。

**重点结论：资源释放用 try-with-resources，资源对象实现 `AutoCloseable`。** 需要「自管理资源」的类（自实现连接池、缓冲器）就实现 `AutoCloseable` 的 `close()`，把释放逻辑收进去。

## 6 核心对比表：三种对象创建方式

纯概念主题用**核心对比表**替代公式解析，把「对象怎么造」摆开：

| 维度 | 构造器 | 静态工厂 | Builder |
| --- | --- | --- | --- |
| 有无名字 | 无（只能叫类名） | **有**（语义化） | 有 |
| 必建新对象 | 是 | 可返回缓存实例 | 是 |
| 返回子类型 | 否 | **可** | 否 |
| 参数多时 | 重叠构造器灾难 | 尚可 | **最优雅** |
| 不可变对象 | 可 | 可 | **天然支持** |
| 样板代码 | 最少 | 少 | 较多 |

**重点结论：按参数数量与语义选创建方式——参数少用构造器/静态工厂，参数多且可选用 Builder。** 配合「避免重复创建」「及时释放资源」，你就能把对象的「生」管好。至于对象的「死」——GC 负责回收，你负责切断引用与释放资源，各司其职。

## 7 小结

- **静态工厂**有名字、可返回缓存实例、可返回子类型，优先于构造器。
- **Builder** 解决多参数：链式 + 不可变，参数 ≥ 4 且可选时用它。
- **避免不必要的对象**：循环外提取 `Pattern`、别 `new String("...")`、优先基本类型。
- **消除过期引用**：自管理内存的类要在失效时置 `null`；缓存、监听器是泄漏高发区。
- **资源用 try-with-resources** 释放，绝不依赖终结器。

在下一节，我们处理「对象如何被比较、如何被打印」——**覆盖 equals、hashCode 与 toString**。
