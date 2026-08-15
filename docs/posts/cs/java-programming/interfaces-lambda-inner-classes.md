---
title: 接口、lambda 表达式与内部类
date: 2026-08-07
---

# 接口、lambda 表达式与内部类

<div class="epigraph">
<p>接口定义「能做什么」，lambda 让「怎么做」变成一行代码，内部类把「辅助逻辑」藏进宿主类。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第6章 ｜ 2026-08-07</p>
</div>

## 为什么从接口开始

继承给了你「一个类扩展另一个类」的能力，但 Java 是**单继承**——一个类只能有一个父类。现实需求却常是「既是 X 又是 Y」：一个 `Employee` 既能被 `Comparable` 排序，又能被 `Runnable` 丢进线程跑。**接口（interface）**就是解决「单继承不够用」的方案：它只声明「能做什么」（方法的签名），不管「怎么做」，一个类可以**实现任意多个接口**。<span class="marginnote">接口是 Java 对 C++「多继承」的改造——只继承「契约」而不继承「实现」，从根上避开菱形继承的混乱。接口的成员天然是 `public`，因为它存在的意义就是被别人调用。</span>这一章再进一步：接口 + lambda 表达式让「传一段行为给方法」成为可能，内部类则在背后处理各种「局部辅助」场景。

## 1 接口：行为契约

**接口（interface）**用 `interface` 声明，用 `implements` 实现。接口里的方法默认是 **`public abstract`**（无需显式写），字段默认是 **`public static final`**（常量）。

```java
public interface Comparable<T> {
    int compareTo(T other);          // 抽象方法：只签名，无实现
}

public class Employee implements Comparable<Employee> {
    private double salary;
    @Override
    public int compareTo(Employee other) {
        return Double.compare(salary, other.salary);   // 负、零、正
    }
}
```

**重点结论：接口 = 契约。「实现接口」=「承诺我能提供这些行为」。** 调用方只依赖接口，不关心具体实现——`Collections.sort(list)` 只要 `list` 里的元素实现了 `Comparable` 就能排，不管元素是 `Employee` 还是 `String`。这就是「面向接口编程」：依赖抽象，不依赖细节。

**Java 8 之后接口有了两个新成员：**

- **默认方法（default method）**：带 `default` 关键字、有方法体的方法。作用是**平滑演化**——给已发布的接口加方法时，用默认方法提供兜底实现，已有的实现类不用改就能编译。
- **静态方法（static method）**：接口里也能有 `static` 方法，如 `Comparator.comparing(...)`，是工具方法的合法栖身地。

**辨析｜易错点：接口「多实现」遇到两个同名默认方法会冲突。** 一个类实现了两个都有 `default void log()` 的接口，编译器要求该类必须**显式重写** `log()` 解决冲突，否则报错。默认方法只是「省事」，不是「免责」。

## 2 抽象类 vs 接口：两种「半成品」

Java 里还有一类「半成品」——**抽象类（abstract class）**：用 `abstract` 修饰、不能被实例化的类，它把部分方法实现好、把部分方法留给子类。

| 维度 | 抽象类 | 接口 |
| --- | --- | --- |
| 关键字 | `abstract class` | `interface` |
| 继承方式 | `extends`（单继承） | `implements`（多实现） |
| 构造器 | 有 | 没有 |
| 字段 | 可以有实例字段 | 只能是常量 |
| 方法 | 抽象 + 已实现混搭 | 默认都是抽象的（Java 8 后有 default/static） |
| 语义 | 「is-a」：本质是同类 | 「can-do」：能力清单 |

**重点结论：能用接口就用接口，需要共享字段/构造器/已有实现时才用抽象类。** 接口描述「对象能做什么」，抽象类描述「对象是什么」。一个 `Dog` 可以同时实现 `Swimmable` 和 `Runnable`（两个能力），但只能 `extends Animal`（一个本质）——「能力可以叠加，本质只有一个」。

## 3 lambda 表达式：把行为当参数

**lambda 表达式**是「一段可传参的代码块」，它的本质是**匿名函数**：只保留参数、箭头与方法体，不写类名、方法名。

```java
// 传统匿名内部类：啰嗦
Runnable r1 = new Runnable() {
    public void run() { System.out.println("hello"); }
};

// lambda：一行
Runnable r2 = () -> System.out.println("hello");

// 带参数与返回：Comparator 的 lambda 版
Comparator<Employee> bySalary =
        (a, b) -> Double.compare(a.getSalary(), b.getSalary());
```

**lambda 能赋值给接口类型的先决条件：该接口必须是「函数式接口（functional interface）」**——**只有一个抽象方法**的接口。`Runnable`（一个 `run`）、`Comparator`（一个 `compare`）都是函数式接口。编译器据此推断 lambda 就是那个唯一抽象方法的实现。<span class="marginnote">Java 8 用 `@FunctionalInterface` 注解标记这类接口——它不强制，但让编译器帮你检查「是不是真的只有一个抽象方法」。`java.util.function` 包预置了大量函数式接口：`Function<T,R>`、`Predicate<T>`、`Consumer<T>`、`Supplier<T>`，是 lambda 与 Stream 的通用插座。</span>

**lambda 对「捕获的变量」有要求——必须是「事实上不可变」的（effectively final）**：lambda 内部使用的局部变量，在 lambda 定义后不能被重新赋值，否则编译错误。这是为了保证线程安全与可预测性：lambda 可能在其他线程执行，可变的外部变量会让行为不可测。

## 4 内部类：藏在类里的类

**内部类（inner class）**是定义在另一个类内部的类，用于组织「只服务于宿主类」的逻辑。三种常见形态：

**成员内部类**：作为宿主的普通成员，可以访问宿主的所有成员（包括私有）——它隐式持有宿主对象的引用。

**局部内部类**：定义在方法内部，作用域只限该方法。

**匿名内部类（anonymous class）**：没有类名的内部类，在 `new` 时现场定义。它正是 lambda 的「前身」——lambda 只是把「只有一个方法」的匿名内部类写得更短。

**静态内部类（static nested class）**：用 `static` 修饰、**不持有宿主引用**的内部类，如 `Map.Entry`、`Integer` 等包装类的内部实现。

**辨析｜易错点：非静态内部类隐式持有宿主引用。** 这意味着「内部类对象」会阻止「宿主对象」被垃圾回收——宿主还在使用中的假象会造成**内存泄漏**（长期存活的内部类对象把已无用的宿主钉在堆上）。用不到宿主实例成员时，**一律用静态内部类**。

**公式解析：lambda 的「单抽象方法」判定**

lambda 能否替代某个接口，取决于它是不是函数式接口——判定式：

$$
\underbrace{\text{接口抽象方法数}}_{=1} \;\Longrightarrow\; \text{可被 lambda 实现} \qquad
\underbrace{\text{抽象方法数}}_{\ge 2} \;\Longrightarrow\; \text{只能写匿名内部类}
$$

数一数抽象方法：`Runnable` 有 1 个 `run()`，所以 `() -> ...` 合法；`MouseListener` 有 5 个抽象方法，所以它没有 lambda 版，只能匿名内部类。**「一方法即函数式」**这个计数规则，是判断「能不能写 lambda」的唯一标准。

## 5 核心对比表：匿名内部类 vs lambda

| 维度 | 匿名内部类 | lambda 表达式 |
| --- | --- | --- |
| 语法 | 完整类定义（啰嗦） | 参数 -> 方法体（简洁） |
| 适用接口 | 任意接口/类 | 只能函数式接口 |
| 捕获变量 | 同规则（effectively final） | 同规则 |
| 编译产物 | 生成 `.class` 文件 | 生成一个方法（`invokedynamic`） |
| 访问实例字段 | 可以 | 可以 |
| 可读性 | 嵌套深时难读 | 一行看懂 |

**重点结论：函数式接口用 lambda，多方法接口才用匿名内部类。** lambda 是「代码即数据」的最小表达，也让 `java.util.function` 与 Stream 的管道式写法成为可能——这正是下一节《Lambda 与 Stream 流式编程》和 Effective Java 第 42–44 条要深入的主题。内部类则在「需要局部辅助对象、但不想污染命名空间」时仍不可替代。

## 6 小结

- **接口**是行为契约，可多实现；Java 8 起支持 default/静态方法。
- 抽象类 `extends`（is-a），接口 `implements`（can-do）；优先接口。
- **lambda** = 匿名函数，只能赋给**函数式接口**（恰一个抽象方法）。
- lambda 捕获的局部变量必须 effectively final。
- 非静态内部类持有宿主引用，小心内存泄漏；**静态内部类**是默认选择。

在下一节，我们将看到接口与类的「特殊形态」——**枚举类型与注解**，它们把「固定取值集合」与「元数据」变成了一等公民。
