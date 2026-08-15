---
title: 对象与类
date: 2026-08-07
---

# 对象与类

<div class="epigraph">
<p>面向对象不是语法糖，而是一种看待世界的态度：把数据与操作它的行为封装在一起。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第4章 ｜ 2026-08-07</p>
</div>

## 为什么从对象与类开始

前面你学会了数组与字符串——它们是「现成的容器」，拿来即用。但真实程序里的数据很少是裸的整数或字符串：一个员工有工号、姓名、部门、工资，一张订单有编号、商品、金额、状态。把这样一组「相关的数据」和「操作这些数据的方法」捆成一个整体，就是**对象（object）**；而定义对象「长什么样、能干什么」的模板，就是**类（class）**。<span class="marginnote">Java 是「万物皆对象」的忠实信徒——除了八个基本类型，一切数据都被装进类里。你学的 `String`、`ArrayList` 都是类，从这章起你开始<strong>自己造类</strong>，从消费者变成生产者。</span>这一章是面向对象的真正起点：字段、构造器、方法、封装与静态成员。学会它，后面的继承、多态、接口才有砖可垒。

## 1 类与对象：模板与实例

**类（class）**是描述一类对象的模板，**对象（object）**是这个模板的**实例（instance）**。打个比方：类是「建筑设计图」，对象是「按图盖出的房子」。一张图纸能盖出无数栋房子，一个类能 new 出无数个对象。

```java
public class Employee {
    // 字段（field）：对象的数据
    private String name;
    private double salary;

    // 构造器（constructor）：创建对象时初始化
    public Employee(String name, double salary) {
        this.name = name;
        this.salary = salary;
    }

    // 方法（method）：对象的行为
    public double getSalary() { return salary; }
    public void raiseSalary(double byPercent) {
        salary *= 1 + byPercent / 100;
    }
}
```

用 `new` 创建对象，用**点号（.）**访问成员：

```java
Employee alice = new Employee("Alice", 8000);   // new 调构造器，在堆上造一个对象
Employee bob = new Employee("Bob", 9000);
alice.raiseSalary(10);                          // 调 alice 的方法
System.out.println(alice.getSalary());          // 8800.0
```

**重点结论：`alice` 是引用，不是对象本体。** `new Employee(...)` 在堆内存里创建对象并返回其**引用（reference）**，`alice` 这个变量只保存引用。`new` 一次就有一个独立的对象——`alice` 和 `bob` 的数据互不影响，这是「每个实例有自己的一份字段」的内存基础。

**辨析｜易错点：字段与方法都属实例，但通过 `this` 分清身份。** 构造器里 `this.name = name` 的 `this` 指「当前正在创建的那个对象」——参数 `name` 遮蔽了字段 `name`，用 `this` 明确告诉编译器「我写的是字段」。没有命名冲突时 `this` 可省略，但写 `this` 总是更清晰。

## 2 构造器：对象诞生的规则

**构造器（constructor）**是与类同名、没有返回类型（连 `void` 都不写）的特殊方法，在 `new` 时被自动调用。它的职责只有一个：**把新对象的字段初始化到合法状态**。

- 构造器名必须与类名**完全一致**。
- 构造器**没有返回类型**——不是返回 void，而是「什么都不返回」。
- 构造器可以**重载（overload）**：多个构造器参数列表不同，`new` 时按实参自动匹配。
- 没写任何构造器时，编译器送一个**默认无参构造器**（字段全部为默认值：数值 0、引用 null、布尔 false）。

```java
public Employee() { }                        // 无参构造器，字段为默认值
public Employee(String name) { this.name = name; }   // 一个参数的重载
public Employee(String name, double salary) {        // 两个参数的重载
    this.name = name;
    this.salary = salary;
}
```

**辨析｜易错点：一旦手写了任何构造器，默认无参构造器就消失了。** 只写了 `Employee(String, double)` 后，`new Employee()` 会编译报错——因为编译器不再替你生成无参版本。需要无参构造时务必显式写上。这在「用框架反序列化对象」时是高频坑（很多框架要求无参构造器）。

**构造器里调用另一个构造器**用 `this(...)`，且必须是第一行：

```java
public Employee(String name) {
    this(name, 0);        // 委托给两参构造器
}
```

## 3 封装与访问控制：字段私有，方法公开

**封装（encapsulation）**是面向对象的第二根支柱：**字段设为私有（`private`），通过公开方法（`public`）读写**。它的直接好处是「**数据不裸奔，规则握在类手里**」。

```java
public class BankAccount {
    private double balance;                     // 私有字段：外部摸不到

    public void deposit(double amount) {        // 公开方法：规则的唯一入口
        if (amount <= 0) throw new IllegalArgumentException("金额必须为正");
        balance += amount;
    }
    public double getBalance() { return balance; }
}
```

如果不封装，`account.balance = -10000` 直接就把账改成了负数；封装后，外部只能走 `deposit`，而 `deposit` 里的校验把「余额不能为负」变成了类的**不变量（invariant）**。<span class="marginnote">封装是「职责划分」的工程化表达：类对外承诺「我的状态永远合法」，代价是你不能直接碰它的内部。这与操作系统的「内核态/用户态」、数据库的「事务」是同一个思想——把合法性检查收敛到少数入口。</span>

Java 有四个访问级别，从宽到窄：

| 修饰符 | 同包内 | 子类 | 任意位置 | 含义 |
| --- | --- | --- | --- | --- |
| `public` | 可见 | 可见 | **可见** | 完全公开 |
| `protected` | 可见 | **可见** | 不可见 | 包内 + 子类 |
| （默认）包私有 | **可见** | 不可见 | 不可见 | 仅同包 |
| `private` | 不可见 | 不可见 | 不可见 | 仅本类 |

**重点结论：字段一律 `private`，方法按需 `public`/`private`。** 这是 Effective Java 第 15 条「最小化类和成员的可访问性」的日常版本——暴露面越小，日后改内部实现越自由，破坏调用方的风险越低。访问级别是「设计契约」，不是「性能开关」。

## 4 静态成员：属于类的部分

被 `static` 修饰的成员**属于类本身**，不属于任何实例。静态成员用**类名**访问：`Math.PI`、`Integer.parseInt`。

**静态字段（static field）**：所有实例共享**一份**，而不是各存一份。

```java
public class Employee {
    private static int nextId = 1;      // 属于类：所有员工共享下一个可用编号
    private int id;                     // 属于实例：每个员工自己的编号

    public Employee() { this.id = nextId++; }
}
```

**静态方法（static method）**：不访问任何实例字段的方法，如工具函数 `Math.max`。因为不碰 `this`，静态方法没有 `this`——它不需要对象就能调用。

**辨析｜易错点：静态方法里不能访问实例字段。** 实例字段存在具体的对象里，静态方法调用时可能根本没有对象。`public static void f() { System.out.println(salary); }` 编译失败——编译器不知道 `salary` 属于谁。想访问实例字段，就把它写成实例方法。

**`main` 为什么是 `static`**：JVM 启动时还没有任何对象，它必须能「不创建对象」就调用入口——所以 `main` 必须是静态方法。

## 5 核心对比表：实例成员与静态成员

纯概念主题用**核心对比表**替代公式解析，把最容易混淆的两类成员摆在一起：

| 维度 | 实例成员（字段/方法） | 静态成员（字段/方法） |
| --- | --- | --- |
| 归属 | 每个对象一份 | 类唯一一份 |
| 访问方式 | `对象.成员` | `类名.成员` |
| 内存 | 随对象创建/回收 | 类加载时分配，常驻 |
| 能否访问实例字段 | 能 | 不能 |
| 典型用途 | 对象的状态与行为 | 常量、工具方法、计数器 |
| 调用前提 | 先有对象 | 无需对象 |

**重点结论：静态成员表达「类级别的数据与能力」，实例成员表达「对象级别的状态与行为」。** 判断一个字段该不该 `static`：问「它是不是所有实例共享、且与具体对象无关」——员工编号的生成器是共享的（`static`），而每个员工自己的编号是独立的（实例）。这个「类 vs 实例」的二象性，是理解对象模型的钥匙，也是下一章继承里「静态方法没有多态」的伏笔。

## 6 小结

- **类**是模板，**对象**是实例；`new` 在堆上创建对象并返回引用。
- **构造器**与类同名、无返回类型、可重载；手写构造器后默认无参构造器消失。
- **封装**：字段 `private`、方法 `public`，把合法性检查收敛进类的方法。
- 访问级别从宽到窄：`public` → `protected` → 包私有 → `private`。
- **静态成员**属于类：静态字段共享一份，静态方法无 `this`、不能访问实例字段。

在下一节，我们将回答「一个类如何派生出另一个类」——**继承、多态与抽象类**，面向对象的第三根支柱。
