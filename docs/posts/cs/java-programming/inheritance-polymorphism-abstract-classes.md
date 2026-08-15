---
title: 继承、多态与抽象类
date: 2026-08-07
---

# 继承、多态与抽象类

<div class="epigraph">
<p>继承不是代码复用，而是「是-a」关系的建模；多态让代码对扩展开放、对修改封闭。</p>
<footer>—— 改编自 Joshua Bloch《Effective Java》与开闭原则</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第5章 ｜ 2026-08-07</p>
</div>

## 为什么从继承开始

对象与类解决了「一个类怎么造」，但真实世界的关系是**分层的**：经理是员工的一种，员工是人，人是 Object。**继承（inheritance）**让你用一个类去「扩展」另一个类，把共性上提、把差异下放；**多态（polymorphism）**让同一个方法调用在不同子类上表现出不同行为——这是面向对象的第三根支柱，也是框架设计（Spring 里依赖注入的对象能替换实现）的语法地基。这一篇把 `extends`、方法重写、动态绑定、抽象类一次讲透。

## 1 extends：继承的语法与构造

**继承**用 `extends` 关键字：子类（subclass）继承父类（superclass）的字段与方法，再添加自己的扩展。

```java
public class Manager extends Employee {
    private double bonus;

    public Manager(String name, double salary) {
        super(name, salary);     // 调用父类构造器，必须第一行
        bonus = 0;
    }

    public double getSalary() {   // 方法重写（override）
        return super.getSalary() + bonus;   // super 调用父类版本
    }
}
```

四个关键点：

- **`super`** 有两个用途：调用父类构造器（必须写在子类构造器**第一行**），以及调用被重写的父类方法。
- **方法重写（override）**要求签名（方法名 + 参数列表）与父类完全一致；重写方法的访问级别不能比父类更窄（`public` 不能重写成 `protected`）。
- **子类不继承构造器**，但每个子类构造器都必须直接或间接调用父类构造器——若子类构造器不写 `super(...)`，编译器会隐式调用父类的无参构造器；父类没有无参构造器时就会编译报错。
- **`@Override` 注解**建议加上：它让编译器检查「你是否真的重写了父类方法」——签名写错时立刻报错，而不是静默地变成重载。<span class="marginnote">Java 的继承是<strong>单继承</strong>：一个类只能有一个直接父类。多继承的语法被放弃了，改为「接口多实现」——这是下一章《接口》的内容。C++ 的多继承在 Java 里被刻意简化，以减少菱形继承的混乱。</span>

**继承与「组合」之争**：Effective Java 第 18 条警告「优先考虑组合而非继承」——继承会暴露父类的所有公开行为，父类改动可能悄悄破坏子类。这属于设计层面的权衡，后面《类和接口的设计规范》会展开；此刻先掌握继承的机制。

## 2 多态与动态绑定

**多态**的字面意思是「多种形态」：同一个引用类型指向不同子类对象，调用同一个方法却执行不同版本。

```java
Employee e = new Manager("Carol", 10000);   // 父类引用指向子类对象
System.out.println(e.getSalary());           // 运行的是 Manager.getSalary()
```

这里 `e` 的**编译期类型**是 `Employee`，**运行期类型**是 `Manager`。当调用 `e.getSalary()` 时，JVM 在运行期根据**对象的实际类型**决定执行哪个方法——这叫**动态绑定（dynamic binding）**，也叫虚方法调用。

**辨析｜易错点：编译看左边，运行看右边。** 一行 `e.getSalary()` 能调用什么方法，由编译期类型 `Employee` 决定（`e` 看不到 Manager 独有的 `setBonus`）；但执行哪份实现，由运行期类型 `Manager` 决定。若 `Employee` 里根本没有 `getSalary`，编译直接报错——即便运行时它是 Manager 也不行。

**为什么多态重要**？因为有了它，代码可以**面向父类编程**：

```java
public void payAll(Employee[] staff) {
    for (Employee e : staff) {
        e.getSalary();   // 普通员工算基本工资，经理算基本工资+奖金
    }
}
```

`payAll` 只认识 `Employee`，却能为所有子类正确发薪——将来新增一个 `Contractor` 子类，`payAll` 一行都不用改。这就是「对扩展开放、对修改封闭」的开闭原则，多态是它的语法基石。

## 3 公式解析：动态绑定的查找链

动态绑定在运行时具体怎么找到正确的方法？本质是在继承链上自底向上查找：

$$

\text{调用 } e.\text{method() } \Rightarrow \text{从 } e \text{ 的实际类开始，沿继承链向上找到第一个定义该方法的类}

$$

对这条公式做三步拆解：

- **第一步，取实际类**：运行期先看 `e` 实际指向的对象属于哪个类（`Manager`），从它开始找。
- **第二步，沿继承链向上**：实际类里没有 `method`，就找它的父类、父类的父类……直到 `Object`。
- **第三步，命中即止**：找到第一个定义该方法的类，执行它的实现——**越「下」层的重写越优先**。

这解释了多态的正确性来源：JVM 不靠「变量声明的类型」选方法，而靠「对象实际的类型」。所以 `Manager` 重写了 `getSalary`，即便你用 `Employee` 引用持有时，调的也是 `Manager` 的版本。

**再澄清一个高频混淆：重载（overload）与重写（override）。**

| 维度 | 重写（override） | 重载（overload） |
| --- | --- | --- |
| 位置 | 子类重定义父类方法 | 同类/子类中同名不同参 |
| 签名 | 与父类完全一致 | 参数列表必须不同 |
| 绑定 | 动态（运行期） | 静态（编译期） |
| 关键字 | `@Override` 建议 | 无 |

**辨析｜易错点：重载是「编译期决定、静态绑定」，重写是「运行期决定、动态绑定」。** 两个重载方法选哪个，编译器看实参类型就定了；两个重写方法执行哪个，JVM 看对象实际类型才定。把「重写」当「重载」写（参数列表写错），`@Override` 会当场报错——这就是它存在的意义。

## 4 抽象类：把「怎么做」留给子类

**抽象类（abstract class）**用 `abstract` 修饰，不能 `new`——它是「半成品模板」：把子类共有的实现写进来，把需要差异化的方法留成**抽象方法（abstract method）**，强制子类实现。

```java
public abstract class Shape {
    protected String name;
    public Shape(String name) { this.name = name; }      // 共享的构造器与字段
    public abstract double area();                        // 抽象方法：只有声明，无实现

    public String describe() { return name + " 面积 " + area(); }   // 已实现，可复用
}
public class Circle extends Shape {
    private double r;
    public Circle(double r) { super("圆"); this.r = r; }
    @Override public double area() { return Math.PI * r * r; }      // 子类必须实现
}
```

**抽象类的三个要点：**

- **抽象方法没有方法体**，只有签名，以分号结束；子类**必须实现所有抽象方法**（除非子类也是抽象的）。
- 抽象类**可以有字段、构造器、已实现的方法**——这是它与接口最大的区别。
- **不能 `new Shape()`**——它不完整，实例化它没有意义；但 `Shape s = new Circle(1)` 合法，多态照常工作。

**什么时候用抽象类**：当多个子类要**共享字段与实现**（模板方法模式：`describe` 写好骨架，`area` 由子类填）时。抽象类表达「is-a 且共享实现」，接口表达「can-do 能力清单」——下一章《接口、lambda 表达式与内部类》会把两者的分工讲透。

**`final` 关键词的三个用途**（与抽象相反的方向）：`final` 类不能继承（如 `String`）、`final` 方法不能重写、`final` 字段/变量不能重新赋值。**「abstract 强制重写」与「final 禁止重写」是同一把锁的两面。**<span class="marginnote">把类做成 `final` 不只是性能优化（JVM 能做去虚化），更是「不可继承」的设计声明——`String` 就是靠 `final` 保证了「所有 String 的行为都一致、不会被子类偷偷改变」。Effective Java 第 17 条建议：能用 `final` 就用。</span>

## 5 小结

- **继承**用 `extends`：`super` 调父类构造器（第一行）与父类方法；重写签名一致、访问级别不收紧。
- **多态 + 动态绑定**：编译看左边、运行看右边；对象实际类型决定执行哪个方法。
- 重写是**运行期动态绑定**，重载是**编译期静态绑定**——两者别混。
- **抽象类**是半成品模板：字段、构造器、实现方法 + 抽象方法；不能 `new`，子类必须实现抽象方法。
- 继承适合「is-a」，能复用实现；但「复用方法」优先考虑组合——见《类和接口的设计规范》。

在下一节，我们给类加上「能力清单」——**接口、lambda 表达式与内部类**。