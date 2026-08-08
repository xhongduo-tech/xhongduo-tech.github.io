---
title: 接口（Interface）、协议（Protocol）与混入（Mixin）
date: 2026-08-07
---

# 接口（Interface）、协议（Protocol）与混入（Mixin）

<div class="epigraph">
<p>继承决定「你是谁」，接口决定「你能做什么」——现代 OO 越来越偏向后者。</p>
<footer>—— 佚名（OO 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第11章 + 现代 OO 机制 ｜ 2026-08-07</p>
</div>

## 为什么从接口与混入开始

多继承的复杂性让现代语言另辟蹊径：**继承实现**保持单链，而**继承契约**可以多路。于是有了三个密切相关的机制：**接口（interface）**——纯契约（只有方法签名）；**协议（protocol）**——接口在不同语言中的变体（Swift/Objective-C）；**混入（mixin）**——可复用实现的「半接口」（带默认实现但不参与 is-a）。这三者共同回答了「如何在不付出多继承复杂度的情况下复用多种能力」。<span class="marginnote">三者的定位差异：<strong>接口</strong>说「你必须会做 X」（契约，无实现），<strong>混入</strong>说「我可以帮你做 X」（实现，可复用），<strong>协议</strong>是 Swift 里两者皆可的统称。从「纯契约」到「带实现的复用」是一条光谱，各语言在此光谱上取点。</span>

## 1 接口：纯契约

**接口（interface）**：一组**方法签名**的集合，不含实现（或仅有默认实现）。类「实现」接口 = 承诺提供这些方法。

```java
public interface Comparable<T> {
    int compareTo(T other);       // 只有签名，没有实现
}

public class Person implements Comparable<Person> {
    private int age;

    public int compareTo(Person other) {
        return Integer.compare(this.age, other.age);
    }
}
```

接口的价值：

**多态契约**：`sort` 只依赖 `Comparable`——任何实现它的类都能排序。**「面向接口编程」**：依赖抽象而非具体实现。
**多重「继承」**：一个类可实现多个接口——获得「契约多继承」而无菱形问题（接口无字段）。<span class="marginnote">接口是「类型即行为」的体现：`接口` 不是「一种对象」而是「一种能力」。这是对「is-a」的扩展——「can-do」关系。`继承` 与 `接口` 描述的是不同的维度：前者是类型层次，后者是能力契约。</span>

## 2 协议：接口的语言变体

**协议（protocol）**：Swift、Objective-C 中与接口等价的概念，但功能更丰富。

```swift
protocol Greetable {
    var name: String { get }
    func greet() -> String
}

struct Person: Greetable {           // 结构体也能遵循协议
    let name: String
    func greet() -> String { "Hello, \(name)!" }
}
```

协议的扩展能力：

**扩展（extension）**：可以为协议提供**默认实现**——写在 `extension` 里，遵循者不写也可用。
**可选的协议方法**：`@objc optional`——遵循者可不实现。
**结构化遵循**：Swift 中 `struct`/`enum` 也能遵循协议（不仅类）。<span class="marginnote">协议 + 扩展 = 「带默认实现的接口」——这是接口走向「混入」的一步：接口不仅约束「必须做」，还能提供「默认怎么做」。Swift 的协议扩展让「能力」可以带「现成实现」，接近 Rust trait 的默认方法。</span>

## 3 混入：可复用的实现

**混入（mixin）**：一段**可复用实现**的单元，可以「混入」类中，但不构成「is-a」继承。

- **Ruby**：`module` + `include`——模块定义方法，类 include 后获得这些方法。

```ruby
module Jsonable
  def to_json
    { class: self.class.name }.to_json
  end
end

class Order
  include Jsonable      # 混入：获得 to_json 方法，但不继承任何东西
end
```

- **Python**：多重继承实现混入（如 `JSONMixin`）——Python 的 mixin 就是「不用于独立实例化的类」。
- **Kotlin/Swift**：接口默认方法、协议扩展实现混入功能。

**辨析｜易错点：** 混入 vs 接口：接口是「契约」（强制实现），混入是「实现」（直接获得）。**「接口问你会不会，混入直接给你」**。混入 vs 继承：混入不建立「is-a」关系——`Order` 是 mixin `Jsonable` 的接收者，但「订单是一种 Jsonable」这个说法很别扭——它是「订单具有 Jsonable 的能力」。**混入是「has-a 的实现复用 + 无继承的层次」**。

## 4 公式解析：接口与多态

接口驱动的多态可以形式化：「接口」是方法签名集合，类型满足接口 = 提供这些方法：

$$
T \;\text{satisfies}\; I \;\Longleftrightarrow\; \forall\ m \in I : T \text{ 提供 } m
$$

面向接口的函数：

$$
f : (T \text{ satisfies } I) \Rightarrow f \text{ 可调用 } T \text{ 的值} \quad \text{且只依赖 } I \text{ 中的方法}
$$

三步拆解：

- **第一步，满足关系**：`T` 当且仅当 `T` 实现了接口里的全部方法——这是编译期检查的「契约验证」。
- **第二步，依赖只限接口**：函数 `f` 的签名约束 `T`——`f` 内部只能调用 `I` 里的方法，不能假设 `T` 的专有方法。
- **第三步，看解耦**：`f` 与具体 `T` 解耦——任何满足 `I` 的类型都能用 `f`。**「接口是函数与实现之间的最小依赖面」**：换实现不换函数，这就是面向接口编程的形式化。

**辨析｜易错点：** 接口与**抽象类（abstract class）**：抽象类可带**字段与构造函数**（部分实现），接口不能（Java 8 前）；抽象类建立「is-a」，接口建立「can-do」。**「抽象类管『是什么』，接口管『能做什么』」**——一个类只能继承一个抽象类，但可实现多个接口。

## 5 现代语言的能力组合

| 语言 | 契约机制 | 实现复用机制 | 组合方式 |
| --- | --- | --- | --- |
| Java | `interface` | `default` 方法（Java 8） | 单类 + 多接口 |
| C# | `interface` | `default` 实现 | 单类 + 多接口 |
| Swift | `protocol` + `extension` | 协议扩展默认实现 | 值类型/类都可遵循 |
| Ruby | （无接口） | `module` mixin | 单继承 + include |
| Rust | `trait` | trait 默认方法 | 泛型 + trait bound（无类继承） |

<span class="marginnote">终极演化：Rust 用 trait 统一了「接口 + 混入 + 泛型约束」——trait 既约束（`trait bound`）又提供默认实现，且可多实现（无类继承）。Swift/Kotlin 的协议/接口默认方法也在向这个方向靠拢。「能力即类型」的哲学，正在取代「继承即类型」。</span>


## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 接口（interface） | 接口（interface）：一组方法签名的集合，不含实现（或仅有默认实现）。类「实现」接口 = 承诺提供这些方法。 |
| 方法签名 | 接口（interface）：一组方法签名的集合，不含实现（或仅有默认实现）。类「实现」接口 = 承诺提供这些方法。 |
| 多态契约 | 多态契约：sort 只依赖 Comparable——任何实现它的类都能排序。「面向接口编程」：依赖抽象而非具体实现。 |
| 「面向接口编程」 | 多态契约：sort 只依赖 Comparable——任何实现它的类都能排序。「面向接口编程」：依赖抽象而非具体实现。 |
| 协议（protocol） | 协议（protocol）：Swift、Objective-C 中与接口等价的概念，但功能更丰富。 |
| 扩展（extension） | 扩展（extension）：可以为协议提供默认实现——extension Greetable { func greet() -> String { "H |
| 默认实现 | 接口（interface）：一组方法签名的集合，不含实现（或仅有默认实现）。类「实现」接口 = 承诺提供这些方法。 |
| 可选的协议方法 | 可选的协议方法：@objc optional——遵循者可不实现。 |
| 混入（mixin） | 混入（mixin）：一段可复用实现的单元，可以「混入」类中，但不构成「is-a」继承。 |
| 可复用实现 | 混入（mixin）：一段可复用实现的单元，可以「混入」类中，但不构成「is-a」继承。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **接口** = 纯契约（方法签名集合），提供「can-do」多态，面向接口编程解耦函数与实现。
- **协议**（Swift） = 接口 + 扩展默认实现，`struct`/`enum` 也能遵循。
- **混入** = 可复用实现的单元，不建立 is-a，直接给类「现成能力」。
- 接口/抽象类/混入是「纯契约 → 带实现」的光谱；Rust trait 是「契约 + 默认实现 + 泛型约束」的统合形态。

在下一节，我们将进入第十二篇——**函数式编程**，先看基本概念：数学函数与引用透明性。
