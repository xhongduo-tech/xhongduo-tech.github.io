---
title: 参数化抽象数据类型（泛型容器）
date: 2026-08-07
---

# 参数化抽象数据类型（泛型容器）

<div class="epigraph">
<p>一个栈，装任意类型——把「栈」与「栈里装什么」解耦，泛型的价值由此而来。</p>
<footer>—— 佚名（PLT 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第10章 §10.7 + 泛型编程 ｜ 2026-08-07</p>
</div>

## 为什么从参数化 ADT 开始

ADT（如栈）写一次，却要服务整数、字符串、自定义类型——为每种类型复制一份代码显然荒谬。**参数化 ADT（parametric ADT）**把「元素类型」变成参数：只需一个定义，`Stack<Integer>`、`Stack<String>` 取任意类型。这是第五篇「参数多态」在 ADT 层的落地，也是**泛型（generics）**的主题。各语言的实现机制差异巨大——Java 的**类型擦除**、C++ 的**模板实例化**、Rust 的**单态化**——它们决定了运行时开销、类型安全与二进制兼容性的不同画像。<span class="marginnote">泛型要回答的根本问题：<strong>「Stack&lt;T&gt; 编译后是什么？」</strong>——Java 说「一份代码，擦除 T」，C++ 说「每个 T 生成一份专用代码」，Rust 说「需要时再为每个 T 生成」。同一语义，三种实现哲学。</span>

## 1 泛型的基本形式

**参数化 ADT / 泛型（generic）**：类型定义带一个或多个**类型参数**，实例化时用具体类型替换。

```java
class Stack<T> {                    // 类型参数 T：实例化时才确定
    private List<T> items = new ArrayList<>();
    public void push(T item) { items.add(item); }
    public T pop() { return items.remove(items.size() - 1); }
}
// 一个定义，任意类型实例化
Stack<Integer> intStack = new Stack<>();
Stack<String>  strStack = new Stack<>();
```

**辨析｜易错点：** 泛型的类型参数在**实例化时**替换——`Stack<Integer>` 与 `Stack<String>` 是**两个不同实例**（尽管 Java 擦除后共享代码）。**「泛型定义」与「泛型实例化」必须区分**：前者是模板，后者是具体类型。

## 2 三种实现机制

**Java 类型擦除（type erasure）**：编译期检查类型参数，编译后把 `T` 全部擦除为 `Object`（或上界），运行时只有一份代码。

```java
Stack<String> s = new Stack<>();
s.push("hello");
String x = s.pop();
// 擦除后等价于：
Stack s = new Stack();        // 类型参数消失
s.push("hello");
String x = (String) s.pop();  // 编译器插入强转
```

优点：**二进制兼容**（旧代码能链接新泛型代码）、运行期零开销；缺点：无法在运行期获知 `T`（不能 `new T()`）、`instanceof` 泛型受限。

**C++ 模板实例化（template instantiation）**：每个实例化生成一份专用代码（代码膨胀的代价，换来最大优化空间）。

```cpp
template <typename T>
class Stack {
    std::vector<T> items;
public:
    void push(const T& item) { items.push_back(item); }
    T pop() { T top = items.back(); items.pop_back(); return top; }
};
// 每个 T 生成一份专用代码：Stack<int> 与 Stack<Point> 互不相同
Stack<int> intStack;
Stack<Point> pointStack;
```

**Rust 单态化（monomorphization）**：类似 C++——`Stack<i32>` 与 `Stack<String>` 各自生成专用代码；配合 trait bound 做「编译期多态」。<span class="marginnote">三种机制是「共享 vs 专用」的权衡：擦除共享代码、省空间但失去类型信息；单态化生成专用代码、可极致优化但增大二进制。Rust 选择单态化是因为它要「零成本抽象」——专用代码让泛型与手写等价。</span>

## 3 公式解析：泛型的擦除语义

Java 的类型擦除可以形式化为「把类型参数映射到其擦除」的代换。设泛型类 $C<T_1, \dots, T_k>$，擦除函数 $\text{erase}$：

$$
\text{erase}(C<T_1, \dots, T_k>) = C\ \text{（类名保留，类型参数全变为上界 } \text{Object} \text{）}
$$

$$
\text{erase}(T_i) = \text{upper bound of } T_i
$$

三步拆解：

- **第一步，擦除类**：`Stack<T>` 擦除为 `Stack`——类名不变，类型参数消失。
- **第二步，擦除类型参数**：每个 `T` 替换为其**上界**（`T extends Number` → `Number`，无界 → `Object`）。方法签名里所有 `T` 也替换。
- **第三步，看桥接**：为保持多态（子类覆盖父类的 `push(Object)`），编译器可能生成**桥接方法**（bridge method）做类型转换——擦除的隐藏代价。**「擦除后代码仍正确」是 Java 泛型的设计目标，靠编译期检查 + 运行期强转（cast）保证**。

**辨析｜易错点：** 擦除的代价：`Stack<String>` 与 `Stack<Integer>` 在**运行期是同一个类**（都擦为 `Stack`）——所以 `new Stack<T>()` 非法（无法区分）。**「擦除让泛型信息只存在于编译期」**——所有依赖运行期类型信息的操作（构造泛型对象、泛型数组）都受限。

## 4 泛型的约束：上界与边界

泛型参数往往需要约束——不是所有类型都合法。**上界（upper bound）**限制类型参数必须是某类型的子类：

```java
public class MaxFinder<T extends Comparable<T>> {
    public T max(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;  // 约束保证 compareTo 可用
    }
}
```

```rust
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a >= b { a } else { b }              // trait bound 保证 >= 可用
}
```

约束的价值：**让泛型代码能调用被约束类型的方法**——`T extends Comparable<T>` 保证 `compareTo` 可用，`T: PartialOrd` 保证 `>=` 可用。<span class="marginnote">「泛型 + 约束」是「通用性」与「可用性」的平衡：无约束太通用（啥都不能做），太窄约束失去通用。Rust 的 trait bound 与 Haskell 的 type class 把「约束」变成类型系统的一等概念——「泛型算法 + 需求接口」是泛型编程的完整形态。</span>

## 5 泛型在现代语言中的进化

**协变/逆变**：Java `? extends`/`? super`、C# `out`/`in`、Kotlin `out`/`in`——让泛型在「只读/只写」位置上安全地放宽（第五篇协变逆变已详讲）。
**关联类型**：Rust 的 `type Item`、Swift 的 `associatedtype`——类型参数不仅可以是「输入」，还可能是「输出」（迭代器产出什么元素）。
**泛型推导**：Java 的 `<>`（diamond）、C++17 的 CTAD、Rust 的类型推导——实例化时省去手写类型参数。<span class="marginnote">泛型的进化方向是「更安全 + 更顺手」：约束更精细（trait bound）、变体更合理（协变/逆变）、推导更自动（CTAD）。「写一次、处处用、且类型安全」——泛型的理想在逐步接近。</span>



## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| 参数化 ADT / 泛型（generic） | 参数化 ADT / 泛型（generic）：类型定义带一个或多个类型参数，实例化时用具体类型替换。 |
| 类型参数 | 参数化 ADT / 泛型（generic）：类型定义带一个或多个类型参数，实例化时用具体类型替换。 |
| Java 类型擦除（type erasure） | Java 类型擦除（type erasure）：编译期检查类型参数，编译后把 T 全部擦除为 Object（或上界），运行时只有一份代码。 |
| 二进制兼容 | 优点：二进制兼容（旧代码能链接新泛型代码）、运行期零开销；缺点：无法在运行期获知 T（不能 new T()）、instanceof 泛型受限。 |
| C++ 模板实例化（template instantiation） | C++ 模板实例化（template instantiation）：每个实例化生成一份专用代码（代码膨胀的代价，换来最大优化空间）。 |
| 上界 | Java 类型擦除（type erasure）：编译期检查类型参数，编译后把 T 全部擦除为 Object（或上界），运行时只有一份代码。 |
| 桥接方法 | 第三步，看桥接：为保持多态（子类覆盖父类的 push(Object)），编译器可能生成桥接方法（bridge method）做类型转换——擦除的隐藏代价 |
| 「擦除后代码仍正确」是 Java 泛型的设计目标，靠编译期检查 + 运行期强转（cast）保证 | 第三步，看桥接：为保持多态（子类覆盖父类的 push(Object)），编译器可能生成桥接方法（bridge method）做类型转换——擦除的隐藏代价 |
| 上界（upper bound） | 泛型参数往往需要约束——不是所有类型都合法。上界（upper bound）限制类型参数必须是某类型的子类： |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 6 小结

- **参数化 ADT / 泛型**把元素类型变成参数：一个定义、任意类型实例化。
- 三种实现机制：**Java 擦除**（共享代码、运行期无类型信息）、**C++/Rust 单态化**（每类型专用代码、可极致优化）。
- 擦除语义：`T` 替换为上界、类名保留；桥接方法补多态；运行期无法区分 `Stack<String>` 与 `Stack<Integer>`。
- 泛型 + 约束（上界/trait bound）= 通用与可用的平衡；协变逆变、关联类型、推导是泛型的现代进化。

在下一节，我们将进入第十一篇——**面向对象程序设计**，先看基本概念：对象、类与消息传递。
