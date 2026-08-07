---
title: Ada、C++ 与 Java 中的数据抽象实现
date: 2026-08-07
---

# Ada、C++ 与 Java 中的数据抽象实现

<div class="epigraph">
<p>三种语言，三种对「隐藏表示」的回答——它们共同画出了数据抽象的进化地图。</p>
<footer>—— 佚名（PLT 格言）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言 ｜ Sebesta《程序设计语言原理》第10章 §10.4–10.5 ｜ 2026-08-07</p>
</div>

## 为什么从三种语言看数据抽象

抽象数据类型的设计要求（封装、隐藏、类型安全）在真实语言里各有各的实现方式。塞贝斯塔用三种经典语言对照：**Ada**（把 ADT 作为一等公民的 package）、**C++**（把 ADT 作为 class + 访问控制的混合体）、**Java**（把 ADT 作为 class + 引用的纯面向对象）。这一节通过同一「栈」的三种实现，看「隐藏表示」的机制如何演化——以及每种机制付出的代价。这是理解「语言机制如何落实设计原则」的最佳标本。<span class="marginnote">三种语言正好对应 ADT 实现的三个世代：Ada 是「ADT 专项语言」（军用语言，强调可靠），C++ 是「C + ADT + OO」的混合（性能与多范式），Java 是「纯 OO」（引用语义 + GC）。三代的设计取舍映照出语言演化。</span>

## 1 Ada：package 是 ADT 的一等公民

Ada 用 **package** 实现 ADT：可见部分（specification）声明类型与操作接口，私有部分（private）与包体（body）隐藏表示与实现。

```ada
package Stack is
    type Stack_Type is limited private;   -- 类型公开，表示私有
    procedure Push(S: in out Stack_Type; V: in Integer);
    function  Pop(S: in out Stack_Type) return Integer;
private
    type Stack_Type is record
        Items : array(1..100) of Integer;
        Top   : Integer := 0;
    end record;
end Stack;
```

Ada 的特点：**类型与表示分离**——用户能用 `Stack_Type` 声明变量，但看不到它的结构；`limited private` 连赋值都禁止（更严格的封装）。<span class="marginnote">Ada 的 `limited private` 是最强的封装：不仅隐藏表示，连「拷贝赋值」都不允许（因为是 limited）——你无法把栈整个拷走，只能通过 Push/Pop 操作。这是「操作即唯一入口」的极致。</span>

## 2 C++：class + 访问控制

C++ 用 **class** 实现 ADT：成员默认私有，`public:`/`private:`/`protected:` 控制可见性。

```cpp
class Stack {
private:
    int items[100];      // 私有表示
    int top = 0;
public:
    void push(int v) { items[top++] = v; }
    int pop() { return items[--top]; }
};
```

C++ 的特点：**访问控制分粒度**——`private` 完全隐藏，`protected` 允许子类访问，`friend` 允许指定外部函数访问。<span class="marginnote">C++ 的 `friend` 是「可定制的信任」：默认隐藏，但允许特定函数/类突破封装。这是「封装」与「效率/便利」之间的灵活妥协——friend 用得克制是封装的好帮手，滥用则破坏封装。</span>

**辨析｜易错点：** C++ 的 class 与 struct 默认访问不同：`class` 默认 `private`，`struct` 默认 `public`。且 C++ 的封装是「语法级」——指针算术仍可突破（`*(int*)(&s + ...)` 能摸到私有成员），因为 C++ 保留底层内存自由。**「语言提供封装」与「物理上不可访问」是两回事**。

## 3 Java：纯面向对象的引用语义

Java 用 **class** 实现 ADT，但加上了引用语义与 GC——对象经引用访问，封装与内存安全天然结合。

```java
public class Stack {
    private int[] items;      // 私有表示
    private int top;
    public Stack(int n) { items = new int[n]; }
    public void push(int v) { items[top++] = v; }
    public int pop() { return items[--top]; }
}
```

Java 的特点：**强封装 + 引用共享**——`private` 编译期强制；对象是引用，传对象 = 共享同一 ADT 实例。<span class="marginnote">Java 的封装比 C++ 更「干净」：没有指针算术，`private` 字段物理上无法从类外访问（反射除外）。代价是引用语义——`Stack s2 = s1;` 后 `s1`/`s2` 指向同一栈，别名问题需注意。这是「封装 + 安全性」与「值语义」的取舍。</span>

## 4 公式解析：三者的封装强度对比

封装强度可以形式化为「外部代码能对 ADT 内部表示执行的操作集合」的大小。设 ADT 的表示 $\text{Rep}$ 有公开操作集 $P$（接口），封装强度 $S$：

$$
S(ADT) \;\propto\; |\{\text{外部代码可直接访问的 Rep 操作}\}|
$$

（$S$ 越大，封装越弱。）

| 语言 | 外部可访问表示的方式 | 封装强度 |
| --- | --- | --- |
| Ada `limited private` | 无（连赋值都禁） | 最强 |
| Java `private` | 反射可突破（运行时） | 强 |
| C++ `private` | 指针算术可突破（未定义但可能） | 中 |
| C `struct` | 完全公开 | 无 |

三步拆解：

- **第一步，看访问面**：封装强度 = 「外部能直接摸到表示」的通道数量。通道越少，封装越强。
- **第二步，看语言机制**：Ada 在语法层堵死，Java 在类型层堵死（反射是后门），C++ 有指针后门，C 不设防。
- **第三步，看权衡**：最强封装（Ada）牺牲灵活性，最弱（C）牺牲安全。「封装强度」不是越高越好——它取决于语言定位（可靠 vs 灵活）与使用场景。

**辨析｜易错点：** 「接口」与「实现」的分离程度：Ada 把接口放 spec、实现放 body，**编译单元级分离**；Java 把接口（方法签名）与实现（方法体）放在同一个 class 文件——物理分离较弱。**「接口稳定」的关键不是物理分离，而是「表示私有」**——只要表示不被外部依赖，改实现就安全。

## 5 三者的共同遗产与现代演化

- **共同点**：三者都实现了「类型公开、表示私有、操作受限」的 ADT 核心。
- **演化方向**：接口/协议进一步抽象——Java 的 interface 让「操作集合」独立于具体类；Rust 的 trait 结合「关联类型 + 私有字段 + 模块可见性」提供了更精细的封装。<span class="marginnote">现代语言的封装正走向「组合 + 契约」：Rust 的「私有字段 + `pub` 方法 + trait 抽象」、Go 的「小接口 + 包级封装」、Kotlin 的「data class + sealed class」。ADT 的「表示隐藏」原则不变，实现机制却越来越灵活。</span>


## 6 Ada：package 是 ADT 的一等公民 |
| class | ## 2 C++：class + 访问控制 |
| 辨析｜易错点： | 辨析｜易错点： C++ 的 class 与 struct 默认访问不同：class 默认 private，struct 默认 public。且 C++ 的封 |
| 「语言提供封装」与「物理上不可访问」是两回事 | 辨析｜易错点： C++ 的 class 与 struct 默认访问不同：class 默认 private，struct 默认 public。且 C++ 的封 |
| 第一步，看访问面 | - 第一步，看访问面：封装强度 = 「外部能直接摸到表示」的通道数量。通道越少，封装越强。 |
| 第二步，看语言机制 | - 第二步，看语言机制：Ada 在语法层堵死，Java 在类型层堵死（反射是后门），C++ 有指针后门，C 不设防。 |
| 第三步，看权衡 | - 第三步，看权衡：最强封装（Ada）牺牲灵活性，最弱（C）牺牲安全。「封装强度」不是越高越好——它取决于语言定位（可靠 vs 灵活）与使用场景。 |
| 编译单元级分离 | 辨析｜易错点： 「接口」与「实现」的分离程度：Ada 把接口放 spec、实现放 body，编译单元级分离；Java 把接口（方法签名）与实现（方法体）放在 |
| 「接口稳定」的关键不是物理分离，而是「表示私有」 | 辨析｜易错点： 「接口」与「实现」的分离程度：Ada 把接口放 spec、实现放 body，编译单元级分离；Java 把接口（方法签名）与实现（方法体）放在 |
| 共同点 | - 共同点：三者都实现了「类型公开、表示私有、操作受限」的 ADT 核心。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。


## 术语速查

本节出现的关键术语已整理为速查表——它们也是后续各篇反复使用的核心词汇。读第二遍时，可以只看此表回忆每项的含义，想不起的再回正文对应小节。

| 术语 | 一句话定位 |
| --- | --- |
| package | ## 1 Ada：package 是 ADT 的一等公民 |
| class | ## 2 C++：class + 访问控制 |
| 共同点 | 共同点：三者都实现了「类型公开、表示私有、操作受限」的 ADT 核心。 |
| Ada | # Ada、C++ 与 Java 中的数据抽象实现 |
| C++ | # Ada、C++ 与 Java 中的数据抽象实现 |
| Java | # Ada、C++ 与 Java 中的数据抽象实现 |
| 命名封装：命名空间与包 | 在下一节，我们将看封装的另一种形态——命名封装：命名空间与包。 |

**辨析｜易错点：** 术语速查的价值不在「背定义」，而在「建立联系」——表中的每一条都对应正文的一个核心概念。复习时把表格当「目录」，顺着每条术语回忆它的定义、示例与易错点，比反复读正文更高效。「术语是知识的锚点」——记住术语，就记住了它背后的整个概念簇。

## 7 小结

- **Ada** 用 package 实现 ADT，`limited private` 是最强封装——类型公开、表示私有、连赋值都禁。
- **C++** 用 class + 访问控制，`private`/`protected`/`friend` 分粒度；指针算术是语法级封装的破口。
- **Java** 用 class + 引用语义，`private` 编译期强制；反射是运行时后门。
- 封装强度 = 外部访问表示的通道数；「接口稳定」的核心是「表示私有」，语言机制决定封装的强制程度。

在下一节，我们将看封装的另一种形态——**命名封装：命名空间与包**。
