---
title: 智能指针：Box、Rc 与 RefCell
date: 2026-08-07
---

# 智能指针：Box、Rc 与 RefCell

<div class="epigraph">
<p>指针会伤人，除非它知道自己是谁、拥有什么、能借给谁。</p>
<footer>—— 对 Rust 智能指针家族的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第15章 ｜ 2026-08-07</p>
</div>

## 为什么从智能指针开始

普通的引用 `&T` 只是「借用」，不拥有数据。**智能指针（smart pointer）**是「拥有数据的指针」——它指向堆上的数据，并附带额外的能力与元数据。其实我们已经用过智能指针了：`String` 是「拥有 UTF-8 字节的堆字符串」，`Vec<T>` 是「拥有元素的堆数组」。这一章把概念系统化，介绍三个最基础的智能指针：

- **`Box<T>`**：最简单的拥有型指针，指向堆上单个值。
- **`Rc<T>`**：引用计数指针，让一个值有多个所有者。
- **`RefCell<T>`**：把「借用检查」从编译期挪到运行期。

它们组合起来，能表达那些「单纯所有权」表达不了的数据结构——比如链表、图、双向引用。这也是后续并发章节（`Arc`/`Mutex`）与 unsafe 章节的铺垫。

## 1 Box：堆上单个值

### 为什么需要 Box

`Box<T>` 把值放到堆上，栈上只留指针。最直接的用途是解决**递归类型**问题：一个类型直接或间接包含自己，大小无法确定。经典例子是链表结点：

```rust
enum List {
    Cons(i32, List),    // 错误：无限递归，类型大小无穷
}
```

`List` 里嵌套 `List`，编译器无法确定它占多少字节——它会无限递归。用 `Box` 打破递归：

```rust
enum List {
    Cons(i32, Box<List>),   // Box 是指针，大小固定
    Nil,
}

use crate::List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
}
```

`Box<List>` 是一个固定大小的指针，指向堆上的下一个结点——类型大小从此有界，递归得以表达。这是「把动态大小的类型装进固定大小的指针」的经典解法，与 C 里「链表结点存指向下一个结点的指针」同理，但 Rust 的 `Box` 自动管理堆内存的分配与释放。<span class="marginnote">`Box` 的价值不止递归类型：它还用于「把大对象移到堆上以减小栈占用」「把不透明的类型藏起来（返回 `Box<dyn Trait>`）」等场景。凡是「需要拥有堆上数据」而 `String`/`Vec` 又不够精确的地方，`Box` 是最简选择。</span>

### Box 的基本操作

```rust
let b = Box::new(5);      // 堆上放一个 5
println!("{}", b);        // 自动解引用，等价于 *b
```

`Box::new(5)` 在堆上分配 `5`，`b` 拥有它。`println!("{}", b)` 里 `b` 被自动解引用为 `5`。`Box` 实现 `Deref` trait（解引用）与 `Drop` trait（析构时释放堆内存）。

### 解引用运算符 `*`

`Box<T>` 能当 `&T` 用，靠的是 `Deref` trait。`*b` 显式解引用拿到堆上的值：

```rust
let x = 5;
let y = Box::new(x);   // y: Box<i32>

assert_eq!(5, x);
assert_eq!(5, *y);     // 解引用 y 得到 5
```

`Deref` trait 让智能指针能像普通引用一样解引用。**解引用强制转换（deref coercion）**进一步让 `&Box<T>` 自动转成 `&T`：给函数传 `&Box<MyType>`，它会自动变成 `&MyType`。这解释了为什么 `&String` 能传给收 `&str` 的函数——`String` 实现了 `Deref<Target = str>`。

## 2 Drop：析构钩子

### 自动清理

`Drop` trait 定义值离开作用域时的清理动作。`Box`、`String`、`Vec` 都实现了它来释放堆内存。可以自己实现：

```rust
struct CustomSmartPointer {
    data: String,
}

impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("清理 {} 的资源", self.data);
    }
}

fn main() {
    let c = CustomSmartPointer { data: String::from("my stuff") };
    // 作用域结束时自动打印清理信息
}
```

`drop` 方法在值离开作用域时自动调用——**确定性析构**，这就是第4篇说的「Rust 不需要 GC」的实现机制。手动提前清理用 `std::mem::drop`：

```rust
let c = CustomSmartPointer { data: String::from("临时") };
drop(c);   // 提前调用析构
```

## 3 Rc：引用计数共享所有权

### 多个所有者

`Rc<T>`（Reference Counted）让一个值有**多个所有者**：每克隆一次 `Rc`，引用计数 +1；每个 `Rc` 离开作用域，计数 -1；计数归零时数据被释放。适合「数据被多处共享读取」的场景：

```rust
use std::rc::Rc;

enum List {
    Cons(i32, Rc<List>),
    Nil,
}

fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    let b = Cons(3, Rc::clone(&a));   // a 的所有权被共享
    let c = Cons(4, Rc::clone(&a));   // a 再次被共享
}
```

`b` 和 `c` 都指向 `a` 的数据，`a` 的引用计数是 3（`a` 自己 + 两次克隆）。只有三个 `Rc` 全部离开作用域，数据才被释放。<span class="marginnote">`Rc::clone` 只增加计数，不深拷贝数据——这就是「共享所有权」：多个所有者一起负责，最后一个离开的负责释放。它与 C++ 的 `shared_ptr` 同源，但 Rust 的 `Rc` 只能单线程使用（线程间用 `Arc`，第17篇），且不可变。</span>

### Rc 的限制：只读共享

`Rc<T>` 内部的值**不可变**——因为「共享的」值被多个所有者读取，谁也不能独占修改（与第5篇的 `&`/`&mut` 规则呼应）。要让共享的 `Rc` 还能改，就得和 `RefCell` 组合。

## 4 RefCell：运行期借用检查

### 内部可变性

`RefCell<T>` 提供**内部可变性（interior mutability）**：即使外部只有不可变引用，内部数据仍可修改。它把借用检查从**编译期**推迟到**运行期**：

```rust
use std::cell::RefCell;

let data = RefCell::new(5);

// borrow_mut 在运行期检查：同一时刻只能有一个可变借用
*data.borrow_mut() = 6;

// borrow 只读借用，可多个并存
let r1 = data.borrow();
let r2 = data.borrow();
println!("{r1} {r2}");
```

`borrow_mut()` 违反规则时（已有可变借用再借）**运行期 panic**——编译期放行，运行期兜底。这与第5篇「引用规则编译期检查」形成对照：`RefCell` 把同样的规则挪到了运行期，换来了「结构上看似不可变却能改」的灵活性。<span class="marginnote">为什么需要内部可变性？典型场景：一个测试替身（mock）要记录自己被调用了多少次，但测试代码只持有不可变引用——`RefCell<usize>` 让计数在「不可变外壳」里可变。这是「接口不可变、内部可记账」的经典需求。</span>

### 与 Box/Rc 的组合拳

三个智能指针各管一件事，常常组合使用：

- `Box<T>`：编译期检查的拥有型指针（单一所有者）。
- `Rc<T>`：编译期检查的多所有者共享。
- `RefCell<T>`：运行期检查的内部可变性。

**`Rc<RefCell<T>>`** 是最常见的组合：多个所有者共享，且能修改共享的值。比如可变的共享链表：

```rust
use std::rc::Rc;
use std::cell::RefCell;

let value = Rc::new(RefCell::new(5));

let a = Rc::clone(&value);
let b = Rc::clone(&value);

*value.borrow_mut() += 10;   // 通过任一 Rc 都能改
println!("{}", value.borrow());   // 15
```

`value` 的引用计数管理「谁还活着」，`RefCell` 管理「此刻谁能写」——两者叠加，实现了「共享 + 可变」。

## 5 核心对比：智能指针家族

| 指针 | 拥有方式 | 借用检查时机 | 可变性 | 线程 |
| --- | --- | --- | --- | --- |
| `Box<T>` | 单一所有者 | 编译期 | 可（自身可变时） | 可 Send |
| `Rc<T>` | 多所有者（引用计数） | 编译期 | 不可变 | 单线程 |
| `RefCell<T>` | 单一所有者 | 运行期 | 内部可变 | 单线程 |
| `Rc<RefCell<T>>` | 多所有者 + 运行期借用 | 运行期 | 内部可变 | 单线程 |
| `Arc<T>` / `Mutex<T>` | 多所有者 + 互斥锁 | 运行期 | 锁内可变 | 多线程（第17篇） |

选型直觉：**默认用 `Box`；要共享只读用 `Rc`；要共享且可变用 `Rc<RefCell<T>>`；要跨线程共享用 `Arc` + `Mutex`**。`RefCell` 是「用运行期 panic 换编译期灵活性」的权衡——能用编译期检查就不用它。

## 6 公式解析：Rc 的引用计数生命周期

`Rc<T>` 的行为可以写成「计数器随克隆增减」：

$$
n_{\text{owners}} = 1 + \#\text{clone} - \#\text{drop}
$$

$$
\text{释放数据} \iff n_{\text{owners}} = 0
$$