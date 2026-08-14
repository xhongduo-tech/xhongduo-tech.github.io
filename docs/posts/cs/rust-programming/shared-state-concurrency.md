---
title: 共享状态并发：Mutex 与 Arc
date: 2026-08-07
---

# 共享状态并发：Mutex 与 Arc

<div class="epigraph">
<p>锁不是万能的，但配合所有权，它可以被编译器编排成一场不会出错的演出。</p>
<footer>—— 对 Rust 共享状态并发的概括</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第16章 ｜ 2026-08-07</p>
</div>

## 为什么从 Mutex 与 Arc 开始

上一章的消息传递哲学是「不共享，发消息」。但有些场景天生适合共享状态：一个计数器被多个线程递增、一份配置被多个线程读取。Rust 对这类场景的回答是 **`Mutex<T>`（互斥锁）** + **`Arc<T>`（原子引用计数）**。

`Mutex` 让「同一时刻只有一个线程能访问数据」成为显式规则；`Arc` 让「多个线程共享同一份数据的所有权」成为可能。两者的组合 `Arc<Mutex<T>>` 是 Rust 共享状态并发的标准形态。这一章会看到 Rust 如何把「加锁忘了解锁」「锁内数据裸奔」这些并发灾难变成编译期错误——**锁的类型系统保证你无法忘了解锁，因为解锁是作用域结束自动发生的**。

## 1 Mutex：互斥锁

### 基本用法

**`Mutex<T>`** 包裹数据，`lock()` 加锁、返回数据的可变引用：

```rust
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);

    {
        let mut num = m.lock().unwrap();  // 加锁，得到 MutexGuard
        *num = 6;                          // 通过 guard 修改数据
    }                                      // 作用域结束，guard drop，自动解锁

    println!("m = {m:?}");   // Mutex { data: 6 }
}
```

`m.lock()` 返回 `MutexGuard`——它解引用到 `T` 的可变引用（`DerefMut`），因此 `*num = 6` 能改数据。**`MutexGuard` 离开作用域时自动解锁**：不用手写 `unlock()`，也不存在「忘了解锁」——解锁是析构的一部分，由所有权保证。<span class="marginnote">`MutexGuard` 的自动解锁是「确定性析构」的又一次体现：锁的释放时机由作用域决定，而非人手。对比 C 的 `pthread_mutex_lock`/`unlock`——忘了 unlock 就是死锁，Rust 从类型层面消灭了这种错误。</span>

`lock()` 返回 `Result`：如果持有锁的线程 panic，`Mutex` 会进入中毒（poisoned）状态，`lock()` 返回 `Err`。`.unwrap()` 是「panic 就崩溃」的快速处理，生产代码可用 `.unwrap_or_else(|e| e.into_inner())` 恢复中毒锁。

### 两个线程共享计数器

用 `Arc<Mutex<T>>` 让多个线程共享一个可变计数器：

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);   // 每个线程一份 Arc
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;                        // 临界区：互斥访问
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("结果 = {}", *counter.lock().unwrap());   // 10
}
```

关键三步：

1. **`Arc::new(Mutex::new(0))`**：`Mutex` 提供互斥，`Arc` 提供多线程共享所有权。
2. **`Arc::clone(&counter)`**：每个线程持有一份 `Arc`，引用计数管理「最后一个线程退出时才释放」。
3. **`counter.lock()`**：进入临界区前加锁，其他线程在此阻塞，直到当前 guard 释放。

`Arc` 与第16篇的 `Rc` 的区别只有一个：**`Arc` 的引用计数是原子的**，可以在线程间共享；`Rc` 的计数非原子，单线程专用。<span class="marginnote">为什么 `Rc` 不能跨线程？因为 `Rc::clone` 是「计数 +1」的非原子操作，两个线程同时 clone 会竞态损坏计数。`Arc` 用原子操作（`fetch_add`/`fetch_sub`）保证计数增减在多线程下安全——这是 `Send`/`Sync` trait 要管的事，见第22篇。</span>

## 2 Send 与 Sync：线程安全的类型契约

### 两个 marker trait

为什么 `Arc<Mutex<i32>>` 能跨线程，而 `Rc<i32>` 不行？标准库用两个**标记 trait（marker trait）**给出类型层面的契约：

**`Send`**：该类型的**所有权**可以转移到另一个线程。`i32`、`String`、`Vec` 都 `Send`；`Rc` 不是（多线程转移 `Rc` 会破坏计数）。

**`Sync`**：该类型的**引用**可以安全地跨线程共享。`&T` 可被多个线程同时持有而不产生数据竞争。`Mutex<T>` 是 `Sync`（锁保证互斥）；`RefCell<T>` 不是（运行期借用检查非线程安全）。<span class="marginnote">`Send` 与 `Sync` 的关系：`T: Sync` 当且仅当 `&T: Send`。直觉：能安全共享引用，等价于「引用能安全发送」。一个类型 `Sync` 意味着它的共享引用可以跨线程，`Send` 意味着它的所有权可以跨线程。</span>

### 编译器自动推导

`Send`/`Sync` 是**自动 trait**：编译器根据字段自动实现。一个结构体只要所有字段都是 `Send`，它自己就是 `Send`；含 `Rc` 或 `RefCell` 的结构体自动不是。

这带来一个惊人的结果：**线程安全不是「你记得加锁」，而是「类型不允许你做不安全的事」**。尝试 `thread::spawn` 一个捕获了 `Rc` 的闭包，编译器直接拒绝——`Rc` 不满足 `Send`。

## 3 Mutex 内部的 `MutexGuard` 与死锁

### 为什么不会忘解锁

`MutexGuard` 实现 `Deref` 与 `DerefMut`（所以能 `*num` 访问数据），并实现 `Drop`（解锁）。解锁不是方法调用，而是 **guard 析构的副作用**：

```rust
{
    let mut num = counter.lock().unwrap();  // 加锁
    *num += 1;
}   // guard 在这里 drop → 自动解锁
```

**Rust 的锁没有「忘解锁」的可能**：guard 一定会在作用域结束时被 drop（除非你显式 `mem::forget`，那是有意的泄漏）。这把死锁的第一大来源——忘解锁——从语言层面消除了。

### 手动提前解锁

极少数情况下需要提前解锁（避免持锁太久）：

```rust
let mut num = counter.lock().unwrap();
*num += 1;
drop(num);            // 提前解锁，释放锁给其他线程
```

`drop(num)` 手动析构 guard，立即解锁。之后其他线程不再阻塞。

### 死锁：锁顺序问题

虽然忘解锁被消灭，但**死锁**仍可能发生——当两个线程各自持有一把锁、又都想获取对方的锁时：

```rust
// 线程 A：lock(m1) → 然后想 lock(m2)
// 线程 B：lock(m2) → 然后想 lock(m1)
// 双方都在等对方释放，谁也进不去
```

死锁的根源是**锁的顺序不一致**。解法是约定「所有线程按同一顺序获取多把锁」。Rust 的编译器不检查锁顺序（这超出静态分析能力），需要开发者遵守纪律——这与《操作系统》课程的死锁预防策略（锁排序、一次性申请全部锁）完全一致。<span class="marginnote">Rust 消灭了「忘解锁」，但没消灭「锁顺序死锁」。好消息是消息传递（第16章）在多数场景下可以完全避开锁——设计并发系统时优先考虑通道，只在确实需要共享状态时才上锁，这是 Rust 社区的并发设计共识。</span>

## 4 公式解析：共享状态并发的组合语义

`Arc<Mutex<T>>` 的两个「壳」各管一件事，可以分别写成不变量：

$$
\text{Mutex 保证：} \quad \forall t,\ \#\text{持有锁的线程}(t) \le 1
$$

$$
\text{Arc 保证：} \quad \text{数据存活} \iff \text{引用计数} > 0
$$