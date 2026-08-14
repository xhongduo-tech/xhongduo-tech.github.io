---
title: 常用集合：Vec、String 与 HashMap
date: 2026-08-07
---

# 常用集合：Vec、String 与 HashMap

<div class="epigraph">
<p>数据结构选对了，一半的算法问题已经解决。</p>
<footer>—— 传统编程格言（数据结构课程共识）</footer>
</div>

<div class="article-byline">
<p>第三级 · Rust 编程 ｜ The Rust Book 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从常用集合开始

真实程序几乎都要处理「一堆数据」：一列待办事项、一段文本、一份名字到电话的映射。Rust 标准库提供三种最常用的**集合（collection）**——**`Vec<T>`**（动态数组）、**`String`**（可增长 UTF-8 文本）、**`HashMap<K, V>`**（键值映射）。它们都存放在堆上、可增长，但各自的适用场景与内存布局截然不同。<span class="marginnote">集合与前面学的数组不同：数组定长且栈上存放，集合动态增长且堆上存放。选对集合类型，是性能与正确性的双重前提——这与第三级《数据结构》课程的「选数据结构先于写算法」一脉相承。</span>

这一章同时会看到所有权与集合如何相互作用：遍历时元素是借用还是被取走、插入时值是被移动还是被复制——这些细节决定了你能写出多少「编译不过」的代码。

## 1 Vec：动态数组

### 创建与更新

`Vec<T>` 是连续内存上的动态数组，用 `vec!` 宏或 `Vec::new` 创建：

```rust
let v = vec![1, 2, 3];          // vec![1, 2, 3]，类型推断为 Vec<i32>
let mut v = Vec::new();         // 空 Vec，需要后面让类型可推断
v.push(5);                      // 追加
v.push(6);
```

`push` 在末尾追加，数组容量不够时自动扩容。`Vec` 拥有堆上的元素，离开作用域时元素与数组一起被 drop。

### 读取：索引与 get

两种读取方式：

```rust
let v = vec![1, 2, 3, 4, 5];

let third: &i32 = &v[2];          // 索引：越界直接 panic
let third: Option<&i32> = v.get(2); // get：越界返回 None
```

`&v[2]` 越界时 panic（程序崩溃），`v.get(2)` 越界时返回 `None`——前者适合「确定不会越界」，后者适合「不确定，需要处理不存在的情况」。这里 `Option` 的用处又一次出现：`get` 用 `Some(&value)` / `None` 表达「可能取不到」。<span class="marginnote">「`v[2]` panic vs `v.get(2)` 返回 None」是 Rust 对边界问题的两种立场：急停（快速失败）还是软处理（显式分支）。生产代码里 `get` + `match` 比裸索引更常见，因为它不把「越界」留给崩溃。</span>

### 遍历：借用 vs 取走

`for` 遍历有两种语义：

```rust
let mut v = vec![100, 32, 57];

for i in &v {             // 不可变借用：只读遍历
    println!("{i}");
}

for i in &mut v {         // 可变借用：可修改
    *i += 50;
}

for i in v {              // 取走所有权：v 之后不可用
    println!("{i}");
}
```

三种遍历对应三种所有权姿势：借用、可变借用、整体取走。第三种会把元素逐个移出 `v`，循环结束后 `v` 已不可用。

### Vec 存不同数据类型：枚举的妙用

`Vec` 元素类型必须相同，但配合枚举可以装「不同类型」：

```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("blue")),
    SpreadsheetCell::Float(10.12),
];
```

所有变体都是 `SpreadsheetCell`，所以 `Vec<SpreadsheetCell>` 合法；取出时用 `match` 区分具体类型。

## 2 String：可增长 UTF-8 文本

### String 是字节的集合

`String` 是对 `Vec<u8>` 的封装，加上「必须是合法 UTF-8」的约束。三种创建方式：

```rust
let mut s = String::new();
let s2 = String::from("hello");
let s3 = "hello".to_string();   // &str → String
```

追加与拼接：

```rust
let mut s = String::from("foo");
s.push_str("bar");          // 追加 &str，s 变成 "foobar"
s.push('!');                // 追加单个字符

let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2;          // s1 被移动，s2 被借用
```

**`+` 运算符的签名**：`fn add(self, s: &str) -> String`。`s1 + &s2` 里 `s1` 的所有权被 `add` 吞掉（所以之后不能再用 `s1`），`&s2` 是借用。这个签名是历史遗留，但解释了为什么拼接后 `s1` 失效。

### 为什么 String 不支持索引

`String` 不能像 `s[0]` 那样索引。因为字符串是 **UTF-8 字节序列**，`s[0]` 是「第 0 个字节」，可能恰好是某个字符的一部分。Rust 拒绝给你一个「可能切开字符」的索引，转而提供三种迭代视角：<span class="marginnote">这背后是 Rust 的「诚实」设计：索引可能返回半个中文字符的字节，在别的语言里是静默乱码，在 Rust 里是编译错误或显式方法。三种视角——字节、字符、字形簇——各有各的用途。</span>

```rust
let s = String::from("你好");
s.bytes()        // 字节：每 3 字节一组，共 6 项
s.chars()        // 字符：'你' '好'
s.graphemes()    // 字形簇：需要 unicode-segmentation crate
```

`chars()` 按 Unicode 标量值迭代，是「数有几个字」的标准方式。`s.len()` 返回的是**字节数**（`"你好"` 是 6），不是字符数。

## 3 HashMap：键值映射

### 创建与更新

`HashMap<K, V>` 是哈希表，键到值的映射。来自标准库，需要显式 `use`：

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
```

读取：

```rust
let team = String::from("Blue");
let score: Option<&i32> = scores.get(&team);   // Some(&10)

for (key, value) in &scores {   // 遍历：键值对，借用
    println!("{key}: {value}");
}
```

`get` 返回 `Option<&V>`——键不存在时是 `None`，而不是崩溃或 null。

### 所有权语义

插入键值对时，**键与值都被移动进 HashMap**：

```rust
let field_name = String::from("Favorite color");
let field_value = String::from("Blue");

map.insert(field_name, field_value);
// field_name 与 field_value 之后不可用，所有权已移交 map
```

`&str`/整数等 `Copy` 类型则是复制进 map，原变量仍可用。<span class="marginnote">HashMap 的所有权规则与 Vec 一致：insert 就是「把值交给容器保管」。这再次体现 Rust 的「容器拥有内容」模型——容器 drop 时内容一起 drop，没有悬挂的引用。</span>

### 覆盖、查询不存在时插入、更新旧值

三种常见更新模式：

```rust
map.insert(String::from("Blue"), 10);
map.insert(String::from("Blue"), 25);   // 覆盖：现在 Blue → 25

map.entry(String::from("Yellow")).or_insert(50);  // 不存在才插入
map.entry(String::from("Blue")).or_insert(50);    // 已存在，不变

let count = map.entry(String::from("Yellow")).or_insert(0);
*count += 1;   // 统计词频的惯用法
```

`entry` API 是「查-插」的原子操作：返回一个 `Entry`，`.or_insert(value)` 在键不存在时插入，已存在时返回已有值的可变引用。词频统计用它能一行搞定「没有就置 0，有就 +1」。

## 4 核心对比：三种集合的选型

| 集合 | 数据结构 | 适用场景 | 有序？ | 查单个 | 遍历 |
| --- | --- | --- | --- | --- | --- |
| `Vec<T>` | 动态数组 | 一串同类元素、按下标访问 | 是 | 下标 O(1) | 快（连续内存） |
| `String` | UTF-8 字节序列 | 文本、拼接、格式化 | 是 | 按字节索引（危险） | `chars()`/`bytes()` |
| `HashMap<K,V>` | 哈希表 | 键值查找、词频统计 | 否 | 键 O(1) 平均 | 无序 |

选型直觉：**要保序按位置访问选 `Vec`，要处理文本选 `String`，要按名字查东西选 `HashMap`**。三者都在堆上、都拥有内容、离开作用域自动释放——这是 Rust 集合的统一底色。

## 5 公式解析：Vec 扩容的均摊复杂度

`Vec` 的 `push` 为什么是「均摊 O(1)」？设容量为 $C$，元素个数为 $n$：

$$
\text{push 的成本} = \begin{cases} O(1) & n \lt  C \quad (\text{直接写入}) \\ O(n) & n = C \quad (\text{整体搬入新堆块，容量翻倍}) \end{cases}
$$

拆解：

- **第一步，`push` 平时是 O(1)**：容量够时，把新元素写进下一个空闲槽位并更新长度，一次写入。
- **第二步，满了就扩容**：容量不足时，`Vec` 分配一块更大的堆内存（通常是容量翻倍），把现有 $n$ 个元素搬过去，然后追加。这一次是 $O(n)$。
- **第三步，翻倍策略让均摊 O(1)**：若每次扩容翻倍，扩容发生的次数是 $O(\log n)$ 次，搬运总成本 $1+2+4+\cdots+n = 2n-1 = O(n)$，除以 $n$ 次 `push`，均摊每次 $O(1)$