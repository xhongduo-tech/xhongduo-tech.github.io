---
title: 结构体与联合体
date: 2026-08-07
---

# 结构体与联合体

<div class="epigraph">
<p>结构体让不同的数据聚合成一个整体——这是 C 里「对象」的最初雏形。</p>
<footer>—— 对 C 数据抽象的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ K&R《C 程序设计语言》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从结构体讲起

到目前为止，我们处理的数据都是「单一类型」的：一个 `int`、一串 `char`。但现实对象往往是**多种属性的聚合**——一本书有书名、作者、价格、页数；一个学生有学号、姓名、成绩。C 用**结构体（structure）** 把这些不同类型的成员打包成一个自定义类型。<span class="marginnote">结构体是 C 迈向「数据抽象」的关键一步：它把零散数据组织成有名字的整体，是链表节点、树节点、配置记录、文件头部等一切复合数据的基础。在 Python/Java 里与之对应的是 class/对象——C 用结构体 + 函数指针实现了最朴素的「面向对象」。</span>这一节讲清结构体的定义、访问、嵌套、传参与动态分配，并辨析它的近亲**联合体（union）**。

## 1 定义结构体：打包多种数据

用 `struct` 关键字定义一个结构体类型：

```c
struct Book {
    char  title[100];
    char  author[50];
    float price;
    int   pages;
};
```

这个定义创建了一个**类型** `struct Book`，但**还没有分配任何内存**——它只是一张「蓝图」。声明变量后才有真正的存储：

```c
struct Book b;            /* 声明一个 Book 变量 */
struct Book books[10];    /* 结构体数组 */
struct Book *pb;          /* 指向结构体的指针 */
```

访问成员用**点运算符 `.`**：

```c
b.price = 59.9;
b.pages = 320;
strcpy(b.title, "C Primer Plus");
```

**结构体变量整体赋值是合法的**：`struct Book b2 = b;` 把 b 的所有成员逐字节复制给 b2（浅拷贝）。

**`typedef` 给结构体起别名**，省去每次写 `struct`：

```c
typedef struct {
    char name[30];
    int  age;
    float score;
} Student;

Student s1;            /* 不用再写 struct Student */
s1.age = 18;
```

## 2 访问结构体成员：点号与箭头

结构体变量用 `.`，**指向结构体的指针**用 `->`（箭头）——它是 `(*p).member` 的语法糖：

```c
struct Book b;
struct Book *pb = &b;

(*pb).price = 59.9;    /* 先解引用再取成员 */
pb->price = 59.9;      /* 等价，且更常用 */
```

`->` 需要记住的规则：**左边必须是指针**。`b->price` 会编译报错（`b` 不是指针），`pb.price` 也报错——两者不能混用。这个约定贯穿后续所有指针操作复合数据的代码。

**结构体作为函数参数**：默认按值传递（复制整个结构体），函数内修改不影响调用者。要修改或避免复制开销，传指针：

```c
void update_price(struct Book *pb, float new_price)
{
    pb->price = new_price;   /* 通过指针修改外部结构体 */
}
```

结构体很大时传值会复制整块内存（性能开销），传指针只传 8 字节地址——**大数据结构一律传指针**是工程惯例。

## 3 结构体的嵌套

结构体成员本身可以是另一个结构体：

```c
typedef struct {
    int year;
    int month;
    int day;
} Date;

typedef struct {
    char   name[30];
    Date   birthday;     /* 嵌套结构体 */
} Person;

Person p;
p.birthday.year = 2000;   /* 层层点号访问 */
```

也可以嵌套**指向结构体的指针**，这正是链表节点的雏形（第 5 篇《链表与基本数据结构实现》会展开）：

```c
typedef struct Node {
    int data;
    struct Node *next;    /* 自引用：指向下一个节点 */
} Node;
```

`struct Node *next` 里为什么必须写 `struct Node`？因为 `typedef` 别名 `Node` 在结构体**定义完成之前**还不存在，而 `struct Node` 这个标签从 `struct Node {` 出现那一刻就可用了。

## 4 联合体：让多种类型共享同一块内存

**联合体（union）** 与结构体语法几乎相同，但语义完全不同：联合体的所有成员**共享同一块内存**，同一时刻只有一个成员有意义。

```c
union Value {
    int    i;
    float  f;
    char   c;
};
```

`sizeof(union Value)` 是**最大成员**的大小（4 字节，取 `int`/`float` 的较大者），而不是三者之和。写入 `v.f = 3.14;` 后，再读 `v.i` 得到的是 `3.14` 的位模式按 `int` 解释——毫无意义的值。

**联合体的典型用途**：

**节省内存**：一个字段在不同场景下类型不同，但不会同时用到。
**类型双关（type punning）**：把 `float` 的位当作 `int` 来看（不过这属于未定义行为，工程上慎用）。
- **变体记录**：配合一个「类型标签」字段，实现「有时是整数、有时是浮点」的灵活数据：

```c
typedef struct {
    int   tag;        /* 0 表示 int，1 表示 float */
    union {
        int   i;
        float f;
    } value;
} Variant;
```

这种「**标签 + 联合体**」的组合是 C 里实现「多态数据」的经典手法。<span class="marginnote">联合体的成员共用内存，意味着你「写一个、读另一个」会得到位级重解释。它高效但危险：必须靠外面的 `tag` 记住当前到底是什么类型。C 语言在数据类型灵活性上走了条完全不同于 Python 的路线——一切由程序员负责。</span>

## 5 公式解析：结构体与联合体的大小

`sizeof` 是结构体的经典考点。结构体的总大小**不是**成员大小简单相加，因为要满足**对齐（alignment）**：

$$
\text{sizeof(struct)} = \text{所有成员对齐后所占的空间}，\text{且是最大对齐的倍数}
$$

看这个结构体：

```c
struct S {
    char   c;     /* 1 字节，对齐 1 */
    int    i;     /* 4 字节，对齐 4 */
    char   d;     /* 1 字节，对齐 1 */
};
```

- **第一步，成员顺次摆放并填对齐空隙**：`c` 占偏移 0；`i` 必须从 4 的倍数开始，于是偏移 1~3 空着，`i` 放在偏移 4~7；`d` 放偏移 8。
- **第二步，总大小要对齐到最大对齐的倍数**：目前用了 9 字节（偏移 0~8），最大对齐是 4，`9` 向上取 4 的倍数得 **12**。
- **第三步，结论**：`sizeof(struct S) == 12`，而不是直觉的 $1+4+1=6$