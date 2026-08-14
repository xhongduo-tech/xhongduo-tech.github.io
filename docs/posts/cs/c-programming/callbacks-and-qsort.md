---
title: 回调函数与 qsort
date: 2026-08-07
---

# 回调函数与 qsort

<div class="epigraph">
<p>排序不必知道它排的是什么——规则由调用者提供，算法只负责编排。</p>
<footer>—— 对 qsort 回调设计的中文转述</footer>
</div>

<div class="article-byline">
<p>第三级 · C 语言编程 ｜ K&R《C 程序设计语言》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从回调与 qsort 讲起

上一节讲函数指针时已经预演了 `qsort`。这一节把它讲透，因为它集中体现了 C 最重要的设计思想之一：**通用算法与具体行为解耦**。<span class="marginnote">`qsort` 是 C 标准库里最优雅的函数之一：一个排序函数，能排 `int`、`double`、结构体、字符串——一切类型。秘密就在第四个参数：一个「比较函数指针」（回调）。算法不知道类型，只知道「按这个函数去比较」。这比给每种类型各写一个排序函数高明得多。</span>这一节读懂 `qsort` 的签名、写出正确的比较函数，并把这个模式推广到自己的通用函数。

## 1 qsort 的签名与回调

`qsort` 声明在 `stdlib.h`：

```c
void qsort(void *base, size_t nmemb, size_t size,
           int (*compar)(const void *, const void *));
```

四个参数：

**`base`**：待排序数组的首地址（`void *` 表示「任何类型的数组」都能传）。
**`nmemb`**：元素个数。
- **`size`**：每个元素的大小（字节）——`qsort` 需要它来移动元素。
- **`compar`**：**比较函数指针**（回调）——比较规则完全由调用者定义。

`compar` 的约定：传入两个元素的**地址**（`const void *`），返回：

- **负数**：第一个元素应排在第二个**前面**；
- **0**：两者相等；
- **正数**：第一个应排在第二个**后面**。

`qsort` 内部在需要比较任意两个元素时调用 `compar`，并依据返回值决定先后——**它把自己完全蒙在鼓里，只知道「调比较函数 + 搬元素」**。

## 2 写出正确的比较函数

比较函数的骨架是「先把 `void*` 转回具体指针，再解引用比较」。排 `int`：

```c
int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a;    /* a 是元素地址，转为 int* 再解引用 */
    int y = *(const int *)b;
    if (x < y) return -1;
    if (x > y) return  1;
    return 0;
}
```

可以缩写成 `return (x > y) - (x < y);`——返回三态值。**不要写成 `return x - y;`**：当 `x`、`y` 的差超过 `int` 范围时溢出，结果是未定义行为（`2147483647 - (-2147483647)` 直接溢出）。安全写法是三态比较。

排 `double`：

```c
int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);   /* 浮点 NaN 需另作处理，此处从简 */
}
```

排字符串（注意是 `char **` 而不是 `char *`）：

```c
int cmp_str(const void *a, const void *b)
{
    /* a 指向数组元素，元素本身是 char*，所以解引用得到 char** 再取值 */
    return strcmp(*(const char **)a, *(const char **)b);
}
```

**`*(const char **)a` 是字符串比较最容易错的一行**：`a` 是「元素地址」，元素是 `char *`，所以 `*(const char **)a` 才是那个字符串指针。<span class="marginnote">字符串数组 `char *words[]` 的元素是 `char*`，`qsort` 传给比较函数的 `a` 是指向这个元素的指针——即「指向 `char` 的指针的指针」。解引用一次得 `char*`，再交给 `strcmp`。忘了这层「指针的指针」，要么编译报错、要么比较的是指针地址本身。</span>

## 3 排结构体：按任意字段

`qsort` 排结构体同样简单——比较函数决定「按哪个字段、升序还是降序」：

```c
typedef struct {
    char  name[30];
    int   score;
} Student;

int cmp_by_score_asc(const void *a, const void *b)
{
    int x = ((const Student *)a)->score;
    int y = ((const Student *)b)->score;
    return (x > y) - (x < y);
}

int cmp_by_score_desc(const void *a, const void *b)
{
    return cmp_by_score_asc(b, a);   /* 交换参数 = 降序 */
}

Student class[] = {{"Alice", 92}, {"Bob", 78}, {"Carol", 85}};
qsort(class, 3, sizeof(Student), cmp_by_score_asc);
```

两个技巧：

**降序的捷径**：比较函数调 `asc(b, a)`（参数交换）即得降序，不用重写逻辑。
**多字段排序**：先比主字段，相等再比次字段：

```c
int cmp_student(const void *a, const void *b)
{
    const Student *s1 = (const Student *)a;
    const Student *s2 = (const Student *)b;
    if (s1->score != s2->score)
        return (s1->score > s2->score) - (s1->score < s2->score);
    return strcmp(s1->name, s2->name);   /* 分数相同按姓名 */
}
```

## 4 把回调模式用到自己身上

`qsort` 的模式可以迁移到任何「通用遍历/处理」的函数。写一个通用的「对每个元素应用操作」：

```c
void apply(int *arr, size_t n, void (*fn)(int *))
{
    for (size_t i = 0; i < n; i++)
        fn(&arr[i]);       /* 把每个元素的地址交给回调 */
}

void double_it(int *x) { *x *= 2; }
void print_it(int *x)  { printf("%d ", *x); }

int a[] = {1, 2, 3};
apply(a, 3, double_it);   /* 全部翻倍 */
apply(a, 3, print_it);    /* 2 4 6 */
```

回调让「**遍历框架**」与「**具体操作**」分离：`apply` 只管循环与调用，做什么由调用者传入的函数决定。这个模式再往前走一步，就是 C++ 的 `std::for_each`、Python 的 `map`/`filter`、以及一切事件系统。

**回调与状态**：普通函数指针回调无法携带「额外上下文」。需要上下文时，惯用做法是给回调加一个 `void *ctx` 参数，把用户数据传进去：

```c
typedef struct { int count; } Ctx;
void count_positives(int *x, void *ctx) {
    if (*x > 0) ((Ctx *)ctx)->count++;
}
```

`qsort` 的 C 标准版本没有 `ctx` 参数（早期的接口局限），很多库函数（如 `bsearch`）也如此；现代 C 库普遍提供带 `void *` 上下文的回调变体。<span class="marginnote">「回调不带状态」是老式 C 回调的一大痛点：想在比较时用外部参数（如排序方向）只能靠全局变量。工程解法是给回调约定一个 `void *` 参数传上下文——这个模式在 C 里如此常见，以至于它直接催生了 C++ 的 this 指针与 lambda 捕获。</span>

## 5 核心对比表：`qsort` 与手写排序

| 维度 | `qsort` | 手写冒泡/选择排序 |
| --- | --- | --- |
| 类型通用性 | 任意类型（`void *` + 回调） | 每种类型写一遍 |
| 平均复杂度 | O(n log n)（快排） | O(n²) |
| 比较规则 | 调用者自定义 | 写死在代码里 |
| 代码量 | 一次写比较函数 | 每种类型一套循环 |
| 适用场景 | 工程通用排序 | 教学、小数据、特殊需求 |

**`qsort` 的工程价值不是「更快」而是「更少」**：一个排序框架 + 几个短小的比较函数，覆盖全部类型。它的内部实现是快速排序（C 标准只保证「排好序」，不规定算法），平均 O(n log n)。理解回调，就能理解 `qsort` 的全部设计——**算法与策略分离，是它长青至今的原因**。

## 6 公式解析：`qsort` 的移动模型

`qsort` 不知道自己排的是什么类型，却能把元素搬来搬去——靠的是 `size` 参数：

$$
\text{元素 } j \text{ 的地址} = \text{base} + j \times \text{size}
$$

- **第一步，算偏移**：第 j 个元素的字节偏移是 $j \times size$