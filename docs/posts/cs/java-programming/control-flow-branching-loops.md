---
title: 流程控制：分支、循环与跳转语句
date: 2026-08-07
---

# 流程控制：分支、循环与跳转语句

<div class="epigraph">
<p>程序 = 数据结构 + 算法；而流程控制，是算法在代码里的骨架。</p>
<footer>—— 改编自 Niklaus Wirth《Algorithms + Data Structures = Programs》</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第1卷第3章 ｜ 2026-08-07</p>
</div>

## 为什么从流程控制开始

变量与运算符让程序「有东西可算」，但真正的计算从来不是一条直线：要根据条件决定走哪条路、要反复执行同一段操作直到某个条件成立。**流程控制（control flow）**就是程序里的「红绿灯与高速公路」——**分支**（`if`/`switch`）决定往哪走，**循环**（`while`/`for`）决定走多少遍，**跳转**（`break`/`continue`）决定何时刹车换道。<span class="marginnote">流程控制是所有命令式语言的心脏，Java 的这套骨架直接继承自 C。你会在 C、C++、JavaScript、Go 里看到几乎一模一样的 `if`、`while`、`for`——学一次，处处通用。</span>这一篇把三件套讲透，并揪出那些「编译能过、逻辑是错的」高频坑。

## 1 分支：if-else 与 switch

**`if` 语句**按条件执行不同分支：

```java
int score = 85;
if (score >= 90) {
    System.out.println("优秀");
} else if (score >= 60) {
    System.out.println("及格");
} else {
    System.out.println("不及格");
}
```

**重点结论：条件必须是布尔表达式。** Java 与 C 不同——`if (x)` 在 C 里合法（非零即真），在 Java 里**编译错误**，必须写成 `if (x != 0)`。类型安全在这个细节上体现得淋漓尽致：Java 拒绝「整数当布尔」这种隐式转换。

**`switch` 语句**按一个值精确匹配多个分支（Java 7 起支持 `String`，Java 14 起支持**箭头语法**）：

```java
// 传统语法：每个 case 末尾要 break，否则「落穿」
switch (day) {
    case "MON":
    case "FRI":
        System.out.println("工作日"); break;
    case "SAT":
    case "SUN":
        System.out.println("周末"); break;
    default:
        System.out.println("未知"); break;
}

// 箭头语法：无需 break，自动不落穿，还能用作表达式
String kind = switch (day) {
    case "MON", "FRI" -> "工作日";
    case "SAT", "SUN" -> "周末";
    default -> "未知";
};
```

**辨析｜易错点：传统 `switch` 的「落穿（fall-through）」。** 忘了写 `break`，匹配成功后还会继续执行下一个 `case` 的代码——这是经典 bug 源。箭头语法（`->`）从设计上消灭了落穿，**新代码一律用箭头语法**。

**`switch` vs `if-else`**：`switch` 适合「一个变量对多个离散值的精确匹配」；`if-else` 适合「范围、复合布尔条件」。范围判断（`>= 90`）写不成 `switch`，精确匹配用 `switch` 更清晰。

## 2 循环：while、do-while 与 for

**`while` 循环**：先判断条件，为真才执行——可能一次都不执行。

```java
int i = 0;
while (i < 10) {
    System.out.println(i);
    i++;          // 别忘了推进条件！否则死循环
}
```

**`do-while` 循环**：先执行一次，再判断条件——**至少执行一次**。适合「必须做一次，然后问是否继续」的场景：

```java
int answer;
do {
    answer = askUser();          // 至少问一次
} while (!isValid(answer));      // 无效就再问
```

**`for` 循环**：把「初始化、条件、更新」收进一对括号，紧凑且不易漏更新：

```java
for (int i = 0; i < 10; i++) {
    System.out.println(i);
}
```

**for-each 循环**：遍历数组或集合的增强形式，没有下标、没有越界：

```java
int[] scores = {85, 92, 78};
int sum = 0;
for (int s : scores) {
    sum += s;                    // 遍历所有元素，无需下标
}
```

**重点结论：能 for-each 就别手写下标。** for-each 去掉了下标、边界检查与迭代器样板——它读起来就是「对每个元素做什么」。只有在需要下标（改元素、双数组对齐、倒序遍历）时才回落到传统 `for`。

**死循环的三种写法等价**：`while (true)`、`for (;;)`、`do { ... } while (true)`。死循环本身不是错——服务器的主循环就是死循环，关键是循环体内要有**退出路径**（`break` 或条件推进）。

## 3 跳转：break、continue 与带标签

**`break`**：跳出**当前**循环（或 `switch`）。**`continue`**：跳过本次迭代的剩余部分，直接进入下一次。

```java
for (int i = 0; i < 10; i++) {
    if (i == 3) continue;      // i==3 时跳过打印，继续下一轮
    if (i == 7) break;         // i==7 时整个循环终止
    System.out.println(i);     // 打印 0 1 2 4 5 6
}
```

**带标签的跳转（labeled break）**：`break` 默认只跳出一层循环。要跳出**多层嵌套循环**，给外层循环贴标签：

```java
outer:
for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
        if (i * j == 6) break outer;   // 直接跳出两层
    }
}
```

**辨析｜易错点：`break` 只跳一层。** 嵌套循环里想一次跳出所有层，不带标签的 `break` 只会结束内层循环，外层继续——结果程序「行为诡异但不报错」。要么用带标签的 `break`，要么用一个「是否已找到」的布尔标志，二者选一，别混。

**`continue` 也有带标签版本**：`continue outer` 跳过外层循环的当前迭代，直接进入外层的下一次——用法与 `break outer` 对称。

**公式解析：循环次数的「刚好」账**

循环最容易错的是「多算一次或少算一次」。把一个循环的迭代次数写成公式，对照着数就不容易错：

$$
\text{迭代次数} = \frac{\text{结束值} - \text{起始值}}{\text{步长}} + 1 \quad (\text{条件为 } i < \text{结束值}，\text{步长} > 0)
$$

比如 `for (int i = 0; i < 10; i++)`：起始 0、结束 10、步长 1，次数 $= (10 - 0)/1 + 1 - 1 = 10$。<span class="marginnote">更稳的直觉：`i < n` 的循环恰执行 $n$ 次（从 0 到 $n-1$）。把「循环从 0 数到 $n-1$ 共 $n$ 次」这个事实钉死，配合<strong>左闭右开</strong>区间的心智，就能预判任何循环的迭代次数——这也是后面算法分析里「$O(n)$」直观含义的来源。</span>

## 4 核心对比表：三种循环与两种跳转

纯概念主题用**核心对比表**替代公式解析的展开，把「选哪个」一次说清：

| 维度 | `while` | `do-while` | `for` | `for-each` |
| --- | --- | --- | --- | --- |
| 先判断 | 是 | 否（先做一次） | 是 | 是 |
| 最少执行 | 0 次 | **1 次** | 0 次 | 0 次 |
| 需要下标 | 自己管 | 自己管 | 自带 | 无下标 |
| 适用场景 | 条件驱动 | 必须做一次 | 计数驱动 | 遍历集合/数组 |

| 跳转 | 含义 | 典型用途 |
| --- | --- | --- |
| `break` | 结束当前循环/switch | 找到就退出 |
| `continue` | 跳过本次迭代 | 过滤某些元素 |
| `break outer` | 结束带标签的多层循环 | 在嵌套循环里提前收工 |

**重点结论：循环的选择有默认答案——遍历选 for-each，计数选 `for`，条件驱动选 `while`，至少一次选 `do-while`。** 别把 `while` 硬写成 `for` 的下标形式，也别为了「少一行」把逻辑压缩到不可读。可读性是流程控制的第一优先级，因为这里的 bug 最隐蔽：编译器从不帮你查「循环少算一遍」。

## 5 小结

- 分支用 `if-else`（范围/复合条件）与 `switch`（离散精确匹配）；新代码用箭头语法，避免落穿。
- 循环四兄弟：`while`（0+ 次）、`do-while`（1+ 次）、`for`（计数）、`for-each`（遍历）。
- 条件必须是布尔表达式：`if (x)` 在 Java 里编译错误，须写 `if (x != 0)`。
- `break` 只跳一层；多层退出用带标签的 `break outer`；`continue` 跳过本轮迭代。
- 迭代次数按「起始、结束、步长」记账，养成「从 0 到 n-1 共 n 次」的直觉。

在下一节，我们将把「一坨数据」装进变量与数组，并学会处理文本——**数组与字符串处理**。
