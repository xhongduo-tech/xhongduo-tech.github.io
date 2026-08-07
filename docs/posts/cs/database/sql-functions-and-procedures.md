---
title: 函数与过程：PL/SQL 风格的存储过程
date: 2026-08-07
---

# 函数与过程：PL/SQL 风格的存储过程

<div class="epigraph">
<p>计算机科学中的所有问题，都可以通过增加一层间接层来解决。</p>
<footer>—— 大卫·惠勒（David Wheeler）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第5章 §5.1 ｜ 2026-08-07</p>
</div>

## 为什么从函数与过程开始

到目前为止，SQL 扮演的是一个「声明式查询语言」：你说**要什么**（SELECT），数据库负责**怎么算**（执行计划）。但现实世界里的数据库不只有查询——还有**业务逻辑**：发工资时要按规则扣税、选课时要检查先修与容量、下单时要锁库存。这些逻辑放在应用代码里，每次都要把数据搬出数据库再搬回来；放在数据库里，却能靠近数据、批量执行、被多个应用共享。

数据库把这部分能力称为**函数（function）与过程（procedure）**。它们把「一段可复用的、可能带参数与局部变量的程序」写进数据库——这就是常说的**存储过程（stored procedure）**，而最著名的方言是 Oracle 的 **PL/SQL**。这一节讲清楚：函数与过程怎么定义、参数怎么进出、控制流怎么写、以及「函数 vs 过程」的分界在哪里。

## 1 SQL 函数：标量函数与表函数

**SQL 函数（function）**：一段被命名的、可带参数、必定**返回一个值**的数据库程序。最简单的是标量函数，返回单个值。比如「统计某个系有多少教师」：

```sql
CREATE FUNCTION dept_count(dept_name VARCHAR(20))
RETURNS INTEGER
BEGIN
    DECLARE d_count INTEGER;
    SELECT COUNT(*) INTO d_count
    FROM instructor
    WHERE instructor.dept_name = dept_name;
    RETURN d_count;
END;
```

定义之后，函数可以出现在普通查询里，像内置函数一样使用：

```sql
SELECT dept_name, dept_count(dept_name)
FROM department;
```

这样「统计系人数」的逻辑被写了一次，任何查询都能复用。<span class="marginnote">标量函数返回单个值，可以在 SELECT 列表、WHERE 条件里到处用；注意参数名 `dept_name` 与列名同名时，函数体内要小心遮蔽——这是 SQL 函数最容易踩的坑之一。</span>

SQL 标准还支持**表函数（table function）**：返回的是一整张表（关系），可以直接放进 FROM：

```sql
CREATE FUNCTION instructor_of(dept VARCHAR(20))
RETURNS TABLE (ID VARCHAR(5), name VARCHAR(20))
...
```

表函数把「一段查询」封装成「一张可查的虚表」，比视图更进一步——它可以带参数。

## 2 SQL 过程：CREATE PROCEDURE 与参数进出

**过程（procedure）**与函数的关键差别：**过程不一定返回单值，它通过参数与外部的状态交互**，常用于「执行一段修改数据的操作」。定义用 `CREATE PROCEDURE`：

```sql
CREATE PROCEDURE dept_count_proc(IN dept VARCHAR(20),
                                 OUT count INTEGER)
BEGIN
    SELECT COUNT(*) INTO count
    FROM instructor
    WHERE instructor.dept_name = dept;
END;
```

参数有三种模式：

| 模式 | 含义 | 用途 |
| --- | --- | --- |
| `IN` | 输入参数，只读 | 把值传进过程 |
| `OUT` | 输出参数，只写 | 把结果带回调用方 |
| `INOUT` | 既可读也可写 | 双向传递 |

调用过程用 `CALL`：

```sql
CALL dept_count_proc('Music', @c);
SELECT @c;  -- 读出输出参数
```

**辨析｜易错点：** 函数与过程的混淆是经典入门错误。记忆口诀：**函数有返回值、过程靠参数**。函数被调用在「表达式里」（SELECT / WHERE），过程被调用在「语句里」（CALL）。对函数执行 `CALL`、或把过程塞进 SELECT 列表，都会报错——它们在语言里是两种不同的可调用体。<span class="marginnote">这对应着编程语言里「纯函数 vs 语句 / 副作用」的区分：函数偏向「算一个值」，过程偏向「做一件事」。第三级《程序设计语言》会从类型系统与副作用的角度再深挖一次。</span>

## 3 PL/SQL 风格：控制流与异常

过程体内是一段**过程式程序**，支持变量、条件、循环与异常处理——这就是「SQL 语言里长出编程语言」的地方：

```sql
CREATE PROCEDURE update_salary(IN amount NUMERIC(8,2))
BEGIN
    DECLARE total NUMERIC(8,2);
    IF amount < 0 THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'salary cannot be negative';
    ELSE
        UPDATE instructor
        SET salary = salary + amount;
    END IF;
END;
```

常用的控制结构：

- **条件**：`IF ... THEN ... ELSE ... END IF;` 与 `CASE ... WHEN ...`。
- **循环**：`LOOP ... END LOOP`、`WHILE condition DO ... END WHILE`、`FOR ... IN ... DO`。
- **异常**：`DECLARE EXIT HANDLER FOR ...` 捕获错误，`SIGNAL SQLSTATE` 主动抛出错误。<span class="marginnote">`SQLSTATE` 是五字符状态码：`'45000'` 是用户自定义异常的约定值。抛出与捕获让过程能「把错误留在数据库里处理」，而不是粗暴地打断整个事务。</span>

这些结构让过程能做「查一步、算一步、写一步」的复合操作，而这一切发生在数据库进程内、离数据最近的地方——省去了往返传输，也便于把一组操作包成一个事务。

**辨析｜易错点：** 过程体内的 SQL 与外部交互要小心**上下文**：过程里默认跑在「一个事务」里，`COMMIT` / `ROLLBACK` 往往由调用方统一控制。把 COMMIT 写进过程内部、又在外层包事务，会出现「事务边界漂移」——一半在事务里、一半不在，出错时难以回滚。**存储过程不是忘掉事务的理由，恰恰更要注意事务边界。**

## 4 外部语言函数：把 Python 请进来

纯 SQL 写循环并不舒服。现代数据库（PostgreSQL、SQLite、MySQL 等）允许用**外部语言**写函数——最常见的是 PL/Python、PL/Java、PL/pgSQL 与 JavaScript：

```sql
CREATE FUNCTION add_one(x INTEGER)
RETURNS INTEGER
LANGUAGE plpython3u
AS $$
    return x + 1
$$;
```

外部函数把「数据库擅长的事」（集合查询）与「编程语言擅长的事」（复杂算法、字符串处理、科学计算）结合起来。<span class="marginnote">代价是<strong>安全与隔离</strong>：外部函数运行在数据库进程内或近旁，写不好可能造成内存泄漏、卡死连接甚至越权访问。生产系统对外部函数要像对待第三方依赖一样：最小权限、限资源、可审计。</span>

**辨析｜易错点：** 外部函数不是银弹。每次调用有**跨语言边界**的开销，复杂计算未必比纯 SQL 快；而且外部函数里的任何异常都可能让整个语句失败。**工程建议：集合逻辑用 SQL，单条复杂计算再考虑外部语言，性能问题先测再优化。**

## 5 公式解析：阶乘函数的递归实现

函数最漂亮的一面是它可以**递归**——过程语言里写递归，与我们在第2章《关系代数》、第4章《递归视图》里见到的递归是同一种思想。用 PL/SQL 风格实现阶乘：

$$
n! =
\begin{cases}
1, & n = 0\\
n \cdot (n-1)!, & n > 0
\end{cases}
$$

```sql
CREATE FUNCTION fact(n INTEGER)
RETURNS INTEGER
BEGIN
    IF n = 0 THEN
        RETURN 1;
    ELSE
        RETURN n * fact(n - 1);
    END IF;
END;
```

逐项拆解这条递归：

- **基例（递归出口）**：`IF n = 0 THEN RETURN 1`——当 $n=0$ 时直接给答案，不再调用自己。没有这一行，递归会无限下潜。
- **递归步**：`RETURN n * fact(n-1)`——把「算 $n!$」化小为「算 $(n-1)!$ 再乘 $n$」，问题规模每次减一。
- **调用栈**：`fact(3)` 触发 `fact(2)`，再触发 `fact(1)`、`fact(0)`；`fact(0)` 返回 1 后，依次回填：$1 \to 1 \to 2 \to 6$。<span class="marginnote">递归依赖运行时的调用栈保存每一层状态。层级太深会栈溢出——迭代版本（WHILE 累乘）没有这个风险，这也是为什么生产代码里很多「递归」最终写成迭代。</span>
- **与查询递归的对照**：这里每层只调一次自己（线性递归），而 `WITH RECURSIVE` 里每层迭代把结果集合扩大一轮——两者共享「基例 + 递推」的骨架，只是数据从「单个值」变成了「整个关系」。

理解了函数与过程的语法与语义，我们就有了「把逻辑住进数据库」的全部积木。而下一节将引入一个更「自动」的机制：**触发器**——不用你调用，事件一发生，数据库自己就去执行。

## 6 小结

- **函数（function）**：可带参数、返回一个值的数据库程序，标量函数用于表达式，表函数用于 FROM。
- **过程（procedure）**：靠 `IN` / `OUT` / `INOUT` 参数与外界交互，用 `CALL` 调用，适合执行修改操作。
- **函数 vs 过程**：函数有返回值、出现在表达式里；过程靠参数、出现在语句里。
- **PL/SQL 风格**：过程体内支持 `IF` / `LOOP` / `WHILE` / 异常处理，SQL 语言里长出过程式程序。
- **外部语言函数**：用 Python / Java 等写数据库函数，扩展能力强但要注意安全与隔离。
- **递归函数**：基例 + 递归步；递归简洁但占用调用栈，迭代版本在深度大时更稳妥。

在下一节，我们将进入数据库的「自动反应」机制——**触发器**：定义、事件与语义。
