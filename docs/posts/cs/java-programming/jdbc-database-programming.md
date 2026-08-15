---
title: JDBC 数据库编程
date: 2026-08-07
---

# JDBC 数据库编程

<div class="epigraph">
<p>应用程序与数据库之间，JDBC 是那座标准化的桥——一次学习，连接任何数据库。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第2卷第5章 ｜ 2026-08-07</p>
</div>

## 为什么从 JDBC 开始

一个现代后端系统，数据几乎都存在**关系型数据库**（MySQL、PostgreSQL、Oracle）里。Java 程序要读写数据库，靠的是 **JDBC（Java Database Connectivity）**——Java 官方定义的「数据库访问标准接口」。JDBC 的意义在于**统一**：你写的代码面向 JDBC 接口，底层驱动（`mysql-connector`、`postgresql` 驱动）各不相同，但 API 一致——**换数据库只需换驱动与连接串，业务代码一行不改**。这一篇讲透 JDBC 的四步流程、`PreparedStatement` 防注入的原理、事务控制，并预览连接池与 Spring 的封装——它们是 JDBC 在生产环境的真正形态。

## 1 JDBC 的四步流程

JDBC 的使用模式高度固定，四步走：

**第一步，加载驱动并连接**。驱动负责把 JDBC 调用翻译成数据库协议：

```java
String url = "jdbc:mysql://localhost:3306/shop?useSSL=false&serverTimezone=UTC";
String user = "root", password = "secret";
try (Connection conn = DriverManager.getConnection(url, user, password)) {
    // 连接已建立
}
```

`url` 是 JDBC 连接串：`jdbc:<数据库厂商>://<主机>:<端口>/<库名>?<参数>`。`DriverManager` 根据 `url` 自动选择合适的驱动。**连接是重量级资源**，用 try-with-resources 确保关闭——否则连接泄漏会让数据库「连接数打满」而拒绝服务。

**第二步，创建语句（Statement）**：

```java
try (Statement stmt = conn.createStatement()) { ... }
```

**第三步，执行并拿结果**。查询用 `executeQuery`，返回 **`ResultSet`**（结果集，一个游标）：

```java
try (Statement stmt = conn.createStatement();
     ResultSet rs = stmt.executeQuery("SELECT id, name FROM employee")) {
    while (rs.next()) {                  // 游标逐行移动
        int id = rs.getInt("id");        // 按列名取值
        String name = rs.getString("name");
        System.out.println(id + ": " + name);
    }
}
```

`ResultSet` 是一个**游标**：初始指向第一行之前，每次 `next()` 移到下一行，返回 `false` 表示没有更多行了。<span class="marginnote">按列取值的方法家族：`getInt`、`getString`、`getDouble`、`getDate` 等，既可按<strong>列名</strong>（可读性好）也可按<strong>列下标</strong>（快一点，从 1 开始）。`next()` 循环是 JDBC 的「for-each」——它和你用 `iterator` 遍历集合是同一个心智模型。</span>

**第四步，处理 `SQLException`**。JDBC 的所有操作都可能抛 `SQLException`（受检异常）——连接失败、SQL 语法错、约束冲突都会冒出来。它必须被 try-catch 或 `throws` 处理。

**更新数据**用 `executeUpdate`（返回受影响的行数）：

```java
int rows = stmt.executeUpdate("UPDATE employee SET salary = 10000 WHERE id = 1");
```

## 2 PreparedStatement：预编译与防注入

把参数拼进 SQL 字符串是**危险的**。考虑登录查询：

```java
// 反例：字符串拼接 SQL
String sql = "SELECT * FROM users WHERE name = '" + input + "'";
```

如果 `input` 是 `' OR '1'='1`，拼出来的 SQL 变成：

$$

\text{SELECT * FROM users WHERE name = '' OR '1'='1'}

$$

这个 SQL 的 `WHERE name = '' OR '1'='1'`——`'1'='1'` 恒为真，于是**整张 users 表被无条件选中**。攻击者只需输入 `' OR '1'='1`，登录校验就形同虚设；换成 `'; DROP TABLE users; --` 甚至能**删库**。这就是著名的 **SQL 注入（SQL injection）**。

**正解：用 `PreparedStatement` 占位符，把参数与 SQL 结构分开**：

```java
String sql = "SELECT * FROM users WHERE name = ? AND password = ?";
try (PreparedStatement ps = conn.prepareStatement(sql)) {
    ps.setString(1, input);       // 参数通过 setter 绑定，绝不拼进 SQL 字符串
    ps.setString(2, password);
    try (ResultSet rs = ps.executeQuery()) {
        // 无论 input 是什么，都只是「一个字符串值」，不是 SQL 代码
    }
}
```

**为什么安全**：`?` 占位符让数据库把「SQL 结构」与「参数值」**分开解析**——输入里的引号、分号都只是**数据**，不会成为 SQL 语法的一部分。**PreparedStatement 同时是「防注入」与「预编译性能优化」的合体**：同一 SQL 重复执行时，数据库只解析一次、复用执行计划，批量操作快得多。

**公式解析：PreparedStatement 如何拆开「结构」与「数据」**

SQL 注入的本质是「数据被当成了语法」。PreparedStatement 的防御是把两者在协议层分开：

$$
\text{拼接：} \quad \text{SQL 字符串} \oplus \text{用户输入} \to \text{输入可变成语法}
$$

$$
\text{预编译：} \quad \text{SQL 结构（占位符固定）} \;\|\; \text{参数（按类型绑定）} \to \text{输入永远是数据}
$$

数据库先编译 `SELECT ... WHERE name = ?` 这个固定结构，再按类型把参数值「作为值」填入——无论参数里藏什么引号，都不会被重新解释成 SQL 语法。

## 3 事务：让一批操作「全成或全败」

**事务（transaction）**把多条 SQL 绑成一个原子单元：**要么全部成功提交，要么全部回滚**——绝不允许「转账扣了钱、收款没到账」的中间状态。JDBC 的默认行为是**每条 SQL 自动提交**；要事务必须手动关闭自动提交：

```java
try (Connection conn = DriverManager.getConnection(url, user, pwd)) {
    conn.setAutoCommit(false);              // 关闭自动提交，进入事务模式
    try {
        stmt.executeUpdate("UPDATE account SET balance = balance - 100 WHERE id = 1");
        stmt.executeUpdate("UPDATE account SET balance = balance + 100 WHERE id = 2");
        conn.commit();                       // 两条都成功，一起提交
    } catch (SQLException e) {
        conn.rollback();                     // 任何一条失败，全部回滚
        throw e;
    } finally {
        conn.setAutoCommit(true);            // 恢复默认
    }
}
```

**事务的四个性质 ACID**：

| 性质 | 含义 |
| --- | --- |
| **原子性（Atomicity）** | 全成或全败 |
| **一致性（Consistency）** | 数据库从合法状态到合法状态（约束恒成立） |
| **隔离性（Isolation）** | 未提交的修改对其他事务不可见 |
| **持久性（Durability）** | 提交后修改永久生效 |

**隔离级别**（从低到高）：读未提交 → 读已提交 → 可重复读 → 串行化。级别越高越安全、并发越低。**「读已提交」**是多数数据库默认（MySQL 默认可重复读）。隔离级别取舍属于数据库课程，见第三级《数据库系统》。

## 4 连接池与生产形态

**为什么需要连接池**：建立数据库连接是**重量级操作**（TCP 握手 + 认证 + 分配资源），每次请求都建/拆连接，系统很快被拖垮。**连接池（connection pool）**预建一批连接、请求时借用、用完归还——与线程池同一思想。

生产环境几乎不用裸 JDBC，而用 **HikariCP 等连接池 + Spring JDBC / MyBatis 等框架**封装——但它们**底层都是 JDBC**：连接池管理 `Connection`，框架帮你写 `PreparedStatement` 与结果映射。**理解 JDBC 四步流程，是读懂任何数据访问框架的前提。**

**重点结论：JDBC 是数据访问的「底层协议」，框架是它的「工程化封装」。** 学好 JDBC 的四步、PreparedStatement 与事务，你就拿到了「无论什么框架都通用」的地基。<span class="marginnote">MyBatis 的 `#{...}` 占位符、JPA 的 `@Query`，最终都编译成 PreparedStatement；HikariCP 只是替你管理 `Connection` 的获取与归还。读懂 JDBC，框架对你就不再是「黑盒」——报错、调优、换库时都能对症下药。</span>

## 5 小结

- JDBC 四步：**连接 → 建 Statement → 执行拿 ResultSet → 处理 SQLException**。
- **永远用 `PreparedStatement`**：占位符防 SQL 注入，兼得预编译性能。
- **事务**：`setAutoCommit(false)` + 成功 `commit` + 失败 `rollback`；ACID 四性质。
- `ResultSet` 是游标，`next()` 逐行移动；连接是重量级资源，务必 try-with-resources 关闭。
- 生产用连接池（HikariCP）+ 框架封装，底层都是 JDBC。

在下一节，我们处理「时间这个又麻烦又重要的话题」——**日期时间 API 与国际化**。