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