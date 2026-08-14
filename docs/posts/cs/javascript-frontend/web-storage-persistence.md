---
title: Web Storage 与本地持久化
date: 2026-08-07
---

# Web Storage 与本地持久化

<div class="epigraph">
<p>刷新页面就丢数据，是浏览器留给互联网最深刻的「失忆」；本地存储把记忆还给网页。</p>
<footer>—— 史蒂夫 · 萨德斯（Steve Souders）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 存储 ｜ 2026-08-07</p>
</div>

## 为什么从本地存储开始

页面一刷新，JS 的内存数据就全部归零——这是单页应用最大的「失忆」问题。**本地持久化**让浏览器把数据留在用户设备上，刷新、关页、甚至隔天再开都还在。前端最常用的三种存储：**`localStorage`**（长期）、**`sessionStorage`**（会话级）、**`IndexedDB`**（大数据）。

「数据存哪」是前端工程的基本决策之一：主题偏好、登录态、购物车、草稿、离线数据……各有适合的存储。而存储背后还牵着**安全**这条线——`localStorage` 里的数据一旦被 XSS 拿到就全量泄露（第29篇），所以「什么能存、什么不能存」是本节红线。<span class="marginnote">「Web Storage」是 `localStorage`/`sessionStorage` 的官方统称（WHATWG 规范）。它比 Cookie 大得多、不随每个请求自动发送（省流量、更安全），是现代前端本地状态的默认存储。</span>

## 1 localStorage：同步、永久的键值库

**`localStorage`** 是最常用的本地存储：**同步读写、字符串键值对、无过期时间**。

```js
// 写入
localStorage.setItem("theme", "dark");
// 读取（没有则返回 null）
const theme = localStorage.getItem("theme");
// 删除
localStorage.removeItem("theme");
// 清空
localStorage.clear();
// 键的数量
localStorage.length;
```

三条关键事实：

1. **值只能是字符串**——存对象必须 `JSON.stringify`，读出来再 `JSON.parse`。
2. **同步**：读写是阻塞的，但数据量小（约 5MB）时无感；**别用 localStorage 存大文件**。
3. **永久**：不手动清就一直在（除非用户清浏览器数据）——这是与 `sessionStorage` 的语义分界。

**存对象的完整姿势**（封装成工具函数更稳）：

```js
const KEY = "user_prefs";
const prefs = { theme: "dark", fontSize: 16 };

localStorage.setItem(KEY, JSON.stringify(prefs));     // 序列化
const loaded = JSON.parse(localStorage.getItem(KEY)); // 反序列化
```

**辨析｜易错点：** 反序列化要防「解析失败」——存的数据可能被用户清空、被旧版本写过、甚至被篡改：

```js
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;          // JSON 坏了就回退默认值，别让整站崩
  }
}
```

「读取即崩溃」是 localStorage 应用最常见的健壮性问题——**任何来自存储的数据都不可信，都要 try/catch 兜底**。<span class="marginnote">`localStorage` 的键值<strong>按源（origin）隔离</strong>：`https://a.com` 存的数据，`https://b.com` 读不到——这是浏览器的安全沙箱。所以同源下换子路径（`/shop` 与 `/blog`）是共享的，换域名/端口则完全隔离。</span>

## 2 sessionStorage：标签页级的会话记忆

**`sessionStorage`** 与 `localStorage` 的 API 完全一样，区别只在**生命周期**：

**按标签页隔离**：每个标签页一份独立存储，新开标签页不共享。
**会话结束即清**：关闭标签页/窗口就清空；刷新页面**不清空**（还在同一会话里）。

```js
sessionStorage.setItem("draft", draftContent);   // 草稿：防刷新丢失
```

**典型场景**：

- **表单草稿**：用户输到一半刷新，草稿还在（但新开标签页不共享——正好符合「每页独立草稿」）。
- **一次性向导**：多步流程的中间状态，会话结束即清理。
- **页面内临时标记**：同会话内跨页传递，不污染长期存储。

**辨析｜易错点：** `sessionStorage` 的「会话」边界——刷新保留、关标签页清除、新开标签页不继承。浏览器「恢复上次会话」功能会把 `sessionStorage` 一起恢复，所以「关掉就一定没了」并不绝对。设计上把它当「短命记忆」即可。

**Storage 的通用事件**：`storage` 事件在**其他标签页**修改 localStorage 时触发（当前页不触发），可用于「多标签页同步」：

```js
window.addEventListener("storage", (e) => {
  if (e.key === "theme") applyTheme(e.newValue);   // 另一标签页改了主题，这边跟着变
});
```

`e` 携带 `key`、`oldValue`、`newValue`——这是跨标签页通信的免费通道。<span class="marginnote">`storage` 事件只在<strong>其他</strong>同源标签页触发，且只在「值真正变化」时触发——这天然避免了「自己改自己监听」的循环。做「多标签同步登录态」「主题实时联动」时它是首选。</span>

## 3 Cookie：与 Storage 的取舍

**Cookie** 是更古老的存储，但**定位完全不同**——它主要是「随请求自动发送的凭据」：

| 维度 | Cookie | localStorage | sessionStorage |
| --- | --- | --- | --- |
| 容量 | ~4KB | ~5MB | ~5MB |
| 生命周期 | 可设过期时间 | 永久 | 会话结束 |
| 随请求发送 | **是**（每请求都带） | 否 | 否 |
| 作用域 | 可按路径/域名 | 按源 | 按标签页 |
| JS 访问 | 默认可（HttpOnly 则不可） | 可 | 可 |
| 场景 | 会话凭据、服务端需要的数据 | 客户端偏好、缓存 | 短命草稿 |

**关键决策**：

**服务端需要读的数据**（登录态、会话 id）→ Cookie（随请求自动带，`HttpOnly` 防 JS 读）。
**纯客户端的数据**（主题、设置、本地缓存）→ Web Storage（不占请求带宽）。
- **敏感数据**（密码、令牌）→ **都不存前端**，或存内存/HttpOnly Cookie——`localStorage` 的 XSS 泄露风险让敏感数据必须远离它。

**Cookie 的安全属性**（第29篇详细展开）：`HttpOnly`（JS 读不到，防 XSS 窃取）、`Secure`（仅 HTTPS）、`SameSite`（防 CSRF）。**凡是给 Cookie 的，都要把这三个属性想一遍**。<span class="marginnote">为什么 2000 年代「什么都塞 Cookie」行不通了？Cookie 每请求都发送，5KB 的 Cookie × 每页 30 个请求 = 上百 KB 冗余流量，拖慢加载（第28篇性能）。Storage 诞生正是为了把「客户端自己的数据」从请求路径里挪出去。</span>

## 4 IndexedDB：浏览器里的数据库

数据超过几 MB、需要结构化查询、需要异步操作——`localStorage` 力不从心，这时候用 **IndexedDB**：浏览器内置的**事务型对象数据库**。

它比 Storage 复杂得多，但核心模型值得了解：

**对象仓库（object store）**：类似表，存「对象」而非字符串。
**索引（index）**：按字段建立查询通道，支持范围查询。
- **异步 API**：不阻塞主线程，可存大体积数据（数百 MB）。
- **事务**：一组操作要么全成要么全回滚。

原生 API 繁琐，实践中多用封装（如 `idb` 库）或更上层的 **Cache API**（Service Worker 用）：

```js
// 概念示意（idb 库风格）
await db.put("users", { id: 1, name: "小明" });
const user = await db.get("users", 1);
```

**IndexedDB 的典型场景**：离线应用的数据层（PWA）、超大缓存（地图瓦片）、富文本草稿、文件 Blob 存储。**判断标准**：数据 > 5MB、需查询、要异步 → IndexedDB；小键值 → Storage。<span class="marginnote">IndexedDB 是「客户端存储的终点站」：Storage 是「字符串抽屉」，IndexedDB 是「对象数据库」。它的事件驱动 API 回调地狱严重，现代做法是包一层 Promise 封装（idb 库几百行就搞定）。PWA 离线缓存的主数据层就是它。</span>

## 5 安全红线：什么能存，什么不能存

本地存储最大的坑**不在技术，而在安全**。把「能不能存」当成存储决策的第一问：

**❌ 绝对不能存**：

密码、密钥、令牌——`localStorage` 一旦被 XSS 读取（第29篇），攻击者能 `localStorage.getItem("token")` **直接带走全部凭据**。
敏感个人信息——同样的 XSS 泄露路径。

**✅ 可以存**：

- 主题偏好、语言设置、用户非敏感偏好。
- 公开数据的缓存（可重新拉取）。
- 会话凭据的替代品——**前提**是放 HttpOnly Cookie（JS 读不到，XSS 偷不走）。

**为什么 HttpOnly Cookie 比 localStorage 存 token 安全？** XSS 注入的脚本能执行 JS——能读 `localStorage`、但**读不到 HttpOnly Cookie**（浏览器层面拒绝 JS 访问）。攻击者拿到的是「能执行的 JS」，`localStorage` 对它门户大开，HttpOnly Cookie 则锁着门。<span class="marginnote">这是前端安全最重要的一句话：<strong>XSS 能读 localStorage，读不到 HttpOnly Cookie</strong>。所以「登录 token 放哪」的答案是「HttpOnly Cookie 或内存」，不是 localStorage。市面上「localStorage 存 JWT」的教程是公认反模式。</span>

## 6 公式解析：存储读写与序列化

本地存储的一切操作，最终都归结为「**字符串进、字符串出**」——序列化是它的枢纽公式：

$$
\text{store}(v) = \text{String}(\text{serialize}(v)), \qquad \text{load}(k) = \text{parse}(\text{storage}.getItem(k))
$$