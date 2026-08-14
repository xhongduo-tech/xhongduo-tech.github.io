---
title: 前端安全基础
date: 2026-08-07
---

# 前端安全基础

<div class="epigraph">
<p>在 Web 上，没有所谓「只影响前端的漏洞」——前端是攻击者进入系统的第一道门。</p>
<footer>—— 特伦特 · 福西斯（Trenton Ivey）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 安全 ｜ 2026-08-07</p>
</div>

## 为什么从安全开始

页面做完了、优化好了——但**攻击者也在看你的页面**。Web 安全的前端防线是每个前端工程师的必修课：**XSS**（脚本注入）、**CSRF**（跨站请求伪造）、**点击劫持**，三大经典威胁都直接作用于浏览器。第21篇的 `innerHTML`、第24篇的 CORS、第25篇的存储红线，都在这一节汇总成完整的安全心智。

安全的第一原则是**默认不信（zero trust）**：所有来自用户的输入、来自网络的数据，都不可信。前端安全的本质，就是「如何给不可信数据划定边界」。这一节建立威胁模型——**谁会攻击你、怎么攻、如何防**——而不只是背几个「不要用 innerHTML」的清单。<span class="marginnote">OWASP（开放 Web 应用安全项目）的 Top 10 是 Web 安全的权威基线，XSS 与「失效访问控制」常年榜上有名。前端的职责是「第一道防线」，后端才是「最后防线」——两层都要做，任何一层失守都出事故。</span>

## 1 XSS：脚本注入

**XSS（Cross-Site Scripting，跨站脚本）** 是前端第一大威胁：攻击者把**可执行的脚本**注入到你的页面里，让它在**受害者的浏览器**里运行——偷 Cookie、改页面、窃取数据。

**XSS 的三种形态**：

| 类型 | 注入点 | 例子 |
| --- | --- | --- |
| 存储型（stored） | 数据存在服务器，所有人中招 | 评论区存了 `<script>`，每人都执行 |
| 反射型（reflected） | 请求参数拼进页面，一次性 | `?q=<script>` 反射进搜索页 |
| DOM 型（DOM-based） | 纯前端，不经过服务器 | `location.hash` 直接进 `innerHTML` |

**核心原理**：用户输入被当成 HTML/JS **执行**了，而不是当作「文本」显示。看这条公式（下节详解）：

```js
// 危险：把用户输入当 HTML 解析
el.innerHTML = `<p>${userComment}</p>`;
// 用户输入：</p><script>偷数据()</script><p>
// 结果：脚本被执行了！
```

**怎么防——三道防线**：

1. **转义/纯文本**：默认把用户内容当**文本**——`textContent` 而非 `innerHTML`；框架自动转义（React 的 `{expr}` 默认转义）。
2. **净化（sanitize）**：确实需要富文本时，用净化库（DOMPurify）白名单过滤标签属性，**绝不手写正则过滤**。
3. **CSP（内容安全策略）**：给浏览器下「只允许哪些脚本」的规则（见第3节）——即使注入成功也执行不了。

**辨析｜易错点：** `innerHTML` 不是唯一入口——`document.write`、`eval`、`setAttribute("href", 用户输入)`（`javascript:` 协议）、`onclick` 属性拼接，都是注入面。**规则：任何「把不可信字符串拼进可执行上下文」的地方都是 XSS 入口**。<span class="marginnote">「可执行上下文」的直觉：HTML 标签属性、`javascript:` URL、事件属性、脚本内容，这些地方「字符串即代码」。前端框架（React/Vue）的默认转义覆盖了大部分渲染路径，但 `dangerouslySetInnerHTML`、`v-html` 这类「逃生舱」一旦用错就直通 XSS——用之前先问：能不能不拼 HTML？</span>

## 2 公式解析：XSS 的注入模型

XSS 能成立，本质上是一个**「边界混淆」公式**——数据流本该是「文本」，却被解析成「代码」：

$$
\text{render}(\text{input}) = \begin{cases} \text{text} & \text{若转义正确} \\ \text{code} & \text{若 input 含 } \texttt{<script>} \text{ 且未转义} \end{cases}
$$

**逐步拆解：**

- **第一步，数据流进入**：用户输入（评论、搜索词、URL 参数）进入渲染路径——此时它只是「字符串」。
- **第二步，渲染通道选择**：`textContent` 通道把它当**纯文本**（特殊字符显示为字面量）；`innerHTML` 通道把它当 **HTML 源码**解析。
- **第三步，攻击载荷生效**：走 `innerHTML` 且输入含 `<script>alert(1)</script>` → 浏览器**解析并执行** → 攻击代码以受害者身份运行。
- **第四步，防线生效**：走 `textContent`（或框架转义）→ `<` 变成 `&lt;` 显示为文本 → 脚本只是「字」，永远不执行。

**代入一个实例（评论区）：** 攻击者发布 `你好 <img src=x onerror=偷Cookie()>` → 存储型 XSS 入库 → 每个用户浏览评论时，`img` 加载失败触发 `onerror` → 偷 Cookie 脚本执行。若渲染用 `textContent`，这串输入只显示为「你好 <img…>」的普通文字。

**直觉是什么？** XSS 的根因是「**数据与代码混淆**」——同一个字符串，你以为是数据，解析器当成了代码。防御的本质是**强制澄清边界**：要么把数据放进「文本」通道（转义），要么白名单净化后只放安全的子集（sanitize），要么用 CSP 让「即使当代码也跑不了」。<span class="marginnote">「数据与代码混淆」是安全领域的大母题：SQL 注入是「数据被当成 SQL」，命令注入是「数据被当成 shell 命令」，XSS 是「数据被当成 HTML/JS」——防御套路同源：<strong>参数化/转义/白名单</strong>，永远不信任字符串拼接。</span>

## 3 纵深防御：CSP 与 HttpOnly

单点防御可能漏，所以安全讲究**纵深防御（defense in depth）**——每层防线独立，一层被破还有下一层。

**CSP（Content Security Policy）**：响应头告诉浏览器「本页面允许加载/执行什么」，从源头掐断注入：

```http
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'
```

`script-src 'self'`：脚本只允许来自本站——外部域名脚本、内联脚本、`eval` 全被拒绝。
即使 XSS 注入成功，`<script>` 也无法执行（不是本站源）。

**严格 CSP 的代价**：内联脚本/样式默认被禁——现代构建产物通常是外部文件，正好兼容；但用 `eval`、`new Function` 的库会被挡，需要调整。CSP 是「一次配置、长期受益」的高性价比防线。

**HttpOnly Cookie**（第25篇的红线）：Cookie 加 `HttpOnly` 后，JS 读不到——XSS 偷 Cookie 的经典路径被堵死：

```http
Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Lax
```

**三个属性一条链**：

- **`HttpOnly`**：JS 不可读 → 防 XSS 窃取。
- **`Secure`**：仅 HTTPS 发送 → 防明文拦截。
- **`SameSite=Lax`**：跨站请求不带 Cookie → 防 CSRF（见下节）。<span class="marginnote">纵深防御的思维：XSS 靠「转义」防，CSP 是「即使转义漏了也执行不了」，HttpOnly 是「即使执行了也偷不到 Cookie」——三层各自独立失效面。安全没有银弹，只有「多道闸门」。</span>

## 4 CSRF：跨站请求伪造

**CSRF（Cross-Site Request Forgery，跨站请求伪造）**：攻击者在**别的网站**构造请求，**借你的身份**操作你的账号。

**攻击流程**：

1. 你已登录银行网站 `bank.com`（Cookie 在浏览器里）。
2. 你访问恶意网站 `evil.com`，它藏着一个表单/图片请求：`<img src="https://bank.com/api/transfer?to=attacker&amount=1000">`。
3. 浏览器发请求时**自动带上 bank.com 的 Cookie** → 银行以为是「你本人」操作 → 转账成功。

**关键漏洞**：Cookie 是「浏览器自动携带」的——请求是否合法，服务器单看 Cookie 分不清「你主动点的」还是「恶意站替你点的」。

**怎么防——三道主流防线**：

1. **`SameSite=Lax` Cookie**（最简单）：跨站请求（含 img/表单 POST）不携带 Cookie——攻击者「无凭据可用」，CSRF 直接失效。现代浏览器默认 `Lax`。
2. **CSRF Token**：表单里放一个服务器生成的随机 token，提交时校验——攻击者拿不到 token（跨域读不到）。
3. **校验自定义头**：要求请求带 `X-Requested-With` 之类自定义头——跨站发请求带不了自定义头（需 CORS 预检）。

**辨析｜易错点：** CSRF 与 XSS 的定位差别——**XSS 攻击你的「页面」（在页面里执行脚本），CSRF 攻击你的「身份」（借你的凭据发请求）**。防 CSRF 的核心是「让服务器能区分『用户本人主动的请求』与『别处伪造的请求』」——SameSite、token、自定义头都是这个目的。<span class="marginnote">同源策略（第24篇）是 CSRF 的天然缓冲：恶意网站<strong>读不到</strong>银行的响应（跨域拦截），只能「发」不能「读」。所以 CSRF 擅长「搞破坏」（改设置、转账），不擅长「偷数据」（读不到结果）。这解释了为什么「转账金额」这类写操作是 CSRF 的重灾区。</span>

## 5 点击劫持与依赖安全

**点击劫持（clickjacking）**：攻击者把目标网站**嵌进透明 iframe**，盖在诱饵按钮上——你点的「抽奖」实际点中了底下页面的「转账」。

```
你看得见： [ ✨ 点我抽奖 ✨ ]   ← 诱饵层（透明 iframe 里的目标页）
你看不见： [ 确认转账 1000 ]   ← 目标页真实按钮
```

**防御**：让目标页面**不允许被 iframe 嵌入**——两个响应头二选一：

```http
X-Frame-Options: DENY
# 或更强：
Content-Security-Policy: frame-ancestors 'none'
```

`frame-ancestors` 是 CSP 的现代写法，可精确指定「只允许哪些站点嵌入」。

**依赖安全（supply chain）**：你 import 的第三方包，本身可能被植入恶意代码——供应链攻击是近年最热的攻击面。防线：

用 `npm audit` / GitHub Dependabot 扫描依赖漏洞。
锁版本（lockfile）防「依赖漂移」。
- 只信任活跃维护的包。

**HTTPS 与混合内容**：页面是 HTTPS，却加载了 HTTP 资源——**混合内容（mixed content）**。浏览器会拦截「活动的」混合内容（脚本、iframe）；被动内容（图片）也建议升级。**全程 HTTPS 是安全底线**——Cookie 的 `Secure`、CSP、所有请求都建立在它之上。<span class="marginnote">供应链攻击的著名案例：`event-stream` 包被恶意提交（2018）、`ua-parser-js` 被投毒（2021）——几行恶意代码通过 npm 分发到数十万项目。依赖最小化（少装包）、审计常跑、版本锁死，是成本最低的供应链防线。</span>

## 6 核心对比表：三大前端威胁总览

| 威胁 | 攻击什么 | 手段 | 前端防线 |
| --- | --- | --- | --- |
| XSS | 你的页面/用户数据 | 注入脚本执行 | 转义、净化、CSP、HttpOnly |
| CSRF | 用户的身份/写操作 | 借 Cookie 伪造请求 | SameSite、CSRF Token |
| 点击劫持 | 用户的点击 | 透明 iframe 覆盖 | `X-Frame-Options` / `frame-ancestors` |

**综合防线的「最小安全清单」**（前端工程建议逐条打勾）：

1. 所有用户内容走**文本渲染**，富文本用净化库。
2. 上线启用 **CSP**。
3. 登录 Cookie 全部 `HttpOnly; Secure; SameSite=Lax`。
4. 全站 **HTTPS**，无混合内容。
5. 敏感数据**不存 localStorage**（第25篇红线）。
6. 页面设 **frame-ancestors** 防点击劫持。
7. 依赖**定期审计**、版本锁死。
8. 写操作接口配 **CSRF Token**（若无法依赖 SameSite）。

**辨析｜易错点：** 前端安全最大的误区是「有 HTTPS 就安全」「有框架就安全」——HTTPS 只保传输，防不了 XSS/CSRF；框架转义覆盖了渲染，但 `v-html`/`dangerouslySetInnerHTML` 一用就破功。**安全是「默认不信」的纪律**，不是某个开关。<span class="marginnote">安全是「能力越大责任越大」的领域：越熟悉前端，越要敬畏「不可信输入」。OWASP Top 10 每年更新，MDN 的安全章节常读常新——把「安全自查清单」放进发布流程，比出了事再补救便宜一个量级。</span>

## 7 小结

- **XSS** 是「数据被当代码执行」：存储/反射/DOM 三型；防线 = 转义（textContent）+ 净化（DOMPurify）+ CSP。
- **CSP** 用响应头声明「允许加载什么」——注入即使成功也执行不了，是纵深防御的核心层。
- **HttpOnly + Secure + SameSite** 三属性管住 Cookie：防 XSS 窃取、防明文、防 CSRF。
- **CSRF** 是「借身份发请求」：防线 = `SameSite=Lax`、CSRF Token、自定义头校验。
- **点击劫持**用透明 iframe 骗点击，防 `X-Frame-Options`/`frame-ancestors`；依赖安全靠审计与锁版本。
- 安全心智：**默认不信一切输入**；前端是第一道防线，后端是最后防线，两层都要做。

在下一节，我们收束全专题——**Web 标准、浏览器兼容性与特性检测**。从规范到浏览器实现、从特性检测到 polyfill，这是把「会写」变成「写得出能在真实世界运行」的最后一课。
