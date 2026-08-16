---
title: Web 安全（XSS/CSRF/CSP、供应链攻击）
date: 2026-08-07
---

# Web 安全（XSS/CSRF/CSP、供应链攻击）

<div class="epigraph">
<p>安全不是某一次加固，而是每个请求、每段输出、每次信任都要回答的问题。</p>
<footer>—— OWASP 基金会（Open Worldwide Application Security Project）</footer>
</div>

<div class="article-byline">
<p>第三级 · Web 前端与全栈工程（浏览器渲染/框架/工程化） ｜ OWASP 指南 + Flanagan《JavaScript 权威指南》第7版 第15章（同源策略） ｜ 2026-08-07</p>
</div>

## 为什么从 Web 安全开始

前面的所有能力——渲染、事件、框架、状态、构建、SSR——都建立在一个前提上：**这段代码是可信的**。可当代码里混进了攻击者注入的脚本、当浏览器替你自动带上 Cookie 去发请求、当 npm 包名被人「仿冒」……一切性能优化都失去了意义。Web 安全不是「加个 WAF 就完事」，它是一套从**信任边界**推导出来的规则。

本节把这些规则的**原理**讲透——理解了同源策略、XSS、CSRF、CSP 为什么存在，你才能在任何框架、任何工具里都守住底线。

## 1 信任的边界：同源策略与 CORS

浏览器安全模型的基石是**同源策略（Same-Origin Policy，SOP）**：**一个源的脚本只能访问同一源（协议 + 域名 + 端口三者一致）的 DOM、Cookie 与网络资源**。`https://a.com` 的页面读不到 `https://b.com` 的 `document`，也改不了它的数据——这是浏览器给「跨源隔离」画的第一个圈。

但现代 Web 必然需要跨源通信（你的前端要调另一个域的 API）。于是浏览器提供了**CORS（跨源资源共享，Cross-Origin Resource Sharing）** 这个**受控的例外**：服务器用响应头 `Access-Control-Allow-Origin: https://a.com` 明确声明「我允许这个源读取我」。浏览器不信任来源，它只信任**服务器声明的允许名单**。<span class="marginnote">CORS 最容易被误解的一点：<strong>它防的是「浏览器代发」的读取，而不是「攻击者服务器直接发」</strong>。攻击者的服务器照样能打你的 API——CORS 只决定「浏览器要不要把响应交给页面 JS」。真正的防线是服务端的认证与授权，CORS 只是浏览器侧的护栏。</span>

CORS 还会触发**预检（preflight）**：带 `Authorization` 头、自定义头、非简单方法（`PUT`/`DELETE`）的请求，浏览器先发一个 `OPTIONS` 请求问服务器「允许吗」，得到允许才发真正的请求。配置错 CORS（如 `Access-Control-Allow-Origin: *` + 需要凭据）是安全漏洞，配置错成「不该允许的允许了」更是直接开洞。

同源策略之上还有一层传输层信任：**HTTPS/TLS**。它同时保证**机密性**（内容加密）、**完整性**（防篡改）、**身份验证**（防伪冒网站）。现代浏览器对非 HTTPS 页面的能力持续收窄（`Secure` Cookie 只在 HTTPS 生效、`window` 敏感 API 被限制），「无 HTTPS 不出网」已近乎硬性规定。HTTPS 与《HTTP 演进》的 HTTP/2、HTTP/3 也是绑定的：**H2 必须 TLS，HTTP/3 内嵌 TLS 1.3**——安全与性能在这里是同一件事。

## 2 XSS：让别人的脚本替你执行

**XSS（跨站脚本，Cross-Site Scripting）** 是 Web 上最经典也最常见的漏洞：**攻击者把脚本注入到你的页面里，让它在用户的浏览器、你的源上执行**。注入的脚本和你的脚本拥有同等权限——能读 Cookie、能发请求、能改页面。

三大类型：

- **反射型（Reflected XSS）**：恶意脚本通过 URL 参数「反射」回响应。例：搜索框把 `q=<script>...</script>` 原样回显，受害者点开构造好的链接就中招。
- **存储型（Stored XSS）**：脚本被**存进服务器**（评论区、用户名），任何浏览该页面的用户都会执行。危害最大，一次注入、万人受害。
- **DOM 型（DOM-based XSS）**：不经过服务器，直接由前端 JS 把不可信输入（`location.hash`、`postMessage` 数据）写进 `innerHTML` 时触发。

防御分三层，缺一不可：

1. **输出转义（encoding）**：把不可信内容当作文本而非 HTML 输出——`&lt;` 转成 `&lt;`，脚本就成了无害字符。现代框架默认转义，这也是「别用 `v-html`/`dangerouslySetInnerHTML`」的原因。
2. **消毒（sanitization）**：确实需要富文本时，用 DOMPurify 之类的库剥离脚本标签，只保留白名单标签。
3. **纵深防御（CSP）**：即使注入成功，也让它执行不了——见下节。

<span class="marginnote">XSS 的本质是「<strong>数据与代码的边界</strong>被打破了」：用户输入本是数据，却被当成了代码执行。所以审计时只需问一句：「这段输入从哪里来，会被当做什么来用？」凡是「不可信输入 → HTML/URL/JS 上下文」，就要转义或消毒。这套「不可信数据即代码」的思维，与《前端智能化》里「模型输出当指令执行」的危险是同构的。</span>

## 3 CSRF 与 Cookie 安全

**CSRF（跨站请求伪造，Cross-Site Request Forgery）**：攻击者让受害者的浏览器**自动携带受害者自己的凭据**去发请求。经典场景：你登录了银行 A（Cookie 已种下），此时打开攻击者的站点 B，B 里的一个 `<img src="https://bank-a.com/transfer?to=attacker&amount=1000">` 就会让浏览器带上 A 的 Cookie 发出转账请求——**浏览器无法区分这个请求是不是你本人「点击」的**。

防御三件套：

- **CSRF Token**：服务器给每个表单下发一次性随机 token，请求时必须带上。攻击者无法预知 token，伪造请求就会失败。
- **SameSite Cookie 属性**：`SameSite=Lax`（默认）让「跨站、非顶层导航」的请求不携带 Cookie——把「自动带凭据」这个漏洞源头直接堵上。这是现代浏览器默认开启的第一道防线。
- **自定义请求头 + 服务端校验**：攻击者跨站发请求无法带自定义头，服务端校验「这个头在不在」即可区分。

Cookie 本身还有三个标志位要记牢：**`HttpOnly`**（JS 读不到，XSS 偷不走，但引出了「如何记住登录态」——交给 `httpOnly` Cookie 而非 `localStorage`）、**`Secure`**（只走 HTTPS）、**`SameSite`**（防 CSRF）。**永远不要把敏感 token 存 `localStorage`**——它没有 HttpOnly，任何 XSS 都能读。

## 4 CSP：给脚本画一个围栏

**CSP（内容安全策略，Content-Security-Policy）** 是一个响应头，它声明「页面允许加载哪些来源的资源」：

```
Content-Security-Policy: default-src 'self'; script-src 'self' https://cdn.example.com
```

这条头意思是：默认只允许同源资源，脚本只允许从本站与 `cdn.example.com` 加载。**没有显式允许的内联脚本、`eval()`、外站脚本全部被拦**——就算 XSS 注入成功，脚本也执行不了。CSP 是「纵深防御」的最后一堵墙：**让「代码执行」本身变得需要许可**。

CSP 有两件配套武器：

- **Nonce（一次性随机数）**：`script-src 'self' 'nonce-abc123'`，只有带这个 nonce 的 `<script>` 标签才允许执行——适合「必须用内联脚本」的场景。
- **Hash**：对单个内联脚本做哈希并列入白名单，脚本内容变了哈希就对不上。

CSP 的代价是**严格**：开启后，任何漏配的资源都会被拦，反馈是「功能突然挂了」而非「报错提示」——所以上线要先用 `Content-Security-Policy-Report-Only` 观察违规报告，再正式启用。它是把「默认允许」翻转为「默认拒绝」的范式转换，代价高，收益也最高。

## 5 供应链攻击：最隐蔽的信任危机

现代前端一半以上代码来自 npm。当攻击者不攻击你的代码，而是攻击你**依赖的依赖**，你就成了「供应链攻击」的受害者。真实案例触目惊心：`event-stream` 包被植入窃取比特币私钥的后门（2018）、`ua-parser-js` 等三个流行包被投毒（2021）、`es5-ext` 被利用维护者邮箱发起攻击（2025）。攻击手法包括：

- **仿冒（typosquatting）**：注册与热门包仅一字之差的包名（`lodash` vs `lodahs`），靠手误骗人安装。
- **依赖混淆（dependency confusion）**：利用私有包名与公共 registry 的优先级差，在公开源上传同名恶意包。
- **劫持维护者**：社工/钓鱼拿到维护者账号，直接向合法包注入恶意代码。

防御不是「别用 npm」，而是**把供应链当作攻击面管理**：

- **锁文件（lockfile）**：锁定每个依赖的精确版本与哈希，`npm ci` 按锁文件安装，拒绝漂移。
- **`overrides`/`resolutions`**：显式覆盖有漏洞的传递依赖版本。
- **审计与扫描**：`npm audit`、GitHub Dependabot、`osv-scanner`（基于 OSV 漏洞库）持续监控。
- **最小依赖原则**：能少用一个包就少用一个；越少的依赖 = 越小的攻击面。
- **私有 registry / 包签名**：企业级场景用私有仓库并校验发布者身份。

<span class="marginnote">供应链安全还牵扯到一个组织问题：<strong>前端依赖的「一半代码来自 npm」意味着你的安全预算一半花在别人的代码上</strong>。SBOM（软件物料清单）把「我们依赖了谁」变成可审计的清单，是合规与应急响应的共同基础——它和《构建工程化》里「依赖图」是同一种东西的两种用途：性能看它找热点，安全看它找风险。</span>

## 6 核心对比表：三大漏洞速查

| 漏洞 | 攻击方式 | 危害 | 首要防御 |
| --- | --- | --- | --- |
| XSS（反射/存储/DOM） | 注入脚本在受害者源执行 | 偷 Cookie、劫持会话 | 输出转义 + 消毒 + CSP |
| CSRF | 伪造请求自动带受害者凭据 | 以受害者身份执行操作 | SameSite + CSRF Token |
| 供应链攻击 | 投毒依赖/仿冒包名 | 后门、数据窃取 | 锁文件 + 审计 + 最小依赖 |

## 7 小结

- **同源策略**是浏览器信任的基石，**CORS** 是受控例外；CORS 防浏览器代读，防不了攻击者直发。
- **XSS** 是「数据被当代码执行」：转义、消毒、CSP 三层防线缺一不可。
- **CSRF** 是「浏览器自动带凭据」：`SameSite`、CSRF Token、自定义头组合防御。
- Cookie 三标志（`HttpOnly`/`Secure`/`SameSite`）与「别存 `localStorage`」是前端铁律。
- **CSP** 把「默认允许」翻转为「默认拒绝」，是最强的纵深防御，也最需要上线演练。
- **供应链攻击**把攻击面延伸到依赖图：锁文件、审计、最小依赖、SBOM 是日常纪律。
- 安全不是一次配置，而是一种**贯穿开发的思维方式**：每个输入都问「可信吗」，每个输出都问「会执行吗」。
- HTTPS/TLS 是传输层底线，与 HTTP/2、HTTP/3 绑定——安全与性能在此合流。
- 所有防线都指向同一原则：**默认拒绝，显式允许**。

在下一节，我们转向「如何让 Web 应用跑在更多地方」——**跨端技术**：PWA、WebAssembly 与小程序容器，以及它们如何在 Web 与原生之间重新划界。

> 补充：XSS/CSRF/CSP 的定义与防御以 OWASP 指南为准；同源策略与 CORS 以 MDN 与 Flanagan《JavaScript 权威指南》第 7 版第 15 章为准；供应链攻击案例（event-stream 2018、ua-parser-js 2021、es5-ext 2025）为公开报道事实，详见 OSV/Snyk 事件时间线。
