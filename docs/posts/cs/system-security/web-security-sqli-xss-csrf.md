---
title: Web 安全：SQL 注入、XSS 与 CSRF
date: 2026-08-11
---

# Web 安全：SQL 注入、XSS 与 CSRF

<div class="epigraph">
<p>浏览器是现代计算机的真相来源——因此，永远不要相信来自浏览器的任何字符串。</p>
<footer>—— 迈克尔 · 霍华德 与 戴维 · 勒布朗（Michael Howard & David LeBlanc），《编写安全的代码》</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 系统安全 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Web 安全开始

前面的攻击都在「内存」和「网络」，而现代应用最大的攻击面在**浏览器**。SQL 注入、XSS、CSRF 常年霸榜 OWASP Top 10，它们有一个共同的根因：**把「数据」当「代码」执行**——把用户输入拼进 SQL、把用户输入渲染成 HTML、把「用户的浏览器」当作「用户的意愿」来信任。<span class="marginnote">这三类漏洞的逐技术细节（注入语法、payload、变体）在密码学专题 sql-injection、xss、csrf 三篇分别展开，本处聚焦<strong>统一的根因视角与防御链条</strong>——理解了「数据当代码」这一病根，三兄弟其实是一家人。</span>Web 安全是「系统安全」在现代互联网形态下的主战场。

## 1 SQL 注入：把输入拼进查询

**SQL 注入（SQL Injection）**发生在「用户输入被直接拼进 SQL 语句」时。经典例子：

```sql
SELECT * FROM users WHERE username = 'admin' AND password = '$pass';
```

如果 `$pass` 被用户填成 `' OR '1'='1`，拼出来就是：

```sql
SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1';
```

`'1'='1'` 恒真，整条查询对任何账号都放行——**攻击者没猜密码就登录了**。注入还能做到：拼 `' UNION SELECT ...` 读取任意表、拼 `'; DROP TABLE ...` 破坏数据、写文件、读文件。<span class="marginnote">2008 年史上最大规模的 SQL 注入攻击（针对美国多家零售网站，窃取上亿张信用卡）就是利用网站对输入的不设防。SQL 注入直到今天仍是 OWASP 榜上常客，因为「拼字符串」是程序员最顺手的写法。</span>

**防御的根本：参数化查询（prepared statement）**——把 SQL 结构与数据分离：

```sql
-- 数据作为参数传入，数据库引擎负责转义
SELECT * FROM users WHERE username = ? AND password = ?;
```

数据永远是「数据」，永远不会被解析成 SQL 语法。**这一招根治 SQL 注入**，其余手段（过滤 `'`、WAF 规则）都只是「拼字符串」下的补丁。

## 2 XSS：把输入渲染成脚本

**跨站脚本（XSS，Cross-Site Scripting）**发生在「用户输入被直接渲染进 HTML」时。若评论区把用户输入原样插入页面：

```html
<div>评论内容：<script>stealCookies()</script></div>
```

攻击者输入的 `<script>` 就被浏览器**当作代码执行**，能偷 cookie、改页面、在受害者会话里发帖、键盘记录。

XSS 分三类：

**存储型（Stored）**：恶意脚本存进服务器，任何访客加载页面都中招——危害最大。
- **反射型（Reflected）**：脚本藏在 URL 参数里，受害者点击恶意链接即触发，不持久。
- **DOM 型（DOM-based）**：恶意数据只在浏览器端 JavaScript 里被处理，不经过服务器。

**防御：输出编码（output encoding）**——在 HTML、属性、URL、JS 上下文里分别做转义，让输入永远是「显示的文本」而不是「可执行的标记」；再加 **CSP（内容安全策略）**——用 HTTP 头声明「本页只允许执行哪些来源的脚本」，即使脚本被注入也跑不起来。<span class="marginnote">CSP 是现代浏览器对抗 XSS 的最强防线：`Content-Security-Policy: script-src 'self'` 直接宣告「只执行本站脚本」。<strong>「默认拒绝 + 显式白名单」的 CSP 把 XSS 的杀伤半径从「任意代码」压缩到「必须能找到合法脚本链」</strong>——见密码学专题 xss 一篇的 CSP 细节。</span>

## 3 CSRF：把受害者的浏览器当枪

**跨站请求伪造（CSRF，Cross-Site Request Forgery）**不攻击接口本身，而是**借受害者的会话做坏事**。攻击者在自己的网站放一张恶意图片：

```html
<img src="http://bank.com/transfer?to=attacker&amount=99999" />
```

受害者若**刚好登录着** bank.com，浏览器加载图片时自动带上 bank.com 的 cookie，转账请求就**以受害者的身份**发出去了——服务器看到的是「登录用户本人发起的合法请求」。

**关键前提**：① 受害者在目标站点有有效会话（cookie）；② 目标请求是「自动携带凭证 + 无校验」的。

**防御三板斧：**

- **CSRF Token**：页面里放一个随机、与会话绑定的 token，请求必须携带——攻击者无法预知 token，伪造请求缺少它即被拒。
- **SameSite Cookie**：`Set-Cookie: ...; SameSite=Lax` 让跨站请求**默认不带 cookie**——从根上断了「自动携带凭证」。
- **校验来源**：检查 `Origin`/`Referer` 头是否为本站。

**辨析｜易错点：** CSRF 与 XSS 方向相反：XSS 是「攻击者在页面里注入脚本」，CSRF 是「攻击者借浏览器发请求」；XSS 能偷数据，CSRF 只能「以受害者身份执行动作」。但两者也常合体——**XSS 可以绕过 CSRF 防御**（脚本能读 token），所以 CSRF 防御必须与 XSS 防御配合。

## 4 公式解析：把三类漏洞统一为「信任边界」

三兄弟可以放进同一张表：

$$
\underbrace{\text{用户输入}}_{\text{不可信数据}}
\xrightarrow{\ \text{如果被当作}\ } \underbrace{\text{SQL 语法 / HTML 标记 / 用户意愿}}_{\text{可执行代码 / 权威身份}}
$$

- **第一步，找公共根因**：SQL 注入、XSS、CSRF 全都源于**信任了不该信任的边界**。SQL 注入把数据当 SQL；XSS 把数据当 HTML；CSRF 把「浏览器自动带的凭证」当「用户的实时意愿」。
- **第二步，分防御原则**：SQL 用**参数化**（结构分离）；XSS 用**输出编码 + CSP**（数据即文本）；CSRF 用 **token + SameSite**（凭证不能自动且无条件使用）。
- **第三步，看统一解法**：三层各自对症下药，但总纲只有一条——**对每一条「数据→代码 / 数据→身份」的转化，要么在源头划清边界，要么在输出点做转义**。这个「最小信任」原则与系统安全全局的「最小权限」是同一哲学。
- **第四步，看纵深**：单靠其中一招都会漏——参数化查询能挡 SQL 注入，但挡不住 XSS；XSS 有了，CSRF 又有新路。**Web 应用必须四层同守：参数化、编码、CSP、token**。

## 5 Web 安全的整体图景

三兄弟之外，Web 安全还有更多战场，但它们遵循同一套思维：

- **认证与授权缺陷**：会话管理不当、水平越权（访问他人资源 ID）、垂直越权（普通用户调用管理员接口）。
- **反序列化漏洞**：把不可信数据 `unserialize`/`pickle`，数据变代码——与内存侧「数据当代码」完全同构。
- **SSRF / 路径穿越**：让服务器访问攻击者指定的内网资源/路径——信任「输入给到的地址/路径」。
- **供应链与依赖**：npm/pip 依赖里的恶意包——信任「名字」而不是「内容」。

**重点：** Web 安全没有一个「万能补丁」，但有一条万能纪律：**把每一个数据来源标成「不可信」，在每一次「数据变代码、变地址、变身份」的边界上做校验**。OWASP ASVS 与 Top 10 就是这条纪律的工程清单。

## 6 小结

- **SQL 注入**：输入拼进 SQL → 参数化查询根治（结构分离）。
- **XSS**：输入渲染成 HTML → 输出编码 + CSP 根治（数据即文本）。
- **CSRF**：浏览器自动携带凭证 → CSRF Token + SameSite Cookie 根治（凭证不可自动滥用）。
- 三类漏洞共享根因：**把不可信数据当作可执行代码或权威身份**。
- 纵深四层：**参数化、编码、CSP、token**——单招皆可破，合围方周全。

在下一节，我们把安全配置落到**每一台操作系统**上：从账户与补丁到 SELinux 与审计——**操作系统安全加固**如何让一台机器「默认拒绝、最小暴露」。
