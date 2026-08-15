---
title: 万维网（WWW）概述：URL 与 Web 体系
date: 2026-08-07
---

# 万维网（WWW）概述：URL 与 Web 体系

<div class="epigraph">
<p>万维网不是互联网本身，而是互联网上最绚丽的应用——一张由超链接织成的、覆盖全球的「信息之网」。</p>
<footer>—— 蒂姆·伯纳斯-李（Tim Berners-Lee），万维网发明人</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§6.4 ｜ 2026-08-07</p>
</div>

## 为什么从万维网开始

万维网（WWW，World Wide Web）可能是你使用互联网的「全部体验」——浏览器、网页、链接、搜索。它不是互联网本身（互联网是网络，Web 是跑在 HTTP 上的应用），而是互联网上最成功的一项应用。这一节立起 Web 的三根柱子：**URL（统一资源定位符）、HTTP（超文本传送协议）、HTML（超文本标记语言）。**<span class="marginnote">万维网的三要素（常考）：<strong>URL 负责「资源在哪」、HTTP 负责「怎么取」、HTML 负责「长什么样」</strong>。蒂姆·伯纳斯-李 1989 年在 CERN 提出 WWW，把「超文本 + 互联网」结合，从此人类知识被一张网连起来。你正在读的这篇文章，本质就是一个 HTML 文档，通过 HTTP 从某台服务器送到你眼前。</span>

这一节讲：**WWW 的组成、URL 的结构、Web 的工作方式。**

## 1 WWW 的三要素：URL、HTTP、HTML

**万维网（WWW）**：分布式、超媒体、按需连接的信息系统——它由三根柱子支撑：<span class="marginnote"><strong>URL</strong>`（Uniform Resource Locator，统一资源定位符）——给每个资源一个「全球唯一地址」；<strong>HTTP</strong>`（HyperText Transfer Protocol）——规定浏览器与服务器之间「怎么要、怎么给」；<strong>HTML</strong>（HyperText Markup Language）——描述网页内容与结构的标记语言。<strong>「URL 定位、HTTP 传输、HTML 呈现」</strong>三者配合，Web 才能转起来。</span>

| 要素 | 职责 | 类比 |
| --- | --- | --- |
| URL | 定位资源 | 资源的门牌号 |
| HTTP | 传输资源 | 取货的快递员 |
| HTML | 描述内容 | 货物的包装说明书 |

**辨析｜易错点：** **WWW 不等于互联网**——互联网是底层网络（TCP/IP），WWW 是上层应用（HTTP）。「互联网上除了 Web 还有邮件、FTP、P2P」。「上网」在中文里常被误用为「上 Web」，严格说「Web 是互联网的一个应用」。

## 2 URL 的结构：资源的「完整地址」

**URL（统一资源定位符）**：标识互联网上**任何一个资源**的字符串。它的一般格式：

$$
\text{URL} = \text{协议} + \text{主机} + \text{端口} + \text{路径} + \text{查询}
$$

以 `http://www.example.com:8080/path/page.html?id=1` 为例：<span class="marginnote">URL 的五段：<strong><code>http</code>`（协议）、<strong><code>www.example.com</code>`（主机名，可换成 IP）、<strong><code>8080</code>`（端口，默认 http 是 80、https 是 443）、<strong><code>/path/page.html</code>`（路径）、<strong><code>?id=1</code>（查询字符串）。URL 不是随便写的——它精确告诉客户端「用什么协议、去哪个主机、哪个端口、取哪个资源、带什么参数」。</strong></strong></strong></span>

**协议**：http、https、ftp 等，告诉客户端用什么应用协议。
**主机**：域名或 IP，定位服务器。
**端口**：可选，默认 80（http）或 443（https）。
**路径**：服务器上的资源位置。
**查询**：可选，以 `?` 开头，传给服务器的参数。

**辨析｜易错点：** URL 与 URI、URN 的区别是经典辨析题：<span class="marginnote"><strong>URI</strong>`（统一资源标识符）是「标识符」的统称，<strong>URL 是 URI 的一种</strong>（能用「位置」定位的），<strong>URN</strong>`（统一资源名）是 URI 的另一种（用「名字」标识，如 ISBN 书号）。通俗说：<strong>URL 告诉你「去哪找」，URN 告诉你「它叫什么」</strong>。考试通常考「URL 是 URI 的子集」。 简单记：<strong>URL ⊂ URI</strong>，URL 是「能定位的 URI」。</span>

## 3 Web 的工作方式：一次浏览的旅程

你在浏览器输入网址后，发生了一连串事件：<span class="marginnote">一次网页浏览的完整链路：<strong>DNS 解析域名 → TCP 三次握手 → HTTP 请求 → 服务器响应 → 浏览器渲染</strong>。你上一章学的 TCP 握手、这一章学的 DNS，全部在这里「汇合」——<strong>Web 是前面所有协议的「总集成」</strong>。这也是为什么「网页打不开」时，可以从 DNS、TCP、HTTP 三个层面逐层排查。</span>

1. **DNS 解析**：浏览器把域名（如 `www.example.com`）解析成 IP。
2. **TCP 连接**：与服务器建立 TCP 连接（三次握手）。
3. **发送 HTTP 请求**：浏览器发送 HTTP GET 请求。
4. **服务器响应**：服务器返回 HTML 文档（含状态码 200）。
5. **浏览器渲染**：解析 HTML、加载图片等子资源、显示页面。

**辨析｜易错点：** 一个网页往往包含**很多个子资源**（图片、CSS、JS）——浏览器要**并发地**为每个子资源发起 HTTP 请求。**「一个网页 = 一个 HTML + 多个子资源请求」**是理解 Web 性能的关键（HTTP/2 的多路复用就是为此优化，见第 9 章）。**「浏览器是 HTTP 客户端」**——你的浏览器本质上是一台复杂的 HTTP 客户端软件。

## 4 超文本与超链接：Web 的「灵魂」

**超文本（hypertext）**：可以包含**指向其他文档的链接（超链接）**的文本。点击一个链接，就跳到另一个文档——正是这种「链接-跳转」机制，让无数独立文档织成「一张网」。<span class="marginnote">「超文本」的「超」字意为「超越」——超越单篇文本的线性阅读。伯纳斯-李的天才之处，是把「超链接」从实验室概念变成了全世界都可用的 Web 标准。<strong>超链接 = 把分散文档连成网的「胶水」</strong>。你此刻的阅读，就是沿着超链接在信息之网里漫游。</span>

**辨析｜易错点：** 超链接是 Web 的「图结构」：网页是「结点」，超链接是「边」。**搜索引擎（Google）正是利用这张「链接图」来评估网页重要性的**（PageRank 算法）——一个网页被越多的权威页面链接，它就越重要。**「Web 的链接结构 + 搜索算法」**是信息检索与 Web 技术交汇的经典议题（见高级《信息检索》）。

## 5 公式解析：URL 编码——把「非法字符」翻译成能上网的写法

URL 只能出现在 ASCII 码表里的**一小部分安全字符**（字母、数字与若干保留符号）。中文、空格、`&`、`#`、`/` 等在 URL 里都不能裸写，必须**编码（percent-encoding）**：

$$
\text{编码字符} = \text{百分号} + \text{该字节的两位十六进制}
$$

- **第一步，看 UTF-8 字节**：以「中文」为例，UTF-8 编码为 6 个字节，如「中」的字节是 `E4 B8 AD`。
- **第二步，逐字节转十六进制**：每个字节前加 `%`，得到 `%E4%B8%AD`。<span class="marginnote"><strong>URL 编码的作用域不止中文</strong>：空格编码为 <code>%20</code>（或 <code>+</code>）、<code>&</code> 编码为 <code>%26</code>、<code>#</code> 编码为 <code>%23</code>。解码时浏览器会把 <code>%E4%B8%AD</code> 还原成「中」。<strong>「查询字符串里想传什么，先问它安不安全」</strong>——凡是保留字与中文，一律编码。</span>
- **第三步，为什么必须编码**：因为 `?`、`&`、`#` 在 URL 里有「语法职责」——`?` 后接查询、`&` 分隔参数、`#` 指向片段。若参数值里出现这些字符而不编码，URL 会被「切错段」，服务端拿到的参数就错了。

**辨析｜易错点：** URL 编码与 **HTML 实体编码（`&amp;`）** 是两回事：前者在 URL 里转义字符，后者在 HTML 文档里转义字符。**「URL 编码在地址栏里生效，HTML 实体在网页源码里生效」**——两者不可混用。

## 6 HTML：Web 内容的「骨架」

**HTML（HyperText Markup Language）**：描述网页结构与内容的**标记语言**，由**标签（tag）** 组成：<span class="marginnote">一段最小 HTML 的灵魂：<strong><code>&lt;!DOCTYPE html&gt;</code></strong>`（声明 HTML5）、<strong><code>&lt;html&gt;</code></strong>`（根元素）、<strong><code>&lt;head&gt;</code></strong>`（元数据）、<strong><code>&lt;body&gt;</code></strong>`（可见内容）、<strong><code>&lt;a href="…"&gt;</code></strong>`（超链接）。<strong>「HTML 决定『有什么』，CSS 决定『长什么样』，JavaScript 决定『会做什么』」</strong>是前端三件套的分工。</span>

| 标签 | 作用 | 例子 |
| --- | --- | --- |
| `&lt;h1&gt;` ~ `&lt;h6&gt;` | 标题分级 | `&lt;h1&gt;标题&lt;/h1&gt;` |
| `&lt;p&gt;` | 段落 | `&lt;p&gt;正文&lt;/p&gt;` |
| `&lt;a href="url"&gt;` | 超链接 | `&lt;a href="…"&gt;点我&lt;/a&gt;` |
| `&lt;img src="…"&gt;` | 图片 | `&lt;img src="cat.jpg"&gt;` |

HTML 文档由 W3C/WHATWG 制定标准，浏览器负责把 HTML「渲染」成可视页面。**HTML 是静态的**（只描述内容），要动态交互还得靠 JavaScript——这是 Web 从「文档」走向「应用」的关键一步。

**辨析｜易错点：** **HTML 不是编程语言**（没有变量、循环、分支），它是**标记语言**；JavaScript 才是 Web 编程语言。**「HTML 是骨架、CSS 是皮肤、JS 是肌肉」**是记住三者职责的口诀。

## 7 从 Web 到全栈：一次请求背后的「全家桶」

一次看似简单的网页访问，其实调用了前面所有层的协议。把「五层模型」与「一次浏览」对应起来：<span class="marginnote"><strong>应用层</strong>`（HTTP/DNS）→ <strong>运输层</strong>`（TCP）→ <strong>网络层</strong>`（IP/路由器）→ <strong>数据链路层</strong>`（以太网帧/交换机）→ <strong>物理层</strong>`（光/电信号）。<strong>「你在浏览器里的每一次回车，都是五层协议的合奏」</strong>——这就是为什么学完 TCP/IP 你会更懂 Web。</span>

| 层 | 协议/机制 | 在这一步里干什么 |
| --- | --- | --- |
| 应用层 | HTTP、DNS | 解析域名、取回网页 |
| 运输层 | TCP | 可靠地分段传输（三次握手） |
| 网络层 | IP | 逐跳路由转发 |
| 数据链路层 | 以太网 | 在每段链路上封装成帧 |
| 物理层 | 双绞线/光纤 | 传比特 |

**辨析｜易错点：** 很多人学完五层模型却「用不上」。这里给你一个锚点：**每一次浏览器请求，都是从上到下一次「封装修建」、从下到上一次「剥壳还原」**——五层协议不是抽象说教，而是每次上网都在发生的真实流程。**「学完网络不看 Web，等于学完语法不写作文」**——Web 是 TCP/IP 的终极应用舞台。

## 8 小结

- **WWW 三要素**：URL（定位）、HTTP（传输）、HTML（呈现）。
- **WWW ≠ 互联网**：Web 是跑在互联网上最成功的应用。
- **URL 结构**：协议 + 主机 + 端口 + 路径 + 查询；URL ⊂ URI。
- **一次浏览旅程**：DNS 解析 → TCP 握手 → HTTP 请求 → 响应 → 渲染。
- **超文本/超链接**：链接把文档织成网；超链接图也是搜索引擎排名的依据。
- **Web 即总集成**：前面所有层（DNS、TCP、IP）都在一次网页浏览里汇合。

在下一节，我们将深入 Web 的传输语言——**超文本传送协议（HTTP）：报文结构与请求方法**。
