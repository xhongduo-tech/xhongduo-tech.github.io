---
title: 简单邮件传送协议（SMTP）
date: 2026-08-07
---

# 简单邮件传送协议（SMTP）

<div class="epigraph">
<p>SMTP 是邮局的「分拣机器」：它不问邮件内容是什么，只管把信封从 A 邮局搬到 B 邮局。</p>
<footer>—— 网络教材中的通俗说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§6.6 ｜ 2026-08-07</p>
</div>

## 为什么从 SMTP 开始

上一节说 SMTP 是电子邮件「全程负责送」的协议。这一节深入它：**SMTP 是怎么把一封邮件从发件人送到收件人的？它有哪些命令？它有什么局限？**<span class="marginnote">SMTP（Simple Mail Transfer Protocol，简单邮件传送协议）诞生于 1982 年，是互联网邮件传输的标准协议。它的特点是<strong>「简单」</strong>：基于 TCP、文本命令、一问一答。<strong>它的职责是「推」（push）——把邮件从发送方推到接收方的服务器</strong>，与「拉」（pull）的 POP3/IMAP 正好相反。</span>

这一节讲：**SMTP 的通信过程、核心命令、以及它为什么需要 MIME 帮忙。**

## 1 SMTP 的通信过程：三次「对话」

SMTP 基于 **TCP 25 端口**，客户端与服务器之间的通信靠**文本命令 + 数字应答**一问一答。发送一封邮件的过程：

1. **建立连接**：客户端连服务器的 25 端口，服务器回 `220`（服务就绪）。
2. **打招呼**：客户端发 `EHLO`（或 `HELO`），服务器回 `250`。
3. **声明发件人**：`MAIL FROM` → 服务器回 `250`。
4. **声明收件人**：`RCPT TO` → 服务器回 `250`（可多个）。
5. **传输内容**：`DATA` → 服务器回 `354`，客户端发送邮件内容，以「单独一行的 `.`」结束 → 服务器回 `250`。
6. **退出**：`QUIT` → 服务器回 `221`，关闭连接。<span class="marginnote">SMTP 的命令流像「对话剧本」：<strong><code>HELO</code>`（你好）→ <code>MAIL FROM</code>`（我是谁）→ <code>RCPT TO</code>`（给谁）→ <code>DATA</code>`（邮件内容）→ <code>QUIT</code>（再见）。每一步都有明确的数字应答。这套「命令 + 应答」的文本协议风格，是互联网早期协议的标准长相。</strong></span>

**辨析｜易错点：** **SMTP 用命令动词（MAIL FROM、RCPT TO）而非 HTTP 的方法**——它是「邮件专用」的协议。而且 **SMTP 应答与 HTTP 状态码类似**（都是三位数字、首位表类别），但含义不同（220=就绪、250=OK、354=开始输入、221=再见）。**「SMTP 的 250 相当于 HTTP 的 200」**是跨协议类比记忆的捷径。

## 2 SMTP 的三个核心特点

SMTP 的设计有三个关键特点：<span class="marginnote"><strong>① 推模式（push）</strong>——发送方主动把邮件推给接收方服务器，不是接收方来取；<strong>② 纯文本</strong>——SMTP 只能传 7 位 ASCII 文本，不能直接传二进制附件；<strong>③ 服务器之间也用 SMTP</strong>——SMTP 既在客户端↔服务器之间用，也在服务器↔服务器之间中继用。<strong>「SMTP 是全程推送的文本协议」</strong>是三句话的总结。</span>

**推（push）**：主动把邮件送到接收方服务器。
**文本限制**：只支持 7 位 ASCII，不能传二进制（图片、附件）。
**全程使用**：客户端到服务器、服务器到服务器都用 SMTP。

**辨析｜易错点：** SMTP 的「文本限制」是它的历史遗留——1980 年代只有 ASCII。这带来一个连锁后果：**传二进制附件必须靠 MIME（见下）把二进制编码成文本**。**「SMTP 只能传文本，附件靠 MIME 编码」**是邮件技术里的经典链条。

## 3 MIME：让 SMTP 能传附件

**MIME（Multipurpose Internet Mail Extensions，多用途互联网邮件扩展）**：不是替代 SMTP，而是**对 SMTP 的扩展**——把非 ASCII 内容（图片、音频、二进制附件）**编码成 7 位 ASCII 文本**，让 SMTP 能传输。<span class="marginnote">MIME 的核心机制：<strong>Base64 编码</strong>——把二进制数据每 3 个字节编码成 4 个 ASCII 字符（64 字符表），使任意二进制都能变成「安全文本」；再加 <strong><code>Content-Type</code> 首部告诉接收方「这是什么类型」（text/plain、image/jpeg、application/pdf）。<strong>「MIME 让 SMTP 的『只能传文本』变成『什么都能传，只是都编码成文本』」</strong>。</strong></span>

| MIME 机制 | 作用 |
| --- | --- |
| Content-Type | 声明内容的类型（文本、图片、附件） |
| Content-Transfer-Encoding | 编码方式（Base64、quoted-printable） |
| 多部分（multipart） | 一封邮件携带多个部分（正文 + 附件） |

**辨析｜易错点：** MIME **不是独立协议**，它是「邮件内容的标注与编码规范」——SMTP 仍是传输协议，MIME 定义了「传输的内容长什么样」。**「SMTP 是车，MIME 是货的包装规范」**。Web 的 `Content-Type` 首部也是从 MIME 继承来的（HTTP 首部里的 MIME 类型）。

## 4 SMTP vs HTTP：两个文本协议的对比

SMTP 与 HTTP 都是「基于 TCP 的文本协议」，但差异明显：<span class="marginnote">对比要点：<strong>端口</strong>（25 vs 80）、<strong>方向</strong>（SMTP 是推、HTTP 是拉）、<strong>连接数</strong>（SMTP 只推、HTTP 一问一答）、<strong>状态</strong>（都无状态）、<strong>编码</strong>（SMTP 只文本、HTTP 可任意二进制）。最本质的区别是「<strong>推 vs 拉</strong>」：<strong>SMTP 把邮件推到服务器，HTTP 把网页拉给客户端</strong>。</span>

| 对比维度 | SMTP | HTTP |
| --- | --- | --- |
| 端口 | 25 | 80/443 |
| 数据方向 | 推（push） | 拉（pull） |
| 传输内容 | 邮件（文本） | 任意资源 |
| 命令风格 | MAIL FROM/RCPT TO/DATA | GET/POST |
| 使用场景 | 邮件传输 | Web 浏览 |

**辨析｜易错点：** **「推 vs 拉」是最本质的对比**：HTTP 是客户端「拉」资源（请求-响应），SMTP 是发送方「推」邮件（主动投递）。**「推拉之别」**解释了为什么 HTTP 用请求-响应而 SMTP 用「命令-应答」——推送方主动发起一系列命令。

## 5 一封邮件从发到收的完整旅程

把前面几节串起来，看一封邮件如何穿越多个服务器：<span class="marginnote">完整链路：<strong>用户代理（UA）→ 发送方邮件服务器（SMTP）→ 中继服务器（SMTP）→ 接收方邮件服务器（SMTP）→ 用户代理（POP3/IMAP）</strong>。<strong>「发送与中继靠 SMTP，收取靠 POP3/IMAP」</strong>——SMTP 管前半程的「送」，读取协议管后半程的「取」。</span>

1. **发件人 UA**：写信 → 点「发送」。
2. **发送方服务器**：UA 用 SMTP 把邮件推给发送方邮件服务器（此时可能就完成了「发送」）。
3. **DNS 解析**：发送方服务器查收件人域名的 **MX 记录**，找到接收方邮件服务器的地址（DNS 的应用，见《域名系统》）。
4. **服务器间 SMTP 中继**：发送方服务器用 SMTP 把邮件中继给接收方服务器（可经过多个中继）。
5. **接收方服务器**：把邮件存入收件人的邮箱。
6. **收件人 UA**：用 POP3/IMAP 把邮件「拉」下来阅读。

**辨析｜易错点：** 常见误解是「SMTP 直接从发件人电脑送到收件人电脑」。实际是：**发件人电脑通常只负责「推到自己的邮件服务器」，真正的跨域传输发生在服务器之间**。所以「发件人电脑关机，邮件照样能到」——因为传输早已由邮件服务器完成了。**「邮件传输是服务器之间的接力，不是两台 PC 的直连」**。

## 6 SMTP 的历史与演进：为什么它「简单」

SMTP 的历史解释了它的一切「古怪」：<span class="marginnote"><strong>1982 年 RFC 821 定义了 SMTP，2008 年 RFC 5321 修订</strong>，核心命令 30 年几乎未变。<strong>「简单」是它的生存之道</strong>：没有会话状态、没有文件系统、只是逐条命令应答，所以 40 年兼容至今。<strong>SMTP 之后还有扩展：ESMTP（扩展 SMTP）加了认证与 8 位传输，STARTTLS 给 SMTP 加加密</strong>——但核心对话框架不变。</span>

- **为什么只传 7 位 ASCII**：1980 年代的网络只能可靠传文本，非 ASCII 内容当时「不存在」。
- **为什么用 25 端口**：当时没有「端口安全」的概念；今天提交邮件用 587（提交端口）与 465（SMTPS），25 只用于服务器间中继。
- **SMTP 与现代**：垃圾邮件催生了 SMTP 认证（AUTH）与 SPF/DKIM/DMARC 等邮件验证机制——「简单」的协议在 40 年安全对抗中不断打补丁。

**辨析｜易错点：** **SMTP 认证 ≠ 加密**：`AUTH LOGIN` 只做身份验证，`STARTTLS` 才加密传输。**「认证管你是谁，加密管别人看不看得到」**——两者都要配齐才是安全的邮件提交。

## 7 术语速查：电子邮件的概念地图

| 术语 | 含义 | 说明 |
| --- | --- | --- |
| 用户代理（UA） | 发信/收信的客户端 | Outlook、Web 邮箱 |
| 邮件服务器 | 存储与转发邮件的主机 | 发送方/接收方各一个 |
| 推（push） | 主动把邮件送出 | SMTP 的模式 |
| 拉（pull） | 主动去取邮件 | POP3/IMAP 的模式 |
| MX 记录 | 域名→邮件服务器的映射 | 靠 DNS 解析 |
| MIME | 邮件内容扩展 | 编码二进制 + 声明类型 |
| Base64 | 二进制→ASCII 编码 | 每 3 字节变 4 字符 |

**辨析｜易错点：** 邮件协议的分工口诀：**SMTP 管「送」，POP3/IMAP 管「取」**。发送全程（UA→服务器、服务器→服务器）都用 SMTP；收件人读取才用 POP3/IMAP。**「寄信用 SMTP、收信用 POP3/IMAP」**是最不会错的分工记忆。

## 8 小结

- **SMTP**：基于 TCP 25 的邮件推送协议，文本命令 + 数字应答。
- **核心命令流**：HELO → MAIL FROM → RCPT TO → DATA → QUIT。
- **三大特点**：推模式、只传 ASCII 文本、客户端与服务器间都用。
- **MIME**：把二进制编码成文本（Base64）+ 声明类型，让 SMTP 能传附件。
- **SMTP vs HTTP**：推 vs 拉；25 vs 80；命令风格不同。
- **历史地位**：SMTP 是互联网最早的应用协议之一，「简单」是它长盛的原因。

在下一节，我们将看邮件的「取件协议」——**邮件读取协议（POP3 与 IMAP）**。

最后补一个复习钩子：**SMTP 的故事印证了「简单协议能活四十年」——它的命令、端口、文本风格从 1982 年沿用至今，靠的不是花哨，而是把「推」这件事做透**。下一节的 POP3/IMAP 会补上邮件的另一半「拉」，届时「推拉配合」的完整拼图就齐了。
