---
title: HTTP 的持久连接、Cookie 与会话管理
date: 2026-08-07
---

# HTTP 的持久连接、Cookie 与会话管理

<div class="epigraph">
<p>HTTP 天生健忘，但 Web 应用需要记住你——于是有了持久连接省时间，有了 Cookie 补记忆。</p>
<footer>—— 网络教材中的通俗说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§6.5 ｜ 2026-08-07</p>
</div>

## 为什么从持久连接与 Cookie 开始

上一节说 HTTP 是无状态的，且 HTTP/1.x 基于 TCP。这两个特性引出一对矛盾：**无状态 + 频繁建连 = 效率灾难；无状态 + 要记住用户 = 体验灾难**。这一节看 HTTP 怎么化解：**持久连接**解决「频繁建连」，**Cookie** 解决「记住用户」。<span class="marginnote">一次网页加载要请求几十个子资源——如果每个资源都新建 TCP 连接（非持久连接），光是三次握手就浪费大量时间。<strong>持久连接让多个请求复用同一条 TCP 连接</strong>，大幅降低开销。而「购物车」「登录状态」需要服务器记住你是谁——<strong>Cookie 是服务器发给浏览器、浏览器再回传的小凭证</strong>，让无状态的 HTTP 具备了「记忆」能力。</span>

这一节讲：**非持久 vs 持久连接、Cookie 机制、会话（Session）管理。**

## 1 非持久连接 vs 持久连接

**非持久连接（HTTP/1.0 默认）**：**每个请求建立一个 TCP 连接**，响应完就关闭。缺点明显——<span class="marginnote">非持久连接的浪费：一个网页 10 张图 = 至少 10 次 TCP 三次握手 + 10 次四次挥手，握手与挥手的时间占比极高。<strong>「连接的开销 > 传输的开销」</strong>在高频小资源场景下尤其严重。HTTP/1.0 时代 Web 页面简单还能忍，页面一复杂就撑不住了。</span>

- **每个请求一条连接**：请求 1 用连接 1，请求 2 再建连接 2……
- **开销大**：握手/挥手次数 × 请求数，时延高。

**持久连接（HTTP/1.1 默认）**：**多个请求复用同一条 TCP 连接**，连接保持一段时间，减少握手开销。HTTP/1.1 的持久连接 + 管线化（pipelining）是早期的优化；更高效的形态是 HTTP/2 的多路复用（第 9 章）。

**辨析｜易错点：** **HTTP/1.0 默认非持久，HTTP/1.1 默认持久**是经典考点。持久连接还分「流水线（pipelining，一个连接上连续发多个请求）」与「非流水线（发一个等一个）」——HTTP/1.1 的流水线有队头阻塞问题，最终被 HTTP/2 的多路复用取代。**「HTTP/1.0 建连，HTTP/1.1 复用，HTTP/2 并行」**是连接演进的一句话。

## 2 Cookie：无状态 HTTP 的「记忆贴片」

**Cookie**：服务器通过响应报文发给浏览器、浏览器保存并在后续请求中回传的**小型文本数据**，用于标识用户会话。<span class="marginnote">Cookie 的工作循环：<strong>服务器在响应里发 <code>Set-Cookie</code> 首部 → 浏览器存下 → 后续请求带 <code>Cookie</code> 首部 → 服务器认出你是谁</strong>。它让「无状态」的 HTTP 具备「记忆」：购物车、登录态、个性化推荐，全靠它。Cookie 是「服务器发的凭证」，不是浏览器自造的。</span>

Cookie 的典型流程：

1. **首次访问**：浏览器请求登录页。
2. **服务器发 Cookie**：登录成功后，服务器在响应里带 `Set-Cookie: session_id=abc123`。
3. **浏览器保存**：浏览器存下这个 Cookie（关联到该域名）。
4. **后续请求**：浏览器每次请求都带 `Cookie: session_id=abc123`，服务器据此识别「已登录用户」。

**辨析｜易错点：** **Cookie 是「服务器发给浏览器、浏览器存着、每次回传」**——它不存服务器端。与 Cookie 相对的 **Session（会话）** 存在**服务器端**：Cookie 里只放一个「会话 ID」，服务器凭 ID 查 Session 数据。**「Cookie 存客户端、Session 存服务器；Cookie 是钥匙，Session 是柜子」**是两者关系的标准答案。

## 3 会话管理：Cookie + Session 的配合

**会话（Session）**：服务器端保存的「用户状态」，用 Session ID 标识。完整机制：<span class="marginnote">为什么不能「什么都放 Cookie」？因为 Cookie 存在客户端，用户可改、可偷——敏感数据放 Cookie 不安全。<strong>所以只把「会话 ID」放 Cookie，真正的数据（购物车内容、用户资料）放服务器端的 Session</strong>。这个「客户端存凭证、服务器存数据」的分工，是现代 Web 认证的标准架构。</span>

1. 用户登录 → 服务器创建 Session，生成唯一 Session ID。
2. 服务器把 Session ID 通过 `Set-Cookie` 发给浏览器。
3. 浏览器每次请求带 Session ID → 服务器查 Session → 认出用户、拿到状态。
4. 退出登录 → 服务器销毁 Session，Cookie 过期。

**辨析｜易错点：** **Cookie 与 Session 不是「二选一」，而是「配合」**——Cookie 是 Session 的「载体」（传输 Session ID），Session 是 Cookie 的「后盾」（存放真实数据）。**「Cookie 传 ID、Session 存数据」**是最不易错的理解。另外，Session 的过期与销毁由服务器管理，Cookie 的过期由 Expires/Max-Age 控制。

## 4 Cookie 的安全与隐私

Cookie 虽然方便，但也是攻击与隐私问题的焦点：<span class="marginnote">Cookie 相关的两大攻击：<strong>会话劫持</strong>——攻击者偷到你的 Session Cookie，就能冒充你（防御：HttpOnly、Secure、加密传输）；<strong>CSRF</strong>——诱导你的浏览器携带 Cookie 向服务器发「伪造请求」（防御：校验 Referer、加 CSRF Token）。隐私层面，Cookie 可用于跨站追踪用户行为——这正是浏览器「第三方 Cookie 拦截」要解决的。</span>

- **会话劫持**：偷 Cookie = 冒充用户 → 需 HttpOnly、Secure、TLS 保护。
- **CSRF**：借 Cookie 伪造请求 → 需 Token 校验。
- **隐私追踪**：第三方 Cookie 记录浏览行为 → 浏览器默认拦截。

**辨析｜易错点：** **Cookie 本身不是病毒，它是「凭证」**——危险在于「凭证被偷/被滥用」。安全 Cookie 的三件套：<strong>HttpOnly</strong>（脚本读不到，防 XSS 窃取）、<strong>Secure</strong>（只在 HTTPS 传）、<strong>SameSite</strong>（防跨站携带，防 CSRF）。**「Cookie 要防偷、防盗用、防跨站」**是它的安全三防。

## 5 小结

- **非持久 vs 持久连接**：HTTP/1.0 每请求一连接，HTTP/1.1 默认持久复用。
- **连接演进**：HTTP/1.0 建连 → HTTP/1.1 复用 → HTTP/2 并行。
- **Cookie**：服务器发的凭证，浏览器存着、每次回传；`Set-Cookie` 发、`Cookie` 回。
- **Session**：服务器端存用户状态；Cookie 只传 Session ID。
- **黄金分工**：Cookie 传 ID、Session 存数据；Cookie 是钥匙、Session 是柜子。
- **安全三防**：HttpOnly（防 XSS 偷）、Secure（防明文）、SameSite（防 CSRF）。

在下一节，我们将看 Web 的「加速器」——**Web 缓存与代理服务器**。
