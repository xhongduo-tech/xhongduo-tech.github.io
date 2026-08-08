---
title: TLS 握手协议：密钥协商与参数协商
date: 2026-08-07
---

# TLS 握手协议：密钥协商与参数协商

<div class="epigraph">
<p>握手是 TLS 的「婚前协议」：双方先谈妥算法、交换密钥、验明身份，才敢开始正式的加密通信。</p>
<footer>—— 互联网安全研究组（IETF）</footer>
</div>

<div class="article-byline">
<p>第三级 · 密码学与信息安全 ｜ Stallings《密码编码学与网络安全》第十六章 ｜ 2026-08-07</p>
</div>

## 为什么从 TLS 握手开始

上一节看了 TLS 的版本历史，这一节拆开它的核心机制——**握手协议（handshake protocol）**。握手是连接建立时的「安全协商阶段」：双方交换随机数、协商算法、用证书认证服务器身份、通过 ECDHE 协商共享密钥、派生会话密钥。TLS 1.3 的握手只需一次往返（1-RTT），是所有 HTTPS 连接的最初几毫秒。理解握手 = 理解 TLS 如何把前几篇的 DH、签名、证书、KDF 全部缝合在一起。<span class="marginnote">一个全局视角：<strong>TLS 握手是「认证的密钥交换（AKE）」的完整工业实现</strong>——它同时完成三件事：协商参数（算法套件）、认证身份（证书签名）、协商密钥（ECDHE）。这三件事在握手消息里交织进行。</span>

## 1 握手的四类消息

TLS 1.3 握手由四类核心消息组成（正常完整握手，1-RTT）：

1. **ClientHello**：客户端 → 服务器。含客户端随机数、支持的密码套件列表、（ECDHE）客户端公钥。
2. **ServerHello**：服务器 → 客户端。选定的套件、服务器随机数、（ECDHE）服务器公钥。
3. **服务器证书 + 证书验证（CertificateVerify）**：服务器 → 客户端。证书链 + 对握手转录的签名。
4. **Finished**：双方各发一个对完整握手的 MAC/哈希——确认「握手过程未被篡改」。

**1-RTT 的含义**：客户端发 ClientHello → 服务器发完 ServerHello + 证书 + 签名 → 客户端就能算共享密钥并发 Finished + 第一条应用数据——**一次网络往返完成密钥建立**。<span class="marginnote">握手的「转录（transcript）」是 TLS 1.3 的暗线：<strong>所有已交换消息的哈希持续累积，每个签名/MAC 都覆盖 transcript</strong>——任何中间人「改一条消息、重排一条消息」都会改变 transcript，签名验证立即失败。这是防降级与防重排的总闸门。</span>

## 2 密钥协商：ECDHE 如何落地

TLS 1.3 的密钥协商是**ECDHE**（临时椭圆曲线 DH）：

1. 客户端在 ClientHello 里带自己的 ECDHE 公钥 $g^c$。
2. 服务器在 ServerHello 里带自己的 ECDHE 公钥 $g^s$。
3. 双方各自算 $g^{cs}$（共享秘密）。
4. 共享秘密 + 双方随机数 → **HKDF 派生**出一系列密钥：
`client_handshake_traffic_secret`、`server_handshake_traffic_secret`（握手加密）。
`client_application_traffic_secret`、`server_application_traffic_secret`（应用数据加密）。

**为什么要 ECDHE 而不是静态密钥**：ECDHE 是「每次会话新生成临时密钥对」——即使服务器长期私钥未来泄露，也无法解密过去录制的会话（**前向保密**）。TLS 1.3 因此**强制** ECDHE，杜绝了 TLS 1.2 里「RSA 密钥交换无前向保密」的老问题。<span class="marginnote">记忆锚点：<strong>ECDHE 里的 E = ephemeral（临时）</strong>——每次握手都生成新密钥对。前向保密的全部秘密就是「临时」：历史会话的密钥不被长期私钥「牵连」。TLS 1.3 强制这一条，等于把「历史不可解密」变成协议默认性质。</span>

## 3 身份认证：证书与签名验证

服务器在握手里用证书证明「我是这个域名的合法主人」：

1. 服务器发送**证书链**（服务器证书 + 中间 CA）。
2. 客户端验证证书链：逐级验签 → 有效期 → 域名（SAN）→ 吊销（可选）。
3. 服务器发送 **CertificateVerify**：对握手转录的签名。
4. 客户端用证书里的公钥验签——通过则「握手的另一端确实持有该证书的私钥」。

**签名覆盖 transcript** 是防中间人的关键：Eve 可以换掉 ServerHello 里的 DH 值，但她没有服务器的私钥——无法为「被篡改的 transcript」签名。证书认证把「身份」与「会话」焊接在一起。<span class="marginnote">TLS 1.3 还支持「客户端证书」——服务器也验证客户端身份（双向 TLS）。这在企业内网与 mTLS 微服务里常见。握手逻辑对称：客户端也发证书 + 签名，服务器验签。</span>

## 4 公式解析：会话密钥的派生链

$$
\text{HandshakeSecret} = \text{HKDF-Extract}(0, \text{ECDHE 共享秘密})
$$

$$
\text{MasterSecret} = \text{HKDF-Extract}(0, \text{HandshakeSecret} \| \text{transcript})
$$

三步拆解这条「HKDF 密钥派生链」：

- **第一步，提取**：HKDF-Extract 把 ECDHE 共享秘密「压平」成一个高熵的 HandshakeSecret——消除 DH 点的结构偏差。
- **第二步，绑定转录**：MasterSecret 把握手转录（含双方随机数、证书、签名）掺进来——**密钥与会话绑定**，不同会话不同密钥。
- **第三步，扩展**：HKDF-Expand 从 MasterSecret 派生出握手密钥与应用密钥——每把密钥各司其职。

## 5 会话恢复与 0-RTT：握手的速度优化

握手虽只有 1-RTT，但对高频连接（API、页面资源）仍是开销。TLS 1.3 提供两个优化：

- **会话恢复（PSK）**：第一次握手后，双方共享一个**预共享密钥（PSK）**；后续连接用 PSK 快速恢复——握手只需 1-RTT 的简化版。
- **0-RTT**：客户端用上次的 PSK，在第一条消息里就带应用数据——**零往返**。代价：0-RTT 消息**可被重放**，所以只允许幂等请求（如 GET）。

**安全边界**：0-RTT 重放风险是 TLS 1.3 设计里唯一「性能优先于安全」的地方——协议明确要求应用层自行处理幂等性。<span class="marginnote">一个工程教训：<strong>「快」与「防重放」天然冲突</strong>——0-RTT 快，但消息可重放；完整握手安全，但多一次往返。TLS 1.3 的选择是「把选择权留给应用」：需要绝对安全就别用 0-RTT 传非幂等数据。这是协议设计的现实妥协。</span>

## 6 小结

- **TLS 1.3 握手**：ClientHello（含 ECDHE 公钥）→ ServerHello（含 ECDHE 公钥）→ 证书 + CertificateVerify（转录签名）→ 双方 Finished。
- **密钥协商**：ECDHE 强制 → 前向保密；共享秘密经 **HKDF** 派生出握手与应用密钥。
- **身份认证**：证书链验证 + 转录签名——身份与会话焊接，防中间人。
- **transcript 哈希**：所有握手消息进入转录，任何篡改都破坏签名——防降级与重排。
- 优化：会话恢复（PSK）+ 0-RTT（零往返，但可重放，仅限幂等）。

在下一节，我们看握手之后的数据通道——**TLS 记录协议：加密与完整性保护**。
