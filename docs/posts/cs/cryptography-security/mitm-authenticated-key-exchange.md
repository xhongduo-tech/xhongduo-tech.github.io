---
title: 中间人攻击与认证的密钥交换
date: 2026-08-07
---

# 中间人攻击与认证的密钥交换

<div class="epigraph">
<p>裸的密钥交换只能保证「你们两个算出了同一把钥匙」——却不能保证「对方真的是他」。</p>
<footer>—— 威廉 · 斯托林斯（William Stallings）</footer>
</div>

<div class="article-byline">
<p>第三级 · 密码学与信息安全 ｜ Stallings《密码编码学与网络安全》第十五章 ｜ 2026-08-07</p>
</div>

## 为什么从中间人攻击开始

DH 密钥交换解决「公开信道协商密钥」，但它**不认证身份**——这正是中间人攻击（MITM）的温床：攻击者同时与双方建立两段「合法」的 DH，双方都以为在和对方通信。解决 MITM 的唯一办法是**认证的密钥交换（AKE）**：在交换密钥的同时认证双方身份。TLS 握手就是最著名的 AKE——它用证书签名认证服务器身份，从而让 DH 协商安全落地。这一节拆解 MITM 与认证密钥交换的全部逻辑。<span class="marginnote">一句话概括：<strong>DH 回答「我们能否共有一把钥匙」，认证回答「你是谁」，AKE 把两者缝合</strong>。裸 DH + 无认证 = MITM 任意摆布；DH + 签名认证 = TLS。理解 AKE 是理解 TLS 握手的钥匙。</span>

## 1 中间人攻击：DH 的致命缺口

**中间人攻击（Man-in-the-Middle, MITM）** 流程：

1. Eve 截获 Alice 的 DH 值 $g^a$，与 Alice 建立 DH，冒充 Bob（$K_{AE} = g^{ae}$）。
2. Eve 与 Bob 建立另一个 DH，冒充 Alice（$K_{BE} = g^{be}$）。
3. Alice 用 $K_{AE}$ 加密消息，Eve 解密、改用 $K_{BE}$ 加密转发给 Bob。
4. Alice 与 Bob **都不知道被转发了**——Eve 全程透明地偷听/篡改。

**根因**：DH 的消息没有「身份绑定」——$g^a$ 只是数字，不证明「这是 Alice 的」。任何攻击者都能发一个自己的 $g^e$。

## 2 认证的密钥交换：两把武器

防 MITM 的 AKE 有两种主要武器：

**武器一：签名认证（Signature-based）**——TLS 的方式。服务器持有长期签名密钥（证书私钥），在 DH 交换中**对自己的 DH 值签名**：

1. Alice 收到 Bob 的 $g^b$ 与签名 $\sigma = \text{Sign}_{sk_B}(g^b, \ldots)$。
2. Alice 用 Bob 的公钥验证 $\sigma$——通过则「$g^b$ 确实是 Bob 发的」。
3. Eve 无法为「自己的 $g^e$」伪造 Bob 的签名（没有 Bob 的私钥）——MITM 破产。

**武器二：预共享密钥认证（PSK-based）**——双方共享一个长期密钥，用它做 MAC 认证 DH 消息（如 Wi-Fi WPA2 的预共享口令）。

现代协议（TLS、SSH、WireGuard）都用签名认证（配证书），这是公钥基础设施的用武之地。<span class="marginnote">签名认证的核心洞察：<strong>「DH 值」是临时的、可伪造的；「签名」是长期的、不可伪造的</strong>。把临时值交给长期私钥签名，就把「这次会话」钉在了「这个身份」上。Eve 能伪造临时值，却伪造不了签名。</span>

## 3 公式解析：签名认证的 DH

$$
\text{Bob} \to \text{Alice}: \quad g^b, \; \sigma = \text{Sign}_{sk_B}(g^b, \text{transcript})
$$

$$
\text{Alice}: \quad \text{Verify}_{pk_B}(g^b, \text{transcript}, \sigma) = \text{接受} \Rightarrow g^b \text{ 来自 Bob}
$$

三步拆解这条「认证 DH 消息」：

- **第一步，签什么**：Bob 签名的不只是 $g^b$，还有**完整握手转录**（双方 nonce、算法、之前的消息）——签名绑定整个会话，防止中间人「重排/裁剪」握手内容。
- **第二步，验什么**：Alice 用 Bob 的公钥验证签名——通过则「$g^b$ 确系 Bob 发出且会话未被篡改」。
- **第三步，为什么 MITM 失败**：Eve 要冒充 Bob，必须用 Bob 的私钥签名——她没有。Eve 只能提供自己的 $g^e$，但签不了名，Alice 立即识破。

## 4 TLS 握手：认证密钥交换的教科书

TLS 1.3 握手是 AKE 的完整实例（简化）：

1. 客户端发 `ClientHello`：随机数 + 支持的算法列表 + **自己的 DH 公钥** $g^c$（ECDHE）。
2. 服务器发 `ServerHello`：选算法 + **自己的 DH 公钥** $g^s$ + **证书链** + 对握手转录的签名。
3. 客户端验证证书链（信任根）与签名——服务器身份被认证。
4. 双方各自算出共享密钥 $g^{cs}$，用 HKDF 派生会话密钥。
5. 后续消息用 AEAD 加密，记录层带序列号防重放。

**TLS 同时解决三件事**：密钥协商（DH）、身份认证（证书签名）、消息新鲜性（nonce + 序列号）——这正是本节与前两节的缝合点。<span class="marginnote">TLS 1.3 的一个设计细节：<strong>证书与签名在 DH 值之后发送，且签名覆盖全部握手消息（transcript hash）</strong>——这是「签名绑定会话」的完整实现。任何中间人「换 DH 值、改算法、重排消息」都会破坏转录，签名验证立即失败。</span>

## 5 AKE 的现实挑战

认证密钥交换的现实世界远非完美：

- **证书校验失败**：用户忽略证书警告 → MITM 依然可行（公共 Wi-Fi 上的「假证书攻击」）。
- **CA 被攻破**：攻击者获得伪造证书 → 签名认证被架空（DigiNotar）。
- **降级攻击**：攻击者诱导双方用更弱的算法/更短的密钥（如 Downgrade 攻击）→ 需要用版本号/算法协商硬化。
- **前向保密**：AKE 必须用**临时密钥**（ECDHE）——若用静态密钥，长期私钥泄露会解密历史会话。

**现代答案**：TLS 1.3 强制 ECDHE（临时）、完整转录签名、禁止降级到弱套件——把 AKE 的每个缺口都堵上。<span class="marginnote">「降级攻击」是 AKE 的隐藏陷阱：<strong>攻击者不改内容，只改「算法协商」——诱导双方用可被破解的旧算法，然后轻松解密</strong>。TLS 1.3 用「版本号 + transcript 签名」把算法协商也钉进认证——这是从 POODLE 等降级攻击中学到的教训。</span>

## 6 小结

- **MITM**：裸 DH 无身份绑定，Eve 与双方各建一段会话——透明劫持。
- **根因**：DH 值可伪造，不证明身份。
- **认证密钥交换**：签名认证（TLS）或 PSK 认证（WPA2）——把「临时 DH 值」绑定到「长期身份」。
- **TLS 握手**：DH + 证书签名 + nonce/序列号——协商、认证、新鲜性三合一。
- 挑战：证书校验失败、CA 被攻破、降级攻击、前向保密——TLS 1.3 全面硬化。

在下一节，我们看认证因子的现代组合——**多因子认证（MFA）与 FIDO2/WebAuthn**。
