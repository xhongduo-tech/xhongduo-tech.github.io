---
title: WebAuthn / FIDO2
date: 2026-09-08
section: cs
---

# WebAuthn / FIDO2

<div class="epigraph">
<p>WebAuthn 让浏览器把挑战交给认证器：私钥不出盒，站点拿到的是源绑定的公钥凭证。钓鱼站点换源就验不过，这是对共享口令模型的结构性修补。</p>
<footer>—— W3C Web Authentication；FIDO Alliance CTAP；对照 RFC 6238 仍只是第二因素</footer>
</div>

上一课[SAML](/cs/saml)仍常与口令 IdP 搭配。缺口是**源绑定公钥**：凭据属于 `https://example.com`，不是用户可粘贴的秘密。本课收登记与断言，不写破解认证器的步骤。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

口令可被钓鱼页收集。WebAuthn：RP ID、challenge、user verification、attestation（可选）。缺口是丢失设备与恢复流程——恢复若退回邮箱口令，钓鱼面回来。

<span class="marginnote">术语翻译：RP ID 就是「这把钥匙归哪个网站」的标签——认证器签名前先核对浏览器报来的 origin 与标签一致才肯动手；凭据像刻了门牌号的钥匙，换一扇门就插不进锁。</span>

### 不是生物识别本身

指纹/面容是认证器本地解锁。服务器只看见签名成功，不应存储生物模板。FAR/FRR 更后一课。

<span class="marginnote">W3C WebAuthn Level 2/3。平台认证器与只安全密钥。本课不提供克隆密钥的指导。</span>

## 方法

分 ceremony：create 与 get。强调 challenge 防重放、origin 由浏览器填。服务器存公钥与计数器（克隆检测点名）。对照 TOTP：仍是共享秘密，可被钓鱼同时转发。

```mermaid
flowchart TD
  RP["站点 origin"] --> CH["challenge"]
  CH --> AUTH["认证器签"]
  AUTH --> PUB["服务器验公钥"]
  ORIGIN["源绑定"] --> PUB
```

## 机制

身份因素从「知道」转向「持有+本地解锁」。账户恢复、企业托管与备份密钥是治理，不是协议算术。下一课 TOTP 与 MFA：仍广泛部署的共享秘密第二因素。

<span class="marginnote">数字实例：challenge 是服务器随机发的 32 字节 nonce，签名报文里原样带回；同一 challenge 第二次出现即作废——攻击者录下整段签名也无法用它通过下一次登录。</span>

<span class="marginnote">常见误区：以为指纹/面容就是认证凭据——生物特征只在设备本地解锁私钥，服务器只见到一次签名验证结果；生物模板不出设备，指纹可重录，真正要保住的是那把不出盒的私钥。</span>

```mermaid
flowchart TD
  L["真站 example.com 发起 get"] --> LB["浏览器填 origin=example.com"]
  LB --> LM["RP ID 匹配，认证器出签名"]
  PH["钓鱼站 evil.com 复制登录页"] --> PB["浏览器填 origin=evil.com"]
  PB --> PM["RP ID 不匹配，认证器拒签"]
  PM --> X["钓鱼者拿不到可用签名"]
```

## 边界

本课不写 CTAP  hid 细节。TOTP 下一课。

## 小结

- SAML/OAuth 常仍站在口令上；WebAuthn 源绑定密钥。
- 私钥不出认证器；challenge 防重放。
- 恢复流程可能把钓鱼面请回来。
- 下一课 TOTP 与 MFA。
- 出处：W3C Web Authentication；FIDO CTAP；对照 RFC 6238。
