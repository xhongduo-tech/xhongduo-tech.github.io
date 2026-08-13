---
pageClass: plain-doc
---

# 网络安全基础（协议安全/防火墙/IDS/加密通信）

以《Network Security Essentials》为主线，系统讲解从对称加密、公钥密码到 TLS/IPSec 等安全通信协议，再到防火墙、IDS/IPS 等网络边界防御技术的完整知识体系，覆盖协议安全、加密通信与网络入侵防护的基础，是安全从业者的入门必修课。

## 对标教材

- 《Network Security Essentials: Applications and Standards》（William Stallings, 6th Edition）
- 《网络安全基础与应用》（第二版）

## 主题规划

<ProgressGrid cat="cs/network-security" />

### 第1篇 加密与认证基础

- [x] [对称加密与消息机密性](./symmetric-encryption-confidentiality)（Stallings 第2章）
- [x] [公钥密码体制：RSA 与椭圆曲线 ECC](./public-key-crypto-rsa-ecc)（Stallings 第3章）
- [x] [消息认证与哈希函数](./message-authentication-hash)（Stallings 第3章）
- [x] [数字签名与公钥基础设施 PKI](./digital-signature-pki)（Stallings 第3章）
- [x] [密钥分发与用户认证协议](./key-distribution-user-authentication)（Stallings 第4章）

### 第2篇 安全通信协议

- [x] [传输层安全：TLS/SSL 握手与记录协议](./tls-ssl-handshake-record)（Stallings 第6章）
- [x] [HTTPS 与 Web 传输安全](./https-web-security)（Stallings 第6章）
- [x] [网络访问控制 NAC 与云安全](./nac-cloud-security)（Stallings 第5章）
- [x] [IP 安全：IPSec 与 VPN](./ipsec-vpn)（Stallings 第9章）
- [x] [电子邮件安全：S/MIME 与 PGP](./email-security-smime-pgp)（Stallings 第8章）
- [x] [无线局域网安全：802.11i 与 WPA](./wlan-security-80211i-wpa)（Stallings 第7章）

### 第3篇 入侵检测与网络边界防御

- [x] [入侵者与入侵检测系统 IDS](./intruders-ids)（Stallings 第11章）
- [x] [入侵检测技术：特征/异常/蜜罐](./ids-techniques-signature-anomaly-honeypot)（Stallings 第11章）
- [x] [防火墙与包过滤](./firewall-packet-filtering)（Stallings 第12章）
- [x] [入侵防御系统 IPS 与统一威胁管理](./ips-utm)（Stallings 第12章）
- [x] [恶意软件与反恶意软件防护](./malware-protection)（Stallings 第10章）
