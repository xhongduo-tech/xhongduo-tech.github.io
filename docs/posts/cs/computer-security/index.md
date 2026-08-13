---
pageClass: plain-doc
---

# 计算机安全综合

以《Computer Security: Principles and Practice》与 NIST 安全标准为纲，系统覆盖从安全模型、认证与访问控制，到软件漏洞、网络安全、系统加固与安全管理治理的完整计算机安全知识体系。学完这些章节，就写完了计算机安全学科的核心内容。

## 对标教材

- William Stallings & Lawrie Brown, 《Computer Security: Principles and Practice》(4th, 2018)
- NIST 安全标准（SP 800 系列 / 安全与隐私框架）

## 主题规划

<ProgressGrid cat="cs/computer-security" />

### 第1篇 安全基础与模型

- [x] [计算机安全概述与安全需求](./computer-security-overview)（Stallings 第1章）
- [x] [安全攻击面与威胁建模](./attack-surface-threat-modeling)（Stallings 第1章）
- [x] [安全模型与访问控制](./security-models-access-control)（Stallings 第4章）
- [x] [身份认证机制：口令、生物识别与 MFA](./authentication-password-biometric-mfa)（Stallings 第3章）
- [x] [密码学基础：对称与非对称加密](./cryptography-symmetric-asymmetric)（Stallings 第2章）
- [x] [哈希、MAC 与数字签名](./hash-mac-digital-signature)（Stallings 第2章）
- [x] [公钥基础设施 PKI 与数字证书](./pki-digital-certificates)（Stallings 第2章）

### 第2篇 软件与系统安全

- [x] [恶意软件分类与行为分析](./malware-classification-analysis)（Stallings 第6章）
- [x] [缓冲区溢出与内存安全](./buffer-overflow-memory-safety)（Stallings 第10章）
- [x] [内存防护机制：ASLR、DEP 与 CFI](./memory-protection-aslr-dep-cfi)（Stallings 第10章）
- [x] [操作系统安全加固](./os-hardening)（Stallings 第4章）
- [x] [数据库与云安全](./database-cloud-security)（Stallings 第5章）
- [x] [拒绝服务攻击与防护](./denial-of-service)（Stallings 第7章）

### 第3篇 网络安全

- [x] [网络安全基础与协议栈安全](./network-security-basics)（Stallings 第13章）
- [x] [传输层安全与 TLS](./tls-transport-layer-security)（Stallings 第14章）
- [x] [IP 安全与 VPN](./ipsec-vpn)（Stallings 第17章）
- [x] [无线网络安全](./wireless-network-security)（Stallings 第15章）
- [x] [邮件安全：PGP 与 S/MIME](./email-security-pgp-smime)（Stallings 第16章）
- [x] [入侵检测与防御 IDS/IPS](./ids-ips-intrusion-detection)（Stallings 第19章）
- [x] [防火墙与网络边界防护](./firewall-network-boundary)（Stallings 第22章）

### 第4篇 安全管理与治理

- [x] [安全风险评估与审计](./security-risk-assessment-audit)（Stallings 第8章）
- [x] [安全策略、标准与合规](./security-policy-standards-compliance)（NIST SP 800 系列）
- [x] [事件响应与应急管理](./incident-response)（NIST SP 800-61）
- [x] [灾难恢复与业务连续性](./disaster-recovery-bcp)（NIST SP 800-34）
- [x] [安全法规、伦理与责任](./security-law-ethics)（Stallings 第9章）
- [x] [隐私保护与数据合规](./privacy-data-compliance)（NIST 隐私框架）
