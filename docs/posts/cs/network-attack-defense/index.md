---
pageClass: plain-doc
---

# 网络攻防技术（渗透测试/CTF/红蓝对抗/应急响应）

系统学习网络攻防的完整知识体系：从渗透测试的信息收集与漏洞利用，到 Web 应用攻击、CTF 竞赛技能、内网后渗透，再到红蓝对抗与应急响应，构建攻防一体的安全实战能力。

## 对标教材

- Dafydd Stuttard & Marcus Pinto, "The Web Application Hacker's Handbook" (2nd ed., Wiley 2011)
- Patrick Engebretson, "The Basics of Hacking and Penetration Testing" (2nd ed., 2011)（中文版《渗透测试实践指南：必知必会的工具与方法》）
- CTF 竞赛资料（CTF Wiki / picoCTF /《CTF特训营》）

## 主题规划

<ProgressGrid cat="cs/network-attack-defense" />

### 第1篇：渗透测试基础与信息收集

- [x] [渗透测试方法论与测试框架（PTES/OWASP 测试指南）](./penetration-testing-methodology)（渗透测试实践指南 第1章）
- [x] [被动信息收集与 OSINT（Google Hacking/WHOIS/子域名枚举）](./passive-recon-osint)（渗透测试实践指南 第2章）
- [x] [主动侦察：Nmap 端口扫描与指纹识别](./active-recon-nmap)（渗透测试实践指南 第3章）
- [x] [服务枚举与目录爆破（Gobuster/SNMP/SMB 枚举）](./service-enumeration-directory-bruteforce)（渗透测试实践指南 第3章）
- [x] [漏洞扫描与评估（OpenVAS/Nessus/漏洞库）](./vulnerability-scanning)（书目外）
- [x] [网络嗅探与流量分析（Wireshark/tcpdump/ARP 欺骗）](./network-sniffing-traffic-analysis)（书目外）

### 第2篇：Web 应用渗透测试

- [x] [Web 应用技术栈、核心防御机制与攻击面映射（Burp Suite）](./web-defense-mechanisms-attack-surface)（WAHH 第2-4章）
- [x] [绕过客户端控件（前端校验/隐藏字段/JS 混淆）](./bypassing-client-controls)（WAHH 第5章）
- [x] [身份认证攻击（口令爆破/认证绕过/重放）](./authentication-attacks)（WAHH 第6章）
- [x] [会话管理攻击（令牌弱点/会话固定/会话劫持）](./session-management-attacks)（WAHH 第7章）
- [x] [访问控制与越权（IDOR/水平与垂直越权）](./access-control-idor)（WAHH 第8章）
- [x] [SQL 注入与数据存储攻击（联合/盲注/堆叠）](./sql-injection)（WAHH 第9章）
- [x] [业务逻辑漏洞与后端组件攻击（文件上传/XXE/SSRF）](./business-logic-xxe-ssrf)（WAHH 第10-11章）
- [x] [基于浏览器的攻击：XSS 与 CSRF](./xss-csrf)（WAHH 第12章）

### 第3篇：内网渗透与后渗透利用

- [x] [系统漏洞利用与 Metasploit 实战](./metasploit-exploitation)（渗透测试实践指南 第4章）
- [x] [Web 应用漏洞利用](./web-app-exploitation)（渗透测试实践指南 第5章）
- [x] [后渗透：权限提升（Windows/Linux 提权）](./privilege-escalation)（渗透测试实践指南 第7章）
- [x] [横向移动与内网穿透（代理/隧道/域渗透）](./lateral-movement-pivoting)（书目外）
- [x] [权限维持与痕迹清理](./persistence-defense-evasion)（渗透测试实践指南 第7章）

### 第4篇：CTF 竞赛专题

- [x] [CTF 入门与题型总览（Misc/Web/Crypto/Pwn/Reverse）](./ctf-intro-overview)（CTF 竞赛资料）
- [x] [CTF Misc：隐写术与流量取证](./ctf-misc-stego-forensics)（CTF 竞赛资料）
- [x] [CTF Web：Web 题目与漏洞利用](./ctf-web)（CTF 竞赛资料）
- [x] [CTF Crypto：古典密码与现代密码](./ctf-crypto)（CTF 竞赛资料）
- [x] [CTF Pwn：栈溢出、堆利用与 ROP](./ctf-pwn)（CTF 竞赛资料）
- [x] [CTF Reverse：逆向工程与算法还原](./ctf-reverse)（CTF 竞赛资料）

### 第5篇：红蓝对抗与应急响应

- [x] [红蓝对抗方法论与 MITRE ATT&CK 框架](./red-blue-team-attack)（书目外）
- [x] [蓝队防御：检测、监控与安全加固（SIEM/日志分析）](./blue-team-defense-siem)（书目外）
- [x] [应急响应流程与事件处置（分类分级/遏制/清除）](./incident-response)（书目外）
- [x] [数字取证与日志分析（磁盘/内存/网络取证）](./digital-forensics)（书目外）
- [x] [安全评估报告与风险复测](./security-assessment-report)（渗透测试实践指南 第8章）
