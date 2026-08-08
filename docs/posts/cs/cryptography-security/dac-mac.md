---
title: 自主访问控制（DAC）与强制访问控制（MAC）
date: 2026-08-07
---

# 自主访问控制（DAC）与强制访问控制（MAC）

<div class="epigraph">
<p>DAC 让「资源的主人」说了算，MAC 让「系统的安全策略」说了算——前者灵活，后者可靠。</p>
<footer>—— 美国国防部（TCSEC / Orange Book）</footer>
</div>

<div class="article-byline">
<p>第三级 · 密码学与信息安全 ｜ Stallings《密码编码学与网络安全》第二十七章 ｜ 2026-08-07</p>
</div>

## 为什么从 DAC 与 MAC 开始

上一节的 rwx 与 ACL 属于**自主访问控制（DAC）**：资源的主人自己决定谁能访问。而更高安全等级的系统需要**强制访问控制（MAC）**：系统级策略**凌驾于**用户意愿之上——即使文件主人想共享，策略也不允许。SELinux、AppArmor、Windows 的强制完整性都是 MAC 的现代形态。理解 DAC 与 MAC 的差别，是理解「普通系统」与「高安全系统」分界的关键。<span class="marginnote">一个直观比喻：<strong>DAC 像「房间主人决定谁能进」，MAC 像「大楼的消防法规凌驾于房间主人之上」</strong>——MAC 不关心主人愿意与否，只执行全局安全策略。普通 Linux 是 DAC，加固的 SELinux 在 DAC 之上叠 MAC。</span>

## 1 DAC：主人的自主决定

**自主访问控制（Discretionary Access Control）**：资源（客体）的**属主**决定谁能访问、访问什么。Unix 的 rwx、ACL 都是 DAC。

**DAC 的两个特性**：

**属主控制**：文件主人可以授权/收回他人权限（`chmod`、`chown`）。
**权限可传递**：属主可把「授予权限的能力」也传给别人（借文件给他人，他人可再分享）。

**DAC 的弱点**：

**不可控的传播**：主人想「只给 Alice 看」，但 Alice 可以转发给 Bob——**无法防信息扩散**。
**恶意软件的自由**：用户进程以用户身份运行，DAC 挡不住「用户自己允许的操作」——木马在用户权限内想干什么就干什么。
**无全局策略**：每个文件各管各的，没有「系统级」的安全底线。

## 2 MAC：系统策略说了算

**强制访问控制（Mandatory Access Control）**：访问决策由**系统级策略**强制，**不依赖**主体/客体的属主意愿。经典模型是 **BLP（Bell-LaPadula）**：

**主体与客体都有安全标签**（如 绝密 > 机密 > 秘密 > 公开）。
**简单安全性质（no-read-up）**：主体**不能读**安全级别更高的客体。
**星性质（no-write-down）**：主体**不能写**安全级别更低的客体。

这两条规则共同实现「**信息只能从低向高流动**」——防止机密向低级别泄露。BLP 是政府/军队系统的经典模型。

**MAC 的关键**：**策略不可被属主绕过**——即使文件主人想给「绝密」的文件打上「公开」标签，系统策略也不允许（标签修改本身受控）。<span class="marginnote">BLP 的记忆锚点：<strong>「不读高、不写低」</strong>——高机密主体可以读低机密（向下兼容）、写高机密（向上汇总），但绝不允许「低主体读高」或「高主体写低」。这套规则把「信息流方向」钉死，是防泄密（confidentiality）的强制模型。</span>

## 3 SELinux：Linux 的 MAC 实现

**SELinux**（Security-Enhanced Linux，美国 NSA 开发）是最著名的 Linux MAC：

**类型强制（TE）**：每个进程有「域」（domain）、每个文件有「类型」（type），策略定义「域→类型的允许操作」。
**默认拒绝**：**没有明确允许的操作一律禁止**——即使 root 也一样（root 只是域之一，也要遵守策略）。
**策略文件**：`httpd.te` 类型强制策略文件——「HTTP 服务能读网站文件，但（默认）不能碰别的东西」。

**SELinux 的价值**：即使 Web 服务被攻破（拿到 `httpd_t` 域的权限），它也只能读网站文件、不能读 `shadow_t`（口令文件）、不能启动 shell——**把「进程被攻破」的影响锁在最小范围**。这就是 MAC「超越用户意愿的强制」在实战的意义。<span class="marginnote">SELinux 与 DAC 的叠加：<strong>访问必须同时通过 DAC（用户权限）与 MAC（策略）</strong>——双重检查，DAC 通过但 MAC 拒绝仍不能访问。SELinux 常被诟病「难配置」，但它把「默认拒绝」带进了 Linux——这是 MAC 思想最实际的落地。</span>

## 4 公式解析：BLP 的两条规则

$$
\text{no-read-up: } L(S) \ge L(O) \Rightarrow \text{允许 } S \text{ 读 } O
$$

$$
\text{no-write-down: } L(S) \le L(O) \Rightarrow \text{允许 } S \text{ 写 } O
$$

三步拆解这条「BLP 规则」：

- **第一步，标签**：主体 $S$ 与客体 $O$ 都有安全级别 $L$（如 绝密 > 机密）。
- **第二步，读规则**：$S$ 读 $O$ 需要 $L(S) \ge L(O)$——**高/同级可读低**，「低读高」禁止。
- **第三步，写规则**：$S$ 写 $O$ 需要 $L(S) \le L(O)$——**低/同级可写高**，「高写低」禁止。信息流单向向上，防泄密。

## 5 MAC 的现代形态

MAC 思想在现代系统里有多种变体：

- **SELinux / AppArmor**：Linux 的域-类型强制 / 路径配置。
- **Windows 强制完整性（MIC）**：进程与文件有完整性级别（低/中/高/系统），低完整性进程不能写高完整性对象。
- **iOS/Android 沙箱**：每个 App 一个独立用户（UID）+ 权限（Android 的 permission 模型）——应用隔离的 MAC。
- **容器安全**：Docker 默认丢弃能力 + 可挂 SELinux——容器的隔离本质是「系统级策略」。

**MAC vs DAC 的选型**：普通桌面/服务器用 DAC 足够；**高安全、多租户、面向互联网的服务**建议叠 MAC（SELinux/AppArmor）——「默认拒绝」把被攻破的影响最小化。<span class="marginnote">一个现实判断：<strong>DAC 是「方便优先」，MAC 是「安全优先」</strong>——大多数系统用 DAC，因为配置简单；但「被攻破后损失最小化」的需求（Web 服务、多租户）让 MAC 越来越必要。现代容器与云安全的实践，本质是「把 MAC 的隔离做成默认」。</span>

## 6 小结

- **DAC**：属主自主授权（rwx/ACL）——灵活，但不可控传播、无全局底线。
- **MAC**：系统策略强制（BLP 的 no-read-up / no-write-down）——可靠，策略不可绕过。
- **SELinux**：类型强制 + 默认拒绝——root 也要遵守，被攻破影响锁最小。
- **BLP**：「不读高、不写低」——信息单向流动，防泄密的经典模型。
- **现代形态**：AppArmor、Windows MIC、移动沙箱、容器隔离——MAC 正成为默认安全。

在下一节，我们看权限思想的另一支——**权能（Capability）与最小特权原则**。
