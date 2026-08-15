---
title: 系统安全防护与病毒查杀
date: 2026-08-07
---

# 系统安全防护与病毒查杀

<div class="epigraph">
<p>千里之堤，溃于蚁穴。</p>
<footer>——《韩非子·喻老》</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ Microsoft Windows 官方支持文档 ·《鸟哥的Linux私房菜：基础学习篇》 ｜ 2026-08-07</p>
</div>

## 为什么从系统安全开始

电脑装了系统、配了驱动、做了维护，但如果**安全防线**是空的，前面一切努力都可能在一夜之间清零——勒索病毒加密全盘、木马盗走密码、钓鱼邮件骗走账号。系统安全不是「装个杀毒软件就完事」，而是一套由**更新、账号、权限、杀毒、习惯**构成的防御纵深。

本节把三平台的安全防线一层层摆开：Windows 的内置防线、Linux 的权限与 SELinux、macOS 的沙盒，再给出一条「系统安全检查」的实操路径。它承接《系统更新与补丁管理》，也呼应《硬盘健康检测与坏道排查》的「防患于未然」精神。

## 1 威胁全景：先认识敌人

主流威胁分几类，识别它们才有针对性防御：

| 威胁 | 特征 | 典型载体 |
| --- | --- | --- |
| 病毒（Virus） | 感染文件、自我复制 | 可执行文件、文档宏 |
| 木马（Trojan） | 伪装正常程序窃取数据 | 破解软件、捆绑安装 |
| 蠕虫（Worm） | 网络自主传播 | 邮件附件、漏洞利用 |
| 勒索软件（Ransomware） | 加密文件索要赎金 | 邮件附件、漏洞、弱口令 |
| 挖矿木马 | 偷占 CPU/GPU 挖矿 | 网页、破解工具 |
| 钓鱼（Phishing） | 伪造网站/邮件骗取凭证 | 仿冒官网、伪造邮件 |

<span class="marginnote">勒索软件（如 WannaCry、LockBit）是当下最严重的威胁：它把文件加密后索要比特币。对付它没有「灵丹妙药」，唯一可靠的防线是<strong>离线备份</strong>（见《系统备份与还原》）——不依赖系统的备份，勒索病毒拿它没办法。</span>

## 2 Windows：内置防线就够了

现代 Windows 自带完整的安全套件，**不需要额外装第三方杀毒**：

- **Microsoft Defender 防病毒**：系统自带杀毒，默认开启，实时防护。
- **Windows Defender 防火墙**：控制入站出站网络流量。
- **SmartScreen**：拦截可疑下载与网站。
- **BitLocker**：磁盘加密，丢失笔记本时保护数据。

入口：「设置 → 隐私和安全性 → Windows 安全中心」。里面分「病毒和威胁防护」「防火墙和网络保护」「设备安全性」等面板。<span class="marginnote">一个常见误区是「Windows 一定要装 360/电脑管家」。其实 Windows 10/11 自带的 Defender 防护能力已与主流第三方杀毒相当，且无弹窗、无捆绑。第三方杀毒不仅多此一举，还可能拖慢系统。</span>

**重点：Windows 安全的第一层不是杀毒，是「补丁 + 账号 + 习惯」。** 系统保持更新、账号用强密码开双因素（2FA）、不双击来路不明的附件，比装十个杀毒软件都有用。

## 3 Linux 安全：权限与 SELinux

Linux 的安全哲学与 Windows 不同——它靠**权限模型**从源头限制危害：

- **最小权限**：普通用户无权限写系统目录，病毒很难「感染系统」。
- **sudo 提权**：需要管理员操作时临时提权，而不是常驻管理员。
- **SELinux/AppArmor**：强制访问控制（MAC），给进程套「笼子」，即使被攻破也被限制在笼子里。

鸟哥在《基础学习篇》第14章「程序管理与 SELinux 初探」里专门讲了这套机制。日常维护要点：<span class="marginnote">SELinux 是红帽系（Fedora/CentOS/Rocky）的强制访问控制层，很多「权限对了还是不行」的玄学问题其实是 SELinux 在拦。排查用 `getenforce`（看状态）、`ausearch -m avc`（看拦截日志）。桌面 Ubuntu 默认用 AppArmor，理念相同。</span>

```
sudo apt update && sudo apt upgrade   # 保持更新
sudo ufw status                        # Ubuntu 防火墙状态
sudo ufw enable                        # 开启防火墙
```

Linux 中病毒相对少见，但**挖矿木马、被植入后门、弱口令爆破**依然常见——尤其暴露在公网的服务器。

## 4 macOS 安全：沙盒与门禁

macOS 的安全设计是「围墙花园」：

- **Gatekeeper（门禁）**：默认只允许运行来自 App Store 与已认证开发者的应用，拦截未签名程序。
- **沙盒（Sandbox）**：App 运行在受限环境，无法随意访问整个系统。
- **T2 / Apple Silicon 安全芯片**：固件级信任链与磁盘加密（FileVault）。

**FileVault** 全盘加密值得开启：系统设置 → 隐私与安全性 → FileVault → 开启。它加密整个磁盘，笔记本丢失时数据不会泄露。<span class="marginnote">FileVault 开启后务必记住「恢复密钥」（可存进 iCloud 或抄下来）。忘记密码又丢了恢复密钥，数据就真的打不开了——安全与便利的天平，记得往安全这边留一根保险丝。</span>

## 5 核心对比表：三平台安全防线

| 防线 | Windows | Linux | macOS |
| --- | --- | --- | --- |
| 杀毒 | Defender | 少见，需时装 clamav | XProtect |
| 防火墙 | Defender 防火墙 | `ufw`/`firewalld` | 内置防火墙 |
| 权限控制 | UAC 用户账户控制 | sudo + 文件权限 | 沙盒 + 门禁 |
| 磁盘加密 | BitLocker | LUKS | FileVault |
| 更新 | Windows 更新 | 包管理器 | 软件更新 |

## 6 动手：一次系统安全体检

**第一步：确认更新。** 三平台都把系统更新到最新。

**第二步：检查杀毒与防火墙。** Windows 打开 Windows 安全中心确认 Defender 开启、防火墙启用；Linux `sudo ufw status`；macOS 确认 FileVault 开启。

**第三步：扫一次毒。** Windows 安全中心 → 病毒和威胁防护 → 快速扫描/完整扫描；macOS 若装第三方可用其扫描。

**第四步：检查账号。** 管理员账户少用，日常用标准账户；开启密码 + 2FA。

**第五步：审视习惯。** 不双击来路不明的附件、不装破解软件、不从第三方下载站装软件。

## 7 速查表：常见安全操作

| 操作 | 命令/路径 |
| --- | --- |
| Windows 安全中心 | 设置 → 隐私和安全性 → Windows 安全中心 |
| Windows 快速扫描 | Windows 安全中心 → 病毒和威胁防护 |
| Linux 更新 | `sudo apt update && sudo apt upgrade` |
| Linux 防火墙 | `sudo ufw status` / `sudo ufw enable` |
| Linux 查 SELinux | `getenforce` |
| macOS 加密 | 系统设置 → 隐私与安全性 → FileVault |
| 离线备份 | 见《系统备份与还原》 |

## 8 小结

- 主流威胁：**病毒、木马、蠕虫、勒索软件、挖矿木马、钓鱼**；勒索软件靠离线备份防御。
- Windows 内置 **Defender + 防火墙 + SmartScreen**，无需第三方杀毒。
- 安全第一层是**补丁 + 账号 + 习惯**，不是杀毒软件本身。
- Linux 靠**最小权限 + sudo + SELinux/AppArmor**；公网服务器防挖矿与爆破。
- macOS 靠**门禁 + 沙盒 + FileVault**，开 FileVault 记得留恢复密钥。
- 安全体检五步：**更新、杀毒、防火墙、账号 2FA、审视习惯**。

在下一节，我们回到硬件本体——**电脑除尘与硬件日常保养**，用物理手段让电脑长寿。
