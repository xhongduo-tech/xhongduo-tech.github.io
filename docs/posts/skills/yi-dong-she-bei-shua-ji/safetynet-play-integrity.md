---
title: SafetyNet 与 Play Integrity 检测原理与应对
date: 2026-08-07
---

# SafetyNet 与 Play Integrity 检测原理与应对

<div class="epigraph">
<p>Google 在手机上装了一台「测谎仪」——它检测的不是你刷没刷机，而是这台设备「能不能被信任」。</p>
<footer>—— Android 安全体系解读（Android 刷机社区资料）</footer>
</div>

<div class="article-byline">
<p>生活技能树 · 移动设备系统定制与刷机 ｜ Android 刷机社区资料 ｜ 2026-08-07</p>
</div>

## 为什么 Root 用户绕不开 Play Integrity

刷机、解锁、Root 之后，很多用户会发现一个现象：**银行应用打不开、支付功能被禁用、游戏拒绝运行**。原因不是应用「歧视」刷机党，而是它们调用了 Google 的设备完整性检测——早期的 **SafetyNet**，现在的 **Play Integrity API**。这一篇讲清楚：Google 到底在测什么、三种判定各代表什么、刷机用户有哪些合规/不合规的应对手段。**先说清楚伦理底线：应对检测的讨论仅用于学习与自用设备，不用于绕过付费、欺诈或恶意场景。** 从检测机制看，Play Integrity 的「硬件级证明不可伪造」，正是第三级《计算机安全》与《密码学与信息安全》里「非对称签名 + 硬件信任根」的工程应用。

## 1 SafetyNet 到 Play Integrity：Google 的完整性检测演化

**SafetyNet** 是 Google 早期的完整性验证体系，提供「设备完整性」与「CTS 认证」等判定，被大量银行、支付、游戏应用调用。它的核心是一个「兼容性测试」概念：设备是否满足 Google 的兼容定义（CTS），以及是否被篡改。

**Play Integrity API** 是 SafetyNet 的**后继者**，Google 已逐步停用 SafetyNet，转向 Play Integrity。两者的目标一致——**给应用开发者一个「这台设备是否可信」的判定**——但 Play Integrity 更现代、更严格：引入硬件级密钥认证（STRONG 档）、动态更新检测规则。

**检测的调用方式**：应用通过 Google Play 服务调用 Play Integrity API，由 Google 云端完成判定，返回一个或多个判定结果。**检测不在手机本地完成**——应用只能收到「通过/不通过」的结果，拿不到判定细节，这让「欺骗检测」变得更难。<span class="marginnote">Play Integrity 的「云端判定」是它比 SafetyNet 难对付的核心：<strong>本地可以掩盖 root 痕迹，但判定在 Google 服务器上，能看到设备上报的完整状态</strong>。刷机社区对抗的重点，也变成了「让设备上报的数据看起来干净」。</span>

## 2 Play Integrity 的三档判定：BASIC、DEVICE 与 STRONG

Play Integrity 返回的判定主要有三档，层层递进：

**MEETS_BASIC_INTEGRITY（基础完整性）**：设备通过了基本完整性检查——系统未被明显篡改、未被 root（通过常见 root 信号检测）。**这一档是「软件层面」的完整性**。

**MEETS_DEVICE_INTEGRITY（设备完整性）**：在基础之上，额外验证 **Bootloader 锁状态**与系统签名——**解锁的设备过不了这一档**。它检查的是「设备是否以官方认可的方式运行」。

**MEETS_STRONG_INTEGRITY（强完整性）**：最高档，要求**硬件级密钥认证（Key Attestation）**——用设备内置的硬件安全模块（TEE/安全元件）签发证明，确保证明不可伪造。**任何形式的解锁、改机、刷机都过不了这一档**，因为硬件级证据无法被软件伪造。<span class="marginnote">三档判定的「硬度」差别，是理解检测对抗的关键：<strong>BASIC 查软件痕迹（可掩盖），DEVICE 查锁状态（解锁即失败），STRONG 查硬件证明（软件无法伪造）</strong>。刷机社区能「应付」的主要是 BASIC 与部分 DEVICE 场景，STRONG 是硬件级的铜墙铁壁。</span>

**还有一档 MEETS_VIRTUAL_INTEGRITY**：设备运行在模拟器/虚拟机上，用于区分真机。

## 3 检测什么：解锁、Root 与系统指纹

Play Integrity 的检测项，对应刷机用户最关心的几个「雷区」：

**Bootloader 锁状态**：解锁设备上报「未锁定/已验证状态异常」，导致 DEVICE 档失败。**这是硬件层面的事实，刷回官方系统也改变不了解锁历史**（除非回锁且系统完全原版）。

**Root 痕迹**：su 二进制、Magisk 包名、Zygisk 注入痕迹、常见 root 管理器进程等。这些是**软件痕迹**，可以通过隐藏手段掩盖——但 Google 会不断更新检测规则。

**系统指纹**：系统签名、分区哈希、系统属性的异常（如「测试版」「已修改」标记）。**系统指纹是「这套系统是否原厂」的证据**。

**硬件证明**：Key Attestation 里包含 bootloader 状态、系统签名、ROM 指纹的硬件签名版本。**它由 TEE 签发，软件层改不掉**——STRONG 档因此几乎不可伪造。

**对刷机用户的影响梯度**：只解锁（原版系统）→ 可能过 BASIC、DEVICE 失败；解锁 + Root → BASIC 也可能失败；刷第三方 ROM → 三档全挂。**影响大小取决于应用要哪一档**——银行往往要 DEVICE/STRONG，普通应用只查 BASIC。

## 4 应对思路：Zygisk、DenyList 与 Shamiko

刷机社区针对检测的应对，核心目标是「让 BASIC 甚至部分 DEVICE 场景通过」。主流手段如下：

**Magisk 的 Zygisk + DenyList**：在 Zygisk 开启后，把要隐藏 root 的应用加入 **DenyList**——该应用启动时，Zygisk 对它隐藏 Magisk 痕迹（su、magiskd 等）。这是 Magisk 官方内置的隐藏手段。<span class="marginnote">DenyList 的思路是「按应用隐藏」：<strong>不是全局隐藏 root，而是对指定的检测应用抹掉 root 痕迹</strong>。这比「全局隐藏」更精准——普通应用照常能用 root，敏感应用看不到 root。代价是检测应用名单需要自己维护。</span>

**Shamiko 模块**：Magisk 社区开发的增强隐藏模块，在 DenyList 基础上进一步隐藏更深的痕迹（如 Magisk 的包名、挂载痕迹）。**它是 DenyList 的「加强版」**，需要配合 Zygisk 使用。

**「隐藏 + 合规」两手**：隐藏手段只能应对软件层检测，**面对 STRONG 硬件证明与「回锁后系统非原版」的组合场景，没有任何软件手段能过**。所以务实的策略是：**明确自己需要哪一档、评估哪些应用必须过检测、用 DenyList/Shamiko 精准应对软件层**，同时接受「硬件级检测过不了」的现实。

**重要提醒**：应对检测是**猫鼠游戏**——Google 持续更新规则，旧手段随时失效；且**用隐藏手段绕过应用的完整性要求可能违反服务条款**。仅建议用于自用设备的正常使用与学习，不要用于规避付费或恶意用途。

## 5 公式解析：判定结果的「与逻辑」

应用拿到 Play Integrity 判定后，是否放行，是一个「多条件与」的判断。以银行应用为例：

$$
\text{放行} \iff \text{MEETS\_BASIC} \;\land\; \text{MEETS\_DEVICE} \;\land\; \text{（无风险信号）}
$$

逐步拆解：

- **MEETS_BASIC**：软件层完整性通过——Root 痕迹被隐藏后可能满足。
- **MEETS_DEVICE**：锁状态通过——**解锁设备基本过不了这一项**，除非完全回锁且原版系统。
- **无风险信号**：Google 还会综合设备上报的其他信号（如异常的环境、hook 框架）。
- **$\land$（与逻辑）**：**所有条件同时成立才放行**，任一不满足即拒绝。所以「我隐藏了 root，为什么还不行？」——因为可能栽在 DEVICE 或风险信号上。

这个公式解释了两个常见困惑：**为什么隐藏了 root 银行应用仍拒绝**（多半栽在锁状态），以及**为什么「只解锁不改机」反而比「解锁+Root」更可能通过 BASIC**（改得越少，软件痕迹越少）。

## 6 核心要点：检测项与应对对照表

| 检测项 | 对应判定 | 应对手段 | 可逆性 |
| --- | --- | --- | --- |
| Root 软件痕迹 | BASIC | Zygisk + DenyList | 可隐藏 |
| Root 深层痕迹 | BASIC | Shamiko | 可隐藏 |
| 系统签名/指纹 | BASIC/DEVICE | 刷回原版系统 | 可恢复 |
| Bootloader 锁状态 | DEVICE | 回锁（需原版系统） | 有历史痕迹 |
| 硬件 Key Attestation | STRONG | **无法伪造** | 硬件级 |
| 模拟器环境 | VIRTUAL | 无（真机正常） | 无需应对 |

## 7 术语速查表

| 术语 | 含义 | 关键点 |
| --- | --- | --- |
| SafetyNet | 早期完整性检测 | 已被 Play Integrity 取代 |
| Play Integrity | 现代完整性 API | 云端判定 |
| MEETS_BASIC_INTEGRITY | 基础完整性 | 查软件痕迹 |
| MEETS_DEVICE_INTEGRITY | 设备完整性 | 查锁状态 |
| MEETS_STRONG_INTEGRITY | 强完整性 | 硬件证明 |
| Zygisk | 进程注入框架 | 隐藏的基础 |
| DenyList | 按应用隐藏 root | Magisk 官方 |
| Shamiko | 深度隐藏模块 | DenyList 加强版 |
| Key Attestation | 硬件级密钥认证 | 不可伪造 |
| 云端判定 | Google 服务器判定 | 本地难欺骗 |

## 8 快速自查清单

应对检测前，逐条确认：

- 我要让**哪些应用**通过检测？它们要求**哪一档**判定？
- 设备当前是**解锁还是已回锁**？锁状态决定 DEVICE 档是否可能通过。
- 隐藏手段（Zygisk + DenyList / Shamiko）与 Magisk 版本是否**兼容**？
- 是否接受「**STRONG 档无法伪造**」的现实，并据此降低预期？
- 使用场景是否**合规**——仅自用与学习，不用于规避付费或恶意用途？

## 9 小结

- Play Integrity 是 SafetyNet 的后继者，**云端判定、本地难欺骗**。
- 三档判定：**BASIC（查 root 痕迹）、DEVICE（查锁状态）、STRONG（硬件证明）**——硬度递增，可伪造性递减。
- 检测项集中四类：**root 痕迹、系统指纹、锁状态、硬件证明**；前两者可隐藏，后两者基本不可绕。
- 应对手段：**Zygisk + DenyList / Shamiko 按应用隐藏**，只对软件层有效。
- 务实策略：**先搞清要哪一档，精准应对软件层，接受硬件级检测过不了的现实**。

在下一节，我们进入本专题第四篇：**ROM 定制与系统备份恢复**。第一篇：**常见 ROM 类型——官方、类原生与第三方**。
