---
title: 定制 ROM 刷写流程与前置条件
date: 2026-08-07
---

# 定制 ROM 刷写流程与前置条件

<div class="epigraph">
<p>刷第三方 ROM 不是「点一下装系统」，而是「把设备的信任边界整个换掉」——每一步前置条件，都是为这个换掉的过程兜底。</p>
<footer>—— 刷机社区（《手机刷机与系统定制》第5章）</footer>
</div>

<div class="article-byline">
<p>生活技能树 · 移动设备系统定制与刷机 ｜ 《手机刷机与系统定制》第5章 ｜ 2026-08-07</p>
</div>

## 为什么刷第三方 ROM 要讲究「流程」

前面几篇把解锁、TWRP、Magisk、ROM 类型都讲透了，这一篇把它们**串成一条完整流程**：从前置条件到首次开机。刷第三方 ROM 的失败，绝大多数不是「某一步不会」，而是「前置条件没备齐」或「清除分区选错了」——比如没四清就跨版本刷、刷完 GApps 顺序不对导致卡机。这一篇给出可照做的标准流程，以及每一步「为什么这样做」的解释。整个流程里「先备份、再动手、留底牌」的节奏，与相邻专题《手机维修》的维修作业规范如出一辙。

## 1 前置条件清单：解锁、TWRP、备份

刷第三方 ROM 之前，必须确认以下**前置条件全部满足**：

**① Bootloader 已解锁**。第三方 ROM 未签名，锁定设备无法启动。这是最硬的前提——没解锁，后面全免谈。

**② TWRP 已装好且可用**。第三方 ROM 以卡刷包（zip）分发，需要 TWRP 执行。确认 TWRP 版本匹配设备与目标系统版本，`fastboot boot twrp.img` 临时启动也能用，但「永久刷入」更省心。

**③ 数据已备份且验证可恢复**。刷第三方 ROM 通常要清空 `data`（见下节）。Nandroid 备份 + EFS 备份 + 用户数据导出，三层备份缺一不可。<span class="marginnote">前置条件里最容易「省」的是备份，最不该省的也是备份：<strong>刷 ROM 的清 data 是不可逆的，而备份只花几分钟</strong>。社区里「刷完 ROM 发现聊天记录没了」的帖子，几乎都源于「当时觉得不用备份」。别当那个帖子的主角。</span>

**④ 刷机包齐备且匹配**。目标 ROM 卡刷包、对应 GApps 包（需要 Google 服务的话）、对应机型与系统版本的 TWRP、官方线刷包（救砖底牌）。**全部核对机型与版本**。

**⑤ 硬件前提**。电量 ≥ 50%（刷机中途断电是变砖主因）、数据线可靠、电脑驱动就绪。

## 2 标准刷写流程：从四清到刷入

确认前置条件后，按以下步骤执行：

**第一步，进 TWRP**：`adb reboot recovery` 或按键进入，确认 TWRP 正常加载。

**第二步，四清（Clean Flash 的标准动作）**：TWRP 的 `Wipe → Advanced Wipe`，勾选 **Dalvik/ART Cache、Data、Cache、System** 四项清除。**注意：不要勾 Internal Storage**（会清掉相册等个人文件，除非你想全清）。<span class="marginnote">四清 vs 双清的差别在 System：<strong>双清（dalvik+data）不清系统，用于「重装同一个系统」；四清（加 system）把旧系统抹掉，用于「换不同系统」</strong>。跨 ROM 类型（官方→类原生）不清 system，残留的旧系统文件会与新系统打架。</span>

**第三步，刷入 ROM 包**：`Install` 选择 ROM 的 zip，滑动刷入。刷完**先不要急着重启**——通常还要刷 GApps。

**第四步，刷入 GApps（按需）**：需要 Google 服务就接着 `Install` GApps 包。**顺序必须是「先 ROM 后 GApps」**，颠倒会导致系统组件冲突。

**第五步，重启**：`Reboot → System`。首次开机比平时慢（系统初始化），耐心等待；若超过 10 分钟仍卡 Logo，多半是刷写或清除有问题。

**第六步，首次设置**：进入系统后按向导设置。若之前恢复了 Nandroid 的 data，可能需要处理屏锁与 FRP。

## 3 首刷与升级：clean flash 与 dirty flash

刷机社区的两个高频词：**Clean Flash** 与 **Dirty Flash**。

**Clean Flash（干净刷写）**：清 data（+cache+dalvik，视情况加 system）后刷入。**任何「换 ROM 类型」「大版本升级」「跨 Android 大版本」都必须 clean flash**——旧系统残留会引发 FC、设置冲突甚至无法开机。

**Dirty Flash（不清理刷写）**：不清 data，直接在现有系统上刷入新包（通常只清 cache/dalvik）。适用于**同一 ROM 的小版本升级**——保留数据与设置，升级快速。**风险**：版本跨度大时，dirty flash 会留下数据库/配置不兼容的坑。<span class="marginnote">判断该 clean 还是 dirty，问一个问题：<strong>新旧版本的数据格式兼容吗？</strong>同 ROM 小升级（兼容）→ dirty；换 ROM、大版本（不兼容）→ 必须 clean。拿不准就 clean——dirty 刷出问题再 clean，等于白折腾两遍。</span>

**dirty flash 的正确姿势**：TWRP 里只清 `Dalvik/ART Cache + Cache`，然后刷入新 ROM 包（+ 若 GApps 变化也重刷），重启。**别在 dirty flash 时清 data 或 system**——那就不是 dirty 了。

## 4 常见问题排查：bootloop 与功能异常

刷完第三方 ROM，最常遇到三类问题：

**卡 Logo / Bootloop**：反复重启或停在开机动画。排查顺序：**是否四清干净**（跨版本没清 system/data → 先回 TWRP 四清重刷）→ **ROM 包是否与机型匹配** → **内核/补丁是否冲突** → 仍不行则恢复 Nandroid 或刷官方包。

**无信号 / 相机崩溃 / 指纹失效**：硬件适配问题。第三方 ROM 对部分硬件的 HAL 支持不完整。排查：**看 ROM 的已知问题列表**（Known Issues）→ 刷对应内核/固件补丁 → 实在不行换 ROM 或回官方。

**设置强制关闭（FC）**：多半是**数据冲突**（dirty flash 跨版本）或 **GApps 版本不匹配**。排查：先备份数据，清掉出问题应用的缓存/数据；不行则 clean flash。<span class="marginnote">排查问题的总原则是「从数据到系统逐步升级」：<strong>先清应用数据（最轻）→ 再 dirty flash（保留设置）→ 再 clean flash（全清重来）→ 最后恢复备份/官方包</strong>。每一步都能回退，别一上来就全清。</span>

**排查的基本功**：学会抓 **logcat**（`adb logcat` 抓系统日志）与看 **last_kmsg/dmesg**（内核日志）。日志里的红色 `FATAL`/`E/` 行，是定位问题的最快线索——**「读日志」是把刷机从「试运气」变成「工程化」的分水岭**。

## 5 公式解析：刷写前「清什么」的决策

刷 ROM 前到底清哪些分区，可以用一个决策链表达：

$$
\text{同 ROM 小升级} \rightarrow \text{清 cache+dalvik（dirty）}; \qquad \text{否则} \rightarrow \text{清 data+cache+dalvik（+system）（clean）}
$$

逐步拆解：

- **同 ROM 小升级**：数据格式兼容，保留 data 即可，只清运行缓存（cache/dalvik）。
- **换 ROM 或大版本**：数据格式可能不兼容，必须清 data；跨 ROM 类型（官方↔类原生）还要加清 system，抹掉旧系统残留。
- **Internal Storage（个人文件）**：**默认不清**——它不属于「系统数据」，四清也不包含它。
- **判据**：**「数据格式是否兼容」决定 data 清不清，「旧系统残留会不会打架」决定 system 清不清**。

这个决策链帮你在任何刷机场景下快速确定「清什么」，避免「该清没清」与「不该清清掉」。

## 6 核心要点：刷写流程检查清单

| 阶段 | 动作 | 要点 |
| --- | --- | --- |
| 前置 | 确认解锁 | 未解锁免谈 |
| 前置 | TWRP 就绪 | 版本匹配 |
| 前置 | 三层备份 | Nandroid + EFS + 数据 |
| 刷写 | 四清/双清 | 按「数据格式兼容」决策 |
| 刷写 | 刷 ROM | 机型匹配 |
| 刷写 | 刷 GApps | 先 ROM 后 GApps |
| 重启 | 首启等待 | 超时卡 Logo 查清除 |
| 升级 | clean/dirty 选择 | 拿不准就 clean |
| 排查 | 读日志 | logcat/dmesg |

## 7 术语速查表

| 术语 | 含义 | 关键点 |
| --- | --- | --- |
| Clean Flash | 清 data 刷入 | 换系统必用 |
| Dirty Flash | 不清 data 刷入 | 同 ROM 小升级 |
| 四清 | dalvik+data+cache+system | 跨类型换系统 |
| 双清 | dalvik+data | 重装同系统 |
| Internal Storage | 内部存储分区 | 默认不清 |
| bootloop | 开机循环 | 排查四清/机型 |
| logcat | 系统日志 | 排错基本功 |
| dmesg | 内核日志 | 看内核崩溃 |
| GApps | Google 服务包 | 先 ROM 后刷 |
| Known Issues | ROM 已知问题 | 刷前必看 |

## 8 快速自查清单

刷第三方 ROM 前，逐条确认：

- **解锁、TWRP、备份**三项前置条件是否全部满足？
- ROM 与 GApps 包是否**机型与版本匹配**？Known Issues 看了吗？
- 这次是 **clean 还是 dirty**？该清的分区选对没有（没误勾 Internal Storage）？
- 刷入顺序是**先 ROM 后 GApps**？
- 救砖底牌（官方线刷包 + Nandroid）是否在手边？

## 9 小结

- 刷第三方 ROM 的五项前置：**解锁、TWRP、备份、包匹配、硬件前提**。
- 标准流程：**进 TWRP → 四清 → 刷 ROM → 刷 GApps（先 ROM 后 GApps）→ 重启 → 首设**。
- **clean flash** 用于换系统/大版本（清 data+system），**dirty flash** 用于同 ROM 小升级（只清 cache/dalvik），拿不准就 clean。
- 排查三连：**卡 Logo 查四清与机型、功能异常查 Known Issues、FC 查数据冲突**；用 logcat/dmesg 读日志。
- 「清什么」的决策链：**数据格式兼容决定 data，旧系统残留决定 system**。

在下一节，我们把定制 ROM 的「细节」补齐：**GApps、系统签名与本地化深度定制**。
